"""
TUNABLE CONFIG
--------------
All the knobs you may want to tweak are consolidated under the "Config" section
~50 lines below. Look for the block with:
    # =================== Config ===================
and within it, see:

- Market/labels: MU_MODE, FORECAST_ALPHA, YF_PERIOD, YF_INTERVAL, FORECAST_HORIZON_DAYS
- Risk/optimizers: RISK_FREE, GAMMA, MAXITER, REPS (QAOA)
- QNN knobs: FEATURES_N (qubits), QNN_FM_REPS, QNN_ANSATZ_REPS, QNN_TRAIN_ITERS,
             QNN_NOISE_SIGMA, QNN_CLIP_MIN, QNN_CLIP_MAX
"""
# app.py — QAOA + Markowitz + QNN next-month forecaster
# - Fixes inflated risk via unit sanity (kept)
# - Adds Markowitz benchmark (kept)
# - Adds QNN forecast for next-month price per asset and portfolio value paths
#
# Run:
#   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
#
import time, math, json, random, logging, inspect, os
from typing import List, Optional, Dict, Any, Tuple

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, ValidationError

# ---- Qiskit (core) ----
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
try:
    # --- MODIFIED: Use standard, faster, shot-based primitives ---
    from qiskit.primitives import Estimator
    from qiskit.primitives import Sampler
except ImportError as e:
    raise ImportError("Please install Qiskit primitives:\n  pip install 'qiskit>=1.1' qiskit-algorithms") from e

# ---- Optional Qiskit Machine Learning (QNN) + classical fallback ----
QML_AVAILABLE = True
try:
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_algorithms.optimizers import COBYLA as COBYLA_QNN
except Exception:
    QML_AVAILABLE = False

# ---- Optional market data ----
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

# ---- Classical fallback model ----
try:
    from sklearn.linear_model import Ridge
except Exception:
    Ridge = None

SUPABASE_CALLBACK_BEARER = os.getenv("SUPABASE_CALLBACK_BEARER", "")
SUPABASE_FUNCTION_SECRET = os.getenv("SUPABASE_FUNCTION_SECRET", "")

# =================== Config ===================
MU_MODE = "blend"          # "historical" | "forecast" | "blend"
FORECAST_ALPHA = 0.6       # weight on forecast when MU_MODE="blend" (0..1)

USE_YFINANCE = True
PREFER_INPUT_STATS = True
YF_PERIOD = "5y"
YF_INTERVAL = "1mo"
RISK_FREE = 4.3            # % for Sharpe (percent units throughout)
GAMMA = 0.6                # QAOA risk-aversion
MAXITER = 200
REPS = 3                   # QAOA depth
FORECAST_HORIZON_DAYS = 20 # ~1 trading month
FEATURES_N = 5             # QNN features

# ---- Tunable QNN knobs ----
QNN_FM_REPS = 10            # Feature map depth (re-upload features)
QNN_ANSATZ_REPS = 3         # Variational ansatz depth
QNN_TRAIN_ITERS = 120       # Random-search iterations for weights
QNN_NOISE_SIGMA = 0.05      # Initial perturbation scale for random search
QNN_CLIP_MIN = -0.5         # Clip lower bound for 20d return prediction (decimal)
QNN_CLIP_MAX = 0.5          # Clip upper bound for 20d return prediction (decimal)

# ---- Tunable QAOA knobs ----
QAOA_REPS = REPS             # QAOA circuit depth (can override REPS here)
QAOA_MAXITER = MAXITER       # optimizer iterations (can override MAXITER here)
QAOA_RANDOM_RESTARTS = 5     # run QAOA multiple times with different initial points, pick best
QAOA_USE_SAMPLER = True      # try to read bitstring distribution from result metadata
QAOA_SHOTS = 2048            # if a real Sampler backend is used (ignored for StatevectorSampler)
QAOA_WEIGHTS_MODE = "risk_weighted"   # "risk_weighted" | "equal_weight" | "markowitz_subset"
QAOA_K_CARDINALITY = 14       # desired # of selected assets (soft via Step 3 fallback)
QAOA_LAMBDA_K = 0.9          # (unused in this version; penalty happens via Step 3 fallback)

# You can also increase FEATURES_N (qubits) but prefer depth first.

# =================== Models (request unchanged; response extended) ===================
class Asset(BaseModel):
    asset_id: str
    name: str
    expected_return: Optional[float] = None  # % annual
    risk: Optional[float] = None             # % annual stdev
    current_allocation: float

# -------- Step 2: sensible allocation cap defaults --------
class Constraints(BaseModel):
    # Allocation caps in percent; enforce breadth via caps + cardinality
    min_allocation_per_asset: float = 0.0
    max_allocation_per_asset: float = 100.0
    target_metric: str = "sharpe"
# -----------------------------------------------------------

class Portfolio(BaseModel):
    portfolio_id: str
    name: str
    assets: List[Asset]
    constraints: Constraints

class OptimizationRequest(BaseModel):
    optimization_id: str
    callback_url: Optional[str] = "https://ormhkjdakokdoprwahvo.supabase.co/functions/v1/receive-optimization-results"
    portfolio: Portfolio

class AllocationOut(BaseModel):
    asset_id: str
    optimized_allocation: float

class MetricsOut(BaseModel):
    expected_return: float
    portfolio_risk: float
    sharpe_ratio: float

class AlgorithmInfoOut(BaseModel):
    method: str
    iterations: int
    convergence_time_seconds: float

class ResultsOut(BaseModel):
    optimized_allocations: List[AllocationOut]
    metrics: MetricsOut
    algorithm_info: AlgorithmInfoOut

class ComparisonOut(BaseModel):
    qaoa_sharpe: float
    markowitz_sharpe: float
    winner: str  # "QAOA" | "Markowitz" | "Tie"

# ---- Forecast output models ----
class AssetForecastOut(BaseModel):
    asset_id: str
    ticker: str
    current_price: float
    predicted_20d_return_pct: float
    next_month_price_estimate: float

class PathPointOut(BaseModel):
    date: str
    est_value: float

class PortfolioPathsOut(BaseModel):
    qaoa: List[PathPointOut]
    markowitz: List[PathPointOut]

class ForecastOut(BaseModel):
    per_asset: List[AssetForecastOut]
    portfolio_paths: PortfolioPathsOut


class CorrelationMatrixOut(BaseModel):
    tickers: List[str]
    matrix: List[List[float]]  # row-major; matrix[i][j] = corr(tickers[i], tickers[j])

class OptimizationResponse(BaseModel):
    optimization_id: str
    status: str                 # "completed" | "failed"
    progress: int
    results: Optional[ResultsOut] = None            # QAOA
    benchmark: Optional[ResultsOut] = None          # Markowitz
    comparison: Optional[ComparisonOut] = None
    forecast: Optional[ForecastOut] = None
    correlation_matrix: Optional[CorrelationMatrixOut] = None
    error_message: Optional[str] = None

# =================== App + CORS ===================
app = FastAPI(title="QAOA + Markowitz + QNN Forecast (Demo + yfinance)")
log = logging.getLogger("qaoa_demo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
random.seed(42); np.random.seed(42)

LAST_RESULT: dict | None = None
LAST_CALLBACK: dict | None = None

# --- NEW: Add a cache for QNN predictions ---
QNN_CACHE: Dict[str, Tuple[float, float]] = {}

def _excerpt(s: str, n: int = 300) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "…"

# permissive CORS (demo)
@app.middleware("http")
async def permissive_cors(request: Request, call_next):
    origin = request.headers.get("origin", "*")
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": request.headers.get("access-control-request-method", "*") or "*",
                "Access-Control-Allow-Headers": request.headers.get("access-control-request-headers", "*") or "*",
                "Access-Control-Max-Age": "86400",
                "Vary": "Origin",
            },
        )
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = origin
    resp.headers["Access-Control-Allow-Methods"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Vary"] = "Origin"
    return resp

@app.options("/optimize")
async def options_optimize():
    return Response(status_code=200)

# =================== Data helpers (yfinance + stats) ===================
def map_name_to_ticker(name: str) -> str:
    n = name.upper()
    if n in ["GOLD", "XAU", "XAUUSD", "GLD"]:
        return "GLD"
    if n in ["BTC", "BTCUSD", "BITCOIN"]:
        return "BTC-USD"
    return name  # assume it is a ticker

def _collect_pred20d_map(tickers: list[str], opt_id: str = "job") -> dict[str, tuple[float, float]]:
    """
    Returns {ticker: (pred20d_decimal, last_price)} for each ticker.
    Never raises; missing/failed tickers get (0.0, 100.0 or last known).
    """
    out = {}
    n = len(tickers)
    # --- NEW: Progress Log ---
    log.info(f"[{opt_id}] --- (5%) Starting QNN forecast for {n} assets... ---")
    for i, tk in enumerate(tickers):
        try:
            # --- NEW: Progress Log (inside loop) ---
            # Calculate progress: Assume this step is ~20% of the total job (from 5% to 25%)
            progress_pct = 5.0 + (float(i) / float(n)) * 20.0
            log.info(f"[{opt_id}] ({progress_pct:.1f}%) Forecasting {tk} ({i+1}/{n})...")
            
            yhat, p = _predict_qnn_return(tk)  # already robust
        except Exception:
            yhat, p = 0.0, 100.0
        out[tk] = (float(yhat), float(p))
        
    # --- NEW: Progress Log ---
    log.info(f"[{opt_id}] --- (25%) QNN forecasting complete. ---")
    return out

def _annualize_20d_return_pct(r20_dec: float) -> float:
    # Kept for backward-compatibility: now delegates to interval-aware version
    return _annualize_pred_horizon(r20_dec, YF_INTERVAL, FORECAST_HORIZON_DAYS)


def _periods_per_year(interval: str) -> float:
    """Return periods-per-year used for annualization based on yfinance interval."""
    m = (interval or "").lower()
    if m == "1d":
        return 252.0
    if m == "5d":
        return 252.0 / 5.0
    if m in ("1wk", "1w"):
        return 52.0
    if m in ("1mo", "1m"):
        return 12.0
    # Fallback: assume daily-ish
    return 252.0

def _horizon_steps_for_interval(interval: str, horizon_days: int) -> int:
    """
    Convert a 'days' horizon into steps for the chosen interval.
    For monthly data we predict next-month (1 step). For 5d/weekly we scale accordingly.
    """
    m = (interval or "").lower()
    if m == "1d":
        return int(max(1, horizon_days))
    if m == "5d":
        return int(max(1, round(horizon_days / 5.0)))
    if m in ("1wk", "1w"):
        return int(max(1, round(horizon_days / 5.0)))  # ~5 trading days per week
    if m in ("1mo", "1m"):
        return 1  # next month
    return int(max(1, horizon_days))

def _timedelta_step_for_interval(interval: str):
    """Rough step size for forecast path dates based on interval."""
    from datetime import timedelta
    m = (interval or "").lower()
    if m == "1d":
        return timedelta(days=1)
    if m == "5d":
        return timedelta(days=5)
    if m in ("1wk", "1w"):
        return timedelta(days=7)
    if m in ("1mo", "1m"):
        return timedelta(days=30)  # coarse monthly step
    return timedelta(days=1)

def annualize_from_interval(returns: pd.Series, interval: str) -> Tuple[float, float]:
    """Annualize mean/stdev from arbitrary interval returns to percent units."""
    mu_p = returns.mean()
    sigma_p = returns.std()
    t = _periods_per_year(interval)
    mu_a = ((1.0 + mu_p) ** t - 1.0) * 100.0     # %
    sigma_a = sigma_p * math.sqrt(t) * 100.0     # %
    return float(mu_a), float(sigma_a)

def _annualize_pred_horizon(r_h_dec: float, interval: str, horizon_days: int) -> float:
    """
    Annualize a predicted return over 'horizon' steps, where steps depend on interval.
    Returns percent units.
    """
    steps = _horizon_steps_for_interval(interval, horizon_days)
    per_year = _periods_per_year(interval)
    if steps <= 0:
        steps = 1
    if per_year <= 0:
        per_year = 252.0
    return ((1.0 + float(r_h_dec)) ** (float(per_year) / float(steps)) - 1.0) * 100.0

def fetch_mu_sigma_from_yf(ticker: str) -> Tuple[float, float]:
    df = yf.download(ticker, period=YF_PERIOD, interval=YF_INTERVAL, auto_adjust=True, progress=False)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No data for {ticker}")
    rets = df["Close"].pct_change().dropna()
    if len(rets) < 2:
        raise ValueError(f"Not enough data for {ticker}")
    return annualize_from_interval(rets, YF_INTERVAL)
def _to_1d_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Robustly extract a 1-D float Series of Close prices from a yfinance DataFrame.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" in df.columns:
        s = df["Close"]
    else:
        s = None
        try:
            cols = df.columns
            if isinstance(cols, pd.MultiIndex):
                close_cols = [c for c in cols if (isinstance(c, tuple) and c[0] == "Close")]
                if close_cols:
                    s = df[close_cols[0]]
        except Exception:
            s = None
        if s is None:
            return pd.Series(dtype=float)
    if isinstance(s, pd.DataFrame):
        if s.shape[1] >= 1:
            s = s.iloc[:, 0]
        else:
            return pd.Series(dtype=float)
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    try:
        s = s.astype(float)
    except Exception:
        s = pd.to_numeric(s, errors="coerce")
    return s.dropna()

def _series_pct_change_close(ticker: str, period: str, interval: str) -> pd.Series:
    """Safe daily return series from Close; always 1-D Series or empty."""
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception:
        return pd.Series(dtype=float)
    s = _to_1d_close_series(df)
    if s.empty:
        return pd.Series(dtype=float)
    rets = s.pct_change().dropna()
    return rets if len(rets) >= 2 else pd.Series(dtype=float)

def _series_close(ticker: str, period: str, interval: str) -> pd.Series:
    """Safe close-price series fetcher → always 1-D Series or empty."""
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception:
        return pd.Series(dtype=float)
    return _to_1d_close_series(df)

def fetch_covariance_from_yf(tickers: List[str]) -> np.ndarray:
    series_list, names = [], []
    for tk in tickers:
        s = _series_pct_change_close(tk, YF_PERIOD, YF_INTERVAL)
        if not s.empty:
            series_list.append(s)
            names.append(tk)
        else:
            log.warning(f"yfinance: no usable returns for {tk}; skipping in covariance")
    if len(series_list) == 0:
        raise ValueError("No usable return series for covariance")
    returns = pd.concat(series_list, axis=1, join="inner")
    returns.columns = names
    if returns.empty or returns.shape[0] < 2:
        raise ValueError("Insufficient overlapping data to compute covariance")
    t = _periods_per_year(YF_INTERVAL)
    cov_period = returns.cov(min_periods=2)              # decimal^2 per interval
    cov_annual_dec = cov_period * float(t)                # decimal^2 (annual)
    cov_annual_pct2 = cov_annual_dec * (100.0 ** 2)       # percent^2 (annual)
    if not np.isfinite(cov_annual_pct2.values).all():
            raise ValueError("Non-finite covariance")
    log.info(f"covariance: rows={returns.shape[0]} cols={returns.shape[1]} tickers={list(returns.columns)}")
    return cov_annual_pct2.values

# =================== QUBO + postprocessing ===================
def build_sigma_diag(sigma_pct: np.ndarray) -> np.ndarray:
    return np.diag(np.maximum(1e-10, sigma_pct ** 2))

def build_qubo(mu: np.ndarray, Sigma: np.ndarray, gamma: float = GAMMA,
               k_cardinality: int | None = QAOA_K_CARDINALITY,
               lambda_k: float = QAOA_LAMBDA_K) -> SparsePauliOp:
    """
    NOTE: This version encodes -mu^T z + gamma * z^T Sigma z.
    (Cardinality penalty is enforced via Step 3 fallback, not here.)
    """
    n = len(mu)
    terms: Dict[str, float] = {}
    const = -(0.5) * float(np.sum(mu))  # from -mu^T z

    for i in range(n):
        lbl = ''.join('Z' if j == i else 'I' for j in range(n))
        terms[lbl] = terms.get(lbl, 0.0) + 0.5 * mu[i]

    for i in range(n):
        for j in range(n):
            c = gamma * Sigma[i, j] / 4.0
            const += c
            li = ''.join('Z' if k == i else 'I' for k in range(n))
            terms[li] = terms.get(li, 0.0) - c
            if j != i:
                lj = ''.join('Z' if k == j else 'I' for k in range(n))
                terms[lj] = terms.get(lj, 0.0) - c
            lij = ''.join('Z' if (k == i or k == j) else 'I' for k in range(n))
            terms[lij] = terms.get(lij, 0.0) + c

    terms['I' * n] = terms.get('I' * n, 0.0) + const
    labels = list(terms.keys())
    coeffs = np.array(list(terms.values()), dtype=float)
    return SparsePauliOp(labels, coeffs)

def brute_best_bits(mu: np.ndarray, Sigma: np.ndarray, gamma: float) -> np.ndarray:
    n = len(mu)
    best_val, best = float("inf"), np.zeros(n)
    for k in range(1 << n):
        z = np.array([(k >> i) & 1 for i in range(n)], dtype=float)  # little-endian
        val = -float(np.dot(mu, z)) + gamma * float(np.dot(z, Sigma @ z))
        if val < best_val:
            best_val, best = val, z
    return best

def _clamp_box_and_sum_to_one(w: np.ndarray, min_pct: float, max_pct: float) -> np.ndarray:
    w = np.clip(w, min_pct, max_pct)
    s = w.sum()
    if s <= 0:
        n = len(w)
        w[:] = max(min_pct, 0.0)
        left = 100.0 - n * w[0]
        if left > 0 and max_pct > w[0]:
            w += (left / n)
        return w
    return w / s * 100.0

def weights_from_bits(mu: np.ndarray, sigma: np.ndarray, bits: np.ndarray,
                      min_pct: float, max_pct: float) -> np.ndarray:
    idx = np.where(bits > 0.5)[0]
    if len(idx) == 0:
        score = np.divide(mu, np.maximum(1e-8, sigma ** 2))
        idx = [int(np.argmax(score))]
    scores = np.zeros_like(mu)
    scores[idx] = np.maximum(1e-6, mu[idx] / np.maximum(1e-6, sigma[idx] ** 2))
    w = scores / scores.sum() * 100.0
    w = np.clip(w, min_pct, max_pct)
    return (w / w.sum()) * 100.0 if w.sum() > 0 else w

def weights_from_bits_mode(mu: np.ndarray, sigma: np.ndarray, bits: np.ndarray,
                           min_pct: float, max_pct: float,
                           Sigma_pct2: np.ndarray | None = None,
                           rf_pct: float = RISK_FREE) -> np.ndarray:
    mode = (QAOA_WEIGHTS_MODE or "risk_weighted").lower()
    idx = np.where(bits > 0.5)[0]
    if mode == "equal_weight":
        if len(idx) == 0:
            return _clamp_box_and_sum_to_one(np.ones_like(mu) * (100.0/len(mu)), min_pct, max_pct)
        w = np.zeros_like(mu, dtype=float); w[idx] = 100.0 / len(idx)
        return _clamp_box_and_sum_to_one(w, min_pct, max_pct)
    if mode == "markowitz_subset" and Sigma_pct2 is not None and len(idx) > 0:
        sub_mu = mu[idx]
        sub_S = Sigma_pct2[np.ix_(idx, idx)]
        sub_w = markowitz_max_sharpe(sub_mu, sub_S, rf_pct, min_pct, max_pct)
        w = np.zeros_like(mu, dtype=float); w[idx] = sub_w
        return _clamp_box_and_sum_to_one(w, min_pct, max_pct)
    return weights_from_bits(mu, sigma, bits, min_pct, max_pct)

def _portfolio_metrics_core(w_frac: np.ndarray, mu_pct: np.ndarray, Sigma_pct2: np.ndarray, rf_pct: float) -> Dict[str, float]:
    exp_ret = float(np.dot(w_frac, mu_pct))                   # %
    var = float(np.dot(w_frac, np.dot(Sigma_pct2, w_frac)))   # %^2
    risk = math.sqrt(max(var, 0.0))                           # %
    sharpe = (exp_ret - rf_pct) / (risk + 1e-8)
    return {"expected_return": exp_ret, "portfolio_risk": risk, "sharpe_ratio": sharpe}

def portfolio_metrics(w_pct: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float = RISK_FREE) -> Dict[str, float]:
    w = w_pct / 100.0
    m = _portfolio_metrics_core(w, mu, Sigma, rf)
    if m["portfolio_risk"] > 300.0:
        Sigma_rescaled = Sigma / (100.0 ** 2)
        m = _portfolio_metrics_core(w, mu, Sigma_rescaled, rf)
    return {
        "expected_return": round(m["expected_return"], 4),
        "portfolio_risk": round(m["portfolio_risk"], 4),
        "sharpe_ratio": round(m["sharpe_ratio"], 6),
    }

def extract_quasi_dist(sampler_result: Any) -> Dict[str, float]:
    if hasattr(sampler_result, "quasi_dists"):
        d = sampler_result.quasi_dists[0]
        try: return dict(d)
        except Exception: pass
    try:
        md0 = sampler_result.metadata[0]
        if "counts" in md0:
            c = md0["counts"]; tot = sum(c.values())
            return {k: v/tot for k, v in c.items()} if tot else {}
        if "measure" in md0 and "counts" in md0["measure"]:
            c = md0["measure"]["counts"]; tot = sum(c.values())
            return {k: v/tot for k, v in c.items()} if tot else {}
    except Exception:
        pass
    return {}

def _make_qaoa(optimizer, reps=REPS):
    try:
        sig = inspect.signature(QAOA.__init__)
        params = list(sig.parameters.keys())
        if "estimator" in params:
            return QAOA(estimator=Estimator(), optimizer=optimizer, reps=reps)
        if len(params) > 1 and params[1] in ("estimator",):
            return QAOA(Estimator(), optimizer, reps)
        if "quantum_instance" in params:
            try:
                from qiskit import Aer
                from qiskit.utils import QuantumInstance
                qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))
                return QAOA(optimizer=optimizer, reps=reps, quantum_instance=qi)
            except Exception as e:
                print(f"[QAOA legacy init failed]: {e}")
    except Exception as e:
        print(f"[QAOA init introspection failed]: {e}")
    return None

# =================== Markowitz (classical max-Sharpe) ===================
def markowitz_max_sharpe(mu_pct: np.ndarray, Sigma_pct2: np.ndarray, rf_pct: float,
                         min_pct: float, max_pct: float) -> np.ndarray:
    n = len(mu_pct)
    ones = np.ones(n)
    Sigma = Sigma_pct2.copy()
    try:
        inv = np.linalg.pinv(Sigma, rcond=1e-8)
    except Exception:
        inv = np.linalg.pinv(Sigma + 1e-6 * np.eye(n), rcond=1e-8)
    excess = mu_pct - rf_pct * ones
    raw = inv @ excess
    if not np.any(np.isfinite(raw)):
        w = np.ones(n) / n * 100.0
        return _clamp_box_and_sum_to_one(w, min_pct, max_pct)
    w = raw / np.sum(raw)
    w = np.where(np.isfinite(w), w, 0.0)
    w_pct = w * 100.0
    w_pct = _clamp_box_and_sum_to_one(w_pct, min_pct, max_pct)
    for _ in range(3):
        w_pct = _clamp_box_and_sum_to_one(w_pct, min_pct, max_pct)
    return w_pct

# =================== QNN Forecaster ===================
def _baseline_yhat(close: pd.Series, horizon_days: int, interval: str) -> float:
    """Deterministic fallback forecast based on simple momentum/mean returns.
    Returns a decimal horizon return (e.g., 0.0123 for +1.23%)."""
    try:
        s = close.dropna()
        if s.size < 3:
            return 0.0
        # horizon steps consistent with existing helper
        steps = max(1, _horizon_steps_for_interval(interval, horizon_days))
        # 1) Momentum: cumulative return over last 'steps'
        if s.size > steps:
            mom = float(s.iloc[-1] / s.iloc[-steps-1] - 1.0)
        else:
            mom = float(s.pct_change().dropna().tail(steps).add(1).prod() - 1.0) if s.size > 1 else 0.0
        # 2) Mean-return × steps as a secondary signal
        rets = s.pct_change().dropna()
        mean_r = float(rets.mean()) if rets.size else 0.0
        mean_h = mean_r * steps
        # Blend conservatively
        y = 0.65 * mom + 0.35 * mean_h
        # Clip to the same bounds as QNN path
        return float(np.clip(y, QNN_CLIP_MIN, QNN_CLIP_MAX))
    except Exception:
        return 0.0

def _make_features(close: pd.Series) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build supervised dataset:
      X: [r5, r20, vol5, vol20]
      y: future horizon-step return (decimal), where horizon depends on YF_INTERVAL
    Returns X, y, last_price
    Robust to short/degenerate inputs.
    """
    if not isinstance(close, pd.Series):
        close = pd.Series(close, dtype=float)
    close = close.astype(float).dropna()
    horizon_steps = _horizon_steps_for_interval(YF_INTERVAL, FORECAST_HORIZON_DAYS)
    min_len = max(horizon_steps + 25, 36)  # ensure enough samples for monthly too
    if close.size < min_len:
        raise ValueError("Insufficient history for feature engineering")
    rets = close.pct_change()
    r5    = close.pct_change(min(5, max(1, horizon_steps))).rename("r5")
    r20   = close.pct_change(max(2*horizon_steps, 2)).rename("r20")
    vol5  = rets.rolling(min(5, max(2, horizon_steps))).std().rename("vol5")
    vol20 = rets.rolling(max(2*horizon_steps, 3)).std().rename("vol20")
    feats = pd.concat([r5, r20, vol5, vol20], axis=1)
    future = (close.shift(-horizon_steps) / close - 1.0).rename("y")
    df = pd.concat([feats, future], axis=1).dropna()
    if df.shape[0] < 10:
        raise ValueError("Insufficient samples after dropna")
    X = df[["r5", "r20", "vol5", "vol20"]].values.astype(float)
    y = df["y"].values.astype(float)
    last_price = float(close.iloc[-1])
    return X, y, last_price

def _predict_qnn_return(ticker: str) -> Tuple[float, float]:
    """
    Returns: (pred_20d_return_decimal, current_price)
    Uses Qiskit QNN if available; else Ridge fallback.
    Never raises — returns (0.0, last_price or 100.0) on any data/model issue.
    --- USES QNN_CACHE ---
    """
    # --- NEW: Check cache first ---
    if ticker in QNN_CACHE:
        return QNN_CACHE[ticker]
    # -----------------------------

    close = _series_close(ticker, YF_PERIOD, YF_INTERVAL)
    last_price = float(close.iloc[-1]) if close.size else 100.0
    min_needed = max(_horizon_steps_for_interval(YF_INTERVAL, FORECAST_HORIZON_DAYS) + 25, 36)
    
    if close.size < min_needed:
        yhat_b = _baseline_yhat(close, FORECAST_HORIZON_DAYS, YF_INTERVAL)
        log.info(f"[QNN] insufficient history ({close.size} < {min_needed}); using baseline yhat={yhat_b:.6f}")
        QNN_CACHE[ticker] = (float(yhat_b), last_price) # Cache baseline
        return QNN_CACHE[ticker]
    try:
        X, y, last_price = _make_features(close)
    except Exception as e:
        log.warning(f"[QNN] feature engineering failed ({e}); using baseline")
        yhat_b = _baseline_yhat(close, FORECAST_HORIZON_DAYS, YF_INTERVAL)
        QNN_CACHE[ticker] = (float(yhat_b), last_price) # Cache baseline
        return QNN_CACHE[ticker]

    mu = X.mean(axis=0); sig = X.std(axis=0) + 1e-12
    Xn = (X - mu) / sig
    n = len(Xn)
    split = max(int(n * 0.8), 1)
    Xtr, ytr = Xn[:split], y[:split]
    x_live = ((X[-1] - mu) / sig).reshape(1, -1)

    # ---- NEW: Scale y-target to [-1, 1] for QNN ----
    y_min = float(ytr.min())
    y_max = float(ytr.max())
    y_range = y_max - y_min
    if y_range < 1e-9:
        y_range = 1.0  # Avoid division by zero if all y are the same
        y_min = y_min - 0.5 # Center it
    
    # Scale ytr from [y_min, y_max] -> [0, 1] -> [-1, 1]
    ytr_scaled = 2.0 * ((ytr - y_min) / y_range) - 1.0
    # -----------------------------------------------

    try:
        if QML_AVAILABLE:
            from qiskit import QuantumCircuit
            feature_map = ZZFeatureMap(FEATURES_N, reps=QNN_FM_REPS)
            ansatz = RealAmplitudes(FEATURES_N, reps=QNN_ANSATZ_REPS)
            qc = QuantumCircuit(FEATURES_N)
            qc.compose(feature_map, inplace=True)
            qc.compose(ansatz, inplace=True)
            
            # --- MODIFIED: Let QNN create its own Estimator ---
            qnn = EstimatorQNN(
                circuit=qc,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters
                # 'estimator=est' removed
            )
            # ------------------------------------------------
            
            # ---- MODIFIED: Use ytr_scaled in loss ----
            def mse(w):
                preds = np.array([float(qnn.forward(row, w)) for row in Xtr])
                # Compare QNN output [-1, 1] to scaled target [-1, 1]
                return float(np.mean((preds - ytr_scaled) ** 2))
            # ------------------------------------------

            weights = np.random.uniform(-0.5, 0.5, size=len(ansatz.parameters))
            best_w, best_loss = weights.copy(), mse(weights)
            for _ in range(QNN_TRAIN_ITERS):
                probe = best_w + np.random.normal(scale=QNN_NOISE_SIGMA, size=best_w.shape)
                loss = mse(probe)
                if loss < best_loss:
                    best_loss, best_w = loss, probe
            
            # ---- NEW: Inverse-scale the prediction ----
            # Get the scaled prediction in [-1, 1]
            yhat_scaled = float(qnn.forward(x_live.flatten(), best_w))
            # Convert yhat_scaled from [-1, 1] -> [0, 1]
            yhat_01 = (yhat_scaled + 1.0) / 2.0
            # Convert from [0, 1] -> [y_min, y_max] (original return range)
            yhat = yhat_01 * y_range + y_min
            # -----------------------------------------
        
        else:
            if Ridge is None:
                log.info("[QNN] Ridge not available, using baseline")
                yhat = _baseline_yhat(close, FORECAST_HORIZON_DAYS, YF_INTERVAL)
            else:
                # Ridge regression trains on the original ytr, which is correct
                model = Ridge(alpha=1.0, fit_intercept=True)
                model.fit(Xtr, ytr) 
                yhat = float(model.predict(x_live)[0])
    
    except Exception as e:
        log.warning(f"[QNN] model execution failed ({e}); using baseline")
        yhat = _baseline_yhat(close, FORECAST_HORIZON_DAYS, YF_INTERVAL)

    # --- MODIFIED: Just clip, don't replace near-zero values ---
    yhat = float(np.clip(yhat, QNN_CLIP_MIN, QNN_CLIP_MAX))
    
    # [REMOVED BLOCK for near-zero check]
    # ----------------------------------------------------

    # --- NEW: Store result in cache ---
    QNN_CACHE[ticker] = (yhat, last_price)
    return QNN_CACHE[ticker]

def _make_path_from_allocs(allocs_pct: np.ndarray, tickers: List[str]) -> List[PathPointOut]:
    """
    Convert per-asset predicted horizon returns into a portfolio value path (base=100).
    Horizon steps depend on YF_INTERVAL; for monthly interval it's ~next month (1 step).
    """
    pred_map = {}
    for i, tk in enumerate(tickers):
        yhat, _ = _predict_qnn_return(tk)
        pred_map[tk] = yhat  # decimal over horizon_steps
    horizon_steps = _horizon_steps_for_interval(YF_INTERVAL, FORECAST_HORIZON_DAYS)
    # Convert total horizon return to per-step return
    daily_ret_per_asset = {}
    for tk, rH in pred_map.items():
        per_step = (1.0 + rH) ** (1.0 / float(horizon_steps)) - 1.0
        daily_ret_per_asset[tk] = per_step
    w = allocs_pct / 100.0
    port_step_ret = 0.0
    for i, tk in enumerate(tickers):
        port_step_ret += w[i] * daily_ret_per_asset[tk]
    base = 100.0
    path = []
    day = datetime.utcnow().date()
    step = _timedelta_step_for_interval(YF_INTERVAL)
    val = base
    for d in range(1, max(1, horizon_steps) + 1):
        val *= (1.0 + port_step_ret)
        path.append(PathPointOut(date=(day + d * step).isoformat(), est_value=round(float(val), 4)))
    return path

def _per_asset_forecasts(tickers: List[str], asset_ids: List[str]) -> List[AssetForecastOut]:
    out = []
    for aid, tk in zip(asset_ids, tickers):
        yhat, last_price = _predict_qnn_return(tk)
        next_month_price = float(last_price * (1.0 + yhat))
        out.append(AssetForecastOut(
            asset_id=aid,
            ticker=tk,
            current_price=round(last_price, 6),
            predicted_20d_return_pct=round(100.0 * yhat, 4),
            next_month_price_estimate=round(next_month_price, 6),
        ))
    return out

# =================== Routes ===================
@app.get("/")
async def root():
    return {"message": "QAOA + Markowitz + QNN Forecast up. POST to /optimize",
            "yfinance_available": YF_AVAILABLE, "qml_available": QML_AVAILABLE, "use_yfinance": USE_YFINANCE}

@app.get("/debug/last-result")
async def debug_last_result():
    return LAST_RESULT or {"info": "no result produced yet"}

@app.get("/debug/last-callback")
async def debug_last_callback():
    return LAST_CALLBACK or {"info": "no callback attempted yet"}

# --- NEW: Add a route to clear the QNN cache ---
@app.get("/debug/clear-cache")
async def debug_clear_cache():
    global QNN_CACHE
    count = len(QNN_CACHE)
    QNN_CACHE = {}
    return {"status": "cache_cleared", "items_removed": count}

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: Request):
    t0 = time.time()
    try:
        raw = await request.body()
        try:
            req = OptimizationRequest.model_validate_json(raw.decode("utf-8"))
        except ValidationError as ve:
            log.error(f"422 validation: {ve}")
            return OptimizationResponse(
                optimization_id="unknown", status="failed", progress=0,
                error_message=f"Validation error: {ve.errors()}"
            )

        assets = req.portfolio.assets
        n = len(assets)
        if n == 0:
            return OptimizationResponse(
                optimization_id=req.optimization_id, status="failed", progress=0,
                error_message="No assets provided."
            )
        
        # --- NEW: Progress Log ---
        log.info(f"[{req.optimization_id}] --- (0%) Starting optimization for {n} assets: {[a.name for a in assets]} ---")

        names = [a.name for a in assets]
        tickers = [map_name_to_ticker(x) for x in names]

        # --- MODIFIED: Pass the opt_id to the helper function ---
        pred_map = _collect_pred20d_map(tickers, req.optimization_id)
        mu_forecast = np.array([_annualize_pred_horizon(pred_map[tk][0], YF_INTERVAL, FORECAST_HORIZON_DAYS) for tk in tickers], dtype=float)  # % annualized from QNN
        
        # --- NEW: Progress Log ---
        log.info(f"[{req.optimization_id}] --- (30%) QNN forecasts annualized. Fetching historical data... ---")

        # ---- Build mu and sigma from payload, optionally override with yfinance ----
        mu = np.array([(a.expected_return if a.expected_return is not None else np.nan) for a in assets], dtype=float)   # %
        sigma = np.array([(a.risk            if a.risk            is not None else np.nan) for a in assets], dtype=float) # %
        if USE_YFINANCE and YF_AVAILABLE:
            for i, tk in enumerate(tickers):
                if (not PREFER_INPUT_STATS) or (np.isnan(mu[i]) or np.isnan(sigma[i])):
                    try:
                        m, s = fetch_mu_sigma_from_yf(tk)
                        if np.isnan(mu[i]) or not PREFER_INPUT_STATS:   mu[i] = m
                        if np.isnan(sigma[i]) or not PREFER_INPUT_STATS: sigma[i] = s
                    except Exception as e:
                        log.warning(f"yfinance mu/sigma failed for {tk}: {e}")
        mu_hist = np.where(np.isfinite(mu), mu, 5.0)
        sigma   = np.where(np.isfinite(sigma), sigma, 20.0)

        if MU_MODE == "forecast":
            mu_used = mu_forecast
        elif MU_MODE == "blend":
            mu_used = (1.0 - FORECAST_ALPHA) * mu_hist + FORECAST_ALPHA * mu_forecast
        else:  # "historical"
            mu_used = mu_hist

        # --- NEW: Progress Log ---
        log.info(f"[{req.optimization_id}] --- (40%) Historical data complete. Fetching covariance matrix... ---")

        # ---- Covariance ----
        Sigma = None
        if USE_YFINANCE and YF_AVAILABLE:
            try:
                Sigma = fetch_covariance_from_yf(tickers)  # %^2
            except Exception as e:
                log.warning(f"yfinance covariance failed: {e}")
        if Sigma is None:
            # --- NEW: Progress Log ---
            log.warning(f"[{req.optimization_id}] (45%) Covariance fetch failed. Using diagonal fallback.")
            Sigma = build_sigma_diag(sigma)  # diagonal fallback in %^2
        
        # --- NEW: Progress Log ---
        log.info(f"[{req.optimization_id}] --- (50%) Covariance matrix complete. Building QUBO... ---")


        # ---- Correlation matrix (from Sigma) ----
        try:
            stds = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
            denom = np.outer(stds, stds)
            Corr = np.divide(Sigma, denom, out=np.zeros_like(Sigma), where=(denom > 0))
            Corr = np.clip(Corr, -1.0, 1.0)
            corr_out = CorrelationMatrixOut(tickers=tickers, matrix=Corr.round(6).tolist())
        except Exception:
            Corr = np.eye(len(tickers))
            corr_out = CorrelationMatrixOut(tickers=tickers, matrix=Corr.round(6).tolist())
        # ---- QUBO & QAOA ----
        H = build_qubo(mu_used, Sigma, gamma=GAMMA,
                       k_cardinality=QAOA_K_CARDINALITY, lambda_k=QAOA_LAMBDA_K)
        optimizer = COBYLA(maxiter=QAOA_MAXITER)
        qaoa = _make_qaoa(optimizer, reps=QAOA_REPS)

        selected = None
        qres = None
        best_energy = None
        best_result = None

        if qaoa is not None:
            restarts = max(int(QAOA_RANDOM_RESTARTS), 1)
            # --- NEW: Progress Log ---
            log.info(f"[{req.optimization_id}] --- (60%) Starting QAOA execution ({restarts} restarts)... ---")
            for r in range(restarts):
                try:
                    # --- NEW: Progress Log (inside loop) ---
                    # Assume QAOA is ~25% of the job (from 60% to 85%)
                    progress_pct = 60.0 + (float(r) / float(restarts)) * 25.0
                    log.info(f"[{req.optimization_id}] ({progress_pct:.1f}%) Running QAOA restart {r+1}/{restarts}...")

                    # Randomize initial point if supported
                    if hasattr(qaoa, "initial_point") and hasattr(qaoa.ansatz, "num_parameters"):
                        import numpy as _np
                        qaoa.initial_point = _np.random.uniform(
                            -_np.pi, _np.pi, qaoa.ansatz.num_parameters
                        )
                    _res = qaoa.compute_minimum_eigenvalue(operator=H)
                    _energy = (
                        float(getattr(_res, "eigenvalue", 0.0).real)
                        if hasattr(_res, "eigenvalue")
                        else 0.0
                    )
                    
                    # --- NEW: Progress Log (inside loop) ---
                    log.info(f"[{req.optimization_id}] ({(60.0 + (float(r+1) / float(restarts)) * 25.0):.1f}%) Restart {r+1} complete. Energy={_energy:.4f}")

                    if best_energy is None or _energy < best_energy:
                        best_energy, best_result = _energy, _res
                        # --- NEW: Progress Log ---
                        log.info(f"[{req.optimization_id}] New best energy found: {best_energy:.4f}")

                except Exception as e:
                    print(f"[QAOA restart {r}] failed:", e)
                    continue

            qres = best_result
            
            # --- NEW: Progress Log ---
            log.info(f"[{req.optimization_id}] --- (85%) QAOA execution finished. Best energy: {best_energy}. Sampling for bitstring... ---")

            optimal = getattr(qres, "optimal_point", None)
            try:
                trained = qaoa.ansatz.bind_parameters(optimal) if optimal is not None else qaoa.ansatz
                sampler = Sampler()
                job = sampler.run([trained], shots=4096)
                sres = job.result()
                dist = extract_quasi_dist(sres)
                if dist:
                    best_str = max(dist, key=dist.get)  # little-endian
                    bits = list(map(int, best_str[::-1]))  # reverse to match Z-order
                    selected = np.array(bits, dtype=float)
            except Exception as e:
                log.warning(f"[Sampler fallback]: {e}")

            if selected is None:
                # --- NEW: Progress Log ---
                log.warning(f"[{req.optimization_id}] (88%) Sampler failed. Using brute_best_bits fallback.")
                selected = brute_best_bits(mu_used, Sigma, GAMMA)
        else:
            # --- NEW: Progress Log ---
            log.warning(f"[{req.optimization_id}] (88%) QAOA instance is None. Using brute_best_bits fallback.")
            selected = brute_best_bits(mu_used, Sigma, GAMMA)

        # -------- Step 3: Guaranteed ≥K diversification fallback --------
        try:
            K_MIN = int(QAOA_K_CARDINALITY) if QAOA_K_CARDINALITY is not None else 0
        except Exception:
            K_MIN = 0
        if K_MIN and int(np.sum(selected)) < K_MIN:
            # --- NEW: Progress Log ---
            log.info(f"[{req.optimization_id}] (90%) QAOA selected {int(np.sum(selected))} assets. Applying fallback to reach K={K_MIN}.")
            # rank unused assets by simple utility μ/σ^2 and add top to reach K
            util = np.divide(mu_used, np.maximum(1e-8, sigma ** 2))
            need = K_MIN - int(np.sum(selected))
            candidates = np.where(selected == 0)[0]
            if candidates.size > 0 and need > 0:
                add_idx = candidates[np.argsort(util[candidates])[::-1]][:need]
                selected[add_idx] = 1
        else:
            # --- NEW: Progress Log ---
            log.info(f"[{req.optimization_id}] (90%) QAOA selection complete ({int(np.sum(selected))} assets). Calculating weights...")
        # ----------------------------------------------------------------

        # ---- Weights & metrics: QAOA ----
        c = req.portfolio.constraints
        w_qaoa_pct = weights_from_bits_mode(mu_used, sigma, selected,
                                            c.min_allocation_per_asset,
                                            c.max_allocation_per_asset,
                                            Sigma_pct2=Sigma,
                                            rf_pct=RISK_FREE)

        # (Step 2 reinforcement) Re-clamp & renormalize to honor caps before returning
        w_qaoa_pct = _clamp_box_and_sum_to_one(
            w_qaoa_pct,
            c.min_allocation_per_asset,
            c.max_allocation_per_asset
        )

        metrics_qaoa = portfolio_metrics(w_qaoa_pct, mu_used, Sigma, rf=RISK_FREE)
        allocations_qaoa = [
            AllocationOut(asset_id=assets[i].asset_id, optimized_allocation=round(float(w_qaoa_pct[i]), 2))
            for i in range(n) if w_qaoa_pct[i] > 1e-6
        ]
        iters = None
        try:
            iters = getattr(getattr(qres, "optimizer_result", None), "eval_count", None) if qres else None
        except Exception:
            pass
        if iters is None:
            iters = MAXITER
        elapsed_q = round(time.time() - t0, 2)

        qaoa_results = ResultsOut(
            optimized_allocations=allocations_qaoa,
            metrics=MetricsOut(**metrics_qaoa),
            algorithm_info=AlgorithmInfoOut(
                method="QAOA",
                iterations=int(iters),
                convergence_time_seconds=elapsed_q
            ),
        )
        
        # --- NEW: Progress Log ---
        log.info(f"[{req.optimization_id}] --- (95%) QAOA results complete. Running Markowitz benchmark... ---")

        # ---- Markowitz benchmark ----
        w_mkv_pct = markowitz_max_sharpe(mu_used, Sigma, RISK_FREE,
                                         c.min_allocation_per_asset,
                                         c.max_allocation_per_asset)
        metrics_mkv = portfolio_metrics(w_mkv_pct, mu_used, Sigma, rf=RISK_FREE)
        allocations_mkv = [
            AllocationOut(asset_id=assets[i].asset_id, optimized_allocation=round(float(w_mkv_pct[i]), 2))
            for i in range(n) if w_mkv_pct[i] > 1e-6
        ]
        elapsed_total = round(time.time() - t0, 2)
        mkv_results = ResultsOut(
            optimized_allocations=allocations_mkv,
            metrics=MetricsOut(**metrics_mkv),
            algorithm_info=AlgorithmInfoOut(
                method="Markowitz",
                iterations=1,
                convergence_time_seconds=elapsed_total
            ),
        )
        
        # --- NEW: Progress Log ---
        log.info(f"[{req.optimization_id}] --- (98%) Markowitz benchmark complete. Generating final QNN forecast paths... ---")

        # ---- Comparison ----
        qS = metrics_qaoa["sharpe_ratio"]
        mS = metrics_mkv["sharpe_ratio"]
        winner = "Tie" if abs(qS - mS) < 1e-6 else ("QAOA" if qS > mS else "Markowitz")
        cmp_out = ComparisonOut(qaoa_sharpe=round(qS, 6), markowitz_sharpe=round(mS, 6), winner=winner)

        # ---- QNN Forecasts ----
        asset_ids = [a.asset_id for a in assets]
        per_asset = _per_asset_forecasts(tickers, asset_ids)
        path_qaoa = _make_path_from_allocs(w_qaoa_pct, tickers)
        path_mkv  = _make_path_from_allocs(w_mkv_pct, tickers)
        forecast = ForecastOut(
            per_asset=per_asset,
            portfolio_paths=PortfolioPathsOut(qaoa=path_qaoa, markowitz=path_mkv)
        )
        
        # --- NEW: Progress Log ---
        log.info(f"[{req.optimization_id}] --- (100%) All calculations complete. Assembling final response. ---")

        # ---- Final Response ----
        resp = OptimizationResponse(
            optimization_id=req.optimization_id,
            status="completed",
            progress=100,
            results=qaoa_results,
            benchmark=mkv_results,
            comparison=cmp_out,
            forecast=forecast,
            correlation_matrix=corr_out
        )

        # ---- Debug tap ----
        global LAST_RESULT
        LAST_RESULT = {
            "when": datetime.utcnow().isoformat() + "Z",
            "optimization_id": req.optimization_id,
            "results": json.loads(resp.model_dump_json()),
        }
        log.info(f"[RESULT READY] id={req.optimization_id} "
                 f"QAOA ret={metrics_qaoa['expected_return']} risk={metrics_qaoa['portfolio_risk']} sharpe={metrics_qaoa['sharpe_ratio']} | "
                 f"MKV ret={metrics_mkv['expected_return']} risk={metrics_mkv['portfolio_risk']} sharpe={metrics_mkv['sharpe_ratio']}")

        # ---- Optional callback ----
        global LAST_CALLBACK
        LAST_CALLBACK = None
        if req.callback_url:
            try:
                log.info(f"[CALLBACK →] id={req.optimization_id} url={req.callback_url}")
                r = requests.post(
                    req.callback_url,
                    json=json.loads(resp.model_dump_json()),
                    timeout=15,
                    headers={
                        "Content-Type": "application/json",
                        "X-Optimization-Id": req.optimization_id,
                    },
                )
                LAST_CALLBACK = {
                    "when": datetime.utcnow().isoformat() + "Z",
                    "optimization_id": req.optimization_id,
                    "url": req.callback_url,
                    "status_code": r.status_code,
                    "response_excerpt": _excerpt(r.text, 500),
                }
                log.info(f"[CALLBACK ←] id={req.optimization_id} status={r.status_code} "
                         f"body={_excerpt(r.text, 200)}")
            except Exception as e:
                LAST_CALLBACK = {
                    "when": datetime.utcnow().isoformat() + "Z",
                    "optimization_id": req.optimization_id,
                    "url": req.callback_url,
                    "error": str(e),
                }
                log.warning(f"[CALLBACK ✗] id={req.optimization_id} error={e}")

        return resp

    except Exception as e:
        log.exception("Unhandled /optimize error")
        return OptimizationResponse(
            optimization_id="unknown", status="failed", progress=0,
            error_message=f"{type(e).__name__}: {e}"
        )
