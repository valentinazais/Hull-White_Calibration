"""
Hull-White One-Factor Model Calibration Engine
───────────────────────────────────────────────
dr(t) = [θ(t) - a·r(t)] dt + σ·dW(t)

Calibrates (a, σ) to swaption normal vols, then θ(t) analytically from the yield curve.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.stats import norm


# ─────────────────────────────────────────────
# 1. Yield Curve Utilities
# ─────────────────────────────────────────────

def build_zero_curve(tenors: np.ndarray, yields: np.ndarray):
    """
    Build a continuous zero-rate curve (cubic spline) from market par yields.
    Assumes yields are continuously compounded zero rates (simplified bootstrap).
    Returns a callable spline: T → zero_rate(T).
    """
    # Prepend T=0 with the shortest tenor rate for continuity
    t = np.concatenate([[0.0], tenors])
    y = np.concatenate([[yields[0]], yields])
    return CubicSpline(t, y, bc_type='natural')


def discount_factor(T: float, zero_spline) -> float:
    """P(0,T) = exp(-r(T)·T)"""
    if T <= 0:
        return 1.0
    r = float(zero_spline(T))
    return np.exp(-r * T)


def forward_rate(T: float, zero_spline, dt=1e-4) -> float:
    """Instantaneous forward rate f(0,T) = -d/dT ln P(0,T)."""
    if T < dt:
        T = dt
    ln_p1 = -float(zero_spline(T - dt / 2)) * (T - dt / 2)
    ln_p2 = -float(zero_spline(T + dt / 2)) * (T + dt / 2)
    return -(ln_p2 - ln_p1) / dt


def forward_rate_derivative(T: float, zero_spline, dt=1e-4) -> float:
    """∂f(0,T)/∂T — slope of the forward curve."""
    f1 = forward_rate(T - dt / 2, zero_spline, dt)
    f2 = forward_rate(T + dt / 2, zero_spline, dt)
    return (f2 - f1) / dt


# ─────────────────────────────────────────────
# 2. Hull-White Analytic Formulas
# ─────────────────────────────────────────────

def B(a: float, T: float) -> float:
    """B(a,T) = (1 - e^{-aT}) / a"""
    if abs(a) < 1e-10:
        return T
    return (1.0 - np.exp(-a * T)) / a


def A(a: float, sigma: float, T: float, zero_spline) -> float:
    """
    ln A(0,T) for the HW ZCB price:
    P^{HW}(0,T) = A(0,T) · exp(-B(a,T)·r0)
    ln A = ln P^M(0,T) + B·f(0,0) - σ²/(4a)·B²·(1-e^{-2aT})  (???)

    Actually use the standard formula:
    ln A(0,T) = ln[P^M(0,T)/P^M(0,0)] + B(a,T)·f^M(0,0)
                - (σ²/4a)·B(a,T)²·(1 - e^{-2a·0})  ... evaluated at s=0

    Simplification for t=0:
    P^{HW}(0,T) = P^M(0,T) · exp(...)  which simply equals P^M(0,T) at t=0.
    """
    return np.log(discount_factor(T, zero_spline))


def hw_zcb_price(a: float, sigma: float, t: float, T: float,
                 r_t: float, zero_spline) -> float:
    """
    Hull-White zero-coupon bond price P(t,T) given r(t).
    At t=0, r_t = r(0) and this should recover P^M(0,T).

    P(t,T) = A(t,T)·exp(-B(a,T-t)·r_t)

    where:
      B(a,τ) = (1-e^{-aτ})/a,  τ = T-t
      ln A(t,T) = ln[P^M(0,T)/P^M(0,t)] + B(a,τ)·f^M(0,t)
                  - (σ²/4a)·B(a,τ)²·(1-e^{-2at})
    """
    tau = T - t
    if tau <= 0:
        return 1.0
    b = B(a, tau)
    pm_T = discount_factor(T, zero_spline)
    pm_t = discount_factor(t, zero_spline)
    f_t = forward_rate(t, zero_spline)

    if abs(a) < 1e-10:
        ln_A = np.log(pm_T / pm_t) + b * f_t - 0.5 * sigma**2 * t * tau**2
    else:
        ln_A = (np.log(pm_T / pm_t) + b * f_t
                - (sigma**2 / (4.0 * a)) * b**2 * (1.0 - np.exp(-2.0 * a * t)))

    return np.exp(ln_A - b * r_t)


def theta(a: float, sigma: float, t: float, zero_spline) -> float:
    """
    θ(t) = ∂f^M/∂t + a·f^M(0,t) + (σ²/2a)·(1 - e^{-2at})
    """
    f = forward_rate(t, zero_spline)
    df = forward_rate_derivative(t, zero_spline)
    if abs(a) < 1e-10:
        return df + sigma**2 * t
    return df + a * f + (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t))


# ─────────────────────────────────────────────
# 3. Swaption Pricing (Jamshidian Decomposition)
# ─────────────────────────────────────────────

def _swap_schedule(expiry: float, tenor: float, freq: float = 1.0):
    """Generate payment times for a swap starting at `expiry` with given tenor."""
    dt = 1.0 / freq
    n_payments = int(round(tenor * freq))
    return np.array([expiry + dt * (i + 1) for i in range(n_payments)]), dt


def hw_swaption_vol_analytic(a: float, sigma: float,
                              expiry: float, tenor: float,
                              zero_spline) -> float:
    """
    Analytic normal (basis-point) vol for a European payer swaption
    under Hull-White, using the Jamshidian approach.

    Steps:
      1. Compute par swap rate K and annuity A from the market curve.
      2. Compute the variance of the swap rate under HW.
      3. Return the normal vol = sqrt(variance / expiry).
    """
    if expiry <= 0 or tenor <= 0:
        return 0.0

    payment_times, dt = _swap_schedule(expiry, tenor)

    # Market discount factors at payment times
    dfs = np.array([discount_factor(ti, zero_spline) for ti in payment_times])
    df_expiry = discount_factor(expiry, zero_spline)

    # Annuity
    annuity = np.sum(dfs * dt)
    if annuity < 1e-12:
        return 0.0

    # Par swap rate
    K = (df_expiry - dfs[-1]) / annuity

    # ── Variance of the swap rate under HW ──
    # Under HW the ZCB volatility at time T_expiry for maturity T_i is:
    #   v_i = σ/a · (1 - e^{-a(T_i - T_expiry)})  (for the bond)
    # The swap-rate variance (normal model) is approximately:
    #   Var(S) ≈ (1/A)² · Σ_i Σ_j  c_i·c_j · v_i·v_j · ∫_0^{T_exp} e^{-2a(T_exp-u)} du ...
    #
    # We use a simpler, well-known approximation for normal vol:
    #   σ_N² ≈ (1/A²) · Σ_i Σ_j  w_i·w_j · Cov(P(T_exp,T_i), P(T_exp,T_j))
    #
    # where w_i = dt for coupon payments, w_n includes principal,
    # and the HW covariance of ZCBs is:
    #   Cov_i_j = P(0,T_i)·P(0,T_j) · [exp(B_i·B_j·σ²·(1-e^{-2a·T_exp})/(2a)) - 1]
    #
    # For a cleaner, standard approach we compute the "normal vol" directly
    # from the bond-put decomposition.

    # ── Direct Jamshidian / bond-option formula approach ──
    # HW swaption price via Jamshidian: decompose swaption into bond options.
    # For each bond option, use the HW closed-form.
    # Then convert total price to normal vol.

    # Coupon cashflows of the underlying swap (fixed leg pays K per period)
    cashflows = np.full(len(payment_times), K * dt)
    cashflows[-1] += 1.0  # principal

    # σ_P(T_exp, T_i) = σ · B(a, T_i - T_exp) · √((1-e^{-2a·T_exp})/(2a))
    if abs(a) < 1e-10:
        sig_p_factor = sigma * np.sqrt(expiry)
    else:
        sig_p_factor = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * expiry)) / (2.0 * a))

    b_values = np.array([B(a, ti - expiry) for ti in payment_times])
    sig_bonds = sig_p_factor * b_values  # vol of each bond

    # Price each bond option using Bachelier formula (normal model approx)
    # The swaption price ≈ Σ c_i · BondPut(T_i)
    # Under normal model: BondPut = P(0,T_i) · [d·Φ(d) + φ(d)] · σ_bond
    # where d = (P(0,T_i)/X_i - 1) ... but we need the strike for each bond.

    # ── Simpler & standard: use the swap-rate normal vol formula ──
    # σ_swap_normal ≈ (1/A) · √(Σ_i Σ_j c_i c_j B_i B_j · V)
    # where V = σ² · (1-e^{-2aT}) / (2a), c_i = cashflow_i · P(0,T_i)

    V = sig_p_factor**2  # = σ²(1-e^{-2aT})/(2a) · already

    # weighted cashflows
    w = cashflows * dfs  # c_i · P(0,T_i)

    # Quadratic form: Σ_i Σ_j w_i w_j B_i B_j
    bw = b_values * w
    quad = np.sum(bw)**2  # (Σ w_i B_i)²... no, need Σ_iΣ_j w_i w_j B_i B_j = (Σ w_i B_i)²
    # Actually Σ_i Σ_j w_i·w_j·B_i·B_j = (Σ w_i B_i)²  ✓

    variance_swap_rate = quad * V / (annuity**2)
    if variance_swap_rate <= 0:
        return 0.0

    normal_vol = np.sqrt(variance_swap_rate / expiry)  # annualized normal vol

    return normal_vol * 1e4  # convert to basis points


# ─────────────────────────────────────────────
# 4. Calibration
# ─────────────────────────────────────────────

def calibrate(market_vol_df, zero_spline, expiry_map, tenor_map,
              a0=0.05, sigma0=0.01, callback=None):
    """
    Calibrate (a, σ) by minimizing Σ (model_vol - market_vol)² over a grid
    of (expiry, tenor) points.

    Parameters
    ----------
    market_vol_df : pd.DataFrame
        Swaption normal vols in bps, rows=expiry labels, cols=tenor labels.
    zero_spline : callable
        Zero-rate spline from build_zero_curve.
    expiry_map : dict
        label → year fraction (e.g. '1Mo' → 1/12).
    tenor_map : dict
        label → year fraction (e.g. '1Yr' → 1.0).
    a0, sigma0 : float
        Initial guesses.
    callback : callable or None
        Called with (iteration, a, sigma, rmse) after each optimizer step.

    Returns
    -------
    dict with keys: 'a', 'sigma', 'history', 'model_vols'
    """
    # Select a subset of the grid to calibrate on (all points)
    expiry_labels = list(market_vol_df.index)
    tenor_labels = list(market_vol_df.columns)

    targets = []
    for elbl in expiry_labels:
        for tlbl in tenor_labels:
            e_yr = expiry_map[elbl]
            t_yr = tenor_map[tlbl]
            mv = market_vol_df.loc[elbl, tlbl]
            if np.isfinite(mv) and e_yr > 0 and t_yr > 0:
                targets.append((e_yr, t_yr, mv, elbl, tlbl))

    history = []
    iteration_counter = [0]

    def objective(x):
        a_trial = x[0]
        s_trial = x[1]
        if a_trial <= 1e-6 or s_trial <= 1e-6:
            return 1e12

        total_err = 0.0
        n = 0
        for (e_yr, t_yr, mv, _, _) in targets:
            try:
                model_v = hw_swaption_vol_analytic(a_trial, s_trial, e_yr, t_yr, zero_spline)
                total_err += (model_v - mv) ** 2
                n += 1
            except Exception:
                total_err += 1e8

        mse = total_err / max(n, 1)
        rmse = np.sqrt(mse)

        iteration_counter[0] += 1
        it = iteration_counter[0]
        history.append({
            'iteration': it,
            'a': a_trial,
            'sigma': s_trial,
            'rmse': rmse,
            'total_error': total_err
        })
        if callback is not None:
            callback(it, a_trial, s_trial, rmse)

        return total_err

    result = minimize(
        objective,
        x0=[a0, sigma0],
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-6, 'adaptive': True}
    )

    a_star, sigma_star = result.x

    # Compute final model vol surface
    model_vols = {}
    for (e_yr, t_yr, mv, elbl, tlbl) in targets:
        v = hw_swaption_vol_analytic(a_star, sigma_star, e_yr, t_yr, zero_spline)
        model_vols[(elbl, tlbl)] = v

    return {
        'a': a_star,
        'sigma': sigma_star,
        'history': history,
        'model_vols': model_vols,
        'scipy_result': result
    }
