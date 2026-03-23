# Hull-White One-Factor Calibration Engine

Interactive calibration dashboard for the Hull-White short-rate model.
Calibrates mean reversion and volatility to a swaption normal vol surface,
then recovers the time-dependent drift from the initial yield curve.

---

## Model

### Hull-White Dynamics

$$
dr(t) = [\theta(t) - a\,r(t)]\,dt + \sigma\,dW(t)
$$

| Symbol | Definition |
|--------|-----------|
| $r(t)$ | Short rate |
| $a$ | Mean-reversion speed (constant) |
| $\sigma$ | Volatility (constant) |
| $\theta(t)$ | Time-dependent drift, fitted to the yield curve |
| $W(t)$ | Standard Brownian motion |

---

### Zero-Coupon Bond Price

$$
P(t,T) = A(t,T)\,\exp\!\bigl(-B(a,\,T-t)\,r(t)\bigr)
$$

$$
B(a,\tau) = \frac{1 - e^{-a\tau}}{a}
$$

$$
\ln A(t,T) = \ln\frac{P^M(0,T)}{P^M(0,t)} + B(a,T-t)\,f^M(0,t) - \frac{\sigma^2}{4a}\,B(a,T-t)^2\,(1 - e^{-2at})
$$

---

### Drift Recovery

$$
\theta(t) = \frac{\partial f^M}{\partial t}(0,t) + a\,f^M(0,t) + \frac{\sigma^2}{2a}\,(1 - e^{-2at})
$$

Where $f^M(0,t) = -\frac{\partial}{\partial t}\ln P^M(0,t)$ is the market instantaneous forward rate.

---

### Swaption Normal Vol (Swap-Rate Variance Approximation)

For a European swaption with expiry $T_0$ and swap payments at $T_1, \ldots, T_n$:

$$
\sigma_N^2 \cdot T_0 = \frac{1}{A^2}\left[\sum_{i=1}^{n} c_i\,P(0,T_i)\,B(a,\,T_i - T_0)\right]^2 \cdot \frac{\sigma^2}{2a}\,(1 - e^{-2aT_0})
$$

| Symbol | Definition |
|--------|-----------|
| $A$ | Annuity: $\sum_i P(0,T_i)\,\Delta t$ |
| $K$ | Par swap rate: $[P(0,T_0) - P(0,T_n)] / A$ |
| $c_i$ | Cashflows: $K\,\Delta t$ for $i < n$, $\;1 + K\,\Delta t$ for $i = n$ |
| $\sigma_N$ | Normal (Bachelier) swaption vol in bps |

This is a linear swap-rate approximation (freeze-the-weights), exact under one factor
up to the linearization of the swap rate in bond prices.

---

## Calibration

### Objective

$$
\min_{a,\,\sigma} \sum_{(T_e,\,\tau)} \bigl(\sigma_N^{\text{model}}(a,\sigma;\,T_e,\tau) - \sigma_N^{\text{market}}(T_e,\tau)\bigr)^2
$$

### Optimizer

Nelder-Mead on $(a, \sigma)$ — derivative-free, suitable for a 2-parameter search.

### Inputs

- Market yield curve (tenors + zero rates)
- Swaption normal vol matrix (expiry $\times$ tenor), in bps
- Initial guess $(a_0, \sigma_0)$

### Outputs

- Calibrated $(a^*, \sigma^*)$
- Model vol surface and residuals vs market
- $\theta(t)$ curve
- Convergence diagnostics (RMSE, parameter trajectory)

---

## Known Limitations

- **Constant $(a, \sigma)$**: two free parameters cannot fit a full vol surface exactly.
  Production desks typically use piecewise-constant $\sigma(t)$.
- **One-factor model**: all rates are perfectly correlated.
  A correlation matrix input requires a multi-factor extension.
- **Approximation, not Jamshidian**: the swaption pricer uses a linear swap-rate
  variance formula, not the exact Jamshidian bond-option decomposition.

---

## Features

### Sidebar Parameters
- Mean-reversion speed ($a_0$)
- Volatility ($\sigma_0$)

### Market Data Display
- Swaption normal vol heatmap (expiry × tenor)
- Yield curve plot

### Calibration Output
- Calibrated $a^*$, $\sigma^*$, final RMSE
- RMSE convergence plot
- Parameter trajectory plot
- Iteration log table
- Model vs market vol surface comparison
- Residual heatmap (model − market)
- $\theta(t)$ drift curve

---

## Architecture

```
streamlit (Python)
│
├── app.py           — UI, plots, calibration controls
├── hull_white.py    — Model engine
│   ├── Yield curve utilities (cubic spline, discount factors, forwards)
│   ├── HW analytic formulas (B, A, ZCB price, θ)
│   ├── Swaption normal vol (swap-rate variance approximation)
│   └── Calibration loop (Nelder-Mead)
└── requirements.txt
```

System properties:
- Python backend, Streamlit frontend
- Server-side computation via SciPy / NumPy
- Plotly interactive charts
- Deployed on Streamlit Cloud

---

## Numerical Implementation

- `scipy.interpolate.CubicSpline` for the zero-rate curve
- `scipy.optimize.minimize` (Nelder-Mead) for calibration
- `numpy` for vectorized discount factor and cashflow computations
- `plotly` for interactive heatmaps, line charts, and dual-axis plots

---

## Technology

- Python 3
- Streamlit
- Plotly
- SciPy
- NumPy
- Pandas

---

## Result

A browser-accessible calibration terminal for exploring:
- Hull-White swaption vol fitting
- Model vs market residuals across the vol surface
- Parameter sensitivity via initial guess adjustment
- Time-dependent drift recovery from the yield curve

All directly in the browser without local installation.
