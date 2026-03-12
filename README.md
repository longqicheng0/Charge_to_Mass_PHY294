# Charge_to_Mass_PHY294

Python analysis for a PHY294 charge-to-mass ratio lab using the constant-voltage dataset to estimate the external magnetic field Be from the linearized Helmholtz-coil relation.

## Purpose

This repository contains a directly runnable Python script that:

- converts the beam radius data from mm to m
- propagates measurement uncertainties analytically
- computes the Helmholtz coil field Bc for each current
- fits Bc vs 1/r with manual least-squares formulas
- extracts the external magnetic field from the intercept using Be = -b
- compares the fitted slope to the theoretical alpha value
- reports chi-square, reduced chi-square, and residuals
- generates a plot with the fit and a residual panel

The model used is:

Bc = alpha (1/r) - Be

So for a linear fit

y = mx + b

the variable mapping is:

- y = Bc
- x = 1/r
- m = alpha
- b = -Be

Therefore:

Be = -b

## Files

- [analyze_be_field.py](analyze_be_field.py): main analysis script
- [be_field_fit.png](be_field_fit.png): saved figure when run in a non-interactive backend

## Data Used

Constant-voltage dataset:

- I_A = [1.1176, 1.1953, 1.3234, 1.4650, 1.6398, 1.8637, 2.2050, 2.7050]
- r_mm = [60, 55, 50, 45, 40, 35, 30, 25]

Constants:

- mu0 = 4 pi x 10^-7 T m / A
- n = 130
- R = 0.15 m
- V = 225.4 V

Measurement uncertainties assumed in the script:

- dI = 0.0001 A
- dr = 0.5 mm
- dV = 0.1 V

The script currently treats mu0, n, and R as exact.

## Methods

### 1. Radius conversion

r_m = r_mm x 10^-3

dr_m = dr_mm x 10^-3

### 2. Inverse radius and uncertainty

x = 1/r

The uncertainty is propagated with:

(dx)^2 = (d(1/r)/dr x dr)^2

which gives:

dx = dr / r^2

### 3. Helmholtz coil magnetic field

The script computes:

Bc = (4/5)^(3/2) mu0 n I / R

With K = (4/5)^(3/2) mu0 n / R, this becomes Bc = K I, so:

dBc = K dI

### 4. Manual linear regression

The fit is done with the explicit least-squares formulas requested for:

y = mx + b

where:

- m = (N sum(x_i y_i) - sum(x_i) sum(y_i)) / Delta
- b = y_bar - m x_bar
- Delta = N sum(x_i^2) - (sum(x_i))^2

The fit uncertainty estimates are:

- s_yx^2 = sum((y_i - (m x_i + b))^2) / (N - 2)
- s_m = sqrt(N s_yx^2 / Delta)
- s_b = sqrt(s_yx^2 sum(x_i^2) / Delta)

The coefficient of determination is also computed:

R^2 = 1 - SS_res / SS_tot

### 5. External magnetic field

From the intercept,

Be = -b

and the uncertainty is:

dBe = sb

### 6. Theoretical alpha

The script compares the fitted slope to:

alpha_theory = sqrt((2 m_e / e) V)

using accepted constants for electron mass and charge.

Since only V is treated as uncertain,

d(alpha_theory) = alpha_theory dV / (2V)

### 7. Chi-square

To include both x and y uncertainties in a simple manual way, the script uses an effective uncertainty for each point:

sigma_eff = sqrt((dBc)^2 + (m d(1/r))^2)

Then:

- chi^2 = sum(((y_i - y_fit,i) / sigma_eff,i)^2)
- chi^2_reduced = chi^2 / (N - 2)

## How To Run

From the repository folder:

```bash
python3 analyze_be_field.py
```

Dependencies:

- numpy
- matplotlib

Install them if needed with:

```bash
python3 -m pip install numpy matplotlib
```

If the script is run in a non-interactive environment, it saves the figure as [be_field_fit.png](be_field_fit.png).

## Output

The script prints:

- the experimental constants and uncertainties used
- a table of I, r, 1/r, and Bc with uncertainties
- slope and intercept with uncertainties
- Be in tesla and microtesla with uncertainty
- R^2
- chi-square and reduced chi-square
- alpha_theory with uncertainty
- percent difference between fit and theory
- a short physical interpretation of the Be estimate

The plot includes:

- Bc vs 1/r with error bars
- the best-fit line
- a residual plot below the main graph

## Current Numerical Results

Using the dataset currently hard-coded in the script, the output is approximately:

- slope m = 5.27807e-05 ± 8.8e-07 T m
- intercept b = -2.92594e-05 ± 2.3e-05 T
- Be = 2.92594e-05 ± 2.3e-05 T
- Be = 29.2594 ± 23 microtesla
- R^2 = 0.998330
- chi^2 = 10.6933
- reduced chi^2 = 1.78221
- alpha_theory = 5.06269e-05 ± 1.1e-08 T m
- signed percent difference = 4.25431 ± 1.7 percent

Interpretation from the script:

- the estimated Be is comparable in magnitude to Earth's magnetic field
- the uncertainty is large relative to the Be estimate

## Notes

- The analysis uses manual uncertainty propagation rather than symbolic packages.
- The regression formulas are coded explicitly rather than relying only on numpy fitting helpers.
- If you later want uncertainties in R, n, or additional systematics included, the script can be extended with the same partial-derivative method used here.