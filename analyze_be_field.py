import numpy as np
import matplotlib.pyplot as plt


# Formula summary used in this script:
# 1. Linear fit for y = m x + b using manual least-squares formulas:
#    m = (N * sum(x_i y_i) - sum(x_i) sum(y_i)) / Delta
#    b = y_bar - m x_bar
#    s_yx^2 = sum((y_i - (m x_i + b))^2) / (N - 2)
#    s_m = sqrt(N * s_yx^2 / Delta)
#    s_b = sqrt(s_yx^2 * sum(x_i^2) / Delta)
#    where Delta = N * sum(x_i^2) - (sum(x_i))^2
# 2. Uncertainty propagation for a derived quantity z = f(x, y, ...):
#    (dz)^2 = (df/dx * dx)^2 + (df/dy * dy)^2 + ...
# 3. For x = 1 / r:
#    dx = |d(1/r)/dr| dr = (1 / r^2) dr
# 4. For Bc(I) = K * I, where K = (4/5)^(3/2) * mu0 * n / R:
#    dBc = |dBc/dI| dI = K dI
# 5. For alpha_theory(V) = sqrt((2 m_e / e) * V) = sqrt(C V):
#    dalpha = |d(alpha)/dV| dV = (C / (2 sqrt(C V))) dV = alpha/(2V) dV
# 6. For chi-square using uncertainties in both x and y for a linear model y = m x + b:
#    sigma_eff^2 = (dy)^2 + (m dx)^2
#    chi^2 = sum(((y_i - y_fit,i) / sigma_eff,i)^2)
#    reduced chi^2 = chi^2 / (N - 2)


MU0 = 4 * np.pi * 1e-7
N_TURNS = 130
COIL_RADIUS_M = 0.15
ACCELERATING_VOLTAGE_V = 225.4

CURRENT_A = np.array([1.1176, 1.1953, 1.3234, 1.4650, 1.6398, 1.8637, 2.2050, 2.7050], dtype=float)
RADIUS_MM = np.array([60, 55, 50, 45, 40, 35, 30, 25], dtype=float)

CURRENT_UNCERTAINTY_A = 0.0001
RADIUS_UNCERTAINTY_MM = 0.5
VOLTAGE_UNCERTAINTY_V = 0.1

ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
REPORT_OUTPUT_PATH = "analysis_results.txt"
QUESTION3_PLOT_PATH = "be_field_fit.png"
QUESTION5_PLOT_PATH = "em_fit.png"
E_OVER_M_ACCEPTED = 1.758820e11


def format_value_uncertainty(value, uncertainty, value_precision=6, uncertainty_precision=2, scale=1.0):
    scaled_value = value * scale
    scaled_uncertainty = uncertainty * scale
    return f"{scaled_value:.{value_precision}g} ± {scaled_uncertainty:.{uncertainty_precision}g}"


def append_report_line(report_lines, text=""):
    report_lines.append(text)


def convert_mm_to_m(length_mm, uncertainty_mm):
    length_m = np.asarray(length_mm, dtype=float) * 1e-3
    uncertainty_m = float(uncertainty_mm) * 1e-3
    return length_m, uncertainty_m


def inverse_radius_with_uncertainty(radius_m, radius_uncertainty_m):
    inverse_radius = 1.0 / radius_m
    inverse_radius_uncertainty = radius_uncertainty_m / (radius_m ** 2)
    return inverse_radius, inverse_radius_uncertainty


def helmholtz_field_with_uncertainty(current_a, current_uncertainty_a, mu0, n_turns, coil_radius_m):
    prefactor = ((4.0 / 5.0) ** 1.5) * mu0 * n_turns / coil_radius_m
    field_t = prefactor * np.asarray(current_a, dtype=float)
    field_uncertainty_t = prefactor * float(current_uncertainty_a)
    return field_t, field_uncertainty_t


def manual_linear_fit(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    num_points = len(x_values)
    sum_x = np.sum(x_values)
    sum_y = np.sum(y_values)
    sum_xy = np.sum(x_values * y_values)
    sum_x2 = np.sum(x_values ** 2)

    delta = num_points * sum_x2 - sum_x ** 2
    slope = (num_points * sum_xy - sum_x * sum_y) / delta
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    intercept = y_mean - slope * x_mean

    fitted_y = slope * x_values + intercept
    residuals = y_values - fitted_y
    syx_squared = np.sum(residuals ** 2) / (num_points - 2)
    slope_uncertainty = np.sqrt(num_points * syx_squared / delta)
    intercept_uncertainty = np.sqrt(syx_squared * sum_x2 / delta)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_values - y_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_uncertainty": slope_uncertainty,
        "intercept_uncertainty": intercept_uncertainty,
        "r_squared": r_squared,
        "fitted_y": fitted_y,
        "residuals": residuals,
        "syx_squared": syx_squared,
        "delta": delta,
    }


def alpha_theory_with_uncertainty(voltage_v, voltage_uncertainty_v, electron_mass_kg, elementary_charge_c):
    coefficient = 2.0 * electron_mass_kg / elementary_charge_c
    alpha_theory = np.sqrt(coefficient * voltage_v)
    alpha_uncertainty = alpha_theory * voltage_uncertainty_v / (2.0 * voltage_v)
    return alpha_theory, alpha_uncertainty


def signed_percent_difference_with_uncertainty(measured_value, measured_uncertainty, reference_value, reference_uncertainty):
    percent_difference = 100.0 * (measured_value - reference_value) / reference_value
    partial_measured = 100.0 / reference_value
    partial_reference = -100.0 * measured_value / (reference_value ** 2)
    percent_uncertainty = np.sqrt(
        (partial_measured * measured_uncertainty) ** 2
        + (partial_reference * reference_uncertainty) ** 2
    )
    return percent_difference, percent_uncertainty


def chi_square_with_effective_uncertainty(y_values, fitted_y, y_uncertainty, slope, x_uncertainty, num_fit_parameters=2):
    residuals = np.asarray(y_values, dtype=float) - np.asarray(fitted_y, dtype=float)
    effective_uncertainty = np.sqrt(np.asarray(y_uncertainty, dtype=float) ** 2 + (slope * np.asarray(x_uncertainty, dtype=float)) ** 2)
    chi_square = np.sum((residuals / effective_uncertainty) ** 2)
    degrees_of_freedom = len(residuals) - num_fit_parameters
    reduced_chi_square = chi_square / degrees_of_freedom
    return chi_square, reduced_chi_square, effective_uncertainty


def helmholtz_k_constant(mu0, n_turns, coil_radius_m):
    return (1.0 / np.sqrt(2.0)) * ((4.0 / 5.0) ** 1.5) * mu0 * n_turns / coil_radius_m


def equation10_y_with_uncertainty(voltage_v, voltage_uncertainty_v, radius_m, radius_uncertainty_m):
    sqrt_voltage = np.sqrt(voltage_v)
    y_values = sqrt_voltage / radius_m
    partial_y_partial_v = 1.0 / (2.0 * radius_m * sqrt_voltage)
    partial_y_partial_r = -sqrt_voltage / (radius_m ** 2)
    y_uncertainty = np.sqrt(
        (partial_y_partial_v * voltage_uncertainty_v) ** 2
        + (partial_y_partial_r * radius_uncertainty_m) ** 2
    )
    return y_values, y_uncertainty


def print_data_table(currents_a, current_uncertainty_a, radii_m, radius_uncertainty_m,
                     inverse_radii, inverse_radii_uncertainty, fields_t, field_uncertainty_t,
                     report_lines=None):
    headers = (
        "Point",
        "I (A)",
        "r (m)",
        "1/r (1/m)",
        "Bc (T)",
    )
    table_lines = []
    table_lines.append("\nData Table")
    table_lines.append("-" * 104)
    table_lines.append(f"{headers[0]:>5}  {headers[1]:>20}  {headers[2]:>20}  {headers[3]:>24}  {headers[4]:>20}")
    table_lines.append("-" * 104)
    for index, (current_a, radius_m, inverse_radius, field_t) in enumerate(
        zip(currents_a, radii_m, inverse_radii, fields_t), start=1
    ):
        current_text = format_value_uncertainty(current_a, current_uncertainty_a, value_precision=7, uncertainty_precision=2)
        radius_text = format_value_uncertainty(radius_m, radius_uncertainty_m, value_precision=7, uncertainty_precision=2)
        inverse_text = format_value_uncertainty(
            inverse_radius,
            inverse_radii_uncertainty[index - 1],
            value_precision=7,
            uncertainty_precision=2,
        )
        field_text = format_value_uncertainty(
            field_t,
            field_uncertainty_t,
            value_precision=7,
            uncertainty_precision=2,
        )
        table_lines.append(f"{index:>5}  {current_text:>20}  {radius_text:>20}  {inverse_text:>24}  {field_text:>20}")
    table_lines.append("-" * 104)

    for line in table_lines:
        print(line)
        if report_lines is not None:
            append_report_line(report_lines, line)


def interpret_be_field(be_t, be_uncertainty_t):
    be_microtesla = be_t * 1e6
    be_uncertainty_microtesla = be_uncertainty_t * 1e6
    earth_field_reference_microtesla = 50.0

    if abs(be_microtesla) < 0.5 * earth_field_reference_microtesla:
        magnitude_statement = "smaller than a typical Earth magnetic field"
    elif abs(be_microtesla) > 1.5 * earth_field_reference_microtesla:
        magnitude_statement = "larger than a typical Earth magnetic field"
    else:
        magnitude_statement = "comparable in magnitude to a typical Earth magnetic field"

    relative_uncertainty = abs(be_uncertainty_t / be_t) if be_t != 0 else np.inf
    if relative_uncertainty < 0.1:
        uncertainty_statement = "The uncertainty is small relative to the estimate."
    elif relative_uncertainty < 0.5:
        uncertainty_statement = "The uncertainty is moderate relative to the estimate."
    else:
        uncertainty_statement = "The uncertainty is large relative to the estimate."

    print("\nInterpretation")
    print(
        f"Be = {format_value_uncertainty(be_t, be_uncertainty_t, value_precision=6, uncertainty_precision=2, scale=1e6)} µT, "
        f"which is {magnitude_statement}."
    )
    print(uncertainty_statement)
    print(
        f"For reference, many locations on Earth have magnetic fields on the order of 25 to 65 µT; "
        f"your estimate is {be_microtesla:.3g} ± {be_uncertainty_microtesla:.2g} µT."
    )


def build_interpretation_lines(be_t, be_uncertainty_t):
    be_microtesla = be_t * 1e6
    be_uncertainty_microtesla = be_uncertainty_t * 1e6
    earth_field_reference_microtesla = 50.0

    if abs(be_microtesla) < 0.5 * earth_field_reference_microtesla:
        magnitude_statement = "smaller than a typical Earth magnetic field"
    elif abs(be_microtesla) > 1.5 * earth_field_reference_microtesla:
        magnitude_statement = "larger than a typical Earth magnetic field"
    else:
        magnitude_statement = "comparable in magnitude to a typical Earth magnetic field"

    relative_uncertainty = abs(be_uncertainty_t / be_t) if be_t != 0 else np.inf
    if relative_uncertainty < 0.1:
        uncertainty_statement = "The uncertainty is small relative to the estimate."
    elif relative_uncertainty < 0.5:
        uncertainty_statement = "The uncertainty is moderate relative to the estimate."
    else:
        uncertainty_statement = "The uncertainty is large relative to the estimate."

    return [
        "\nInterpretation",
        (
            f"Be = {format_value_uncertainty(be_t, be_uncertainty_t, value_precision=6, uncertainty_precision=2, scale=1e6)} µT, "
            f"which is {magnitude_statement}."
        ),
        uncertainty_statement,
        (
            f"For reference, many locations on Earth have magnetic fields on the order of 25 to 65 µT; "
            f"your estimate is {be_microtesla:.3g} ± {be_uncertainty_microtesla:.2g} µT."
        ),
    ]


def write_report(report_lines, output_path):
    report_text = "\n".join(report_lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(report_text)
    return report_text


def plot_results(
    inverse_radii,
    inverse_radii_uncertainty,
    fields_t,
    field_uncertainty_t,
    fit_slope,
    fit_intercept,
    residuals_t,
    chi_square,
    reduced_chi_square,
):
    x_line = np.linspace(np.min(inverse_radii) * 0.95, np.max(inverse_radii) * 1.05, 300)
    y_line = fit_slope * x_line + fit_intercept

    figure, (axis_main, axis_residuals) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axis_main.errorbar(
        inverse_radii,
        fields_t,
        xerr=inverse_radii_uncertainty,
        yerr=field_uncertainty_t,
        fmt="o",
        color="tab:blue",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        label="Measured data",
    )
    axis_main.plot(x_line, y_line, color="tab:red", linewidth=2, label="Best-fit line")
    axis_main.set_ylabel("Bc (T)")
    axis_main.set_title("Helmholtz Coil Field vs Inverse Electron Beam Radius")
    annotation_text = (
        f"Best fit: Bc = ({fit_slope:.4e})(1/r) + ({fit_intercept:.4e})\n"
        f"reduced chi^2 = {reduced_chi_square:.3f}"
    )
    axis_main.text(
        0.03,
        0.97,
        annotation_text,
        transform=axis_main.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    axis_main.legend()
    axis_main.grid(True, alpha=0.3)

    axis_residuals.errorbar(
        inverse_radii,
        residuals_t,
        xerr=inverse_radii_uncertainty,
        yerr=field_uncertainty_t,
        fmt="o",
        color="tab:green",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        label="Residuals",
    )
    axis_residuals.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis_residuals.set_xlabel("1/r (1/m)")
    axis_residuals.set_ylabel("Residual (T)")
    axis_residuals.grid(True, alpha=0.3)
    axis_residuals.legend()

    figure.tight_layout()
    figure.savefig(QUESTION3_PLOT_PATH, dpi=300)
    print(f"\nPlot saved to {QUESTION3_PLOT_PATH}")
    backend_name = plt.get_backend().lower()
    if "agg" not in backend_name:
        plt.show()


def plot_question5_results(
    currents_a,
    current_uncertainty_a,
    y_values,
    y_uncertainty,
    fit_slope,
    fit_intercept,
    residuals,
    reduced_chi_square,
):
    x_line = np.linspace(np.min(currents_a) * 0.95, np.max(currents_a) * 1.05, 300)
    y_line = fit_slope * x_line + fit_intercept

    figure, (axis_main, axis_residuals) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axis_main.errorbar(
        currents_a,
        y_values,
        xerr=current_uncertainty_a,
        yerr=y_uncertainty,
        fmt="o",
        color="tab:blue",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        label="Measured data",
    )
    axis_main.plot(x_line, y_line, color="tab:red", linewidth=2, label="Best-fit line")
    axis_main.set_xlabel("I (A)")
    axis_main.set_ylabel("sqrt(V)/r")
    axis_main.set_title("Question 5: Linearized e/m Fit")
    annotation_text = (
        f"Best fit: sqrt(V)/r = ({fit_slope:.4e})I + ({fit_intercept:.4e})\n"
        f"reduced chi^2 = {reduced_chi_square:.3f}"
    )
    axis_main.text(
        0.03,
        0.97,
        annotation_text,
        transform=axis_main.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    axis_main.legend()
    axis_main.grid(True, alpha=0.3)

    axis_residuals.errorbar(
        currents_a,
        residuals,
        xerr=current_uncertainty_a,
        yerr=y_uncertainty,
        fmt="o",
        color="tab:green",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        label="Residuals",
    )
    axis_residuals.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis_residuals.set_xlabel("I (A)")
    axis_residuals.set_ylabel("Residual")
    axis_residuals.grid(True, alpha=0.3)
    axis_residuals.legend()

    figure.tight_layout()
    figure.savefig(QUESTION5_PLOT_PATH, dpi=300)
    print(f"Plot saved to {QUESTION5_PLOT_PATH}")
    backend_name = plt.get_backend().lower()
    if "agg" not in backend_name:
        plt.show()


def main():
    report_lines = []
    radii_m, radius_uncertainty_m = convert_mm_to_m(RADIUS_MM, RADIUS_UNCERTAINTY_MM)
    inverse_radii, inverse_radii_uncertainty = inverse_radius_with_uncertainty(radii_m, radius_uncertainty_m)
    fields_t, field_uncertainty_t = helmholtz_field_with_uncertainty(
        CURRENT_A,
        CURRENT_UNCERTAINTY_A,
        MU0,
        N_TURNS,
        COIL_RADIUS_M,
    )

    fit_results = manual_linear_fit(inverse_radii, fields_t)
    slope = fit_results["slope"]
    intercept = fit_results["intercept"]
    slope_uncertainty = fit_results["slope_uncertainty"]
    intercept_uncertainty = fit_results["intercept_uncertainty"]
    r_squared = fit_results["r_squared"]
    residuals_t = fit_results["residuals"]

    be_t = -intercept
    be_uncertainty_t = intercept_uncertainty

    chi_square, reduced_chi_square, effective_residual_uncertainty = chi_square_with_effective_uncertainty(
        fields_t,
        fit_results["fitted_y"],
        field_uncertainty_t,
        slope,
        inverse_radii_uncertainty,
    )

    alpha_theory, alpha_theory_uncertainty = alpha_theory_with_uncertainty(
        ACCELERATING_VOLTAGE_V,
        VOLTAGE_UNCERTAINTY_V,
        ELECTRON_MASS_KG,
        ELEMENTARY_CHARGE_C,
    )
    percent_difference, percent_difference_uncertainty = signed_percent_difference_with_uncertainty(
        slope,
        slope_uncertainty,
        alpha_theory,
        alpha_theory_uncertainty,
    )

    # Question 5: Equation 10 linearization y = sqrt(V)/r vs x = I.
    k_constant = helmholtz_k_constant(MU0, N_TURNS, COIL_RADIUS_M)
    equation10_x = CURRENT_A
    equation10_x_uncertainty = CURRENT_UNCERTAINTY_A
    equation10_y, equation10_y_uncertainty = equation10_y_with_uncertainty(
        ACCELERATING_VOLTAGE_V,
        VOLTAGE_UNCERTAINTY_V,
        radii_m,
        radius_uncertainty_m,
    )
    question5_fit = manual_linear_fit(equation10_x, equation10_y)
    slope_q5 = question5_fit["slope"]
    intercept_q5 = question5_fit["intercept"]
    slope_q5_uncertainty = question5_fit["slope_uncertainty"]
    intercept_q5_uncertainty = question5_fit["intercept_uncertainty"]
    residuals_q5 = question5_fit["residuals"]
    r_squared_q5 = question5_fit["r_squared"]
    chi_square_q5, reduced_chi_square_q5, _ = chi_square_with_effective_uncertainty(
        equation10_y,
        question5_fit["fitted_y"],
        equation10_y_uncertainty,
        slope_q5,
        equation10_x_uncertainty,
    )
    e_over_m = (slope_q5 / k_constant) ** 2
    e_over_m_uncertainty = abs(2.0 * slope_q5 / (k_constant ** 2)) * slope_q5_uncertainty
    percent_difference_em = 100.0 * abs(e_over_m - E_OVER_M_ACCEPTED) / E_OVER_M_ACCEPTED

    header_lines = [
        "Charge-to-Mass Lab Analysis: External Magnetic Field from Linearized Helmholtz Relation",
        "=" * 88,
        f"Voltage V = {format_value_uncertainty(ACCELERATING_VOLTAGE_V, VOLTAGE_UNCERTAINTY_V, value_precision=7, uncertainty_precision=2)} V",
        f"Current uncertainty dI = {CURRENT_UNCERTAINTY_A:.1e} A",
        f"Radius uncertainty dr = {RADIUS_UNCERTAINTY_MM:.2f} mm = {radius_uncertainty_m:.1e} m",
    ]
    for line in header_lines:
        print(line)
        append_report_line(report_lines, line)

    print_data_table(
        CURRENT_A,
        CURRENT_UNCERTAINTY_A,
        radii_m,
        radius_uncertainty_m,
        inverse_radii,
        inverse_radii_uncertainty,
        fields_t,
        field_uncertainty_t,
        report_lines,
    )

    fit_result_lines = [
        "\nFit Results",
        f"Slope m = {format_value_uncertainty(slope, slope_uncertainty, value_precision=6, uncertainty_precision=2)} T·m",
        f"Intercept b = {format_value_uncertainty(intercept, intercept_uncertainty, value_precision=6, uncertainty_precision=2)} T",
        f"External field Be = -b = {format_value_uncertainty(be_t, be_uncertainty_t, value_precision=6, uncertainty_precision=2)} T",
        f"External field Be = {format_value_uncertainty(be_t, be_uncertainty_t, value_precision=6, uncertainty_precision=2, scale=1e6)} µT",
        f"R^2 = {r_squared:.6f}",
        f"Chi-square = {chi_square:.6g}",
        f"Reduced chi-square = {reduced_chi_square:.6g}",
        "Effective residual uncertainty uses sigma_eff = sqrt((dBc)^2 + (m d(1/r))^2) to include both x and y uncertainties.",
    ]
    for line in fit_result_lines:
        print(line)
        append_report_line(report_lines, line)

    theory_lines = [
        "\nTheory Comparison",
        f"alpha_theory = {format_value_uncertainty(alpha_theory, alpha_theory_uncertainty, value_precision=6, uncertainty_precision=2)} T·m",
        f"Signed percent difference (fit vs theory) = {format_value_uncertainty(percent_difference, percent_difference_uncertainty, value_precision=6, uncertainty_precision=2)} %",
        f"Absolute percent difference = {abs(percent_difference):.6g} %",
    ]
    for line in theory_lines:
        print(line)
        append_report_line(report_lines, line)

    interpretation_lines = build_interpretation_lines(be_t, be_uncertainty_t)
    for line in interpretation_lines:
        print(line)
        append_report_line(report_lines, line)

    consistency_ratio = abs(e_over_m - E_OVER_M_ACCEPTED) / e_over_m_uncertainty if e_over_m_uncertainty > 0 else np.inf
    if consistency_ratio <= 2.0:
        consistency_statement = "The Question 5 e/m result is reasonably consistent with the accepted value within uncertainty."
    else:
        consistency_statement = "The Question 5 e/m result is not reasonably consistent with the accepted value within uncertainty."

    question5_lines = [
        "\nQuestion 5: e/m from Equation 10",
        "Fitted equation: sqrt(V)/r = s I + c",
        f"Slope s = {format_value_uncertainty(slope_q5, slope_q5_uncertainty, value_precision=6, uncertainty_precision=2)}",
        f"Intercept c = {format_value_uncertainty(intercept_q5, intercept_q5_uncertainty, value_precision=6, uncertainty_precision=2)}",
        f"R^2 = {r_squared_q5:.6f}",
        f"Chi-square = {chi_square_q5:.6g}",
        f"Reduced chi-square = {reduced_chi_square_q5:.6g}",
        f"Helmholtz constant k = {k_constant:.8e}",
        f"Computed e/m = {format_value_uncertainty(e_over_m, e_over_m_uncertainty, value_precision=6, uncertainty_precision=2)} C/kg",
        f"Accepted e/m = {E_OVER_M_ACCEPTED:.6e} C/kg",
        f"Percent difference = {percent_difference_em:.6g} %",
        consistency_statement,
    ]
    for line in question5_lines:
        print(line)
        append_report_line(report_lines, line)

    write_report(report_lines, REPORT_OUTPUT_PATH)
    print(f"\nTerminal output also written to {REPORT_OUTPUT_PATH}")
    plot_results(
        inverse_radii,
        inverse_radii_uncertainty,
        fields_t,
        field_uncertainty_t,
        slope,
        intercept,
        residuals_t,
        chi_square,
        reduced_chi_square,
    )
    plot_question5_results(
        equation10_x,
        equation10_x_uncertainty,
        equation10_y,
        equation10_y_uncertainty,
        slope_q5,
        intercept_q5,
        residuals_q5,
        reduced_chi_square_q5,
    )
    print("\nGenerated files:")
    print(f"- {REPORT_OUTPUT_PATH}")
    print(f"- {QUESTION3_PLOT_PATH}")
    print(f"- {QUESTION5_PLOT_PATH}")


if __name__ == "__main__":
    main()