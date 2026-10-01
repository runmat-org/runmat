use super::*;

pub(super) fn spline_coefs_for_series(
    x: &[f64],
    y: &[f64],
    endpoint_slopes: Option<EndpointSlopes>,
) -> BuiltinResult<Vec<[f64; 4]>> {
    let n = x.len();
    if let Some(slopes) = endpoint_slopes {
        return clamped_spline_coefs(x, y, slopes);
    }
    if n == 2 {
        let h = x[1] - x[0];
        return Ok(vec![[0.0, 0.0, (y[1] - y[0]) / h, y[0]]]);
    }
    if n == 3 {
        return Ok(quadratic_as_piecewise_cubic(x, y));
    }

    let h: Vec<f64> = x.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let mut a = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];

    a[0][0] = -h[1];
    a[0][1] = h[0] + h[1];
    a[0][2] = -h[0];
    for i in 1..(n - 1) {
        a[i][i - 1] = h[i - 1];
        a[i][i] = 2.0 * (h[i - 1] + h[i]);
        a[i][i + 1] = h[i];
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }
    a[n - 1][n - 3] = -h[n - 2];
    a[n - 1][n - 2] = h[n - 3] + h[n - 2];
    a[n - 1][n - 1] = -h[n - 3];

    let second = solve_dense(a, rhs)?;
    Ok(second_derivatives_to_coefs(x, y, &second))
}

/// Solve the strictly diagonally dominant tridiagonal second-derivative
/// system for a clamped cubic spline in linear space and time.
fn clamped_spline_coefs(
    x: &[f64],
    y: &[f64],
    slopes: EndpointSlopes,
) -> BuiltinResult<Vec<[f64; 4]>> {
    let n = x.len();
    let widths: Vec<f64> = x.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let secants: Vec<f64> = y
        .windows(2)
        .zip(widths.iter())
        .map(|(pair, &width)| (pair[1] - pair[0]) / width)
        .collect();
    let mut diagonal = vec![0.0; n];
    let mut upper = vec![0.0; n - 1];
    let mut rhs = vec![0.0; n];
    diagonal[0] = 2.0 * widths[0];
    upper[0] = widths[0];
    rhs[0] = 6.0 * (secants[0] - slopes.first);
    for i in 1..n - 1 {
        diagonal[i] = 2.0 * (widths[i - 1] + widths[i]);
        upper[i] = widths[i];
        rhs[i] = 6.0 * (secants[i] - secants[i - 1]);
    }
    diagonal[n - 1] = 2.0 * widths[n - 2];
    rhs[n - 1] = 6.0 * (slopes.last - secants[n - 2]);

    for i in 1..n {
        let lower = widths[i - 1];
        let factor = lower / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    let mut second = vec![0.0; n];
    second[n - 1] = rhs[n - 1] / diagonal[n - 1];
    for i in (0..n - 1).rev() {
        second[i] = (rhs[i] - upper[i] * second[i + 1]) / diagonal[i];
    }
    if second.iter().any(|value| !value.is_finite()) {
        return Err(interp_error(
            "spline",
            "spline: invalid endpoint slope system",
        ));
    }
    Ok(second_derivatives_to_coefs(x, y, &second))
}

fn quadratic_as_piecewise_cubic(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let y0 = y[0];
    let y1 = y[1];
    let y2 = y[2];
    let c2 =
        y0 / ((x0 - x1) * (x0 - x2)) + y1 / ((x1 - x0) * (x1 - x2)) + y2 / ((x2 - x0) * (x2 - x1));
    let c1 = -y0 * (x1 + x2) / ((x0 - x1) * (x0 - x2))
        - y1 * (x0 + x2) / ((x1 - x0) * (x1 - x2))
        - y2 * (x0 + x1) / ((x2 - x0) * (x2 - x1));
    let c0 = y0 * x1 * x2 / ((x0 - x1) * (x0 - x2))
        + y1 * x0 * x2 / ((x1 - x0) * (x1 - x2))
        + y2 * x0 * x1 / ((x2 - x0) * (x2 - x1));
    (0..2)
        .map(|i| {
            let xi = x[i];
            let a = c2 * xi * xi + c1 * xi + c0;
            let b = 2.0 * c2 * xi + c1;
            [0.0, c2, b, a]
        })
        .collect()
}

fn second_derivatives_to_coefs(x: &[f64], y: &[f64], second: &[f64]) -> Vec<[f64; 4]> {
    let mut coefs = Vec::with_capacity(x.len() - 1);
    for i in 0..(x.len() - 1) {
        let h = x[i + 1] - x[i];
        let a = y[i];
        let b = (y[i + 1] - y[i]) / h - h * (2.0 * second[i] + second[i + 1]) / 6.0;
        let c = second[i] / 2.0;
        let d = (second[i + 1] - second[i]) / (6.0 * h);
        coefs.push([d, c, b, a]);
    }
    coefs
}

pub(super) fn pchip_coefs_for_series(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    let n = x.len();
    let h: Vec<f64> = x.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let delta: Vec<f64> = y
        .windows(2)
        .zip(h.iter())
        .map(|(pair, &width)| (pair[1] - pair[0]) / width)
        .collect();

    let mut slopes = vec![0.0; n];
    if n == 2 {
        slopes[0] = delta[0];
        slopes[1] = delta[0];
    } else {
        slopes[0] = endpoint_slope(h[0], h[1], delta[0], delta[1]);
        slopes[n - 1] = endpoint_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        for i in 1..(n - 1) {
            if delta[i - 1] == 0.0 || delta[i] == 0.0 || delta[i - 1].signum() != delta[i].signum()
            {
                slopes[i] = 0.0;
            } else {
                let w1 = 2.0 * h[i] + h[i - 1];
                let w2 = h[i] + 2.0 * h[i - 1];
                slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }
    }

    let mut coefs = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let c0 = y[i];
        let c1 = slopes[i];
        let c2 = (3.0 * delta[i] - 2.0 * slopes[i] - slopes[i + 1]) / h[i];
        let c3 = (slopes[i] + slopes[i + 1] - 2.0 * delta[i]) / (h[i] * h[i]);
        coefs.push([c3, c2, c1, c0]);
    }
    coefs
}

fn endpoint_slope(h0: f64, h1: f64, del0: f64, del1: f64) -> f64 {
    let mut slope = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1);
    if slope.signum() != del0.signum() {
        slope = 0.0;
    } else if del0.signum() != del1.signum() && slope.abs() > (3.0 * del0).abs() {
        slope = 3.0 * del0;
    }
    slope
}

fn solve_dense(mut a: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> BuiltinResult<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..n {
            let candidate = a[row][col].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if pivot_abs <= 1.0e-14 {
            return Err(interp_error(
                "spline",
                "spline: singular interpolation system",
            ));
        }
        if pivot != col {
            a.swap(pivot, col);
            rhs.swap(pivot, col);
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            a[row][col] = 0.0;
            for k in (col + 1)..n {
                a[row][k] -= factor * a[col][k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for col in (row + 1)..n {
            acc -= a[row][col] * solution[col];
        }
        solution[row] = acc / a[row][row];
    }
    Ok(solution)
}

pub(super) fn validate_finite_coefs(coefs: &[f64], name: &'static str) -> BuiltinResult<()> {
    if coefs.iter().any(|value| !value.is_finite()) {
        return Err(interp_error(
            name,
            format!("{name}: interpolation coefficients must be finite"),
        ));
    }
    Ok(())
}
