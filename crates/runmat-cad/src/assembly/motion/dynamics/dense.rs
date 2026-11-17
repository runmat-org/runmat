//! Small dense linear algebra for the equations of motion: the constraint systems of a
//! mechanism have tens of rows, so dense Cholesky factors are cheap and exact enough.

/// Solves `(A + εI) x = b` for a symmetric positive semi-definite `n × n` matrix `a`
/// (row-major), with `ε` a small multiple of the largest diagonal entry so dependent rows (a
/// redundant mate) still factor; the solution is then the minimum-norm one in the limit.
/// `None` when the factorisation breaks down (a non-finite entry).
pub fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(Vec::new());
    }
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0_f64, f64::max).max(1e-300);
    let eps = 1e-12 * scale;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            if i == j {
                s += eps;
            }
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !s.is_finite() {
                    return None;
                }
                // A dependent row's pivot is the regularisation alone.
                l[i * n + i] = if s > 0.0 { s.sqrt() } else { eps.sqrt() };
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// `y = A x` for a row-major `rows × cols` matrix.
pub fn mul(a: &[f64], x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a[r * cols + c] * x[c]).sum())
        .collect()
}

/// `y = Aᵀ x` for a row-major `rows × cols` matrix.
pub fn mul_t(a: &[f64], x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut y = vec![0.0; cols];
    for (row, &xr) in a.chunks(cols).take(rows).zip(x) {
        if xr == 0.0 {
            continue;
        }
        for (yc, &arc) in y.iter_mut().zip(row) {
            *yc += arc * xr;
        }
    }
    y
}

pub fn mat3_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

pub fn mat3_t_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2])
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// The inverse of a symmetric positive definite 3 × 3 matrix (an inertia tensor).
pub fn inverse3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let det = m[0][0] * c(1, 1, 2, 2) - m[0][1] * c(1, 0, 2, 2) + m[0][2] * c(1, 0, 2, 1);
    if det.is_nan() || det.abs() <= 1e-300 || !det.is_finite() {
        return None;
    }
    let inv = [
        [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
        [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
        [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
    ];
    Some(inv.map(|row| row.map(|v| v / det)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spd_solve_and_inverse() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let b = [1.0, 2.0, 3.0];
        let x = solve_spd(&a, &b, 3).unwrap();
        let back = mul(&a, &x, 3, 3);
        for i in 0..3 {
            assert!((back[i] - b[i]).abs() < 1e-9);
        }
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let inv = inverse3(&m).unwrap();
        let e = mat3_vec(&inv, mat3_vec(&m, [1.0, -2.0, 0.5]));
        assert!((e[0] - 1.0).abs() < 1e-12 && (e[1] + 2.0).abs() < 1e-12 && (e[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_repeated_row_still_solves() {
        // Two identical constraint rows (a redundant mate): the system is consistent.
        let a = [2.0, 2.0, 2.0, 2.0];
        let x = solve_spd(&a, &[4.0, 4.0], 2).unwrap();
        assert!((2.0 * x[0] + 2.0 * x[1] - 4.0).abs() < 1e-6, "{x:?}");
    }
}
