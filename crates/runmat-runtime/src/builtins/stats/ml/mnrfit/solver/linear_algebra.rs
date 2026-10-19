use super::*;

pub(super) async fn solve_linear(
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    n: usize,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<f64>> {
    for col in 0..n {
        check()?;
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in col + 1..n {
            let value = a[row * n + col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
            if (row - col).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation
            .completed(((n - col - 1) % 16_384) as u64)
            .await?;
        if best <= EPS || !best.is_finite() {
            return Err(convergence(
                "mnrfit: information matrix is singular or ill-conditioned",
            ));
        }
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
                if (j + 1).is_multiple_of(16_384) {
                    cooperation.completed(16_384).await?;
                }
            }
            cooperation.completed((n % 16_384) as u64).await?;
            b.swap(col, pivot);
        }
        let diag = a[col * n + col];
        for row in col + 1..n {
            let factor = a[row * n + col] / diag;
            a[row * n + col] = 0.0;
            let mut completed = 0u64;
            for j in col + 1..n {
                a[row * n + j] -= factor * a[col * n + j];
                completed += 1;
                if completed == 16_384 {
                    cooperation.completed(completed).await?;
                    completed = 0;
                }
            }
            cooperation.completed(completed).await?;
            b[row] -= factor * b[col];
        }
        cooperation.completed(1).await?;
    }
    let mut x = filled_cooperatively(n, 0.0, "solver result", check, cooperation).await?;
    for row in (0..n).rev() {
        check()?;
        let mut rhs = b[row];
        let mut completed = 0u64;
        for col in row + 1..n {
            rhs -= a[row * n + col] * x[col];
            completed += 1;
            if completed == 16_384 {
                cooperation.completed(completed).await?;
                completed = 0;
            }
        }
        cooperation.completed(completed).await?;
        x[row] = rhs / a[row * n + row];
    }
    Ok(x)
}

pub(super) async fn invert_matrix(
    a: Vec<f64>,
    n: usize,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut out = filled_cooperatively(
        checked_elements(n, n, "covariance matrix")?,
        0.0,
        "covariance matrix",
        check,
        cooperation,
    )
    .await?;
    for col in 0..n {
        check()?;
        let mut rhs = filled_cooperatively(n, 0.0, "covariance solve", check, cooperation).await?;
        rhs[col] = 1.0;
        let solved = solve_linear(
            copy_cooperatively(&a, "covariance solve copy", check, cooperation).await?,
            rhs,
            n,
            check,
            cooperation,
        )
        .await?;
        for row in 0..n {
            out[row + col * n] = solved[row];
            if (row + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((n % 16_384) as u64).await?;
    }
    Ok(out)
}
