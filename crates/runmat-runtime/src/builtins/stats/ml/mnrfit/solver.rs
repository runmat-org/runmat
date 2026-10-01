use super::*;
use crate::builtins::stats::summary::distribution_math;

const RIDGE: f64 = 1.0e-8;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub(super) struct NominalFit {
    pub(super) beta: Vec<f64>,
    pub(super) covb: Vec<f64>,
    pub(super) se: Vec<f64>,
    pub(super) t: Vec<f64>,
    pub(super) p: Vec<f64>,
    pub(super) deviance: f64,
    pub(super) dfe: f64,
}

pub(super) async fn fit_nominal_logit(
    data: &PreparedData,
    max_iter: usize,
    tol_x: f64,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation<'_>,
) -> BuiltinResult<NominalFit> {
    check()?;
    let cols = data.cols;
    let logits = data.outcomes.len();
    let dim = checked_elements(cols, logits, "coefficients")?;
    let matrix_len = checked_elements(dim, dim, "information matrix")?;
    let mut beta = filled_cooperatively(dim, 0.0, "coefficients", check, cooperation).await?;
    let mut information =
        filled_cooperatively(matrix_len, 0.0, "information matrix", check, cooperation).await?;
    let mut gradient = filled_cooperatively(dim, 0.0, "gradient", check, cooperation).await?;
    for _ in 0..max_iter {
        check()?;
        for chunk in information.chunks_mut(16_384) {
            chunk.fill(0.0);
            cooperation.completed(chunk.len() as u64).await?;
        }
        for chunk in gradient.chunks_mut(16_384) {
            chunk.fill(0.0);
            cooperation.completed(chunk.len() as u64).await?;
        }
        for row in 0..data.rows {
            check()?;
            cooperation.completed(1).await?;
            let probs =
                row_probabilities(&data.design, row, cols, &beta, logits, check, cooperation)
                    .await?;
            let weight = data.weights[row];
            for j in 0..logits {
                check()?;
                let residual = weight * (data.outcomes[j][row] - probs[j]);
                for a in 0..cols {
                    gradient[j * cols + a] += residual * data.design[row + a * data.rows];
                    if (a + 1).is_multiple_of(16_384) {
                        cooperation.completed(16_384).await?;
                    }
                }
                cooperation.completed((cols % 16_384) as u64).await?;
            }
            for j in 0..logits {
                for l in 0..logits {
                    check()?;
                    let block = weight * probs[j] * if j == l { 1.0 - probs[l] } else { -probs[l] };
                    for a in 0..cols {
                        check()?;
                        let za = data.design[row + a * data.rows];
                        for b in 0..cols {
                            let zb = data.design[row + b * data.rows];
                            information[(j * cols + a) * dim + (l * cols + b)] += block * za * zb;
                            if (b + 1).is_multiple_of(16_384) {
                                cooperation.completed(16_384).await?;
                            }
                        }
                        cooperation.completed((cols % 16_384) as u64).await?;
                    }
                }
            }
        }
        for idx in 0..dim {
            information[idx * dim + idx] += RIDGE;
        }
        let delta = solve_linear(
            copy_cooperatively(&information, "information solve copy", check, cooperation).await?,
            copy_cooperatively(&gradient, "gradient solve copy", check, cooperation).await?,
            dim,
            check,
            cooperation,
        )
        .await?;
        let mut max_step = 0.0f64;
        for (index, step) in delta.iter().enumerate() {
            max_step = max_step.max(step.abs());
            if (index + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((delta.len() % 16_384) as u64).await?;
        for (index, (coef, step)) in beta.iter_mut().zip(delta.iter()).enumerate() {
            *coef += *step;
            if (index + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((beta.len() % 16_384) as u64).await?;
        if max_step <= tol_x.max(EPS) {
            break;
        }
    }
    for (index, value) in beta.iter().enumerate() {
        if !value.is_finite() {
            return Err(convergence(
                "mnrfit: solver produced non-finite coefficients",
            ));
        }
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((beta.len() % 16_384) as u64).await?;
    for chunk in information.chunks_mut(16_384) {
        chunk.fill(0.0);
        cooperation.completed(chunk.len() as u64).await?;
    }
    for row in 0..data.rows {
        check()?;
        cooperation.completed(1).await?;
        let probs =
            row_probabilities(&data.design, row, cols, &beta, logits, check, cooperation).await?;
        let weight = data.weights[row];
        for j in 0..logits {
            check()?;
            for l in 0..logits {
                check()?;
                let block = weight * probs[j] * if j == l { 1.0 - probs[l] } else { -probs[l] };
                for a in 0..cols {
                    check()?;
                    let za = data.design[row + a * data.rows];
                    for b in 0..cols {
                        let zb = data.design[row + b * data.rows];
                        information[(j * cols + a) * dim + (l * cols + b)] += block * za * zb;
                        if (b + 1).is_multiple_of(16_384) {
                            cooperation.completed(16_384).await?;
                        }
                    }
                    cooperation.completed((cols % 16_384) as u64).await?;
                }
            }
        }
    }
    for idx in 0..dim {
        information[idx * dim + idx] += RIDGE;
    }
    let covb = invert_matrix(information, dim, check, cooperation).await?;
    let mut se = reserved(dim, "standard errors")?;
    let mut t = reserved(dim, "coefficient statistics")?;
    let mut p = reserved(dim, "coefficient probabilities")?;
    for idx in 0..dim {
        let error = covb[idx * dim + idx].max(0.0).sqrt();
        se.push(error);
        t.push(if error > 0.0 {
            beta[idx] / error
        } else {
            f64::NAN
        });
        let stat = t[idx];
        p.push(if stat.is_finite() {
            2.0 * (1.0 - distribution_math::standard_normal_cdf(stat.abs()))
        } else {
            f64::NAN
        });
        if (idx + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((dim % 16_384) as u64).await?;
    let loglik = log_likelihood(data, &beta, check, cooperation).await?;
    let saturated = saturated_log_likelihood(data, check, cooperation).await?;
    check()?;
    let dfe = (data.observations - dim as f64).max(0.0);
    Ok(NominalFit {
        beta,
        covb,
        se,
        t,
        p,
        deviance: (2.0 * (saturated - loglik)).max(0.0),
        dfe,
    })
}

async fn row_probabilities(
    design: &[f64],
    row: usize,
    cols: usize,
    beta: &[f64],
    logits: usize,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation<'_>,
) -> BuiltinResult<Vec<f64>> {
    let rows = design.len() / cols;
    let mut eta = reserved(logits, "row scores")?;
    let mut max_eta = 0.0f64;
    for j in 0..logits {
        check()?;
        let mut value = 0.0;
        for col in 0..cols {
            value += design[row + col * rows] * beta[j * cols + col];
            if (col + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((cols % 16_384) as u64).await?;
        max_eta = max_eta.max(value);
        eta.push(value);
    }
    let base = (-max_eta).exp();
    let mut denom = base;
    let mut exp_eta = reserved(logits, "row probabilities")?;
    for (index, value) in eta.into_iter().enumerate() {
        let exp = (value - max_eta).exp();
        denom += exp;
        exp_eta.push(exp);
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((logits % 16_384) as u64).await?;
    for (index, value) in exp_eta.iter_mut().enumerate() {
        *value /= denom;
        if (index + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((logits % 16_384) as u64).await?;
    Ok(exp_eta)
}

async fn log_likelihood(
    data: &PreparedData,
    beta: &[f64],
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation<'_>,
) -> BuiltinResult<f64> {
    let logits = data.outcomes.len();
    let mut ll = 0.0;
    for row in 0..data.rows {
        check()?;
        let probs = row_probabilities(
            &data.design,
            row,
            data.cols,
            beta,
            logits,
            check,
            cooperation,
        )
        .await?;
        cooperation.completed(1).await?;
        let mut probability_total = 0.0;
        for (index, probability) in probs.iter().enumerate() {
            probability_total += probability;
            if (index + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((probs.len() % 16_384) as u64).await?;
        let mut baseline = 1.0 - probability_total;
        baseline = baseline.max(EPS);
        for (j, prob) in probs.iter().enumerate() {
            let count = data.weights[row] * data.outcomes[j][row];
            if count > 0.0 {
                ll += count * prob.max(EPS).ln();
            }
            if (j + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((logits % 16_384) as u64).await?;
        let mut outcome_total = 0.0;
        for j in 0..logits {
            outcome_total += data.outcomes[j][row];
            if (j + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((logits % 16_384) as u64).await?;
        let baseline_count = data.weights[row] * (1.0 - outcome_total);
        if baseline_count > 0.0 {
            ll += baseline_count * baseline.ln();
        }
    }
    Ok(ll)
}

async fn saturated_log_likelihood(
    data: &PreparedData,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation<'_>,
) -> BuiltinResult<f64> {
    let logits = data.outcomes.len();
    let mut ll = 0.0;
    for row in 0..data.rows {
        check()?;
        cooperation.completed(1).await?;
        let weight = data.weights[row];
        let mut outcome_total = 0.0;
        for j in 0..logits {
            let p = data.outcomes[j][row];
            outcome_total += p;
            if p > 0.0 {
                ll += weight * p * p.max(EPS).ln();
            }
            if (j + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((logits % 16_384) as u64).await?;
        let baseline_p = (1.0 - outcome_total).max(0.0);
        if baseline_p > 0.0 {
            ll += weight * baseline_p * baseline_p.max(EPS).ln();
        }
    }
    Ok(ll)
}

async fn solve_linear(
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    n: usize,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation<'_>,
) -> BuiltinResult<Vec<f64>> {
    for col in 0..n {
        check()?;
        cooperation.completed(1).await?;
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in col + 1..n {
            check()?;
            cooperation.completed(1).await?;
            let value = a[row * n + col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
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

async fn invert_matrix(
    a: Vec<f64>,
    n: usize,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation<'_>,
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
