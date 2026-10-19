use super::*;

mod linear_algebra;
mod probability;

use self::linear_algebra::{invert_matrix, solve_linear};
use self::probability::{log_likelihood, row_probabilities, saturated_log_likelihood};
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
    cooperation: &mut FitCooperation,
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
            if (row + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((data.rows % 16_384) as u64).await?;
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
