use super::*;

const RIDGE: f64 = 1.0e-8;

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

pub(super) fn fit_nominal_logit(
    data: &PreparedData,
    max_iter: usize,
    tol_x: f64,
) -> BuiltinResult<NominalFit> {
    let cols = data.cols;
    let logits = data.outcomes.len();
    let dim = cols
        .checked_mul(logits)
        .ok_or_else(|| internal("mnrfit: coefficient dimension overflow"))?;
    let matrix_len = dim
        .checked_mul(dim)
        .ok_or_else(|| internal("mnrfit: information matrix dimension overflow"))?;
    let mut beta = vec![0.0; dim];
    let mut information = vec![0.0; matrix_len];
    let mut gradient = vec![0.0; dim];
    for _ in 0..max_iter {
        information.fill(0.0);
        gradient.fill(0.0);
        for row in 0..data.rows {
            let probs = row_probabilities(&data.design, row, cols, &beta, logits);
            let weight = data.weights[row];
            for j in 0..logits {
                let residual = weight * (data.outcomes[j][row] - probs[j]);
                for a in 0..cols {
                    gradient[j * cols + a] += residual * data.design[row + a * data.rows];
                }
            }
            for j in 0..logits {
                for l in 0..logits {
                    let block = weight * probs[j] * if j == l { 1.0 - probs[l] } else { -probs[l] };
                    for a in 0..cols {
                        let za = data.design[row + a * data.rows];
                        for b in 0..cols {
                            let zb = data.design[row + b * data.rows];
                            information[(j * cols + a) * dim + (l * cols + b)] += block * za * zb;
                        }
                    }
                }
            }
        }
        for idx in 0..dim {
            information[idx * dim + idx] += RIDGE;
        }
        let delta = solve_linear(information.clone(), gradient.clone(), dim)?;
        let max_step = delta.iter().map(|value| value.abs()).fold(0.0, f64::max);
        for (coef, step) in beta.iter_mut().zip(delta.iter()) {
            *coef += *step;
        }
        if max_step <= tol_x.max(EPS) {
            break;
        }
    }
    if beta.iter().any(|value| !value.is_finite()) {
        return Err(convergence(
            "mnrfit: solver produced non-finite coefficients",
        ));
    }
    information.fill(0.0);
    for row in 0..data.rows {
        let probs = row_probabilities(&data.design, row, cols, &beta, logits);
        let weight = data.weights[row];
        for j in 0..logits {
            for l in 0..logits {
                let block = weight * probs[j] * if j == l { 1.0 - probs[l] } else { -probs[l] };
                for a in 0..cols {
                    let za = data.design[row + a * data.rows];
                    for b in 0..cols {
                        let zb = data.design[row + b * data.rows];
                        information[(j * cols + a) * dim + (l * cols + b)] += block * za * zb;
                    }
                }
            }
        }
    }
    for idx in 0..dim {
        information[idx * dim + idx] += RIDGE;
    }
    let covb = invert_matrix(information, dim)?;
    let se = (0..dim)
        .map(|idx| covb[idx * dim + idx].max(0.0).sqrt())
        .collect::<Vec<_>>();
    let t = beta
        .iter()
        .zip(se.iter())
        .map(|(coef, se)| if *se > 0.0 { coef / se } else { f64::NAN })
        .collect::<Vec<_>>();
    let p = t
        .iter()
        .map(|stat| {
            if stat.is_finite() {
                2.0 * (1.0 - distribution_math::standard_normal_cdf(stat.abs()))
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    let loglik = log_likelihood(data, &beta);
    let saturated = saturated_log_likelihood(data);
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

fn row_probabilities(
    design: &[f64],
    row: usize,
    cols: usize,
    beta: &[f64],
    logits: usize,
) -> Vec<f64> {
    let rows = design.len() / cols;
    let mut eta = Vec::with_capacity(logits);
    let mut max_eta = 0.0f64;
    for j in 0..logits {
        let mut value = 0.0;
        for col in 0..cols {
            value += design[row + col * rows] * beta[j * cols + col];
        }
        max_eta = max_eta.max(value);
        eta.push(value);
    }
    let base = (-max_eta).exp();
    let mut denom = base;
    let mut exp_eta = Vec::with_capacity(logits);
    for value in eta {
        let exp = (value - max_eta).exp();
        denom += exp;
        exp_eta.push(exp);
    }
    exp_eta.into_iter().map(|value| value / denom).collect()
}

fn log_likelihood(data: &PreparedData, beta: &[f64]) -> f64 {
    let logits = data.outcomes.len();
    let mut ll = 0.0;
    for row in 0..data.rows {
        let probs = row_probabilities(&data.design, row, data.cols, beta, logits);
        let mut baseline = 1.0 - probs.iter().sum::<f64>();
        baseline = baseline.max(EPS);
        for (j, prob) in probs.iter().enumerate() {
            let count = data.weights[row] * data.outcomes[j][row];
            if count > 0.0 {
                ll += count * prob.max(EPS).ln();
            }
        }
        let baseline_count =
            data.weights[row] * (1.0 - (0..logits).map(|j| data.outcomes[j][row]).sum::<f64>());
        if baseline_count > 0.0 {
            ll += baseline_count * baseline.ln();
        }
    }
    ll
}

fn saturated_log_likelihood(data: &PreparedData) -> f64 {
    let logits = data.outcomes.len();
    let mut ll = 0.0;
    for row in 0..data.rows {
        let weight = data.weights[row];
        for j in 0..logits {
            let p = data.outcomes[j][row];
            if p > 0.0 {
                ll += weight * p * p.max(EPS).ln();
            }
        }
        let baseline_p = (1.0 - (0..logits).map(|j| data.outcomes[j][row]).sum::<f64>()).max(0.0);
        if baseline_p > 0.0 {
            ll += weight * baseline_p * baseline_p.max(EPS).ln();
        }
    }
    ll
}

fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> BuiltinResult<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in col + 1..n {
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
            }
            b.swap(col, pivot);
        }
        let diag = a[col * n + col];
        for row in col + 1..n {
            let factor = a[row * n + col] / diag;
            a[row * n + col] = 0.0;
            for j in col + 1..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut rhs = b[row];
        for col in row + 1..n {
            rhs -= a[row * n + col] * x[col];
        }
        x[row] = rhs / a[row * n + row];
    }
    Ok(x)
}

fn invert_matrix(a: Vec<f64>, n: usize) -> BuiltinResult<Vec<f64>> {
    let mut out = vec![0.0; n * n];
    for col in 0..n {
        let mut rhs = vec![0.0; n];
        rhs[col] = 1.0;
        let solved = solve_linear(a.clone(), rhs, n)?;
        for row in 0..n {
            out[row + col * n] = solved[row];
        }
    }
    Ok(out)
}
