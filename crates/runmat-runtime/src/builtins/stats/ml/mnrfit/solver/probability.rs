use super::*;

pub(super) async fn row_probabilities(
    design: &[f64],
    row: usize,
    cols: usize,
    beta: &[f64],
    logits: usize,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
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

pub(super) async fn log_likelihood(
    data: &PreparedData,
    beta: &[f64],
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
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
        if (row + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((data.rows % 16_384) as u64).await?;
    Ok(ll)
}

pub(super) async fn saturated_log_likelihood(
    data: &PreparedData,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<f64> {
    let logits = data.outcomes.len();
    let mut ll = 0.0;
    for row in 0..data.rows {
        check()?;
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
        if (row + 1).is_multiple_of(16_384) {
            cooperation.completed(16_384).await?;
        }
    }
    cooperation.completed((data.rows % 16_384) as u64).await?;
    Ok(ll)
}
