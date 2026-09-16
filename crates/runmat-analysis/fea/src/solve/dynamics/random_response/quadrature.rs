use std::collections::HashMap;

use crate::contracts::FeaRunError;

use super::{RandomResponseIntegrationEvidence, RandomResponseIntegrationPolicy};

pub(super) struct IntegratedResponse {
    pub values: Vec<f64>,
    pub evidence: RandomResponseIntegrationEvidence,
}

pub(super) fn integrate_breakpoints<F>(
    breakpoints: &[f64],
    authored_interval_count: usize,
    policy: RandomResponseIntegrationPolicy,
    mut evaluate: F,
) -> Result<IntegratedResponse, FeaRunError>
where
    F: FnMut(f64) -> Result<Vec<f64>, FeaRunError>,
{
    if breakpoints.len() < 2 || breakpoints.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(FeaRunError::Solve(
            "random-response integration breakpoints are not strictly increasing".into(),
        ));
    }
    let mut cache = HashMap::<u64, Vec<f64>>::new();
    let mut total = Vec::<f64>::new();
    let mut compensation = Vec::<f64>::new();
    let mut maximum_level = 0;
    let mut maximum_absolute_error = 0.0_f64;
    let mut maximum_relative_error = 0.0_f64;

    for interval in breakpoints.windows(2) {
        let left = interval[0];
        let right = interval[1];
        let left_values = evaluate_cached(left, &mut cache, &mut evaluate)?;
        let right_values = evaluate_cached(right, &mut cache, &mut evaluate)?;
        if left_values.len() != right_values.len() || left_values.is_empty() {
            return Err(FeaRunError::Solve(
                "random-response integrand changed shape".into(),
            ));
        }
        let mut prior = left_values
            .iter()
            .zip(&right_values)
            .map(|(left_value, right_value)| 0.5 * (right - left) * (left_value + right_value))
            .collect::<Vec<_>>();
        let mut accepted = None;
        for level in 1..=policy.maximum_refinement_levels {
            let panel_count = 1_usize << level;
            let step = (right - left) / panel_count as f64;
            let mut midpoint_sum = vec![0.0; prior.len()];
            for odd_index in (1..panel_count).step_by(2) {
                let frequency = left + odd_index as f64 * step;
                let values = evaluate_cached(frequency, &mut cache, &mut evaluate)?;
                if values.len() != midpoint_sum.len() {
                    return Err(FeaRunError::Solve(
                        "random-response integrand changed shape".into(),
                    ));
                }
                for (sum, value) in midpoint_sum.iter_mut().zip(values) {
                    *sum += value;
                }
            }
            let refined = prior
                .iter()
                .zip(midpoint_sum)
                .map(|(coarse, midpoints)| 0.5 * coarse + step * midpoints)
                .collect::<Vec<_>>();
            let mut converged = true;
            let mut refinement_absolute_error = 0.0_f64;
            let mut refinement_relative_error = 0.0_f64;
            for (coarse, value) in prior.iter().zip(&refined) {
                let absolute_error = (value - coarse).abs() / 3.0;
                let relative_error = absolute_error / value.abs().max(f64::MIN_POSITIVE);
                refinement_absolute_error = refinement_absolute_error.max(absolute_error);
                refinement_relative_error = refinement_relative_error.max(relative_error);
                if absolute_error
                    > policy.absolute_variance_tolerance
                        + policy.relative_variance_tolerance * value.abs()
                {
                    converged = false;
                }
            }
            maximum_level = maximum_level.max(level);
            if converged {
                maximum_absolute_error = maximum_absolute_error.max(refinement_absolute_error);
                maximum_relative_error = maximum_relative_error.max(refinement_relative_error);
                accepted = Some(refined);
                break;
            }
            prior = refined;
        }
        let interval_values = accepted.ok_or_else(|| {
            FeaRunError::Solve(format!(
                "random-response variance integration did not converge on [{left}, {right}] Hz after {} refinements",
                policy.maximum_refinement_levels
            ))
        })?;
        if total.is_empty() {
            total.resize(interval_values.len(), 0.0);
            compensation.resize(interval_values.len(), 0.0);
        }
        for ((sum, correction), value) in
            total.iter_mut().zip(&mut compensation).zip(interval_values)
        {
            let corrected = value - *correction;
            let next = *sum + corrected;
            *correction = (next - *sum) - corrected;
            *sum = next;
        }
    }
    if total.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(FeaRunError::Solve(
            "random-response integration produced invalid variance".into(),
        ));
    }
    Ok(IntegratedResponse {
        values: total,
        evidence: RandomResponseIntegrationEvidence {
            authored_interval_count,
            resonance_split_count: breakpoints
                .len()
                .saturating_sub(authored_interval_count + 1),
            evaluated_frequency_count: cache.len(),
            maximum_refinement_level_used: maximum_level,
            maximum_estimated_absolute_variance_error: maximum_absolute_error,
            maximum_estimated_relative_variance_error: maximum_relative_error,
        },
    })
}

fn evaluate_cached<F>(
    frequency: f64,
    cache: &mut HashMap<u64, Vec<f64>>,
    evaluate: &mut F,
) -> Result<Vec<f64>, FeaRunError>
where
    F: FnMut(f64) -> Result<Vec<f64>, FeaRunError>,
{
    let key = frequency.to_bits();
    if let Some(values) = cache.get(&key) {
        return Ok(values.clone());
    }
    let values = evaluate(frequency)?;
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(FeaRunError::Solve(
            "random-response integrand is not finite and nonnegative".into(),
        ));
    }
    cache.insert(key, values.clone());
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_rule_integrates_quadratic_density() {
        let integrated = integrate_breakpoints(
            &[0.0, 1.0],
            1,
            RandomResponseIntegrationPolicy {
                modal_damping_ratio: 0.05,
                minimum_cumulative_mass_ratio: 0.9,
                relative_variance_tolerance: 1.0e-7,
                absolute_variance_tolerance: 1.0e-12,
                maximum_refinement_levels: 12,
            },
            |frequency| Ok(vec![frequency * frequency]),
        )
        .unwrap();
        assert!((integrated.values[0] - 1.0 / 3.0).abs() < 1.0e-7);
        assert!(integrated.evidence.evaluated_frequency_count > 2);
    }
}
