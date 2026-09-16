use std::collections::HashMap;

use crate::contracts::FeaRunError;

use super::{RandomResponseIntegrationEvidence, RandomResponseIntegrationPolicy};

pub(super) struct IntervalIntegration {
    pub values: Vec<f64>,
    pub evaluated_frequency_count: usize,
    pub refinement_level: u8,
    pub estimated_absolute_error: f64,
    pub estimated_relative_error: f64,
}

pub(super) struct IntegratedResponse {
    pub values: Vec<f64>,
    pub evidence: RandomResponseIntegrationEvidence,
}

pub(super) fn integrate_interval<F>(
    left: f64,
    right: f64,
    policy: RandomResponseIntegrationPolicy,
    mut evaluate: F,
) -> Result<IntervalIntegration, FeaRunError>
where
    F: FnMut(f64) -> Result<Vec<f64>, FeaRunError>,
{
    if !left.is_finite() || !right.is_finite() || left < 0.0 || left >= right {
        return Err(FeaRunError::Solve(
            "random-response integration interval is invalid".into(),
        ));
    }
    let mut cache = HashMap::<u64, Vec<f64>>::new();
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
        let mut maximum_absolute_error = 0.0_f64;
        let mut maximum_relative_error = 0.0_f64;
        for (coarse, value) in prior.iter().zip(&refined) {
            let absolute_error = (value - coarse).abs() / 3.0;
            let relative_error = absolute_error / value.abs().max(f64::MIN_POSITIVE);
            maximum_absolute_error = maximum_absolute_error.max(absolute_error);
            maximum_relative_error = maximum_relative_error.max(relative_error);
            if absolute_error
                > policy.absolute_variance_tolerance
                    + policy.relative_variance_tolerance * value.abs()
            {
                converged = false;
            }
        }
        if converged {
            return Ok(IntervalIntegration {
                values: refined,
                evaluated_frequency_count: cache.len(),
                refinement_level: level,
                estimated_absolute_error: maximum_absolute_error,
                estimated_relative_error: maximum_relative_error,
            });
        }
        prior = refined;
    }
    Err(FeaRunError::Solve(format!(
        "random-response variance integration did not converge on [{left}, {right}] Hz after {} refinements",
        policy.maximum_refinement_levels
    )))
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
    let intervals = breakpoints
        .windows(2)
        .map(|interval| integrate_interval(interval[0], interval[1], policy, &mut evaluate))
        .collect::<Result<Vec<_>, _>>()?;
    let value_count = intervals
        .first()
        .map(|interval| interval.values.len())
        .ok_or_else(|| FeaRunError::Solve("random-response integration is empty".into()))?;
    let mut total = vec![0.0; value_count];
    let mut compensation = vec![0.0; value_count];
    for interval in &intervals {
        if interval.values.len() != value_count {
            return Err(FeaRunError::Solve(
                "random-response interval results have unequal shapes".into(),
            ));
        }
        compensated_add(&mut total, &mut compensation, &interval.values);
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
            evaluated_frequency_count: intervals
                .iter()
                .map(|interval| interval.evaluated_frequency_count)
                .sum(),
            maximum_refinement_level_used: intervals
                .iter()
                .map(|interval| interval.refinement_level)
                .max()
                .unwrap_or(0),
            maximum_estimated_absolute_variance_error: intervals
                .iter()
                .map(|interval| interval.estimated_absolute_error)
                .fold(0.0_f64, f64::max),
            maximum_estimated_relative_variance_error: intervals
                .iter()
                .map(|interval| interval.estimated_relative_error)
                .fold(0.0_f64, f64::max),
        },
    })
}

pub(super) fn compensated_add(total: &mut [f64], compensation: &mut [f64], values: &[f64]) {
    for ((sum, correction), value) in total.iter_mut().zip(compensation).zip(values) {
        let corrected = value - *correction;
        let next = *sum + corrected;
        *correction = (next - *sum) - corrected;
        *sum = next;
    }
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
            &[0.0, 0.5, 1.0],
            2,
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
        assert!(integrated.evidence.evaluated_frequency_count > 4);
    }
}
