//! Roundoff-aware embedded Gauss–Kronrod error estimates.
//!
//! The adjustment follows QUADPACK QK15/QK15I's resabs/resasc policy. We
//! retain scaled terms through cancellation and transformed infinite domains.
//! Source: https://www.netlib.org/quadpack/qk15i.f

use std::sync::Arc;

use super::adaptive::QuadratureEvaluator;
use super::algebra::{
    ArrayEstimate, EmbeddedRuleInputs, QuadratureArithmeticError, COMPONENT_CHUNK,
};
use super::geometry::ScaledEstimate;

const ROUNDOFF_FACTOR: f64 = 50.0 * f64::EPSILON;

pub(super) fn adjusted_error(
    raw: ScaledEstimate,
    resabs: ScaledEstimate,
    resasc: ScaledEstimate,
) -> ScaledEstimate {
    let mut error = raw.abs();
    if !resasc.is_zero() && !error.is_zero() {
        let ratio = error.multiply_f64(200.0).divide(resasc);
        let scale = if ratio.abs_le(ScaledEstimate::from_f64(1.0)) {
            // This branch proves a bounded ratio; underflow to zero is the
            // limiting adjustment, while the independent roundoff floor stays.
            ratio.trace_f64().powf(1.5)
        } else {
            1.0
        };
        error = resasc.multiply_f64(scale);
    }
    if !resabs.abs_le(ScaledEstimate::from_f64(
        f64::MIN_POSITIVE / ROUNDOFF_FACTOR,
    )) {
        error = error.max(resabs.multiply_f64(ROUNDOFF_FACTOR));
    }
    error
}

fn sample_terms(
    value: f64,
    jacobian: ScaledEstimate,
    base_weight: ScaledEstimate,
    mean: ScaledEstimate,
) -> (ScaledEstimate, ScaledEstimate) {
    let transformed = jacobian.multiply_f64(value);
    (
        base_weight.multiply(transformed.abs()),
        base_weight.multiply(transformed.subtract(mean).abs()),
    )
}

pub(super) async fn scalar<E: QuadratureEvaluator<Sample = f64>>(
    inputs: EmbeddedRuleInputs<'_, f64>,
    evaluator: &mut E,
) -> Result<ScaledEstimate, QuadratureArithmeticError> {
    let EmbeddedRuleInputs {
        samples,
        base_weights,
        jacobians,
        width,
        kronrod,
        gauss,
    } = inputs;
    if samples.len() != base_weights.len() || samples.len() != jacobians.len() {
        return Err(QuadratureArithmeticError::InconsistentComponents);
    }
    let mean = kronrod.divide(width);
    let mut resabs = ScaledEstimate::from_f64(0.0);
    let mut resasc = ScaledEstimate::from_f64(0.0);
    for ((&sample, &weight), &jacobian) in samples.iter().zip(base_weights).zip(jacobians) {
        let (absolute, deviation) = sample_terms(sample, jacobian, weight, mean);
        resabs = resabs.add(absolute);
        resasc = resasc.add(deviation);
    }
    evaluator
        .charge(samples.len() as u64)
        .await
        .map_err(QuadratureArithmeticError::execution)?;
    Ok(adjusted_error(kronrod.subtract(*gauss), resabs, resasc))
}

pub(super) async fn array<E: QuadratureEvaluator<Sample = Arc<Vec<f64>>>>(
    inputs: EmbeddedRuleInputs<'_, Arc<Vec<f64>>>,
    evaluator: &mut E,
) -> Result<ArrayEstimate, QuadratureArithmeticError> {
    let EmbeddedRuleInputs {
        samples,
        base_weights,
        jacobians,
        width,
        kronrod,
        gauss,
    } = inputs;
    let Some(first) = samples.first() else {
        return Err(QuadratureArithmeticError::InconsistentComponents);
    };
    if samples.len() != base_weights.len()
        || samples.len() != jacobians.len()
        || kronrod.0.len() != first.len()
        || gauss.0.len() != first.len()
        || samples.iter().any(|sample| sample.len() != first.len())
    {
        return Err(QuadratureArithmeticError::InconsistentComponents);
    }
    let mut errors = Vec::new();
    errors
        .try_reserve_exact(first.len())
        .map_err(QuadratureArithmeticError::Resource)?;
    let mut pending_work = 0_usize;
    for component in 0..first.len() {
        let mean = kronrod.0[component].divide(width);
        let mut resabs = ScaledEstimate::from_f64(0.0);
        let mut resasc = ScaledEstimate::from_f64(0.0);
        for ((sample, &weight), &jacobian) in samples.iter().zip(base_weights).zip(jacobians) {
            let (absolute, deviation) = sample_terms(sample[component], jacobian, weight, mean);
            resabs = resabs.add(absolute);
            resasc = resasc.add(deviation);
            pending_work += 1;
            if pending_work == COMPONENT_CHUNK {
                evaluator
                    .charge(COMPONENT_CHUNK as u64)
                    .await
                    .map_err(QuadratureArithmeticError::execution)?;
                pending_work = 0;
            }
        }
        errors.push(adjusted_error(
            kronrod.0[component].subtract(gauss.0[component]),
            resabs,
            resasc,
        ));
    }
    evaluator
        .charge(pending_work as u64)
        .await
        .map_err(QuadratureArithmeticError::execution)?;
    Ok(ArrayEstimate(Arc::new(errors)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundoff_floor_does_not_accept_unearned_huge_cancellation() {
        let huge = ScaledEstimate::from_f64(1.0e300);
        let error = adjusted_error(ScaledEstimate::from_f64(0.0), huge, huge);
        assert!(error.to_f64().expect("representable floor") > 1.0e285);
    }

    #[test]
    fn resasc_modulates_small_embedded_difference_without_below_floor_claim() {
        let error = adjusted_error(
            ScaledEstimate::from_f64(1.0e-10),
            ScaledEstimate::from_f64(1.0),
            ScaledEstimate::from_f64(1.0),
        );
        let value = error.to_f64().expect("representable estimate");
        assert!(value > ROUNDOFF_FACTOR);
        assert!(value < 1.0e-10);
    }
}
