//! Native-precision terminal losses and adjoints for the shared network trace.
//!
//! Inputs and targets have already been aligned to one admitted semantic axis
//! plan. This module does no provider access and does not mutate a network.

use runmat_value::{NumericDType, NumericStorage, Tensor};

use crate::builtins::deep_learning::{
    deep_learning_error,
    model::{
        execution::host::{numeric::Real, softmax},
        plan::axes::AxisPlan,
    },
};
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PredictionDomain {
    Probabilities,
    Logits,
}

pub(super) struct LossResult {
    pub(super) value: f64,
    pub(super) output_gradient: Tensor,
}

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

fn matching(
    prediction: &Tensor,
    target: &Tensor,
    axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<()> {
    if prediction.shape() != axes.shape()
        || target.shape() != axes.shape()
        || prediction.len() != axes.element_count()
        || target.len() != axes.element_count()
    {
        return Err(fail(
            function,
            "prediction and target must match admitted output axes",
        ));
    }
    if prediction.numeric_dtype() != target.numeric_dtype()
        || !matches!(
            prediction.numeric_dtype(),
            NumericDType::F32 | NumericDType::F64
        )
    {
        return Err(fail(
            function,
            "prediction and target must have matching single or double precision",
        ));
    }
    if prediction.is_empty() {
        return Err(fail(
            function,
            "training loss requires at least one output element",
        ));
    }
    Ok(())
}

/// Validate target values before initializer callbacks or random draws. The
/// terminal loss repeats its checks against the actual minibatch prediction;
/// this admission only moves target-owned failures ahead of those effects.
pub(super) fn admit_target(
    target: &Tensor,
    axes: &AxisPlan,
    classification: bool,
    function: &'static str,
) -> BuiltinResult<()> {
    if target.shape() != axes.shape() || target.len() != axes.element_count() {
        return Err(fail(
            function,
            "target differs from admitted output geometry",
        ));
    }
    match target.numeric_dtype() {
        NumericDType::F32 => admit_target_values(
            target.as_f32_slice().expect("admitted F32"),
            target,
            axes,
            classification,
            1.0e-5_f32,
            function,
        ),
        NumericDType::F64 => admit_target_values(
            target.as_f64_slice().expect("admitted F64"),
            target,
            axes,
            classification,
            1.0e-10,
            function,
        ),
        _ => Err(fail(function, "training target must be single or double")),
    }
}

fn admit_target_values<T: Real>(
    values: &[T],
    target: &Tensor,
    axes: &AxisPlan,
    classification: bool,
    tolerance: T,
    function: &'static str,
) -> BuiltinResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(fail(function, "training target must contain finite values"));
    }
    if !classification {
        return Ok(());
    }
    let geometry =
        softmax::ChannelGeometry::new(target, axes).map_err(|error| fail(function, error))?;
    for sample in 0..geometry.samples() {
        let base = geometry.base(sample);
        let mut sum = T::zero();
        for channel in 0..geometry.channels() {
            let value = values[geometry.offset(base, channel)];
            if value < T::zero() || value > T::one() {
                return Err(fail(function, "classification target must be in [0, 1]"));
            }
            sum += value;
        }
        if !sum.is_finite() || (sum - T::one()).abs() > tolerance {
            return Err(fail(
                function,
                "classification target probabilities must sum to one",
            ));
        }
    }
    Ok(())
}

/// Mean squared error over all output elements. The adjoint is expressed in
/// the prediction's native class, ready for the immutable shared trace.
pub(super) fn mse(
    prediction: &Tensor,
    target: &Tensor,
    axes: &AxisPlan,
    function: &'static str,
) -> BuiltinResult<LossResult> {
    matching(prediction, target, axes, function)?;
    let (value, storage) = match prediction.numeric_dtype() {
        NumericDType::F32 => {
            let (value, gradient) = mse_values(
                prediction.as_f32_slice().expect("admitted F32"),
                target.as_f32_slice().expect("admitted F32"),
                function,
            )?;
            (value as f64, NumericStorage::F32(gradient))
        }
        NumericDType::F64 => {
            let (value, gradient) = mse_values(
                prediction.as_f64_slice().expect("admitted F64"),
                target.as_f64_slice().expect("admitted F64"),
                function,
            )?;
            (value, NumericStorage::F64(gradient))
        }
        _ => unreachable!("admitted floating precision"),
    };
    Ok(LossResult {
        value,
        output_gradient: Tensor::from_numeric_storage(storage, prediction.shape().to_vec())
            .map_err(|error| fail(function, error))?,
    })
}

fn mse_values<T: Real>(
    prediction: &[T],
    target: &[T],
    function: &'static str,
) -> BuiltinResult<(T, Vec<T>)> {
    let divisor = T::from_f64(prediction.len() as f64);
    let mut sum = T::zero();
    let mut gradient = Vec::with_capacity(prediction.len());
    for (&estimate, &expected) in prediction.iter().zip(target) {
        if !estimate.is_finite() || !expected.is_finite() {
            return Err(fail(
                function,
                "training loss requires finite real predictions and targets",
            ));
        }
        let difference = estimate - expected;
        sum += difference * difference;
        gradient.push((T::one() + T::one()) * difference / divisor);
    }
    let value = sum / divisor;
    if !value.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
        return Err(fail(
            function,
            "training loss or adjoint exceeds native precision",
        ));
    }
    Ok((value, gradient))
}

/// Cross entropy over each semantic channel vector. Targets are admitted
/// probability distributions (including one-hot targets). For logits the
/// shared channel-softmax kernel supplies stable probabilities; for a network
/// with a final softmax, the adjoint is with respect to those probabilities.
pub(super) fn crossentropy(
    prediction: &Tensor,
    target: &Tensor,
    axes: &AxisPlan,
    domain: PredictionDomain,
    function: &'static str,
) -> BuiltinResult<LossResult> {
    matching(prediction, target, axes, function)?;
    let geometry =
        softmax::ChannelGeometry::new(prediction, axes).map_err(|error| fail(function, error))?;
    if geometry.samples() == 0 {
        return Err(fail(
            function,
            "crossentropy requires at least one channel vector",
        ));
    }
    let probabilities = match domain {
        PredictionDomain::Probabilities => None,
        PredictionDomain::Logits => {
            Some(softmax::forward(prediction, axes).map_err(|error| fail(function, error))?)
        }
    };
    let values = probabilities.as_ref().unwrap_or(prediction);
    let (value, storage) = match prediction.numeric_dtype() {
        NumericDType::F32 => {
            let (value, gradient) = crossentropy_values(
                values.as_f32_slice().expect("admitted F32"),
                target.as_f32_slice().expect("admitted F32"),
                &geometry,
                domain,
                Tolerance::Single,
                function,
            )?;
            (value as f64, NumericStorage::F32(gradient))
        }
        NumericDType::F64 => {
            let (value, gradient) = crossentropy_values(
                values.as_f64_slice().expect("admitted F64"),
                target.as_f64_slice().expect("admitted F64"),
                &geometry,
                domain,
                Tolerance::Double,
                function,
            )?;
            (value, NumericStorage::F64(gradient))
        }
        _ => unreachable!("admitted floating precision"),
    };
    Ok(LossResult {
        value,
        output_gradient: Tensor::from_numeric_storage(storage, prediction.shape().to_vec())
            .map_err(|error| fail(function, error))?,
    })
}

enum Tolerance {
    Single,
    Double,
}

fn crossentropy_values<T: Real>(
    probability: &[T],
    target: &[T],
    geometry: &softmax::ChannelGeometry,
    domain: PredictionDomain,
    tolerance: Tolerance,
    function: &'static str,
) -> BuiltinResult<(T, Vec<T>)> {
    let one = T::one();
    let inverse_count = one / T::from_f64(geometry.samples() as f64);
    let tolerance = T::from_f64(match tolerance {
        Tolerance::Single => 1.0e-5,
        Tolerance::Double => 1.0e-10,
    });
    let mut value = T::zero();
    let mut gradient = vec![T::zero(); target.len()];
    for sample in 0..geometry.samples() {
        let base = geometry.base(sample);
        let mut target_sum = T::zero();
        let mut probability_sum = T::zero();
        for channel in 0..geometry.channels() {
            let index = geometry.offset(base, channel);
            let expected = target[index];
            let predicted = probability[index];
            if !expected.is_finite()
                || expected < T::zero()
                || expected > one
                || !predicted.is_finite()
                || predicted < T::zero()
                || predicted > one
            {
                return Err(fail(
                    function,
                    "crossentropy requires finite probabilities in [0, 1]",
                ));
            }
            target_sum += expected;
            probability_sum += predicted;
            if expected > T::zero() {
                if predicted <= T::zero() {
                    return Err(fail(
                        function,
                        "crossentropy is undefined for a positive target at zero probability",
                    ));
                }
                value += (T::zero() - expected * predicted.ln()) * inverse_count;
            }
            gradient[index] = match domain {
                PredictionDomain::Probabilities if expected > T::zero() => {
                    (T::zero() - expected / predicted) * inverse_count
                }
                PredictionDomain::Probabilities => T::zero(),
                PredictionDomain::Logits => (predicted - expected) * inverse_count,
            };
        }
        if (target_sum - one).abs() > tolerance || (probability_sum - one).abs() > tolerance {
            return Err(fail(function, "crossentropy channels must each sum to one"));
        }
    }
    if !value.is_finite() || gradient.iter().any(|entry| !entry.is_finite()) {
        return Err(fail(
            function,
            "crossentropy loss or adjoint exceeds native precision",
        ));
    }
    Ok((value, gradient))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::DeepLearningAxisLayout;

    fn axes(format: &str, shape: &[usize]) -> AxisPlan {
        AxisPlan::new(
            DeepLearningAxisLayout::parse(format, shape.len()).unwrap(),
            shape,
        )
        .unwrap()
    }

    #[test]
    fn native_mse_non_square_forward_and_adjoint() {
        let axes = axes("CB", &[2, 3]);
        let prediction = Tensor::from_f32(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
        let target = Tensor::from_f32(vec![0., 2., 2., 4., 6., 4.], vec![2, 3]).unwrap();
        let result = mse(&prediction, &target, &axes, "test").unwrap();
        assert!((result.value - 7.0 / 6.0).abs() < 1.0e-6);
        assert_eq!(
            result.output_gradient.as_f32_slice().unwrap(),
            &[1. / 3., 0., 1. / 3., 0., -1. / 3., 2. / 3.]
        );
    }

    #[test]
    fn channel_crossentropy_respects_semantic_axes_and_native_class() {
        let axes = axes("BC", &[2, 3]);
        let logits = Tensor::new(vec![2., 1., 0., 0., 1., 2.], vec![2, 3]).unwrap();
        let target = Tensor::new(vec![1., 0., 0., 0., 0., 1.], vec![2, 3]).unwrap();
        let result =
            crossentropy(&logits, &target, &axes, PredictionDomain::Logits, "test").unwrap();
        let gradient = result.output_gradient.as_f64_slice().unwrap();
        assert!(result.value > 0.0 && result.value < 1.0);
        assert!(gradient[0] < 0.0 && gradient[1] > 0.0);
        assert!(gradient[4] > 0.0 && gradient[5] < 0.0);
        for index in 0..logits.len() {
            let mut plus = logits.as_f64_slice().unwrap().to_vec();
            let mut minus = plus.clone();
            plus[index] += 1.0e-6;
            minus[index] -= 1.0e-6;
            let objective = |values| {
                crossentropy(
                    &Tensor::new(values, logits.shape().to_vec()).unwrap(),
                    &target,
                    &axes,
                    PredictionDomain::Logits,
                    "test",
                )
                .unwrap()
                .value
            };
            let numerical = (objective(plus) - objective(minus)) / 2.0e-6;
            assert!((gradient[index] - numerical).abs() < 1.0e-8);
        }
    }

    #[test]
    fn malformed_and_nonfinite_targets_reject_without_partial_gradient() {
        let axes = axes("CB", &[2, 1]);
        let estimate = Tensor::new(vec![0.7, 0.3], vec![2, 1]).unwrap();
        for target in [vec![1.0, 1.0], vec![f64::NAN, 1.0], vec![-0.1, 1.1]] {
            let target = Tensor::new(target, vec![2, 1]).unwrap();
            assert!(crossentropy(
                &estimate,
                &target,
                &axes,
                PredictionDomain::Probabilities,
                "test"
            )
            .is_err());
        }
        let wrong = Tensor::from_f32(vec![1., 0.], vec![2, 1]).unwrap();
        assert!(crossentropy(
            &estimate,
            &wrong,
            &axes,
            PredictionDomain::Probabilities,
            "test"
        )
        .is_err());
    }

    #[test]
    fn target_owned_failures_preflight_without_a_prediction_or_initializer() {
        let axes = axes("CB", &[2, 2]);
        let valid = Tensor::from_f32(vec![0.25, 0.75, 0.6, 0.4], vec![2, 2]).unwrap();
        assert!(admit_target(&valid, &axes, true, "trainnet").is_ok());
        for values in [
            vec![0.25, 0.25, 0.6, 0.4],
            vec![f32::NAN, 1.0, 0.6, 0.4],
            vec![-0.1, 1.1, 0.6, 0.4],
        ] {
            let target = Tensor::from_f32(values, vec![2, 2]).unwrap();
            assert!(admit_target(&target, &axes, true, "trainnet").is_err());
        }
        let nonfinite = Tensor::new(vec![1., 2., f64::INFINITY, 4.], vec![2, 2]).unwrap();
        assert!(admit_target(&nonfinite, &axes, false, "trainnet").is_err());
    }
}
