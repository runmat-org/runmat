use runmat_analysis_core::{
    Direction, Signal, SignalAxis, SignalExtrapolation, SignalInterpolation, SignalOrdinate,
    SignalRepresentation, SignalValue,
};

use super::{
    model::{ComplexScalar, ComplexVector3},
    realize::RealizationError,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ExcitationSamplePoint<'a> {
    Static,
    TimeSeconds(f64),
    FrequencyHertz(f64),
    StudyParameter {
        parameter_id: &'a str,
        canonical_value: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum SampledValue {
    Scalar(ComplexScalar),
    Vector3(ComplexVector3),
}

enum SampleLocation {
    Zero,
    Blend {
        left: usize,
        right: usize,
        weight: f64,
    },
}

pub(super) fn sample_signal<Q>(
    signal: &Signal<Q>,
    sample: ExcitationSamplePoint<'_>,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    match &signal.representation {
        SignalRepresentation::Constant(value) => Ok(real_value(value, &component)),
        SignalRepresentation::Tabulated(table) => {
            let abscissa = sample_abscissa(sample, &table.axis)?;
            sample_table(
                axis_values(&table.axis),
                &table.values,
                abscissa,
                table.interpolation,
                table.extrapolation,
                component,
            )
        }
        SignalRepresentation::Waveform(_) => Err(RealizationError::new(
            "waveform signals must be materialized before FEA excitation realization",
        )),
        SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_) => Err(
            RealizationError::new(
                "artifact and generator signals must be materialized before FEA excitation realization",
            ),
        ),
    }
}

pub(super) fn expect_scalar(value: SampledValue) -> Result<ComplexScalar, RealizationError> {
    match value {
        SampledValue::Scalar(value) => Ok(value),
        SampledValue::Vector3(_) => Err(RealizationError::new("excitation requires scalar signal")),
    }
}

pub(super) fn expect_vector3(value: SampledValue) -> Result<ComplexVector3, RealizationError> {
    match value {
        SampledValue::Vector3(value) => Ok(value),
        SampledValue::Scalar(_) => Err(RealizationError::new("excitation requires vector3 signal")),
    }
}

pub(super) fn expect_vector3_or_direction(
    value: SampledValue,
    direction: Option<Direction>,
) -> Result<ComplexVector3, RealizationError> {
    match value {
        SampledValue::Vector3(value) => Ok(value),
        SampledValue::Scalar(value) => {
            let direction = direction.ok_or_else(|| {
                RealizationError::new("scalar directional excitation requires a direction")
            })?;
            Ok(ComplexVector3 {
                real: direction.vector.map(|component| component * value.real),
                imaginary: direction
                    .vector
                    .map(|component| component * value.imaginary),
            })
        }
    }
}

fn axis_values(axis: &SignalAxis) -> &[f64] {
    match axis {
        SignalAxis::TimeSeconds(values) | SignalAxis::FrequencyHertz(values) => values,
        SignalAxis::StudyParameter {
            canonical_values, ..
        } => canonical_values,
    }
}

fn sample_abscissa(
    sample: ExcitationSamplePoint<'_>,
    axis: &SignalAxis,
) -> Result<f64, RealizationError> {
    let value = match (sample, axis) {
        (ExcitationSamplePoint::TimeSeconds(value), SignalAxis::TimeSeconds(_)) => value,
        (ExcitationSamplePoint::FrequencyHertz(value), SignalAxis::FrequencyHertz(_)) => value,
        (
            ExcitationSamplePoint::StudyParameter {
                parameter_id,
                canonical_value,
            },
            SignalAxis::StudyParameter {
                parameter_id: axis_parameter,
                ..
            },
        ) if parameter_id == axis_parameter => canonical_value,
        _ => {
            return Err(RealizationError::new(
                "excitation sample point does not match the signal domain",
            ));
        }
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RealizationError::new(
            "excitation sample point must be finite",
        ))
    }
}

fn real_value<Q>(value: &SignalValue<Q>, component: &impl Fn(&Q) -> f64) -> SampledValue {
    match value {
        SignalValue::Scalar(value) => SampledValue::Scalar(ComplexScalar::real(component(value))),
        SignalValue::Vector3(values) => SampledValue::Vector3(ComplexVector3::real([
            component(&values[0]),
            component(&values[1]),
            component(&values[2]),
        ])),
        SignalValue::Tensor { .. } => unreachable!("FEA actions reject tensor signal values"),
    }
}

fn sample_table<Q>(
    axis: &[f64],
    values: &[SignalOrdinate<Q>],
    x: f64,
    interpolation: SignalInterpolation,
    extrapolation: SignalExtrapolation,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    if axis.len() != values.len()
        || axis.iter().any(|value| !value.is_finite())
        || axis.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(RealizationError::new(
            "tabulated axis must be finite, strictly increasing, and match values",
        ));
    }
    match sample_location(axis, x, interpolation, extrapolation)? {
        SampleLocation::Zero => zero_value(&values[0], &component),
        SampleLocation::Blend {
            left,
            right,
            weight,
        } => {
            let left = ordinate_value(&values[left], &component)?;
            let right = ordinate_value(&values[right], &component)?;
            interpolate_value(left, right, weight)
        }
    }
}

fn sample_location(
    axis: &[f64],
    x: f64,
    interpolation: SignalInterpolation,
    extrapolation: SignalExtrapolation,
) -> Result<SampleLocation, RealizationError> {
    if axis.is_empty() {
        return Err(RealizationError::new("tabulated signal has no samples"));
    }
    if axis.len() == 1 {
        return if x == axis[0] || extrapolation == SignalExtrapolation::HoldEndpoint {
            Ok(SampleLocation::Blend {
                left: 0,
                right: 0,
                weight: 0.0,
            })
        } else if extrapolation == SignalExtrapolation::Zero {
            Ok(SampleLocation::Zero)
        } else {
            Err(RealizationError::new("sample lies outside signal domain"))
        };
    }
    if x < axis[0] || x > axis[axis.len() - 1] {
        return match extrapolation {
            SignalExtrapolation::Reject => {
                Err(RealizationError::new("sample lies outside signal domain"))
            }
            SignalExtrapolation::HoldEndpoint => {
                let index = usize::from(x > axis[axis.len() - 1]) * (axis.len() - 1);
                Ok(SampleLocation::Blend {
                    left: index,
                    right: index,
                    weight: 0.0,
                })
            }
            SignalExtrapolation::Zero => Ok(SampleLocation::Zero),
        };
    }
    let exact_or_right = axis.partition_point(|value| *value < x).min(axis.len() - 1);
    let exact = axis[exact_or_right] == x;
    let (left, right, weight) = match interpolation {
        SignalInterpolation::StepLeft if exact => (exact_or_right, exact_or_right, 0.0),
        SignalInterpolation::StepLeft => (exact_or_right - 1, exact_or_right - 1, 0.0),
        SignalInterpolation::StepRight => (exact_or_right, exact_or_right, 0.0),
        SignalInterpolation::Linear => {
            let left = exact_or_right.saturating_sub(1);
            let right = exact_or_right;
            let weight = if left == right {
                0.0
            } else {
                (x - axis[left]) / (axis[right] - axis[left])
            };
            (left, right, weight)
        }
        SignalInterpolation::CubicHermite => {
            return Err(RealizationError::new(
                "cubic Hermite excitation signals require derivative materialization",
            ));
        }
    };
    Ok(SampleLocation::Blend {
        left,
        right,
        weight,
    })
}

fn zero_value<Q>(
    ordinate: &SignalOrdinate<Q>,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    Ok(match ordinate_value(ordinate, component)? {
        SampledValue::Scalar(_) => SampledValue::Scalar(ComplexScalar::real(0.0)),
        SampledValue::Vector3(_) => SampledValue::Vector3(ComplexVector3::real([0.0; 3])),
    })
}

fn ordinate_value<Q>(
    ordinate: &SignalOrdinate<Q>,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    Ok(match ordinate {
        SignalOrdinate::Real(value) => real_value(value, component),
        SignalOrdinate::Complex { real, imaginary } => complex_value(real, imaginary, component)?,
    })
}

fn complex_value<Q>(
    real: &SignalValue<Q>,
    imaginary: &SignalValue<Q>,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    Ok(match (real, imaginary) {
        (SignalValue::Scalar(real), SignalValue::Scalar(imaginary)) => {
            SampledValue::Scalar(ComplexScalar {
                real: component(real),
                imaginary: component(imaginary),
            })
        }
        (SignalValue::Vector3(real), SignalValue::Vector3(imaginary)) => {
            SampledValue::Vector3(ComplexVector3 {
                real: [
                    component(&real[0]),
                    component(&real[1]),
                    component(&real[2]),
                ],
                imaginary: [
                    component(&imaginary[0]),
                    component(&imaginary[1]),
                    component(&imaginary[2]),
                ],
            })
        }
        _ => {
            return Err(RealizationError::new(
                "complex signal shape is not scalar or vector3",
            ));
        }
    })
}

fn interpolate_value(
    left: SampledValue,
    right: SampledValue,
    weight: f64,
) -> Result<SampledValue, RealizationError> {
    let blend = |a: f64, b: f64| a + weight * (b - a);
    Ok(match (left, right) {
        (SampledValue::Scalar(left), SampledValue::Scalar(right)) => {
            SampledValue::Scalar(ComplexScalar {
                real: blend(left.real, right.real),
                imaginary: blend(left.imaginary, right.imaginary),
            })
        }
        (SampledValue::Vector3(left), SampledValue::Vector3(right)) => {
            let mut real = [0.0; 3];
            let mut imaginary = [0.0; 3];
            for index in 0..3 {
                real[index] = blend(left.real[index], right.real[index]);
                imaginary[index] = blend(left.imaginary[index], right.imaginary[index]);
            }
            SampledValue::Vector3(ComplexVector3 { real, imaginary })
        }
        _ => {
            return Err(RealizationError::new(
                "signal shape changes between samples",
            ))
        }
    })
}
