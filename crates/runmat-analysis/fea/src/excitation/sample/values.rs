use super::{ComplexScalar, ComplexVector3, RealizationError, SampledValue};
use runmat_analysis_core::{SignalOrdinate, SignalValue};

/// Immutable, component-major-within-entity values sampled before numerical assembly.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ComplexTensor {
    pub shape: Vec<usize>,
    pub real: Vec<f64>,
    pub imaginary: Vec<f64>,
}

impl ComplexTensor {
    pub(crate) fn new(
        shape: Vec<usize>,
        real: Vec<f64>,
        imaginary: Vec<f64>,
    ) -> Result<Self, RealizationError> {
        let count = shape.iter().try_fold(1usize, |count, extent| {
            if *extent == 0 {
                None
            } else {
                count.checked_mul(*extent)
            }
        });
        if shape.is_empty()
            || count != Some(real.len())
            || real.len() != imaginary.len()
            || real
                .iter()
                .chain(&imaginary)
                .any(|value| !value.is_finite())
        {
            return Err(RealizationError::new(
                "tensor samples require finite values and an exact positive shape",
            ));
        }
        Ok(Self {
            shape,
            real,
            imaginary,
        })
    }

    fn real(shape: &[usize], real: Vec<f64>) -> Result<Self, RealizationError> {
        let imaginary = vec![0.0; real.len()];
        Self::new(shape.to_vec(), real, imaginary)
    }
}

pub(super) fn real_value<Q>(
    value: &SignalValue<Q>,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    Ok(match value {
        SignalValue::Scalar(value) => SampledValue::Scalar(ComplexScalar::real(component(value))),
        SignalValue::Vector3(values) => SampledValue::Vector3(ComplexVector3::real([
            component(&values[0]),
            component(&values[1]),
            component(&values[2]),
        ])),
        SignalValue::Tensor { shape, values } => SampledValue::Tensor(ComplexTensor::real(
            shape,
            values.iter().map(component).collect(),
        )?),
    })
}

pub(super) fn zero_value<Q>(
    ordinate: &SignalOrdinate<Q>,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    Ok(match ordinate_value(ordinate, component)? {
        SampledValue::Scalar(_) => SampledValue::Scalar(ComplexScalar::real(0.0)),
        SampledValue::Vector3(_) => SampledValue::Vector3(ComplexVector3::real([0.0; 3])),
        SampledValue::Tensor(mut tensor) => {
            tensor.real.fill(0.0);
            tensor.imaginary.fill(0.0);
            SampledValue::Tensor(tensor)
        }
    })
}

pub(super) fn ordinate_value<Q>(
    ordinate: &SignalOrdinate<Q>,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    Ok(match ordinate {
        SignalOrdinate::Real(value) => real_value(value, component)?,
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
        (
            SignalValue::Tensor {
                shape: real_shape,
                values: real,
            },
            SignalValue::Tensor {
                shape: imaginary_shape,
                values: imaginary,
            },
        ) if real_shape == imaginary_shape => {
            let real = real.iter().map(component).collect();
            let imaginary = imaginary.iter().map(component).collect();
            SampledValue::Tensor(ComplexTensor::new(real_shape.clone(), real, imaginary)?)
        }
        _ => {
            return Err(RealizationError::new(
                "complex signal components have different shapes",
            ));
        }
    })
}

pub(super) fn interpolate_value(
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
        (SampledValue::Tensor(left), SampledValue::Tensor(right)) if left.shape == right.shape => {
            let real = left
                .real
                .into_iter()
                .zip(right.real)
                .map(|(a, b)| blend(a, b))
                .collect();
            let imaginary = left
                .imaginary
                .into_iter()
                .zip(right.imaginary)
                .map(|(a, b)| blend(a, b))
                .collect();
            SampledValue::Tensor(ComplexTensor::new(left.shape, real, imaginary)?)
        }
        _ => {
            return Err(RealizationError::new(
                "signal shape changes between samples",
            ))
        }
    })
}
