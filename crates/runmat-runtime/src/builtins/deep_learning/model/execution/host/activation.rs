//! Elementwise host activations, with the same admitted operation used by
//! network forward execution and its reverse pass.

use runmat_value::{NumericDType, NumericStorage, Tensor};

use super::numeric::Real;

#[derive(Clone, Copy)]
pub(in crate::builtins::deep_learning) enum Activation {
    Relu,
    Elu { alpha: f64 },
}

#[derive(Debug)]
pub(in crate::builtins::deep_learning) enum ActivationError {
    InvalidAlpha,
    UnsupportedPrecision,
    GradientMismatch,
    Storage(String),
}

impl std::fmt::Display for ActivationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAlpha => f.write_str("activation alpha must be finite and nonnegative"),
            Self::UnsupportedPrecision => {
                f.write_str("activation requires single or double precision")
            }
            Self::GradientMismatch => f.write_str("activation gradient differs from forward input"),
            Self::Storage(error) => write!(f, "activation tensor storage: {error}"),
        }
    }
}
impl std::error::Error for ActivationError {}

impl Activation {
    pub fn admit(self, dtype: NumericDType) -> Result<(), ActivationError> {
        if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
            return Err(ActivationError::UnsupportedPrecision);
        }
        if let Self::Elu { alpha } = self {
            if !alpha.is_finite()
                || alpha < 0.0
                || (dtype == NumericDType::F32 && !(alpha as f32).is_finite())
            {
                return Err(ActivationError::InvalidAlpha);
            }
        }
        Ok(())
    }
}

pub(in crate::builtins::deep_learning) fn forward(
    input: &Tensor,
    operation: Activation,
) -> Result<Tensor, ActivationError> {
    operation.admit(input.numeric_dtype())?;
    let storage = match input.numeric_dtype() {
        NumericDType::F32 => NumericStorage::F32(values(
            input
                .as_f32_slice()
                .ok_or(ActivationError::UnsupportedPrecision)?,
            operation,
        )),
        NumericDType::F64 => NumericStorage::F64(values(
            input
                .as_f64_slice()
                .ok_or(ActivationError::UnsupportedPrecision)?,
            operation,
        )),
        _ => return Err(ActivationError::UnsupportedPrecision),
    };
    Tensor::from_numeric_storage(storage, input.shape().to_vec()).map_err(ActivationError::Storage)
}

pub(in crate::builtins::deep_learning) fn backward(
    input: &Tensor,
    dy: &Tensor,
    operation: Activation,
) -> Result<Tensor, ActivationError> {
    operation.admit(input.numeric_dtype())?;
    if input.shape() != dy.shape() || input.numeric_dtype() != dy.numeric_dtype() {
        return Err(ActivationError::GradientMismatch);
    }
    let storage = match input.numeric_dtype() {
        NumericDType::F32 => NumericStorage::F32(derivatives(
            input
                .as_f32_slice()
                .ok_or(ActivationError::UnsupportedPrecision)?,
            dy.as_f32_slice()
                .ok_or(ActivationError::UnsupportedPrecision)?,
            operation,
        )),
        NumericDType::F64 => NumericStorage::F64(derivatives(
            input
                .as_f64_slice()
                .ok_or(ActivationError::UnsupportedPrecision)?,
            dy.as_f64_slice()
                .ok_or(ActivationError::UnsupportedPrecision)?,
            operation,
        )),
        _ => return Err(ActivationError::UnsupportedPrecision),
    };
    Tensor::from_numeric_storage(storage, input.shape().to_vec()).map_err(ActivationError::Storage)
}

fn values<T: Real>(input: &[T], operation: Activation) -> Vec<T> {
    input
        .iter()
        .map(|&value| match operation {
            Activation::Relu => {
                if value.is_nan() || value > T::zero() {
                    value
                } else {
                    T::zero()
                }
            }
            Activation::Elu { alpha } => {
                if value >= T::zero() {
                    value
                } else {
                    T::from_f64(alpha) * (value.exp() - T::one())
                }
            }
        })
        .collect()
}

fn derivatives<T: Real>(input: &[T], dy: &[T], operation: Activation) -> Vec<T> {
    input
        .iter()
        .zip(dy)
        .map(|(&value, &gradient)| {
            gradient
                * match operation {
                    Activation::Relu => {
                        if value.is_nan() {
                            value
                        } else if value > T::zero() {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }
                    Activation::Elu { alpha } => {
                        if value >= T::zero() {
                            T::one()
                        } else {
                            T::from_f64(alpha) * value.exp()
                        }
                    }
                }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elu_forward_and_adjoint_keep_single() {
        let x = Tensor::from_f32(vec![-1.0, 0.0, 2.0], vec![3, 1]).unwrap();
        let dy = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let op = Activation::Elu { alpha: 2.0 };
        let y = forward(&x, op).unwrap();
        let dx = backward(&x, &dy, op).unwrap();
        assert_eq!(y.numeric_dtype(), NumericDType::F32);
        assert!((y.as_f32_slice().unwrap()[0] - 2.0 * ((-1.0f32).exp() - 1.0)).abs() < 1e-6);
        assert_eq!(y.as_f32_slice().unwrap()[1..], [0.0, 2.0]);
        assert!((dx.as_f32_slice().unwrap()[0] - 2.0 * (-1.0f32).exp()).abs() < 1e-6);
        assert_eq!(dx.as_f32_slice().unwrap()[1..], [2.0, 3.0]);
    }

    #[test]
    fn relu_nan_stays_nan_in_forward_and_adjoint_at_both_precisions() {
        let x32 = Tensor::from_f32(vec![f32::NAN, -2.0, 3.0], vec![3, 1]).unwrap();
        let dy32 = Tensor::from_f32(vec![1.0; 3], vec![3, 1]).unwrap();
        let y32 = forward(&x32, Activation::Relu).unwrap();
        let dx32 = backward(&x32, &dy32, Activation::Relu).unwrap();
        assert!(y32.as_f32_slice().unwrap()[0].is_nan());
        assert!(dx32.as_f32_slice().unwrap()[0].is_nan());
        assert_eq!(&y32.as_f32_slice().unwrap()[1..], &[0.0, 3.0]);
        assert_eq!(&dx32.as_f32_slice().unwrap()[1..], &[0.0, 1.0]);

        let x64 = Tensor::new(vec![f64::NAN, -2.0, 3.0], vec![3, 1]).unwrap();
        let dy64 = Tensor::new(vec![1.0; 3], vec![3, 1]).unwrap();
        assert!(forward(&x64, Activation::Relu)
            .unwrap()
            .as_f64_slice()
            .unwrap()[0]
            .is_nan());
        assert!(backward(&x64, &dy64, Activation::Relu)
            .unwrap()
            .as_f64_slice()
            .unwrap()[0]
            .is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn forward_and_adjoint_preserve_exact_rank_and_reject_equal_length_shape_mismatch() {
        let scalar = Tensor::new(vec![2.0], vec![]).unwrap();
        let scalar_gradient = Tensor::new(vec![3.0], vec![]).unwrap();
        let scalar_output = forward(&scalar, Activation::Relu).unwrap();
        let scalar_adjoint = backward(&scalar, &scalar_gradient, Activation::Relu).unwrap();
        assert!(scalar_output.shape().is_empty());
        assert!(scalar_adjoint.shape().is_empty());
        assert_eq!(scalar_adjoint.as_f64_slice().unwrap(), &[3.0]);

        let vector = Tensor::from_f32(vec![-2.0, 4.0], vec![2]).unwrap();
        let vector_gradient = Tensor::from_f32(vec![5.0, 6.0], vec![2]).unwrap();
        let vector_output = forward(&vector, Activation::Relu).unwrap();
        let vector_adjoint = backward(&vector, &vector_gradient, Activation::Relu).unwrap();
        assert_eq!(vector_output.shape(), &[2]);
        assert_eq!(vector_output.as_f32_slice().unwrap(), &[0.0, 4.0]);
        assert_eq!(vector_adjoint.shape(), &[2]);
        assert_eq!(vector_adjoint.as_f32_slice().unwrap(), &[0.0, 6.0]);

        let wrong_shape = Tensor::from_f32(vec![5.0, 6.0], vec![1, 2]).unwrap();
        assert!(matches!(
            backward(&vector, &wrong_shape, Activation::Relu),
            Err(ActivationError::GradientMismatch)
        ));
    }
}
