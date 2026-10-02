//! RunMat spectral result precision is a property of the signal, not controls.
//!
//! The host reference FFT/window arithmetic remains binary64. These consuming
//! constructors narrow once at publication for a Single signal; GPU results
//! retain their native payload and are never round-tripped through this adapter.
use num_complex::Complex;
use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, ProviderPrecision,
};
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SignalPrecision {
    Single,
    Double,
}

impl SignalPrecision {
    pub(crate) fn from_numeric_dtype(dtype: NumericDType) -> Self {
        if dtype == NumericDType::F32 {
            Self::Single
        } else {
            Self::Double
        }
    }

    pub(crate) fn of(signal: &Value) -> Self {
        let single = match signal {
            Value::Tensor(value) => value.numeric_dtype() == NumericDType::F32,
            Value::ComplexTensor(value) => value.numeric_dtype() == NumericDType::F32,
            Value::GpuTensor(value) => return Self::of_handle(value),
            _ => false,
        };
        if single {
            Self::Single
        } else {
            Self::Double
        }
    }

    pub(crate) fn of_handle(value: &runmat_accelerate_api::GpuTensorHandle) -> Self {
        if !handle_is_logical(value)
            && handle_integer_type(value).is_none()
            && handle_precision(value) == Some(ProviderPrecision::F32)
        {
            Self::Single
        } else {
            Self::Double
        }
    }

    pub(crate) fn real(self, values: Vec<f64>, shape: Vec<usize>) -> Result<Tensor, String> {
        match self {
            Self::Single => Tensor::from_f32(
                values.into_iter().map(|value| value as f32).collect(),
                shape,
            ),
            Self::Double => Tensor::new(values, shape),
        }
    }

    pub(crate) fn complex(
        self,
        values: Vec<Complex<f64>>,
        shape: Vec<usize>,
    ) -> Result<ComplexTensor, String> {
        match self {
            Self::Single => ComplexTensor::from_f32(
                values
                    .into_iter()
                    .map(|value| (value.re as f32, value.im as f32))
                    .collect(),
                shape,
            ),
            Self::Double => ComplexTensor::new(
                values
                    .into_iter()
                    .map(|value| (value.re, value.im))
                    .collect(),
                shape,
            ),
        }
    }
}
