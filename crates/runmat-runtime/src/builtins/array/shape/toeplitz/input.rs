use super::*;
#[derive(Clone, Debug)]
pub(super) enum InputVector {
    Real {
        values: Vec<f64>,
        dtype: NumericDType,
    },
    TypedInteger(IntegerStorage),
    Complex {
        values: Vec<(f64, f64)>,
        dtype: NumericDType,
    },
    TypedComplex(IntegerComplexStorage),
}

impl InputVector {
    pub(super) async fn from_value(value: Value) -> BuiltinResult<Self> {
        let gathered = gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|err| error_with_detail(&ERROR_INTERNAL, err.message()))?;
        match gathered {
            Value::Num(value) => Ok(Self::Real {
                values: vec![value],
                dtype: NumericDType::F64,
            }),
            Value::Int(value) => Ok(Self::TypedInteger(IntegerStorage::from_scalar(value))),
            Value::Tensor(tensor) => {
                validate_vector_shape(tensor.shape(), tensor_utils::tensor_element_len(&tensor))?;
                let storage = tensor
                    .into_numeric_storage()
                    .map_err(|err| error_with_detail(&ERROR_INTERNAL, err))?;
                Ok(match storage {
                    NumericStorage::F64(values) => Self::Real {
                        values,
                        dtype: NumericDType::F64,
                    },
                    NumericStorage::F32(values) => Self::Real {
                        values: values.into_iter().map(f64::from).collect(),
                        dtype: NumericDType::F32,
                    },
                    storage => Self::TypedInteger(
                        storage
                            .into_integer_storage()
                            .expect("integer numeric storage"),
                    ),
                })
            }
            Value::Complex(re, im) => Ok(Self::Complex {
                values: vec![(re, im)],
                dtype: NumericDType::F64,
            }),
            Value::ComplexTensor(tensor) => {
                validate_vector_shape(
                    tensor.shape(),
                    tensor_utils::complex_tensor_element_len(&tensor),
                )?;
                Ok(match tensor.integer_storage() {
                    Some(storage) => {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &COMPLEX_INTEGER_INPUT_EXTENSION,
                            NAME,
                        )?;
                        Self::TypedComplex(storage.clone())
                    }
                    None => Self::Complex {
                        dtype: tensor.numeric_dtype(),
                        values: tensor.materialize_f64(),
                    },
                })
            }
            other => Err(error_with_detail(
                &ERROR_INVALID_INPUT,
                format!("unsupported input {other:?}"),
            )),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Real { values, .. } => values.len(),
            Self::TypedInteger(storage) => storage.len(),
            Self::Complex { values, .. } => values.len(),
            Self::TypedComplex(storage) => storage.len(),
        }
    }

    pub(super) fn is_complex(&self) -> bool {
        matches!(self, Self::Complex { .. } | Self::TypedComplex(_))
    }

    pub(super) fn real_at(&self, index: usize) -> f64 {
        match self {
            Self::Real { values, .. } => values[index],
            Self::TypedInteger(storage) => storage
                .value_at(index)
                .expect("toeplitz index is in bounds")
                .to_f64(),
            Self::Complex { values, .. } => values[index].0,
            Self::TypedComplex(_) => {
                unreachable!("typed complex integers use toeplitz_typed_complex")
            }
        }
    }

    pub(super) fn complex_at(&self, index: usize) -> (f64, f64) {
        match self {
            Self::Real { values, .. } => (values[index], 0.0),
            Self::TypedInteger(storage) => (
                storage
                    .value_at(index)
                    .expect("toeplitz index is in bounds")
                    .to_f64(),
                0.0,
            ),
            Self::Complex { values, .. } => values[index],
            Self::TypedComplex(_) => {
                unreachable!("typed complex integers use toeplitz_typed_complex")
            }
        }
    }

    pub(super) fn conjugated(&self) -> Self {
        match self {
            Self::Real { values, dtype } => Self::Real {
                values: values.clone(),
                dtype: *dtype,
            },
            Self::TypedInteger(storage) => Self::TypedInteger(storage.clone()),
            Self::Complex { values, dtype } => Self::Complex {
                values: values.iter().map(|&(re, im)| (re, -im)).collect(),
                dtype: *dtype,
            },
            Self::TypedComplex(storage) => Self::TypedComplex(
                IntegerComplexStorage::new(
                    storage.real().clone(),
                    conjugate_integer_imaginary_storage(storage.imag().clone()),
                )
                .expect("existing typed complex storage has matching components"),
            ),
        }
    }

    pub(super) fn dtype(&self) -> Option<NumericDType> {
        match self {
            Self::Real { dtype, .. } => Some(*dtype),
            Self::TypedInteger(_) => None,
            Self::Complex { .. } => None,
            Self::TypedComplex(_) => None,
        }
    }
}
