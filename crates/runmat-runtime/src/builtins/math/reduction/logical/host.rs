use crate::builtins::common::{
    shape::{canonical_scalar_shape, is_scalar_shape, normalize_scalar_shape},
    spec::ReductionNaN,
    tensor,
};
use crate::BuiltinResult;
use runmat_builtins::LogicalReductionKind;
use runmat_value::{CharArray, ComplexTensor, LogicalArray, NumericScalar, Tensor, Value};

use super::{arguments::ReductionSpec, shape, LogicalReductionConfig};

#[derive(Clone)]
struct TruthTensor {
    shape: Vec<usize>,
    data: Vec<TruthValue>,
}

#[derive(Clone, Copy)]
struct TruthValue {
    truthy: bool,
    has_nan: bool,
}

impl TruthValue {
    fn from_bool(truthy: bool) -> Self {
        Self {
            truthy,
            has_nan: false,
        }
    }

    fn from_numeric(value: NumericScalar) -> Self {
        Self {
            truthy: !value.is_zero(),
            has_nan: value.is_nan(),
        }
    }
}

pub(in crate::builtins::math::reduction) async fn reduce(
    config: &LogicalReductionConfig,
    value: Value,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    let truth = TruthTensor::from_value(config, value).await?;
    apply(config, truth, spec, nan_mode)?.into_value(config)
}

impl TruthTensor {
    async fn from_value(config: &LogicalReductionConfig, value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => Ok(Self::from_tensor(tensor)),
            Value::LogicalArray(array) => Ok(Self::from_logical(array)),
            Value::Num(value) => Ok(Self {
                shape: canonical_scalar_shape(),
                data: vec![TruthValue::from_numeric(NumericScalar::F64(value))],
            }),
            Value::Int(value) => Ok(Self {
                shape: canonical_scalar_shape(),
                data: vec![TruthValue::from_bool(!value.is_zero())],
            }),
            Value::Bool(value) => Ok(Self {
                shape: canonical_scalar_shape(),
                data: vec![TruthValue::from_bool(value)],
            }),
            Value::Complex(real, imaginary) => Ok(Self {
                shape: canonical_scalar_shape(),
                data: vec![complex_truth(real, imaginary)],
            }),
            Value::ComplexTensor(tensor) => Ok(Self::from_complex(tensor)),
            Value::CharArray(array) => Ok(Self::from_char(array)),
            Value::GpuTensor(handle) => {
                let tensor = crate::builtins::common::gpu_helpers::gather_tensor_async(&handle)
                    .await?;
                Ok(Self::from_tensor(tensor))
            }
            other => Err(config.error(
                config.invalid_input,
                format!(
                    "unsupported input type {other:?}; expected numeric, logical, complex, or character data"
                ),
            )),
        }
    }

    fn from_tensor(tensor: Tensor) -> Self {
        let data = (0..tensor.len())
            .map(|index| {
                TruthValue::from_numeric(
                    tensor
                        .numeric_value_at(index)
                        .expect("tensor storage length is authoritative"),
                )
            })
            .collect();
        Self {
            shape: normalized_shape(&tensor.shape, tensor.len()),
            data,
        }
    }

    fn from_logical(array: LogicalArray) -> Self {
        Self {
            shape: array.shape.clone(),
            data: array
                .data
                .iter()
                .map(|value| TruthValue::from_bool(*value != 0))
                .collect(),
        }
    }

    fn from_complex(tensor: ComplexTensor) -> Self {
        let data = if let Some(storage) = tensor.integer_storage() {
            (0..storage.len())
                .map(|index| {
                    TruthValue::from_bool(
                        storage
                            .is_nonzero_at(index)
                            .expect("complex integer storage length is authoritative"),
                    )
                })
                .collect()
        } else {
            tensor
                .materialize_f64()
                .iter()
                .map(|(real, imaginary)| complex_truth(*real, *imaginary))
                .collect()
        };
        Self {
            shape: tensor.shape.clone(),
            data,
        }
    }

    fn from_char(array: CharArray) -> Self {
        Self {
            shape: vec![array.rows, array.cols],
            data: array
                .data
                .iter()
                .map(|character| TruthValue::from_bool(u32::from(*character) != 0))
                .collect(),
        }
    }

    fn reduce_dimension(
        &self,
        config: &LogicalReductionConfig,
        dimension: usize,
        nan_mode: ReductionNaN,
    ) -> BuiltinResult<Self> {
        if dimension == 0 {
            return Err(config.error(config.invalid_argument, "dimensions must be positive"));
        }
        if is_scalar_shape(&self.shape) {
            let value = self.data.first().copied();
            return Ok(Self {
                shape: canonical_scalar_shape(),
                data: vec![TruthValue::from_bool(reduce_values(
                    config.kind,
                    value.into_iter(),
                    nan_mode,
                ))],
            });
        }
        if dimension > self.shape.len() {
            return Ok(self.clone());
        }

        let axis = dimension - 1;
        let reduce_len = self.shape[axis];
        let stride_before = shape::product(&self.shape[..axis]);
        let stride_after = shape::product(&self.shape[axis + 1..]);
        let mut output_shape = self.shape.clone();
        output_shape[axis] = 1;
        let mut output = Vec::with_capacity(stride_before.saturating_mul(stride_after));
        if stride_before == 0 || stride_after == 0 {
            return Ok(Self {
                shape: output_shape,
                data: output,
            });
        }

        for after in 0..stride_after {
            for before in 0..stride_before {
                let values = (0..reduce_len).filter_map(|offset| {
                    let index =
                        before + offset * stride_before + after * stride_before * reduce_len;
                    self.data.get(index).copied()
                });
                output.push(TruthValue::from_bool(reduce_values(
                    config.kind,
                    values,
                    nan_mode,
                )));
            }
        }
        Ok(Self {
            shape: output_shape,
            data: output,
        })
    }

    fn into_value(self, config: &LogicalReductionConfig) -> BuiltinResult<Value> {
        if self.data.len() == 1 {
            return Ok(Value::Bool(self.data[0].truthy));
        }
        let shape = normalized_shape(&self.shape, self.data.len());
        let data = self
            .data
            .into_iter()
            .map(|value| u8::from(value.truthy))
            .collect();
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|error| config.error(config.internal, error))
    }
}

fn apply(
    config: &LogicalReductionConfig,
    tensor: TruthTensor,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<TruthTensor> {
    let dimensions = shape::dimensions(&spec, &tensor.shape);
    let mut current = tensor;
    for dimension in dimensions {
        current = current.reduce_dimension(config, dimension, nan_mode)?;
    }
    Ok(current)
}

fn reduce_values(
    kind: LogicalReductionKind,
    values: impl IntoIterator<Item = TruthValue>,
    nan_mode: ReductionNaN,
) -> bool {
    let mut result = matches!(kind, LogicalReductionKind::All);
    for value in values {
        if value.has_nan && matches!(nan_mode, ReductionNaN::Omit) {
            continue;
        }
        match kind {
            LogicalReductionKind::All => {
                result &= value.truthy;
                if !result {
                    break;
                }
            }
            LogicalReductionKind::Any => {
                result |= value.truthy;
                if result {
                    break;
                }
            }
        }
    }
    result
}

fn complex_truth(real: f64, imaginary: f64) -> TruthValue {
    TruthValue {
        truthy: real != 0.0 || imaginary != 0.0 || real.is_nan() || imaginary.is_nan(),
        has_nan: real.is_nan() || imaginary.is_nan(),
    }
}

fn normalized_shape(shape: &[usize], data_len: usize) -> Vec<usize> {
    if tensor::element_count(shape) == data_len {
        normalize_scalar_shape(shape)
    } else if is_scalar_shape(shape) {
        if data_len == 0 {
            Vec::new()
        } else {
            vec![data_len]
        }
    } else {
        shape.to_vec()
    }
}
