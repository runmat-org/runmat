//! Shared logical constructor parsing, allocation and provider ownership.
use crate::builtins::common::shape::normalize_scalar_shape;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinExtensionDescriptor, FALSE_IMPLICIT_PROTOTYPE_EXTENSION, FALSE_LOGICAL_OPTION_EXTENSION,
    FALSE_RESIDENT_SIZE_EXTENSION, FALSE_SINGLE_SIZE_EXTENSION, TRUE_IMPLICIT_PROTOTYPE_EXTENSION,
    TRUE_LOGICAL_OPTION_EXTENSION, TRUE_RESIDENT_SIZE_EXTENSION, TRUE_SINGLE_SIZE_EXTENSION,
};
use runmat_value::{IntValue, LogicalArray, NumericDType, NumericScalar, Value};
fn builtin_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}
pub(super) struct LogicalFillConfig {
    name: &'static str,
    implicit_prototype: &'static BuiltinExtensionDescriptor,
    logical_option: &'static BuiltinExtensionDescriptor,
    single_size: &'static BuiltinExtensionDescriptor,
    resident_size: &'static BuiltinExtensionDescriptor,
}

pub(super) const TRUE_FILL_CONFIG: LogicalFillConfig = LogicalFillConfig {
    name: "true",
    implicit_prototype: &TRUE_IMPLICIT_PROTOTYPE_EXTENSION,
    logical_option: &TRUE_LOGICAL_OPTION_EXTENSION,
    single_size: &TRUE_SINGLE_SIZE_EXTENSION,
    resident_size: &TRUE_RESIDENT_SIZE_EXTENSION,
};

pub(super) const FALSE_FILL_CONFIG: LogicalFillConfig = LogicalFillConfig {
    name: "false",
    implicit_prototype: &FALSE_IMPLICIT_PROTOTYPE_EXTENSION,
    logical_option: &FALSE_LOGICAL_OPTION_EXTENSION,
    single_size: &FALSE_SINGLE_SIZE_EXTENSION,
    resident_size: &FALSE_RESIDENT_SIZE_EXTENSION,
};

struct ParsedLogicalFill {
    shape: Vec<usize>,
    prototype: Option<Value>,
}

pub(super) async fn logical_fill(
    args: crate::call::BuiltinArguments,
    fill: bool,
    config: &LogicalFillConfig,
) -> BuiltinResult<Value> {
    let parsed = ParsedLogicalFill::parse(args, config).await?;
    logical_output(parsed, fill, config.name)
}

impl ParsedLogicalFill {
    async fn parse(
        args: crate::call::BuiltinArguments,
        config: &LogicalFillConfig,
    ) -> BuiltinResult<Self> {
        let name = config.name;
        let mut dims = Vec::new();
        let mut saw_size_vector = false;
        let mut prototype = None;
        let mut saw_like = false;
        let mut implicit_shape = None;
        let mut idx = 0usize;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) =
                crate::builtins::common::arg_tokens::scalar_keyword_of(&arg, "logical")?
            {
                match keyword.as_str() {
                    "like" => {
                        if saw_like {
                            return Err(builtin_error(
                                name,
                                format!("{name}: multiple 'like' specifications are not supported"),
                            ));
                        }
                        let Some(value) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error(
                                name,
                                format!("{name}: expected prototype after 'like'"),
                            ));
                        };
                        ensure_numeric_prototype(&value, name)?;
                        saw_like = true;
                        prototype = Some(value);
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            config.logical_option,
                            name,
                        )?;
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(
                            name,
                            format!("{name}: unrecognised option '{other}'"),
                        ));
                    }
                }
            }

            if value_is_single_size(&arg) {
                crate::compatibility::ensure_builtin_extension_enabled(config.single_size, name)?;
            }
            if matches!(arg, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(config.resident_size, name)?;
            }
            if let Some(parsed) = extract_logical_dims(&arg, name).await? {
                if parsed.is_vector {
                    if saw_size_vector || !dims.is_empty() {
                        return Err(builtin_error(
                            name,
                            format!("{name}: multiple vector size inputs are not supported"),
                        ));
                    }
                    saw_size_vector = true;
                } else if saw_size_vector {
                    return Err(builtin_error(
                        name,
                        format!("{name}: a size vector cannot be combined with other dimensions"),
                    ));
                }
                dims.extend(parsed.values);
                idx += 1;
                continue;
            }

            crate::compatibility::ensure_builtin_extension_enabled(
                config.implicit_prototype,
                name,
            )?;
            if implicit_shape.is_none() {
                implicit_shape = Some(
                    shape_from_value(&arg)
                        .map_err(|error| builtin_error(name, format!("{name}: {error}")))?,
                );
                prototype = Some(arg);
            }
            idx += 1;
        }

        let shape = if !dims.is_empty() || saw_size_vector {
            normalize_logical_shape(dims)
        } else if let Some(shape) = implicit_shape {
            normalize_logical_shape(shape)
        } else {
            vec![1, 1]
        };
        shape
            .iter()
            .try_fold(1usize, |total, dim| total.checked_mul(*dim))
            .ok_or_else(|| builtin_error(name, format!("{name}: output size overflows usize")))?;
        Ok(Self { shape, prototype })
    }
}

fn ensure_numeric_prototype(value: &Value, name: &str) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
    ) {
        Ok(())
    } else {
        Err(builtin_error(
            name,
            format!("{name}: like prototype must be numeric or logical"),
        ))
    }
}

fn logical_output(parsed: ParsedLogicalFill, fill: bool, name: &str) -> BuiltinResult<Value> {
    match parsed.prototype.as_ref() {
        Some(Value::SparseTensor(_)) => {
            let [rows, cols] = parsed.shape.as_slice() else {
                return Err(builtin_error(
                    name,
                    format!("{name}: sparse like output must be two-dimensional"),
                ));
            };
            logical_sparse_output(*rows, *cols, fill, name).map(Value::SparseTensor)
        }
        Some(Value::GpuTensor(prototype)) => {
            logical_gpu_output(prototype, &parsed.shape, fill, name)
        }
        _ => logical_host_output(parsed.shape, fill, name),
    }
}

fn logical_sparse_output(
    rows: usize,
    cols: usize,
    fill: bool,
    name: &str,
) -> BuiltinResult<runmat_value::SparseTensor> {
    if !fill {
        return Ok(runmat_value::SparseTensor::zeros_logical(rows, cols));
    }
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| builtin_error(name, format!("{name}: output size overflows usize")))?;
    let col_ptrs = (0..=cols)
        .map(|col| col.checked_mul(rows))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| builtin_error(name, format!("{name}: output size overflows usize")))?;
    let row_indices: Vec<_> = (0..cols).flat_map(|_| 0..rows).collect();
    runmat_value::SparseTensor::new_logical(rows, cols, col_ptrs, row_indices)
        .map_err(|error| builtin_error(name, format!("{name}: {error}")))
        .and_then(|value| {
            if value.nnz() == len {
                Ok(value)
            } else {
                Err(builtin_error(
                    name,
                    format!("{name}: sparse construction failed"),
                ))
            }
        })
}

fn logical_host_output(shape: Vec<usize>, fill: bool, name: &str) -> BuiltinResult<Value> {
    let len = shape.iter().product();
    if len == 1 && shape == [1, 1] {
        return Ok(Value::Bool(fill));
    }
    LogicalArray::new(vec![u8::from(fill); len], shape)
        .map(Value::LogicalArray)
        .map_err(|error| builtin_error(name, format!("{name}: {error}")))
}

fn logical_gpu_output(
    prototype: &runmat_accelerate_api::GpuTensorHandle,
    shape: &[usize],
    fill: bool,
    name: &str,
) -> BuiltinResult<Value> {
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(prototype)
        .ok_or_else(|| {
            builtin_error(
                name,
                format!("{name}: GPU prototype has no owning provider"),
            )
        })?;
    let result = match provider.logical_fill(shape, fill) {
        Ok(Some(result)) => result,
        Ok(None) => {
            let len = if shape.contains(&0) {
                0
            } else {
                shape
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    .ok_or_else(|| builtin_error(name, format!("{name}: dimensions overflow")))?
            };
            let data = vec![u8::from(fill); len];
            runmat_accelerate_api::upload_logical_protecting(
                provider,
                &runmat_accelerate_api::HostLogicalTensorView { data: &data, shape },
                &[prototype],
            )
            .map_err(|cause| {
                build_runtime_error(format!("{name}: logical upload failed: {cause}"))
                    .with_builtin(name)
                    .with_source(cause)
                    .build()
            })?
        }
        Err(cause) => {
            return Err(crate::runtime_error::with_provider_execution_source(
                build_runtime_error(format!("{name}: logical fill failed: {cause}"))
                    .with_builtin(name)
                    .build(),
                cause.with_unproven_effects(),
            ))
        }
    };
    // Logical intent is certified by the producing allocation owner, never
    // inferred from the current values or attached to a numeric allocation.
    let result = crate::builtins::common::random_args::validate_constructor_gpu_output_protecting(
        name,
        provider,
        result,
        shape,
        runmat_accelerate_api::GpuTensorStorage::Real,
        None,
        None,
        true,
        &[prototype],
    )?;
    crate::builtins::common::gpu_helpers::logical_gpu_value(result)
}

struct LogicalDims {
    values: Vec<usize>,
    is_vector: bool,
}

#[async_recursion::async_recursion(?Send)]
async fn extract_logical_dims(value: &Value, name: &str) -> BuiltinResult<Option<LogicalDims>> {
    match value {
        Value::Num(value) => parse_float_dimension(*value, name).map(|value| {
            Some(LogicalDims {
                values: vec![value],
                is_vector: false,
            })
        }),
        Value::Int(value) => parse_integer_dimension(value, name).map(|value| {
            Some(LogicalDims {
                values: vec![value],
                is_vector: false,
            })
        }),
        Value::Tensor(value) => {
            let len = value.len();
            if len == 0 {
                return Ok(Some(LogicalDims {
                    values: Vec::new(),
                    is_vector: true,
                }));
            }
            let scalar = len == 1;
            let row = value.shape().len() >= 2 && value.shape()[0] == 1;
            let column = value.shape().len() >= 2 && value.shape()[1] == 1;
            if column && !row && !scalar {
                return Err(builtin_error(
                    name,
                    format!("{name}: size vector must be a row vector"),
                ));
            }
            if !(scalar || row || value.shape().len() == 1) {
                return Ok(None);
            }
            let values = (0..len)
                .map(|index| {
                    value
                        .numeric_value_at(index)
                        .ok_or_else(|| builtin_error(name, format!("{name}: missing size value")))
                        .and_then(|value| parse_numeric_dimension(value, name))
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(Some(LogicalDims {
                values,
                is_vector: !scalar,
            }))
        }
        Value::GpuTensor(_) => {
            let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
            extract_logical_dims(&gathered, name).await
        }
        _ => Ok(None),
    }
}

fn parse_numeric_dimension(value: NumericScalar, name: &str) -> BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => parse_float_dimension(value, name),
        NumericScalar::F32(value) => parse_float_dimension(f64::from(value), name),
        integer => parse_integer_dimension(
            &integer.into_int_value().expect("integer numeric scalar"),
            name,
        ),
    }
}

fn parse_float_dimension(value: f64, name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(builtin_error(
            name,
            format!("{name}: dimensions must be finite integer values"),
        ));
    }
    if value <= 0.0 {
        return Ok(0);
    }
    if value >= usize::MAX as f64 {
        return Err(builtin_error(
            name,
            format!("{name}: dimension is outside the supported platform range"),
        ));
    }
    Ok(value as usize)
}

fn parse_integer_dimension(value: &IntValue, name: &str) -> BuiltinResult<usize> {
    let result = match value {
        IntValue::I8(value) => usize::try_from((*value).max(0)),
        IntValue::I16(value) => usize::try_from((*value).max(0)),
        IntValue::I32(value) => usize::try_from((*value).max(0)),
        IntValue::I64(value) => usize::try_from((*value).max(0)),
        IntValue::U8(value) => Ok(usize::from(*value)),
        IntValue::U16(value) => Ok(usize::from(*value)),
        IntValue::U32(value) => usize::try_from(*value),
        IntValue::U64(value) => usize::try_from(*value),
    };
    result.map_err(|_| {
        builtin_error(
            name,
            format!("{name}: dimension is outside the supported platform range"),
        )
    })
}

fn value_is_single_size(value: &Value) -> bool {
    matches!(value, Value::Tensor(value) if value.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(value) if runmat_accelerate_api::handle_integer_type(value).is_none() && runmat_accelerate_api::handle_precision(value) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn normalize_logical_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        return vec![0, 0];
    }
    if shape.len() == 1 {
        return vec![shape[0], shape[0]];
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape().to_vec()),
        Value::ComplexTensor(t) => Ok(t.shape().to_vec()),
        Value::LogicalArray(l) => Ok(l.shape.to_vec()),
        Value::GpuTensor(h) => Ok(normalize_scalar_shape(h.shape())),
        Value::CharArray(ca) => Ok(ca.shape().to_vec()),
        Value::Cell(cell) => Ok(cell.shape().to_vec()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
#[path = "logical_constructor/tests.rs"]
mod tests;
