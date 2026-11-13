//! Class-static empty-array constructors and RunMat shorthand.
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};
use runmat_builtins::{
    BuiltinErrorDescriptor, EMPTY_ERROR_INTERNAL, EMPTY_ERROR_INVALID_DIMS,
    EMPTY_ERROR_NON_EMPTY_SHAPE, EMPTY_ERROR_TOO_MANY_OUTPUTS, EMPTY_ERROR_UNSUPPORTED_CLASS,
    EMPTY_GLOBAL_CALL_EXTENSION, EMPTY_RESIDENT_SIZE_EXTENSION, EMPTY_TYPENAME_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, StringArray, Tensor, Value,
};
const LABEL: &str = "empty";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmptyClass {
    Double,
    Single,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Logical,
    Char,
    String,
    Cell,
    Struct,
    GpuArray,
}

fn invalid_dimensions(message: impl Into<String>) -> crate::RuntimeError {
    empty_descriptor_error(&EMPTY_ERROR_INVALID_DIMS, message)
}

fn unsupported_class(message: impl Into<String>) -> crate::RuntimeError {
    empty_descriptor_error(&EMPTY_ERROR_UNSUPPORTED_CLASS, message)
}
fn internal_error(message: impl Into<String>) -> crate::RuntimeError {
    empty_descriptor_error(&EMPTY_ERROR_INTERNAL, message)
}
fn empty_descriptor_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(LABEL);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "empty",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn empty_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    validate_outputs()?;
    crate::compatibility::ensure_builtin_extension_enabled(&EMPTY_GLOBAL_CALL_EXTENSION, LABEL)?;
    let (class, size_args, had_typename) = split_class_arg(args)?;
    if had_typename {
        crate::compatibility::ensure_builtin_extension_enabled(&EMPTY_TYPENAME_EXTENSION, LABEL)?;
    }
    let shape = parse_shape(&size_args).await?;
    let protected = size_args
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect::<Vec<_>>();
    build_empty_protecting(class, shape, &protected).await
}
fn validate_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(),Some(count) if count > 1) {
        return Err(empty_descriptor_error(
            &EMPTY_ERROR_TOO_MANY_OUTPUTS,
            EMPTY_ERROR_TOO_MANY_OUTPUTS.message,
        ));
    }
    Ok(())
}
#[runtime_builtin(
    name = "int8.empty",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn int8_empty_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    validate_outputs()?;
    let shape = parse_shape(&args).await?;
    build_empty(EmptyClass::Int8, shape).await
}
#[runtime_builtin(
    name = "int16.empty",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn int16_empty_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    validate_outputs()?;
    let shape = parse_shape(&args).await?;
    build_empty(EmptyClass::Int16, shape).await
}
#[runtime_builtin(
    name = "int32.empty",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn int32_empty_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    validate_outputs()?;
    let shape = parse_shape(&args).await?;
    build_empty(EmptyClass::Int32, shape).await
}
#[runtime_builtin(
    name = "int64.empty",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn int64_empty_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    validate_outputs()?;
    let shape = parse_shape(&args).await?;
    build_empty(EmptyClass::Int64, shape).await
}
#[runtime_builtin(
    name = "uint8.empty",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn uint8_empty_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    validate_outputs()?;
    let shape = parse_shape(&args).await?;
    build_empty(EmptyClass::Uint8, shape).await
}
#[runtime_builtin(
    name = "uint16.empty",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn uint16_empty_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    validate_outputs()?;
    let shape = parse_shape(&args).await?;
    build_empty(EmptyClass::Uint16, shape).await
}
#[runtime_builtin(
    name = "uint32.empty",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn uint32_empty_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    validate_outputs()?;
    let shape = parse_shape(&args).await?;
    build_empty(EmptyClass::Uint32, shape).await
}
#[runtime_builtin(
    name = "uint64.empty",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn uint64_empty_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    validate_outputs()?;
    let shape = parse_shape(&args).await?;
    build_empty(EmptyClass::Uint64, shape).await
}
fn class_from_keyword(keyword: &str) -> Option<EmptyClass> {
    Some(match keyword {
        "double" => EmptyClass::Double,
        "single" => EmptyClass::Single,
        "int8" => EmptyClass::Int8,
        "int16" => EmptyClass::Int16,
        "int32" => EmptyClass::Int32,
        "int64" => EmptyClass::Int64,
        "uint8" => EmptyClass::Uint8,
        "uint16" => EmptyClass::Uint16,
        "uint32" => EmptyClass::Uint32,
        "uint64" => EmptyClass::Uint64,
        "logical" => EmptyClass::Logical,
        "char" => EmptyClass::Char,
        "string" => EmptyClass::String,
        "cell" => EmptyClass::Cell,
        "struct" => EmptyClass::Struct,
        "gpuarray" => EmptyClass::GpuArray,
        _ => return None,
    })
}

fn split_class_arg(
    mut args: crate::call::BuiltinArguments,
) -> BuiltinResult<(EmptyClass, crate::call::BuiltinArguments, bool)> {
    for arg in args.iter() {
        if keyword_of(arg)?.as_deref() == Some("like") {
            return Err(unsupported_class(
                "empty: the global 'like' shorthand is not supported; use ClassName.empty",
            ));
        }
    }
    if let Some(class) = args
        .last()
        .map(keyword_of)
        .transpose()?
        .flatten()
        .as_deref()
        .and_then(class_from_keyword)
    {
        args.pop();
        return Ok((class, args, true));
    }
    Ok((EmptyClass::Double, args, false))
}

async fn parse_shape(args: &crate::call::BuiltinArguments) -> BuiltinResult<Vec<usize>> {
    let mut shape = if args.is_empty() {
        vec![0, 0]
    } else if args.len() == 1 {
        let host = host_size_value(&args[0]).await?;
        parse_single_size_argument(&host)?
    } else {
        let mut dims = Vec::with_capacity(args.len());
        for arg in args {
            let host = host_size_value(arg).await?;
            dims.push(parse_size_scalar(&host)?);
        }
        dims
    };
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    if shape.len() == 1 {
        shape.push(shape[0]);
    }
    if !shape.contains(&0) {
        return Err(empty_descriptor_error(
            &EMPTY_ERROR_NON_EMPTY_SHAPE,
            "empty: at least one dimension must be zero to construct an empty array",
        ));
    }
    Ok(shape)
}

async fn host_size_value(value: &Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = value {
        validate_row_or_scalar_shape(handle.shape())?;
        crate::compatibility::ensure_builtin_extension_enabled(
            &EMPTY_RESIDENT_SIZE_EXTENSION,
            LABEL,
        )?;
        return gather_if_needed_async(value)
            .await
            .map_err(|err| internal_error(format!("empty: failed to gather size control: {err}")));
    }
    Ok(value.clone())
}

fn parse_single_size_argument(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(_) | Value::Num(_) => {
            let dim = parse_size_scalar(value)?;
            Ok(vec![dim, dim])
        }
        Value::Tensor(tensor) => parse_size_tensor(tensor),
        other => Err(invalid_dimensions(format!(
            "empty: size input must be a numeric scalar or row vector, got {other:?}"
        ))),
    }
}

fn parse_size_tensor(value: &Tensor) -> BuiltinResult<Vec<usize>> {
    if value.is_empty() {
        return Ok(vec![0, 0]);
    }
    validate_row_or_scalar_shape(value.shape())?;
    if value.len() == 1 {
        let dim = parse_size_scalar(&Value::Tensor(value.clone()))?;
        return Ok(vec![dim, dim]);
    }
    if let Some(storage) = value.integer_storage() {
        return (0..storage.len())
            .map(|index| {
                storage
                    .value_at(index)
                    .ok_or_else(|| internal_error("empty: inconsistent integer size storage"))
                    .and_then(|value| parse_integer_dimension(&value))
            })
            .collect();
    }
    (0..value.len())
        .map(|index| parse_numeric_dimension(tensor::tensor_value_f64(value, index)))
        .collect()
}

fn validate_row_or_scalar_shape(shape: &[usize]) -> BuiltinResult<()> {
    let valid = match shape.len() {
        0 | 1 => true,
        2 => shape[0] <= 1,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid_dimensions(
            "empty: size vector must be a row vector",
        ))
    }
}

fn parse_size_scalar(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(value) => parse_integer_dimension(value),
        Value::Num(value) => parse_numeric_dimension(*value),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                parse_integer_dimension(&value)
            } else {
                parse_numeric_dimension(tensor::tensor_value_f64(tensor, 0))
            }
        }
        other => Err(invalid_dimensions(format!(
            "empty: each dimension must be a numeric scalar, got {other:?}"
        ))),
    }
}

fn parse_integer_dimension(value: &IntValue) -> BuiltinResult<usize> {
    let result = match value {
        IntValue::I8(value) => usize::try_from((*value).max(0) as u64),
        IntValue::I16(value) => usize::try_from((*value).max(0) as u64),
        IntValue::I32(value) => usize::try_from((*value).max(0) as u64),
        IntValue::I64(value) => usize::try_from((*value).max(0) as u64),
        IntValue::U8(value) => usize::try_from(*value as u64),
        IntValue::U16(value) => usize::try_from(*value as u64),
        IntValue::U32(value) => usize::try_from(*value as u64),
        IntValue::U64(value) => usize::try_from(*value),
    };
    result.map_err(|_| invalid_dimensions("empty: dimension exceeds the supported platform range"))
}

fn parse_numeric_dimension(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(invalid_dimensions(
            "empty: dimensions must be finite integers",
        ));
    }
    if value < 0.0 {
        return Ok(0);
    }
    let fits_usize = value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64);
    if value.fract() != 0.0 || !fits_usize {
        return Err(invalid_dimensions(
            "empty: dimension must be an integer within the supported platform range",
        ));
    }
    Ok(value as usize)
}

async fn build_empty(class: EmptyClass, shape: Vec<usize>) -> BuiltinResult<Value> {
    build_empty_protecting(class, shape, &[]).await
}

async fn build_empty_protecting(
    class: EmptyClass,
    shape: Vec<usize>,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Value> {
    match class {
        EmptyClass::Double => Tensor::new(Vec::new(), shape)
            .map(tensor::tensor_into_value)
            .map_err(internal_error),
        EmptyClass::Single => Tensor::new_with_dtype(Vec::new(), shape, NumericDType::F32)
            .map(tensor::tensor_into_value)
            .map_err(internal_error),
        EmptyClass::Int8 => integer_empty(IntegerStorage::I8(Vec::new()), shape),
        EmptyClass::Int16 => integer_empty(IntegerStorage::I16(Vec::new()), shape),
        EmptyClass::Int32 => integer_empty(IntegerStorage::I32(Vec::new()), shape),
        EmptyClass::Int64 => integer_empty(IntegerStorage::I64(Vec::new()), shape),
        EmptyClass::Uint8 => integer_empty(IntegerStorage::U8(Vec::new()), shape),
        EmptyClass::Uint16 => integer_empty(IntegerStorage::U16(Vec::new()), shape),
        EmptyClass::Uint32 => integer_empty(IntegerStorage::U32(Vec::new()), shape),
        EmptyClass::Uint64 => integer_empty(IntegerStorage::U64(Vec::new()), shape),
        EmptyClass::Logical => Ok(Value::LogicalArray(LogicalArray::zeros(shape))),
        EmptyClass::Char => {
            if shape.len() != 2 {
                return Err(invalid_dimensions(
                    "empty: character arrays must be two-dimensional",
                ));
            }
            CharArray::from_code_units(Vec::new(), shape[0], shape[1])
                .map(Value::CharArray)
                .map_err(internal_error)
        }
        EmptyClass::String => StringArray::new(Vec::new(), shape)
            .map(Value::StringArray)
            .map_err(internal_error),
        EmptyClass::Cell => make_cell_with_shape(Vec::new(), shape).map_err(internal_error),
        EmptyClass::Struct => Err(unsupported_class(
            "empty: shaped struct empties require first-class struct-array storage",
        )),
        EmptyClass::GpuArray => {
            let provider = runmat_accelerate_api::provider()
                .ok_or_else(|| internal_error("empty: no acceleration provider is available"))?;
            let dtype = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
                runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
            };
            let host = Tensor::new_with_dtype(Vec::new(), shape, dtype).map_err(internal_error)?;
            let handle = crate::builtins::common::gpu_helpers::upload_tensor_protecting(
                provider, &host, protected,
            )?;
            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .map_err(|cause| {
                    crate::build_runtime_error(format!("empty: output provenance failed: {cause}"))
                        .with_builtin("empty")
                        .with_source(cause)
                        .build()
                })?;
            Ok(Value::GpuTensor(handle))
        }
    }
}

fn integer_empty(storage: IntegerStorage, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new_integer(storage, shape)
        .map(tensor::tensor_into_value)
        .map_err(internal_error)
}

#[cfg(test)]
mod tests;
