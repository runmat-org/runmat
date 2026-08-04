//! MATLAB-compatible `flip` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the `flip` function, mirroring MathWorks MATLAB
//! behaviour for numeric tensors, logical masks, string arrays, complex data,
//! character arrays, and gpuArray handles. It honours dimension vectors,
//! direction keywords such as `'horizontal'`, and gracefully falls back to the
//! host when a registered acceleration provider does not expose a native flip
//! kernel.

use crate::builtins::common::arg_tokens::{tokens_from_values, ArgToken};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, LogicalArray, NumericScalar, NumericStorage, ResolveContext,
    StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::flip")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "flip",
    op_kind: GpuOpKind::Custom("flip"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("flip")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement flip directly; the runtime falls back to gather→flip→upload when the hook is missing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::flip")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "flip",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Flip is a data-reordering boundary; fusion planner treats it as a residency-preserving barrier.",
};

fn preserve_array_type(args: &[Type], _context: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape } => Type::Tensor {
            shape: shape.clone(),
        },
        Type::Logical { shape } => Type::Logical {
            shape: shape.clone(),
        },
        Type::Num | Type::Int | Type::Bool => Type::tensor(),
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

const BUILTIN_NAME: &str = "flip";

const FLIP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array with selected dimensions reversed.",
}];

const FLIP_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array/value to reverse.",
}];

const FLIP_INPUTS_A_DIM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array/value to reverse.",
    },
    BuiltinParamDescriptor {
        name: "dim_or_direction",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Dimension index/vector or direction keyword ('horizontal', 'vertical', 'both').",
    },
];

const FLIP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = flip(A)",
        inputs: &FLIP_INPUTS_A,
        outputs: &FLIP_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = flip(A, dim_or_direction)",
        inputs: &FLIP_INPUTS_A_DIM,
        outputs: &FLIP_OUTPUT,
    },
];

const FLIP_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIP.TOO_MANY_INPUTS",
    identifier: Some("RunMat:flip:TooManyInputs"),
    when: "More than one optional argument is provided after A.",
    message: "flip: too many input arguments",
};

const FLIP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIP.INVALID_INPUT",
    identifier: Some("RunMat:flip:InvalidInput"),
    when: "Input type, dimension argument, or direction token is invalid.",
    message: "flip: invalid input argument",
};

const FLIP_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FLIP.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:flip:UnsupportedInput"),
    when: "Input type is unsupported for flip.",
    message: "flip: unsupported input type",
};

const FLIP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FLIP_ERROR_TOO_MANY_INPUTS,
    FLIP_ERROR_INVALID_INPUT,
    FLIP_ERROR_UNSUPPORTED_INPUT,
];

pub const FLIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FLIP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FLIP_ERRORS,
};

fn flip_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    flip_error_with_message(error.message, error)
}

fn flip_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn flip_error_for(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(builtin).build()
}

#[runtime_builtin(
    name = "flip",
    category = "array/shape",
    summary = "Reverse element order along dimensions.",
    keywords = "flip,reverse,dimension,gpu,horizontal,vertical",
    accel = "custom",
    type_resolver(preserve_array_type),
    descriptor(crate::builtins::array::shape::flip::FLIP_DESCRIPTOR),
    builtin_path = "crate::builtins::array::shape::flip"
)]
async fn flip_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(flip_error(&FLIP_ERROR_TOO_MANY_INPUTS));
    }
    let spec = parse_flip_spec(&rest)?;
    match value {
        Value::Tensor(tensor) => {
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::LogicalArray(array) => {
            let dims = resolve_dims(&spec, &array.shape);
            Ok(flip_logical_array(array, &dims).map(Value::LogicalArray)?)
        }
        Value::ComplexTensor(ct) => {
            let dims = resolve_dims(&spec, &ct.shape);
            Ok(flip_complex_tensor(ct, &dims).map(Value::ComplexTensor)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| flip_error_for("flip", format!("flip: {e}")))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_complex_tensor(tensor, &dims).map(complex_tensor_into_value)?)
        }
        Value::StringArray(strings) => {
            let dims = resolve_dims(&spec, &strings.shape);
            Ok(flip_string_array(strings, &dims).map(Value::StringArray)?)
        }
        Value::CharArray(chars) => {
            let dims = resolve_dims(&spec, &chars.shape);
            Ok(flip_char_array(chars, &dims).map(Value::CharArray)?)
        }
        Value::String(scalar) => Ok(Value::String(scalar)),
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("flip", Value::Num(n))
                .map_err(|e| flip_error_for("flip", e))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("flip", Value::Int(i))
                .map_err(|e| flip_error_for("flip", e))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("flip", Value::Bool(flag))
                .map_err(|e| flip_error_for("flip", e))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => {
            let dims = resolve_dims(&spec, &handle.shape);
            Ok(flip_gpu(handle, &dims).await?)
        }
        Value::Cell(_) => Err(flip_error_with_message(
            "flip: cell arrays are not yet supported",
            &FLIP_ERROR_UNSUPPORTED_INPUT,
        )),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::SparseTensor(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Symbolic(_)
        | Value::OutputList(_) => Err(flip_error(&FLIP_ERROR_UNSUPPORTED_INPUT)),
    }
}

#[derive(Clone, Debug)]
enum FlipSpec {
    Default,
    Dims(Vec<usize>),
}

fn parse_flip_spec(args: &[Value]) -> crate::BuiltinResult<FlipSpec> {
    match args.len() {
        0 => Ok(FlipSpec::Default),
        1 => {
            let tokens = tokens_from_values(args);
            if let Some(token) = tokens.first() {
                if let Some(direction_dims) = parse_direction_token(token)? {
                    return Ok(FlipSpec::Dims(direction_dims));
                }
            }
            if let Some(direction_dims) = parse_direction(&args[0])? {
                return Ok(FlipSpec::Dims(direction_dims));
            }
            let dims = parse_dims_value(&args[0])?;
            if dims.is_empty() {
                Ok(FlipSpec::Default)
            } else {
                Ok(FlipSpec::Dims(dims))
            }
        }
        _ => unreachable!(),
    }
}

fn parse_direction_token(token: &ArgToken) -> crate::BuiltinResult<Option<Vec<usize>>> {
    let ArgToken::String(text) = token else {
        return Ok(None);
    };
    let dims = match text.as_str() {
        "horizontal" | "left-right" | "leftright" | "lr" | "right-left" | "righthoriz" => {
            vec![2]
        }
        "vertical" | "up-down" | "updown" | "ud" | "down-up" => vec![1],
        "both" => vec![1, 2],
        other => {
            return Err(flip_error_for(
                "flip",
                format!("flip: unknown direction '{other}'"),
            ));
        }
    };
    Ok(Some(dims))
}

fn parse_direction(value: &Value) -> crate::BuiltinResult<Option<Vec<usize>>> {
    let text_opt = match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            tensor::value_to_string(value)
        }
        _ => None,
    };
    if let Some(text) = text_opt {
        let lowered = text.trim().to_ascii_lowercase();
        let dims = match lowered.as_str() {
            "horizontal" | "left-right" | "leftright" | "lr" | "right-left" | "righthoriz" => {
                vec![2]
            }
            "vertical" | "up-down" | "updown" | "ud" | "down-up" => vec![1],
            "both" => vec![1, 2],
            other => {
                return Err(flip_error_for(
                    "flip",
                    format!("flip: unknown direction '{other}'"),
                ));
            }
        };
        return Ok(Some(dims));
    }
    Ok(None)
}

fn parse_dims_value(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => parse_dims_tensor(t),
        Value::LogicalArray(la) => {
            let tensor = tensor::logical_to_tensor(la).map_err(|e| {
                flip_error_for(
                    "flip",
                    format!("flip: unable to parse dimension vector: {e}"),
                )
            })?;
            parse_dims_tensor(&tensor)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let dim =
                tensor::parse_dimension(value, "flip").map_err(|e| flip_error_for("flip", e))?;
            Ok(vec![dim])
        }
        Value::GpuTensor(_) => Err(flip_error_for(
            "flip",
            "flip: dimension argument must be specified on the host (numeric or string)",
        )),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                let tmp = Value::StringArray(sa.clone());
                parse_direction(&tmp)?
                    .ok_or_else(|| flip_error_for("flip", "flip: dimension vector must be numeric"))
            } else {
                Err(flip_error_for(
                    "flip",
                    "flip: dimension vector must be numeric",
                ))
            }
        }
        Value::String(_) | Value::CharArray(_) => parse_direction(value)?
            .ok_or_else(|| flip_error_for("flip", "flip: unknown direction string")),
        _ => Err(flip_error_for(
            "flip",
            "flip: dimension vector must be numeric or a direction string",
        )),
    }
}

fn parse_dims_tensor(tensor: &Tensor) -> crate::BuiltinResult<Vec<usize>> {
    if !is_vector(&tensor.shape) {
        return Err(flip_error_for(
            "flip",
            "flip: dimension vector must be a row or column vector",
        ));
    }
    if let Some(parsed) = tensor::integer_tensor_dimension_vector(tensor, "flip", false) {
        return parsed.map_err(|e| flip_error_for("flip", e));
    }
    let len = tensor::tensor_element_len(tensor);
    let mut dims = Vec::with_capacity(len);
    for index in 0..len {
        let entry = match tensor
            .numeric_value_at(index)
            .expect("dimension tensor index is in bounds")
        {
            NumericScalar::F64(value) => value,
            NumericScalar::F32(value) => f64::from(value),
            NumericScalar::I8(_)
            | NumericScalar::I16(_)
            | NumericScalar::I32(_)
            | NumericScalar::I64(_)
            | NumericScalar::U8(_)
            | NumericScalar::U16(_)
            | NumericScalar::U32(_)
            | NumericScalar::U64(_) => {
                unreachable!("integer dimension tensors return through the exact parser")
            }
        };
        if !entry.is_finite() {
            return Err(flip_error_for(
                "flip",
                "flip: dimension indices must be finite",
            ));
        }
        let rounded = entry.round();
        if (rounded - entry).abs() > f64::EPSILON {
            return Err(flip_error_for(
                "flip",
                "flip: dimension indices must be integers",
            ));
        }
        if rounded < 1.0 {
            return Err(flip_error_for(
                "flip",
                "flip: dimension indices must be >= 1",
            ));
        }
        dims.push(rounded as usize);
    }
    Ok(dims)
}

fn is_vector(shape: &[usize]) -> bool {
    let mut non_singleton = 0usize;
    for &dim in shape {
        if dim > 1 {
            non_singleton += 1;
        }
        if non_singleton > 1 {
            return false;
        }
    }
    true
}

fn resolve_dims(spec: &FlipSpec, shape: &[usize]) -> Vec<usize> {
    match spec {
        FlipSpec::Default => vec![default_flip_dim(shape)],
        FlipSpec::Dims(dims) => dims.clone(),
    }
}

fn default_flip_dim(shape: &[usize]) -> usize {
    for (idx, &extent) in shape.iter().enumerate() {
        if extent > 1 {
            return idx + 1;
        }
    }
    1
}

pub(crate) fn flip_tensor(tensor: Tensor, dims: &[usize]) -> crate::BuiltinResult<Tensor> {
    flip_tensor_with("flip", tensor, dims)
}

pub(crate) fn flip_tensor_with(
    builtin: &'static str,
    tensor: Tensor,
    dims: &[usize],
) -> crate::BuiltinResult<Tensor> {
    if tensor::tensor_element_len(&tensor) == 0 || dims.is_empty() {
        return Ok(tensor);
    }
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))?;
    let storage = flip_numeric_storage(storage, &shape, dims, builtin)?;
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

fn flip_numeric_storage(
    storage: NumericStorage,
    shape: &[usize],
    dims: &[usize],
    builtin: &'static str,
) -> crate::BuiltinResult<NumericStorage> {
    macro_rules! flip {
        ($values:expr, $variant:ident) => {
            NumericStorage::$variant(flip_generic(&$values, shape, dims, builtin)?)
        };
    }
    Ok(match storage {
        NumericStorage::F64(values) => flip!(values, F64),
        NumericStorage::F32(values) => flip!(values, F32),
        NumericStorage::I8(values) => flip!(values, I8),
        NumericStorage::I16(values) => flip!(values, I16),
        NumericStorage::I32(values) => flip!(values, I32),
        NumericStorage::I64(values) => flip!(values, I64),
        NumericStorage::U8(values) => flip!(values, U8),
        NumericStorage::U16(values) => flip!(values, U16),
        NumericStorage::U32(values) => flip!(values, U32),
        NumericStorage::U64(values) => flip!(values, U64),
    })
}

pub(crate) fn flip_complex_tensor(
    tensor: ComplexTensor,
    dims: &[usize],
) -> crate::BuiltinResult<ComplexTensor> {
    flip_complex_tensor_with("flip", tensor, dims)
}

pub(crate) fn flip_complex_tensor_with(
    builtin: &'static str,
    tensor: ComplexTensor,
    dims: &[usize],
) -> crate::BuiltinResult<ComplexTensor> {
    if tensor::complex_tensor_element_len(&tensor) == 0 || dims.is_empty() {
        return Ok(tensor);
    }
    let shape = tensor.shape.clone();
    let indices = (0..tensor.len()).collect::<Vec<_>>();
    let indices = flip_generic(&indices, &shape, dims, builtin)?;
    let storage = tensor
        .into_complex_storage()
        .gather(&indices)
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))?;
    ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

pub(crate) fn flip_logical_array(
    array: LogicalArray,
    dims: &[usize],
) -> crate::BuiltinResult<LogicalArray> {
    flip_logical_array_with("flip", array, dims)
}

pub(crate) fn flip_logical_array_with(
    builtin: &'static str,
    array: LogicalArray,
    dims: &[usize],
) -> crate::BuiltinResult<LogicalArray> {
    if array.data.is_empty() || dims.is_empty() {
        return Ok(array);
    }
    let data = flip_generic(&array.data, &array.shape, dims, builtin)?;
    LogicalArray::new(data, array.shape.clone())
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

pub(crate) fn flip_string_array(
    array: StringArray,
    dims: &[usize],
) -> crate::BuiltinResult<StringArray> {
    flip_string_array_with("flip", array, dims)
}

pub(crate) fn flip_string_array_with(
    builtin: &'static str,
    array: StringArray,
    dims: &[usize],
) -> crate::BuiltinResult<StringArray> {
    if array.data.is_empty() || dims.is_empty() {
        return Ok(array);
    }
    let data = flip_generic(&array.data, &array.shape, dims, builtin)?;
    StringArray::new(data, array.shape.clone())
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

pub(crate) fn flip_char_array(array: CharArray, dims: &[usize]) -> crate::BuiltinResult<CharArray> {
    flip_char_array_with("flip", array, dims)
}

pub(crate) fn flip_char_array_with(
    builtin: &'static str,
    array: CharArray,
    dims: &[usize],
) -> crate::BuiltinResult<CharArray> {
    if array.data.is_empty() || dims.is_empty() {
        return Ok(array);
    }
    let shape = array.shape.clone();
    let data = flip_generic(&array.to_column_major(), &shape, dims, builtin)?;
    CharArray::from_column_major(data, shape)
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

pub(crate) async fn flip_gpu(
    handle: GpuTensorHandle,
    dims: &[usize],
) -> crate::BuiltinResult<Value> {
    flip_gpu_with("flip", handle, dims).await
}

pub(crate) async fn flip_gpu_with(
    builtin: &'static str,
    handle: GpuTensorHandle,
    dims: &[usize],
) -> crate::BuiltinResult<Value> {
    if dims.is_empty() {
        return Ok(Value::GpuTensor(handle));
    }
    if dims.contains(&0) {
        return Err(flip_error_for(
            builtin,
            format!("{builtin}: dimension indices must be >= 1"),
        ));
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        let zero_based: Vec<usize> = dims.iter().map(|&d| d - 1).collect();
        if runmat_accelerate_api::handle_integer_type(&handle).is_none() {
            if let Ok(out) = provider.flip(&handle, &zero_based) {
                return Ok(Value::GpuTensor(out));
            }
        }
        let host_tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        let flipped = flip_tensor_with(builtin, host_tensor, dims)?;
        gpu_helpers::upload_tensor(provider, &flipped)
            .map(Value::GpuTensor)
            .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
    } else {
        let host_tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        flip_tensor_with(builtin, host_tensor, dims).map(tensor::tensor_into_value)
    }
}

fn flip_generic<T: Clone>(
    data: &[T],
    shape: &[usize],
    dims: &[usize],
    builtin: &'static str,
) -> crate::BuiltinResult<Vec<T>> {
    if dims.contains(&0) {
        return Err(flip_error_for(
            builtin,
            format!("{builtin}: dimension indices must be >= 1"),
        ));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let total: usize = shape.iter().product();
    if total != data.len() {
        return Err(flip_error_for(
            builtin,
            format!("{builtin}: shape does not match data length"),
        ));
    }
    let mut flip_flags = vec![false; shape.len()];
    for &dim in dims {
        let axis = dim - 1;
        if axis >= flip_flags.len() {
            continue;
        }
        flip_flags[axis] = !flip_flags[axis];
    }
    if !flip_flags.iter().any(|&flag| flag) {
        return Ok(data.to_vec());
    }
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let mut coords = unravel_index(idx, shape);
        for (axis, flag) in flip_flags.iter().enumerate() {
            if *flag && shape[axis] > 1 {
                coords[axis] = shape[axis] - 1 - coords[axis];
            }
        }
        let src_idx = ravel_index(&coords, shape);
        out.push(data[src_idx].clone());
    }
    Ok(out)
}

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for &extent in shape {
        if extent == 0 {
            coords.push(0);
        } else {
            coords.push(index % extent);
            index /= extent;
        }
    }
    coords
}

fn ravel_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut index = 0usize;
    let mut stride = 1usize;
    for (coord, extent) in coords.iter().zip(shape.iter()) {
        if *extent > 0 {
            index += coord * stride;
            stride *= extent;
        }
    }
    index
}

pub(crate) fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor::is_scalar_complex_tensor(&tensor) && tensor.integer_storage().is_none() {
        let value = tensor::complex_tensor_value_complex64(&tensor, 0);
        Value::Complex(value.re, value.im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn flip_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::flip_builtin(value, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{
        HostIntegerDataView, HostIntegerTensorView, HostTensorView, IntegerElementType,
    };
    use runmat_builtins::{
        CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray,
        StringArray, Tensor,
    };

    #[test]
    fn flip_type_preserves_logical_shape() {
        let out = preserve_array_type(
            &[Type::Logical {
                shape: Some(vec![Some(2), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Logical {
                shape: Some(vec![Some(2), Some(1)])
            }
        );
    }

    #[test]
    fn flip_gpu_integer_fallback_preserves_exact_storage_resident() {
        test_support::with_test_provider(|provider| {
            let values = [1_u64, 9_007_199_254_740_993, 4_u64, u64::MAX];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &[2, 2],
                })
                .expect("upload integer");
            let Value::GpuTensor(result) =
                flip_builtin(Value::GpuTensor(handle), Vec::new()).expect("flip integer gpu")
            else {
                panic!("expected resident gpuArray");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&result),
                Some(IntegerElementType::U64)
            );
            let gathered = block_on(gpu_helpers::gather_tensor_async(&result)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    9_007_199_254_740_993,
                    1,
                    u64::MAX,
                    4,
                ]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_vector_defaults_to_first_non_singleton_dim() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let value =
            flip_builtin(Value::Tensor(tensor), Vec::new()).expect("flip row vector default");
        match value {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![4.0, 3.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_preserves_native_single_storage() {
        let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let Value::Tensor(output) = flip_builtin(Value::Tensor(tensor), Vec::new()).expect("flip")
        else {
            panic!("expected tensor output");
        };
        assert_eq!(
            output.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![3.0, 2.0, 1.0])
        );
    }

    #[test]
    fn flip_complex_scalar_keeps_typed_integer_storage_without_mirror() {
        let storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![7]), IntegerStorage::I16(vec![-3]))
                .expect("matching complex integer storage");
        let input = ComplexTensor::new_integer(storage.clone(), vec![1, 1])
            .expect("typed complex integer input");

        let flipped = flip_complex_tensor(input, &[1]).expect("flip typed complex scalar");
        assert_eq!(flipped.integer_storage().cloned(), Some(storage.clone()));

        let value = complex_tensor_into_value(flipped);
        let Value::ComplexTensor(output) = value else {
            panic!("typed complex integer scalar must not collapse to double complex");
        };
        assert_eq!(output.integer_storage().cloned(), Some(storage));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_matrix_vertical_default() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).expect("tensor");
        let value = flip_builtin(Value::Tensor(tensor), Vec::new()).expect("flip matrix");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.materialize_f64(), vec![2.0, 4.0, 1.0, 6.0, 3.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_horizontal_keyword() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).expect("tensor");
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::from("horizontal")])
            .expect("flip horizontal");
        match value {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![5.0, 3.0, 6.0, 1.0, 4.0, 2.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_multiple_dimensions() {
        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let value = flip_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap(),
            )],
        )
        .expect("flip dims");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2, 2]);
                assert_eq!(
                    t.materialize_f64(),
                    vec![6.0, 5.0, 8.0, 7.0, 2.0, 1.0, 4.0, 3.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_both_direction_keyword() {
        let tensor = Tensor::new((1..=6).map(|v| v as f64).collect(), vec![3, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &[1, 2]).expect("cpu flip");
        let value =
            flip_builtin(Value::Tensor(tensor), vec![Value::from("both")]).expect("flip both");
        match value {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), expected.materialize_f64()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_char_array_horizontal() {
        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let value =
            flip_builtin(Value::CharArray(chars), vec![Value::from("horizontal")]).expect("flip");
        match value {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 3);
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "nurtam");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_direction_accepts_char_array_keyword() {
        let keyword = CharArray::new_row("vertical");
        let tensor = Tensor::new((1..=4).map(|v| v as f64).collect(), vec![2, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &[1]).expect("cpu flip");
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::CharArray(keyword)])
            .expect("flip via char");
        match value {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), expected.materialize_f64()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_logical_array_preserves_type() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let expected = flip_logical_array(logical.clone(), &[2]).expect("cpu logical flip");
        let value = flip_builtin(
            Value::LogicalArray(logical),
            vec![Value::from("horizontal")],
        )
        .expect("flip logical");
        match value {
            Value::LogicalArray(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_complex_tensor_defaults_to_first_dim() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (2.0, -1.0), (3.0, 0.5), (4.0, -0.25)],
            vec![2, 2],
        )
        .unwrap();
        let expected = flip_complex_tensor(tensor.clone(), &[1]).expect("cpu complex flip");
        let value = flip_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("flip complex");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.materialize_f64(), expected.materialize_f64())
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_string_array_vertical() {
        let strings =
            StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).expect("string array");
        let value =
            flip_builtin(Value::StringArray(strings), vec![Value::from("vertical")]).expect("flip");
        match value {
            Value::StringArray(out) => {
                assert_eq!(out.data, vec!["b".to_string(), "a".to_string()])
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_accepts_dimension_vector_tensor() {
        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let value =
            flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("flip dims");
        match value {
            Value::Tensor(t) => {
                assert_eq!(
                    t.materialize_f64(),
                    vec![4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_dimension_vector_reads_integer_tensor_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let dims = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![1, large]),
            vec![1, 2],
        )
        .expect("dims");
        let parsed = parse_dims_tensor(&dims).expect("parse dims");
        assert_eq!(parsed, vec![1, large as usize]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_large_integer_dimension_beyond_rank_is_noop() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let original = tensor.clone();
        let large = 9_007_199_254_740_993_u64;
        let dims = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![large]),
            vec![1, 1],
        )
        .expect("dims");
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("flip");
        match value {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), original.materialize_f64()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_dimension_tensor_must_be_vector() {
        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let dims = Tensor::new((1..=4).map(|v| v as f64).collect(), vec![2, 2]).unwrap();
        let err =
            flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect_err("flip fail");
        assert!(err.to_string().contains("row or column vector"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_dimensions_beyond_rank_are_noops() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let original = tensor.clone();
        let dims = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("flip");
        match value {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), original.materialize_f64()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_rejects_zero_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = flip_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
            .expect_err("flip should fail");
        assert!(err.to_string().contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cpu =
                flip_tensor(tensor.clone(), &[default_flip_dim(&tensor.shape)]).expect("cpu flip");
            let value = flip_builtin(Value::GpuTensor(handle), Vec::new()).expect("flip gpu");
            let gathered = test_support::gather(value).expect("gather gpu result");
            assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn flip_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).expect("tensor");
        let cpu = flip_tensor(tensor.clone(), &[1, 3]).expect("cpu flip");
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_value = flip_builtin(
            Value::GpuTensor(handle),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap(),
            )],
        )
        .expect("flip gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
    }
}
