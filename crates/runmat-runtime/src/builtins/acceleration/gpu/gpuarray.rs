//! MATLAB-compatible `gpuArray` builtin that uploads host data to the active accelerator.
//!
//! The implementation mirrors MathWorks MATLAB semantics, including optional
//! size arguments, `'like'` prototypes, and explicit dtype toggles. When no
//! acceleration provider is registered the builtin surfaces a MATLAB-style
//! error, ensuring callers know residency could not be established.

use crate::builtins::acceleration::gpu::type_resolvers::gpuarray_type;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, IntValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "gpuArray";

const GPUARRAY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "GPU-resident handle containing uploaded/converted data.",
}];

const GPUARRAY_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to upload or recast on GPU.",
}];

const GPUARRAY_INPUTS_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value to upload or recast on GPU.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Reshape dimensions (scalar dims or a single size vector tensor).",
    },
];

const GPUARRAY_INPUTS_DTYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value to upload or recast on GPU.",
    },
    BuiltinParamDescriptor {
        name: "dtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"double\""),
        description: "Class tag such as `single`, `int32`, `uint8`, `logical`, or `double`.",
    },
];

const GPUARRAY_INPUTS_LIKE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value to upload or recast on GPU.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal keyword `\"like\"`.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype value whose class drives output conversion.",
    },
];

const GPUARRAY_INPUTS_DIMS_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value to upload or recast on GPU.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Reshape dimensions (scalar dims or a single size vector tensor).",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Class tags and/or `\"like\", prototype` qualifiers.",
    },
];

const GPUARRAY_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X)",
        inputs: &GPUARRAY_INPUTS_BASE,
        outputs: &GPUARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X, dim, ...)",
        inputs: &GPUARRAY_INPUTS_DIMS,
        outputs: &GPUARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X, dtype)",
        inputs: &GPUARRAY_INPUTS_DTYPE,
        outputs: &GPUARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X, \"like\", prototype)",
        inputs: &GPUARRAY_INPUTS_LIKE,
        outputs: &GPUARRAY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = gpuArray(X, dim, ..., option, ...)",
        inputs: &GPUARRAY_INPUTS_DIMS_OPTIONS,
        outputs: &GPUARRAY_OUTPUT,
    },
];

const GPUARRAY_ERROR_NO_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.NO_PROVIDER",
    identifier: Some("RunMat:gpuArray:NoProvider"),
    when: "No acceleration provider is registered for host/device transfers.",
    message: "gpuArray: no acceleration provider registered",
};

const GPUARRAY_ERROR_OPTION_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.OPTION_ARGUMENT",
    identifier: Some("RunMat:gpuArray:OptionArgument"),
    when: "Option tail contains non-text values where class tags/keywords are expected.",
    message: "gpuArray: invalid option argument",
};

const GPUARRAY_ERROR_LIKE_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.LIKE_MISSING",
    identifier: Some("RunMat:gpuArray:LikeMissingPrototype"),
    when: "Keyword `like` is supplied without a following prototype value.",
    message: "gpuArray: expected a prototype value after 'like'",
};

const GPUARRAY_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.LIKE_DUPLICATE",
    identifier: Some("RunMat:gpuArray:LikeDuplicate"),
    when: "Keyword `like` appears more than once.",
    message: "gpuArray: duplicate 'like' qualifier",
};

const GPUARRAY_ERROR_CODISTRIBUTED_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.CODISTRIBUTED_UNSUPPORTED",
    identifier: Some("RunMat:gpuArray:CodistributedUnsupported"),
    when: "Distributed/codistributed qualifiers are requested.",
    message: "gpuArray: codistributed arrays are not supported yet",
};

const GPUARRAY_ERROR_CONFLICTING_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.CONFLICTING_TYPE",
    identifier: Some("RunMat:gpuArray:ConflictingTypeQualifiers"),
    when: "Multiple incompatible class qualifiers are supplied.",
    message: "gpuArray: conflicting type qualifiers supplied",
};

const GPUARRAY_ERROR_UNKNOWN_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.UNKNOWN_OPTION",
    identifier: Some("RunMat:gpuArray:UnknownOption"),
    when: "Text option is not a recognized class/keyword token.",
    message: "gpuArray: unrecognised option",
};

const GPUARRAY_ERROR_SIZE_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.SIZE_ARGUMENT",
    identifier: Some("RunMat:gpuArray:InvalidSizeArgument"),
    when: "Size arguments are malformed (not finite integers, negative, or invalid combinations).",
    message: "gpuArray: invalid size argument",
};

const GPUARRAY_ERROR_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.LIKE_PROTOTYPE",
    identifier: Some("RunMat:gpuArray:InvalidLikePrototype"),
    when: "`like` prototype is unsupported for type inference.",
    message: "gpuArray: invalid 'like' prototype",
};

const GPUARRAY_ERROR_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.INPUT_TYPE",
    identifier: Some("RunMat:gpuArray:UnsupportedInputType"),
    when: "Input value type cannot be uploaded/coerced to supported gpuArray storage.",
    message: "gpuArray: unsupported input type",
};

const GPUARRAY_ERROR_CONVERSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.CONVERSION",
    identifier: Some("RunMat:gpuArray:ConversionFailed"),
    when: "Requested dtype conversion cannot be performed (for example NaN->logical).",
    message: "gpuArray: conversion failed",
};

const GPUARRAY_ERROR_RESHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.RESHAPE",
    identifier: Some("RunMat:gpuArray:ReshapeMismatch"),
    when: "Requested shape does not preserve the element count.",
    message: "gpuArray: cannot reshape gpuArray into requested size",
};

const GPUARRAY_ERROR_PROVIDER_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.PROVIDER_IO",
    identifier: Some("RunMat:gpuArray:ProviderIO"),
    when: "Provider upload/download interaction fails.",
    message: "gpuArray: provider I/O failed",
};

const GPUARRAY_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.INTERNAL",
    identifier: Some("RunMat:gpuArray:InternalError"),
    when: "Internal tensor/container conversion fails.",
    message: "gpuArray: internal error",
};

const GPUARRAY_ERRORS: [BuiltinErrorDescriptor; 14] = [
    GPUARRAY_ERROR_NO_PROVIDER,
    GPUARRAY_ERROR_OPTION_ARGUMENT,
    GPUARRAY_ERROR_LIKE_MISSING,
    GPUARRAY_ERROR_LIKE_DUPLICATE,
    GPUARRAY_ERROR_CODISTRIBUTED_UNSUPPORTED,
    GPUARRAY_ERROR_CONFLICTING_TYPE,
    GPUARRAY_ERROR_UNKNOWN_OPTION,
    GPUARRAY_ERROR_SIZE_ARGUMENT,
    GPUARRAY_ERROR_LIKE_PROTOTYPE,
    GPUARRAY_ERROR_INPUT_TYPE,
    GPUARRAY_ERROR_CONVERSION,
    GPUARRAY_ERROR_RESHAPE,
    GPUARRAY_ERROR_PROVIDER_IO,
    GPUARRAY_ERROR_INTERNAL,
];

pub const GPUARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GPUARRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GPUARRAY_ERRORS,
};

fn gpu_array_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    gpu_array_error_with_message(error.message, error)
}

fn gpu_array_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn gpu_array_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    gpu_array_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::gpuarray")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gpuArray",
    op_kind: GpuOpKind::Custom("upload"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("upload")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Invokes the provider `upload` hook, including complex interleaved uploads, and reuploads gpuArray inputs when dtype conversion is requested. Handles class strings, size vectors, and `'like'` prototypes.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::acceleration::gpu::gpuarray"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gpuArray",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Acts as a residency boundary; fusion graphs never cross explicit host↔device transfers.",
};

#[runtime_builtin(
    name = "gpuArray",
    category = "acceleration/gpu",
    summary = "Move data to the GPU as gpuArray values.",
    keywords = "gpuArray,gpu,accelerate,upload,dtype,like",
    examples = "G = gpuArray([1 2 3], 'single');",
    accel = "array_construct",
    type_resolver(gpuarray_type),
    descriptor(crate::builtins::acceleration::gpu::gpuarray::GPUARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::acceleration::gpu::gpuarray"
)]
async fn gpu_array_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let options = parse_options(&rest)?;
    let incoming_precision = match &value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_precision(handle),
        _ => None,
    };
    let dtype = resolve_dtype(&value, &options)?;
    let dims = options.dims.clone();

    let prepared = match value {
        Value::GpuTensor(handle) => convert_device_value(handle, dtype).await?,
        other => upload_host_value(other, dtype)?,
    };

    let mut handle = prepared.handle;

    if let Some(dims) = dims.as_ref() {
        apply_dims(&mut handle, dims)?;
    }

    let provider_precision = runmat_accelerate_api::provider()
        .map(|p| p.precision())
        .unwrap_or(ProviderPrecision::F64);
    let requested_precision = match dtype {
        DataClass::Single => Some(ProviderPrecision::F32),
        _ => None,
    };
    let final_precision = requested_precision
        .or(incoming_precision)
        .unwrap_or(provider_precision);
    runmat_accelerate_api::set_handle_precision(&handle, final_precision);

    runmat_accelerate_api::set_handle_logical(&handle, prepared.logical);

    Ok(Value::GpuTensor(handle))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DataClass {
    Double,
    Single,
    Logical,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl DataClass {
    fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "double" => Some(Self::Double),
            "single" | "float32" => Some(Self::Single),
            "logical" | "bool" | "boolean" => Some(Self::Logical),
            "int8" => Some(Self::Int8),
            "int16" => Some(Self::Int16),
            "int32" | "int" => Some(Self::Int32),
            "int64" => Some(Self::Int64),
            "uint8" => Some(Self::UInt8),
            "uint16" => Some(Self::UInt16),
            "uint32" => Some(Self::UInt32),
            "uint64" => Some(Self::UInt64),
            "gpuarray" => None, // compatibility no-op
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
struct ParsedOptions {
    dims: Option<Vec<usize>>,
    explicit_dtype: Option<DataClass>,
    prototype: Option<Value>,
}

fn parse_options(rest: &[Value]) -> BuiltinResult<ParsedOptions> {
    let (index_after_dims, dims) = parse_size_arguments(rest)?;
    let mut options = ParsedOptions {
        dims,
        ..ParsedOptions::default()
    };

    let mut idx = index_after_dims;
    while idx < rest.len() {
        let tag = value_to_lower_string(&rest[idx]).ok_or_else(|| {
            gpu_array_error_with_message(
                format!(
                "gpuArray: unexpected argument {:?}; expected a class string or the keyword 'like'",
                rest[idx]
                ),
                &GPUARRAY_ERROR_OPTION_ARGUMENT,
            )
        })?;

        match tag.as_str() {
            "like" => {
                idx += 1;
                if idx >= rest.len() {
                    return Err(gpu_array_error(&GPUARRAY_ERROR_LIKE_MISSING));
                }
                if options.prototype.is_some() {
                    return Err(gpu_array_error(&GPUARRAY_ERROR_LIKE_DUPLICATE));
                }
                options.prototype = Some(rest[idx].clone());
            }
            "distributed" | "codistributed" => {
                return Err(gpu_array_error(&GPUARRAY_ERROR_CODISTRIBUTED_UNSUPPORTED));
            }
            tag => {
                if let Some(class) = DataClass::from_tag(tag) {
                    if let Some(existing) = options.explicit_dtype {
                        if existing != class {
                            return Err(gpu_array_error(&GPUARRAY_ERROR_CONFLICTING_TYPE));
                        }
                    } else {
                        options.explicit_dtype = Some(class);
                    }
                } else if tag != "gpuarray" {
                    return Err(gpu_array_error_with_detail(
                        &GPUARRAY_ERROR_UNKNOWN_OPTION,
                        format!("unrecognised option '{tag}'"),
                    ));
                }
            }
        }

        idx += 1;
    }

    Ok(options)
}

fn parse_size_arguments(rest: &[Value]) -> BuiltinResult<(usize, Option<Vec<usize>>)> {
    let mut idx = 0;
    let mut dims: Vec<usize> = Vec::new();
    let mut vector_consumed = false;

    while idx < rest.len() {
        // Stop at textual qualifiers only; numeric values continue parsing as size args.
        match &rest[idx] {
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => break,
            _ => {}
        }

        match &rest[idx] {
            Value::Int(i) => {
                dims.push(int_to_dim(i)?);
            }
            Value::Num(n) => {
                dims.push(float_to_dim(*n)?);
            }
            Value::Tensor(t) => {
                if vector_consumed || !dims.is_empty() {
                    return Err(gpu_array_error_with_message(
                        "gpuArray: size vectors cannot be combined with scalar dimensions",
                        &GPUARRAY_ERROR_SIZE_ARGUMENT,
                    ));
                }
                dims = tensor_to_dims(t)?;
                vector_consumed = true;
            }
            _ => break,
        }
        idx += 1;
    }

    let dims_option = if dims.is_empty() { None } else { Some(dims) };
    Ok((idx, dims_option))
}

fn value_to_lower_string(value: &Value) -> Option<String> {
    crate::builtins::common::tensor::value_to_string(value).map(|s| s.trim().to_ascii_lowercase())
}

fn int_to_dim(value: &IntValue) -> BuiltinResult<usize> {
    let raw = value.to_i64();
    if raw < 0 {
        return Err(gpu_array_error_with_message(
            "gpuArray: size arguments must be non-negative integers",
            &GPUARRAY_ERROR_SIZE_ARGUMENT,
        ));
    }
    Ok(raw as usize)
}

fn float_to_dim(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(gpu_array_error_with_message(
            "gpuArray: size arguments must be finite integers",
            &GPUARRAY_ERROR_SIZE_ARGUMENT,
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(gpu_array_error_with_message(
            "gpuArray: size arguments must be integers",
            &GPUARRAY_ERROR_SIZE_ARGUMENT,
        ));
    }
    if rounded < 0.0 {
        return Err(gpu_array_error_with_message(
            "gpuArray: size arguments must be non-negative",
            &GPUARRAY_ERROR_SIZE_ARGUMENT,
        ));
    }
    Ok(rounded as usize)
}

fn tensor_to_dims(tensor: &Tensor) -> BuiltinResult<Vec<usize>> {
    let mut dims = Vec::with_capacity(tensor.data.len());
    for value in &tensor.data {
        dims.push(float_to_dim(*value)?);
    }
    Ok(dims)
}

fn resolve_dtype(value: &Value, options: &ParsedOptions) -> BuiltinResult<DataClass> {
    if let Some(explicit) = options.explicit_dtype {
        return Ok(explicit);
    }
    if let Some(prototype) = options.prototype.as_ref() {
        return infer_dtype_from_prototype(prototype);
    }
    if value_defaults_to_logical(value) {
        return Ok(DataClass::Logical);
    }
    Ok(DataClass::Double)
}

fn infer_dtype_from_prototype(proto: &Value) -> BuiltinResult<DataClass> {
    match proto {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle) {
                Ok(DataClass::Logical)
            } else {
                Ok(DataClass::Double)
            }
        }
        Value::LogicalArray(_) | Value::Bool(_) => Ok(DataClass::Logical),
        Value::Int(int) => Ok(match int {
            IntValue::I8(_) => DataClass::Int8,
            IntValue::I16(_) => DataClass::Int16,
            IntValue::I32(_) => DataClass::Int32,
            IntValue::I64(_) => DataClass::Int64,
            IntValue::U8(_) => DataClass::UInt8,
            IntValue::U16(_) => DataClass::UInt16,
            IntValue::U32(_) => DataClass::UInt32,
            IntValue::U64(_) => DataClass::UInt64,
        }),
        Value::Tensor(_) | Value::Num(_) => Ok(DataClass::Double),
        Value::CharArray(_) => Ok(DataClass::Double),
        Value::String(_) => Err(gpu_array_error_with_message(
            "gpuArray: 'like' does not accept MATLAB string scalars; convert to char() first",
            &GPUARRAY_ERROR_LIKE_PROTOTYPE,
        )),
        Value::StringArray(_) => Err(gpu_array_error_with_message(
            "gpuArray: 'like' does not accept string arrays; convert to char arrays first",
            &GPUARRAY_ERROR_LIKE_PROTOTYPE,
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(DataClass::Double),
        other => Err(gpu_array_error_with_message(
            format!(
                "gpuArray: unsupported 'like' prototype type {other:?}; expected numeric or logical values"
            ),
            &GPUARRAY_ERROR_LIKE_PROTOTYPE,
        )),
    }
}

fn value_defaults_to_logical(value: &Value) -> bool {
    match value {
        Value::LogicalArray(_) | Value::Bool(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    }
}

struct PreparedHandle {
    handle: GpuTensorHandle,
    logical: bool,
}

fn upload_host_value(value: Value, dtype: DataClass) -> BuiltinResult<PreparedHandle> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if runmat_accelerate_api::provider().is_none() {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| gpu_array_error(&GPUARRAY_ERROR_NO_PROVIDER))?;

    match value {
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|err| {
                gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_INTERNAL)
            })?;
            upload_complex_host_value(provider, tensor, dtype)
        }
        Value::ComplexTensor(tensor) => upload_complex_host_value(provider, tensor, dtype),
        value => upload_real_host_value(provider, value, dtype),
    }
}

fn upload_real_host_value(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    value: Value,
    dtype: DataClass,
) -> BuiltinResult<PreparedHandle> {
    let tensor = coerce_host_value(value)?;
    let (mut tensor, logical) = cast_tensor(tensor, dtype)?;

    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let new_handle = provider.upload(&view).map_err(|err| {
        gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_PROVIDER_IO)
    })?;

    tensor.data.clear();

    Ok(PreparedHandle {
        handle: new_handle,
        logical,
    })
}

fn upload_complex_host_value(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    mut tensor: ComplexTensor,
    dtype: DataClass,
) -> BuiltinResult<PreparedHandle> {
    match dtype {
        DataClass::Double => {}
        DataClass::Single => {
            for (re, im) in &mut tensor.data {
                *re = (*re as f32) as f64;
                *im = (*im as f32) as f64;
            }
        }
        _ => {
            return Err(gpu_array_error_with_message(
                "gpuArray: complex inputs can only be uploaded as double or single precision",
                &GPUARRAY_ERROR_INPUT_TYPE,
            ));
        }
    }

    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|err| {
        gpu_array_error_with_message(err.to_string(), &GPUARRAY_ERROR_PROVIDER_IO)
    })?;
    let precision = match dtype {
        DataClass::Double => runmat_accelerate_api::ProviderPrecision::F64,
        DataClass::Single => runmat_accelerate_api::ProviderPrecision::F32,
        _ => unreachable!("complex dtype was validated above"),
    };
    runmat_accelerate_api::set_handle_precision(&handle, precision);
    Ok(PreparedHandle {
        handle,
        logical: false,
    })
}

async fn convert_device_value(
    handle: GpuTensorHandle,
    dtype: DataClass,
) -> BuiltinResult<PreparedHandle> {
    let was_logical = runmat_accelerate_api::handle_is_logical(&handle);
    let was_complex = runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
    let current_precision = runmat_accelerate_api::handle_precision(&handle);
    match dtype {
        DataClass::Double => {
            if !(was_complex
                && current_precision == Some(runmat_accelerate_api::ProviderPrecision::F32))
            {
                return Ok(PreparedHandle {
                    handle,
                    logical: false,
                });
            }
        }
        DataClass::Logical => {
            if was_logical {
                return Ok(PreparedHandle {
                    handle,
                    logical: true,
                });
            }
        }
        _ => {}
    }

    let provider = runmat_accelerate_api::provider_for_handle(&handle)
        .or_else(runmat_accelerate_api::provider)
        .ok_or_else(|| gpu_array_error(&GPUARRAY_ERROR_NO_PROVIDER))?;
    if was_complex {
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await
            .map_err(|err| {
                gpu_array_error_with_message(err.to_string(), &GPUARRAY_ERROR_PROVIDER_IO)
            })?;
        let Value::ComplexTensor(tensor) = gathered else {
            return Err(gpu_array_error_with_message(
                "gpuArray: expected complex gpuArray data during conversion",
                &GPUARRAY_ERROR_PROVIDER_IO,
            ));
        };
        let prepared = upload_complex_host_value(provider, tensor, dtype)?;
        provider.free(&handle).ok();
        return Ok(prepared);
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|err| {
            gpu_array_error_with_message(err.to_string(), &GPUARRAY_ERROR_PROVIDER_IO)
        })?;
    let (mut tensor, logical) = cast_tensor(tensor, dtype)?;

    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let new_handle = provider.upload(&view).map_err(|err| {
        gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_PROVIDER_IO)
    })?;

    provider.free(&handle).ok();
    tensor.data.clear();

    Ok(PreparedHandle {
        handle: new_handle,
        logical,
    })
}

fn coerce_host_value(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical).map_err(|err| {
            gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_INTERNAL)
        }),
        Value::Bool(flag) => {
            Tensor::new(vec![if flag { 1.0 } else { 0.0 }], vec![1, 1]).map_err(|err| {
                gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_INTERNAL)
            })
        }
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|err| {
            gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_INTERNAL)
        }),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|err| {
            gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_INTERNAL)
        }),
        Value::CharArray(ca) => char_array_to_tensor(&ca),
        Value::String(text) => {
            let ca = CharArray::new_row(&text);
            char_array_to_tensor(&ca)
        }
        Value::StringArray(_) => Err(gpu_array_error_with_message(
            "gpuArray: string arrays are not supported yet; convert to char arrays with CHAR first",
            &GPUARRAY_ERROR_INPUT_TYPE,
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(gpu_array_error_with_message(
            "gpuArray: internal complex upload routing failed",
            &GPUARRAY_ERROR_INTERNAL,
        )),
        other => Err(gpu_array_error_with_detail(
            &GPUARRAY_ERROR_INPUT_TYPE,
            format!("unsupported input type for GPU transfer: {other:?}"),
        )),
    }
}

fn cast_tensor(mut tensor: Tensor, dtype: DataClass) -> BuiltinResult<(Tensor, bool)> {
    let logical = match dtype {
        DataClass::Logical => {
            convert_to_logical(&mut tensor.data)?;
            true
        }
        DataClass::Single => {
            convert_to_single(&mut tensor.data);
            false
        }
        DataClass::Int8 => {
            convert_to_int_range(&mut tensor.data, i8::MIN as f64, i8::MAX as f64);
            false
        }
        DataClass::Int16 => {
            convert_to_int_range(&mut tensor.data, i16::MIN as f64, i16::MAX as f64);
            false
        }
        DataClass::Int32 => {
            convert_to_int_range(&mut tensor.data, i32::MIN as f64, i32::MAX as f64);
            false
        }
        DataClass::Int64 => {
            convert_to_int_range(&mut tensor.data, i64::MIN as f64, i64::MAX as f64);
            false
        }
        DataClass::UInt8 => {
            convert_to_int_range(&mut tensor.data, 0.0, u8::MAX as f64);
            false
        }
        DataClass::UInt16 => {
            convert_to_int_range(&mut tensor.data, 0.0, u16::MAX as f64);
            false
        }
        DataClass::UInt32 => {
            convert_to_int_range(&mut tensor.data, 0.0, u32::MAX as f64);
            false
        }
        DataClass::UInt64 => {
            convert_to_int_range(&mut tensor.data, 0.0, u64::MAX as f64);
            false
        }
        DataClass::Double => false,
    };

    Ok((tensor, logical))
}

fn convert_to_logical(data: &mut [f64]) -> BuiltinResult<()> {
    for value in data.iter_mut() {
        if value.is_nan() {
            return Err(gpu_array_error_with_message(
                "gpuArray: cannot convert NaN to logical",
                &GPUARRAY_ERROR_CONVERSION,
            ));
        }
        *value = if *value != 0.0 { 1.0 } else { 0.0 };
    }
    Ok(())
}

fn convert_to_single(data: &mut [f64]) {
    for value in data.iter_mut() {
        *value = (*value as f32) as f64;
    }
}

fn convert_to_int_range(data: &mut [f64], min: f64, max: f64) {
    for value in data.iter_mut() {
        if value.is_nan() {
            *value = min;
            continue;
        }
        if value.is_infinite() {
            *value = if value.is_sign_negative() { min } else { max };
            continue;
        }
        let rounded = value.round();
        *value = rounded.clamp(min, max);
    }
}

fn apply_dims(handle: &mut GpuTensorHandle, dims: &[usize]) -> BuiltinResult<()> {
    let new_elems: usize = dims.iter().product();
    let current_elems: usize = if handle.shape.is_empty() {
        new_elems
    } else {
        handle.shape.iter().product()
    };
    if new_elems != current_elems {
        return Err(gpu_array_error_with_message(
            format!(
                "gpuArray: cannot reshape gpuArray of {current_elems} elements into size {:?}",
                dims
            ),
            &GPUARRAY_ERROR_RESHAPE,
        ));
    }
    handle.shape = dims.to_vec();
    Ok(())
}

fn char_array_to_tensor(ca: &CharArray) -> BuiltinResult<Tensor> {
    let rows = ca.rows;
    let cols = ca.cols;
    if rows == 0 || cols == 0 {
        return Tensor::new(Vec::new(), vec![rows, cols]).map_err(|err| {
            gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_INTERNAL)
        });
    }
    let mut data = vec![0.0; rows * cols];
    // Store in row-major to preserve the original character order when interpreted with column-major indexing
    for row in 0..rows {
        for col in 0..cols {
            let idx_char = row * cols + col;
            let ch = ca.data[idx_char];
            data[row * cols + col] = ch as u32 as f64;
        }
    }
    Tensor::new(data, vec![rows, cols]).map_err(|err| {
        gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_INTERNAL)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{GpuTensorStorage, HostTensorView};
    use runmat_builtins::{ComplexTensor, IntValue, LogicalArray, ResolveContext, Type};

    fn call(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(gpu_array_builtin(value, rest))
    }

    fn gather_complex(value: Value) -> ComplexTensor {
        match block_on(crate::dispatcher::gather_if_needed_async(&value)).expect("gather complex") {
            Value::ComplexTensor(tensor) => tensor,
            other => panic!("expected ComplexTensor, got {other:?}"),
        }
    }

    fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, ((ar, ai), (er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (ar - er).abs() < 1e-12 && (ai - ei).abs() < 1e-12,
                "at {idx}: expected ({er}, {ei}), got ({ar}, {ai})"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_transfers_numeric_tensor() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let result = call(Value::Tensor(tensor.clone()), Vec::new()).expect("gpuArray upload");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(handle.shape, tensor.shape);
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather values");
            assert_eq!(gathered.shape, tensor.shape);
            assert_eq!(gathered.data, tensor.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_marks_logical_inputs() {
        test_support::with_test_provider(|_| {
            let logical =
                LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).expect("logical construction");
            let result =
                call(Value::LogicalArray(logical.clone()), Vec::new()).expect("gpuArray logical");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather logical");
            assert_eq!(gathered.shape, logical.shape);
            assert_eq!(gathered.data, vec![1.0, 0.0, 1.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_uploads_complex_tensor() {
        test_support::with_test_provider(|_| {
            let complex = ComplexTensor::new(vec![(1.0, -2.0), (3.5, 4.25)], vec![1, 2]).unwrap();
            let result =
                call(Value::ComplexTensor(complex.clone()), Vec::new()).expect("gpuArray complex");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = gather_complex(Value::GpuTensor(handle.clone()));
            assert_eq!(gathered.shape, complex.shape);
            assert_complex_close(&gathered.data, &complex.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_handles_scalar_bool() {
        test_support::with_test_provider(|_| {
            let result = call(Value::Bool(true), Vec::new()).expect("gpuArray bool");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather bool");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.data, vec![1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_supports_char_arrays() {
        test_support::with_test_provider(|_| {
            let chars = CharArray::new("row1row2".chars().collect(), 2, 4).unwrap();
            let original: Vec<char> = chars.data.clone();
            let result =
                call(Value::CharArray(chars), Vec::new()).expect("gpuArray char array upload");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather chars");
            assert_eq!(gathered.shape, vec![2, 4]);
            let mut recovered = Vec::new();
            for col in 0..4 {
                for row in 0..2 {
                    let idx = row + col * 2;
                    let code = gathered.data[idx];
                    let ch = char::from_u32(code as u32)
                        .expect("valid unicode scalar from numeric code");
                    recovered.push(ch);
                }
            }
            assert_eq!(recovered, original);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_converts_strings() {
        test_support::with_test_provider(|_| {
            let result = call(Value::String("gpu".into()), Vec::new()).expect("gpuArray string");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather string");
            assert_eq!(gathered.shape, vec![1, 3]);
            let expected: Vec<f64> = "gpu".chars().map(|ch| ch as u32 as f64).collect();
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_passthrough_existing_handle() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cloned = handle.clone();
            let result =
                call(Value::GpuTensor(handle.clone()), Vec::new()).expect("gpuArray passthrough");
            let Value::GpuTensor(returned) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(returned.buffer_id, cloned.buffer_id);
            assert_eq!(returned.shape, cloned.shape);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_passthrough_existing_complex_handle() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(vec![(2.0, 3.0), (-4.0, 5.5)], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
            let result =
                call(Value::GpuTensor(handle.clone()), Vec::new()).expect("gpuArray passthrough");
            let Value::GpuTensor(returned) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(returned.buffer_id, handle.buffer_id);
            assert_eq!(
                runmat_accelerate_api::handle_storage(&returned),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = gather_complex(Value::GpuTensor(returned.clone()));
            assert_complex_close(&gathered.data, &complex.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_complex_gpu_to_single_reuploads_complex_handle() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(
                vec![(1.234_567_89, -2.345_678_91), (3.456_789_12, 4.567_891_23)],
                vec![1, 2],
            )
            .unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
            let result = call(
                Value::GpuTensor(handle.clone()),
                vec![Value::from("single")],
            )
            .expect("gpuArray complex single");
            let Value::GpuTensor(returned) = result else {
                panic!("expected gpu tensor");
            };
            assert_ne!(returned.buffer_id, handle.buffer_id);
            assert_eq!(
                runmat_accelerate_api::handle_storage(&returned),
                GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&returned),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            let gathered = gather_complex(Value::GpuTensor(returned.clone()));
            let expected = complex
                .data
                .iter()
                .map(|(re, im)| ((*re as f32) as f64, (*im as f32) as f64))
                .collect::<Vec<_>>();
            assert_complex_close(&gathered.data, &expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_casts_to_int32() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.2, -3.7, 123456.0], vec![3, 1]).unwrap();
            let result =
                call(Value::Tensor(tensor), vec![Value::from("int32")]).expect("gpuArray int32");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather int32");
            assert_eq!(gathered.data, vec![1.0, -4.0, 123456.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_casts_to_uint8() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![-12.0, 12.8, 300.4, f64::INFINITY], vec![4, 1]).unwrap();
            let result =
                call(Value::Tensor(tensor), vec![Value::from("uint8")]).expect("gpuArray uint8");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather uint8");
            assert_eq!(gathered.data, vec![0.0, 13.0, 255.0, 255.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_single_precision_rounds() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.23456789, -9.87654321], vec![2, 1]).unwrap();
            let result =
                call(Value::Tensor(tensor), vec![Value::from("single")]).expect("gpuArray single");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather single");
            let expected = [1.234_567_9_f32 as f64, (-9.876_543_f32) as f64];
            for (observed, expected) in gathered.data.iter().zip(expected.iter()) {
                assert!((observed - expected).abs() < 1e-6);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_like_infers_logical() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![0.0, 2.0, -3.0], vec![3, 1]).unwrap();
            let logical_proto =
                LogicalArray::new(vec![0, 1, 0], vec![3, 1]).expect("logical proto");
            let result = call(
                Value::Tensor(tensor),
                vec![Value::from("like"), Value::LogicalArray(logical_proto)],
            )
            .expect("gpuArray like logical");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            let gathered = test_support::gather(Value::GpuTensor(handle.clone())).expect("gather");
            assert_eq!(gathered.data, vec![0.0, 1.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_like_requires_argument() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let err = call(Value::Tensor(tensor), vec![Value::from("like")]).unwrap_err();
            assert_eq!(err.to_string(), GPUARRAY_ERROR_LIKE_MISSING.message);
            assert_eq!(err.identifier(), GPUARRAY_ERROR_LIKE_MISSING.identifier);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_unknown_option_errors() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let err = call(Value::Tensor(tensor), vec![Value::from("mystery")]).unwrap_err();
            assert!(err
                .to_string()
                .contains(GPUARRAY_ERROR_UNKNOWN_OPTION.message));
            assert_eq!(err.identifier(), GPUARRAY_ERROR_UNKNOWN_OPTION.identifier);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_gpu_to_logical_reuploads() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 0.0, -5.5], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call(
                Value::GpuTensor(handle.clone()),
                vec![Value::from("logical")],
            )
            .expect("gpuArray logical cast");
            let Value::GpuTensor(new_handle) = result else {
                panic!("expected gpu tensor");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&new_handle));
            let gathered =
                test_support::gather(Value::GpuTensor(new_handle.clone())).expect("gather");
            assert_eq!(gathered.data, vec![1.0, 0.0, 1.0]);
            provider.free(&handle).ok();
            provider.free(&new_handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_gpu_logical_to_double_clears_flag() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_logical(&handle, true);
            let result = call(
                Value::GpuTensor(handle.clone()),
                vec![Value::from("double")],
            )
            .expect("gpuArray double cast");
            let Value::GpuTensor(new_handle) = result else {
                panic!("expected gpu tensor");
            };
            assert!(!runmat_accelerate_api::handle_is_logical(&new_handle));
            provider.free(&handle).ok();
            provider.free(&new_handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_applies_size_arguments() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let result = call(
                Value::Tensor(tensor),
                vec![Value::from(2i32), Value::from(2i32)],
            )
            .expect("gpuArray reshape");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(handle.shape, vec![2, 2]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_gpu_size_arguments_update_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call(
                Value::GpuTensor(handle.clone()),
                vec![Value::from(2i32), Value::from(2i32)],
            )
            .expect("gpuArray gpu reshape");
            let Value::GpuTensor(new_handle) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(new_handle.shape, vec![2, 2]);
            provider.free(&handle).ok();
            provider.free(&new_handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_size_mismatch_errors() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let err = call(
                Value::Tensor(tensor),
                vec![Value::from(2i32), Value::from(2i32)],
            )
            .unwrap_err();
            assert!(err.to_string().contains("cannot reshape"));
            assert_eq!(err.identifier(), GPUARRAY_ERROR_RESHAPE.identifier);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn gpu_array_wgpu_roundtrip() {
        use runmat_accelerate_api::AccelProvider;

        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => {
                let tensor = Tensor::new(vec![1.0, 2.5, 3.5], vec![3, 1]).unwrap();
                let result = call(Value::Tensor(tensor.clone()), vec![Value::from("int32")])
                    .expect("wgpu upload");
                let Value::GpuTensor(handle) = result else {
                    panic!("expected gpu tensor");
                };
                let gathered =
                    test_support::gather(Value::GpuTensor(handle.clone())).expect("wgpu gather");
                assert_eq!(gathered.shape, vec![3, 1]);
                assert_eq!(gathered.data, vec![1.0, 3.0, 4.0]);
                provider.free(&handle).ok();
            }
            Err(err) => {
                tracing::warn!("Skipping gpu_array_wgpu_roundtrip: {err}");
            }
        }
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn gpu_array_wgpu_complex_roundtrip() {
        use runmat_accelerate_api::AccelProvider;

        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => {
                let complex =
                    ComplexTensor::new(vec![(1.25, -0.5), (-3.0, 4.0)], vec![1, 2]).unwrap();
                let result =
                    call(Value::ComplexTensor(complex.clone()), Vec::new()).expect("wgpu upload");
                let Value::GpuTensor(handle) = result else {
                    panic!("expected gpu tensor");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_storage(&handle),
                    GpuTensorStorage::ComplexInterleaved
                );
                let gathered = gather_complex(Value::GpuTensor(handle.clone()));
                assert_eq!(gathered.shape, vec![1, 2]);
                assert_complex_close(&gathered.data, &complex.data);
                provider.free(&handle).ok();
            }
            Err(err) => {
                tracing::warn!("Skipping gpu_array_wgpu_complex_roundtrip: {err}");
            }
        }
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_accepts_int_scalars() {
        test_support::with_test_provider(|_| {
            let value = Value::Int(IntValue::I32(7));
            let result = call(value, Vec::new()).expect("gpuArray int");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather int");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.data, vec![7.0]);
        });
    }

    #[test]
    fn gpuarray_type_for_logical_is_logical() {
        assert_eq!(
            gpuarray_type(&[Type::logical()], &ResolveContext::new(Vec::new())),
            Type::logical()
        );
    }
}
