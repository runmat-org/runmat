//! MATLAB-compatible `gpuArray` builtin that uploads host data to the active accelerator.
//!
//! Direct `gpuArray(X)` upload follows MATLAB semantics. Optional size arguments,
//! `'like'` prototypes, and explicit dtype toggles are RunMat-mode extensions.

use crate::builtins::acceleration::gpu::type_resolvers::gpuarray_type;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_accelerate_api::{
    GpuTensorHandle, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView,
    HostTensorView, ProviderPrecision,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, IntValue, IntegerStorage, NumericDType, NumericStorage, Tensor,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "gpuArray";

const GPUARRAY_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gpuarray-size-arguments",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gpuArray size arguments are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GpuArraySizeExtension"),
};

const GPUARRAY_DTYPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gpuarray-dtype-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "gpuArray dtype selectors are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GpuArrayDtypeExtension"),
};

const GPUARRAY_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gpuarray-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the gpuArray \"like\" prototype selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GpuArrayLikeExtension"),
};

const GPUARRAY_TEXT_UPLOAD_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gpuarray-text-upload",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "uploading character vectors or string scalars with gpuArray is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GpuArrayTextUploadExtension"),
};

pub const GPUARRAY_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    GPUARRAY_SIZE_EXTENSION,
    GPUARRAY_DTYPE_EXTENSION,
    GPUARRAY_LIKE_EXTENSION,
    GPUARRAY_TEXT_UPLOAD_EXTENSION,
];

const GPUARRAY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes upload as exact same-class gpuArray storage with the original shape.",
    }];

const GPUARRAY_INTEGER_DTYPE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat-only dtype selectors may explicitly convert X to any supported integer gpuArray class.",
    }];

pub const GPUARRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "G = gpuArray(integer_X)",
        inputs: &GPUARRAY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "The transfer preserves exact values, class, and shape. An existing gpuArray input is returned unchanged and remains valid.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "G = gpuArray(X, integer_dtype)",
        inputs: &GPUARRAY_INTEGER_DTYPE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat-only conversion uses the requested native integer class and never consumes or invalidates a gpuArray input.",
    },
];

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

const GPUARRAY_ERROR_TYPED_COMPLEX_INTEGER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.TYPED_COMPLEX_INTEGER",
    identifier: Some("RunMat:gpuArray:TypedComplexIntegerUnsupported"),
    when: "A typed complex integer array is uploaded without a matching provider storage format.",
    message: "gpuArray: typed complex integer arrays are not supported by the active acceleration provider",
};

const GPUARRAY_ERROR_TYPED_INTEGER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPUARRAY.TYPED_INTEGER",
    identifier: Some("RunMat:gpuArray:TypedIntegerUnsupported"),
    when: "A native integer value or integer GPU class is requested without matching provider storage.",
    message: "gpuArray: native integer storage is not supported by the active acceleration provider",
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

const GPUARRAY_ERRORS: [BuiltinErrorDescriptor; 16] = [
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
    GPUARRAY_ERROR_TYPED_COMPLEX_INTEGER,
    GPUARRAY_ERROR_TYPED_INTEGER,
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
    extensions(crate::builtins::acceleration::gpu::gpuarray::GPUARRAY_EXTENSIONS),
    integer_capabilities(
        crate::builtins::acceleration::gpu::gpuarray::GPUARRAY_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::acceleration::gpu::gpuarray"
)]
async fn gpu_array_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let options = parse_options(&rest)?;
    if options.dims.is_some() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GPUARRAY_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if options.explicit_dtype.is_some() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GPUARRAY_DTYPE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if options.prototype.is_some() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GPUARRAY_LIKE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_) | Value::String(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GPUARRAY_TEXT_UPLOAD_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest.is_empty() {
        if let Value::GpuTensor(handle) = &value {
            // gpuArray(G) is an identity operation. In particular, do not download,
            // re-upload, or release the storage owned by the caller's handle.
            runmat_accelerate_api::mark_handle_explicit(handle);
            return Ok(Value::GpuTensor(handle.clone()));
        }
    }
    let dtype = resolve_dtype(&value, &options)?;
    let dims = options.dims.clone();
    let expected_shape = dims
        .clone()
        .unwrap_or_else(|| gpu_array_value_shape(&value));

    let prepared = match value {
        Value::GpuTensor(handle) => convert_device_value(handle, dtype).await?,
        other => upload_host_value(other, dtype)?,
    };

    let mut handle = prepared.handle;
    if let Some(dims) = dims.as_ref() {
        apply_dims(&mut handle, dims)?;
    }

    if let Err(error) = validate_prepared_handle(
        &handle,
        prepared.provider,
        &expected_shape,
        prepared.storage,
        dtype,
        prepared.logical,
    ) {
        if prepared.owns_handle {
            let _ = prepared.provider.free(&handle);
        }
        return Err(error);
    }
    if dtype.is_integer() {
        runmat_accelerate_api::clear_handle_precision(&handle);
    } else {
        let precision = match dtype {
            DataClass::Single => ProviderPrecision::F32,
            DataClass::Double => ProviderPrecision::F64,
            DataClass::Logical => prepared.provider.precision(),
            _ => unreachable!("noninteger gpuArray class is floating or logical"),
        };
        runmat_accelerate_api::set_handle_precision(&handle, precision);
    }
    runmat_accelerate_api::set_handle_logical(&handle, prepared.logical);
    runmat_accelerate_api::set_handle_class_name(&handle, dtype.class_name());
    runmat_accelerate_api::mark_handle_explicit(&handle);

    Ok(Value::GpuTensor(handle))
}

fn gpu_array_value_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(tensor) => tensor.shape.clone(),
        Value::ComplexTensor(tensor) => tensor.shape.clone(),
        Value::LogicalArray(array) => array.shape.clone(),
        Value::GpuTensor(handle) => handle.shape.clone(),
        Value::CharArray(array) => array.shape.clone(),
        Value::String(text) => vec![1, text.chars().count()],
        _ => vec![1, 1],
    }
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
    fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::UInt8
                | Self::UInt16
                | Self::UInt32
                | Self::UInt64
        )
    }

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

    fn class_name(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::Single => "single",
            Self::Logical => "logical",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
        }
    }

    fn numeric_dtype(self) -> NumericDType {
        match self {
            Self::Double => NumericDType::F64,
            Self::Single => NumericDType::F32,
            Self::Logical => NumericDType::F64,
            Self::Int8 => NumericDType::I8,
            Self::Int16 => NumericDType::I16,
            Self::Int32 => NumericDType::I32,
            Self::Int64 => NumericDType::I64,
            Self::UInt8 => NumericDType::U8,
            Self::UInt16 => NumericDType::U16,
            Self::UInt32 => NumericDType::U32,
            Self::UInt64 => NumericDType::U64,
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
    value.try_to_usize().ok_or_else(|| {
        gpu_array_error_with_message(
            "gpuArray: size arguments must be non-negative integers",
            &GPUARRAY_ERROR_SIZE_ARGUMENT,
        )
    })
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
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(gpu_array_error_with_message(
            "gpuArray: size arguments exceed maximum supported size",
            &GPUARRAY_ERROR_SIZE_ARGUMENT,
        ));
    }
    Ok(rounded as usize)
}

fn tensor_to_dims(tensor: &Tensor) -> BuiltinResult<Vec<usize>> {
    if let Some(storage) = tensor.integer_storage() {
        return storage
            .exact_values()
            .iter()
            .map(int_to_dim)
            .collect::<BuiltinResult<Vec<_>>>();
    }

    let values = tensor::tensor_values_f64_cow(tensor);
    let mut dims = Vec::with_capacity(values.len());
    for value in values.iter() {
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
    if let Value::GpuTensor(handle) = value {
        return dtype_from_gpu_handle(handle);
    }
    if let Value::Int(value) = value {
        return Ok(data_class_from_int_value(value));
    }
    if let Value::Tensor(tensor) = value {
        return Ok(data_class_from_numeric_dtype(tensor.numeric_dtype()));
    }
    if value_defaults_to_logical(value) {
        return Ok(DataClass::Logical);
    }
    Ok(DataClass::Double)
}

fn data_class_from_int_value(value: &IntValue) -> DataClass {
    match value {
        IntValue::I8(_) => DataClass::Int8,
        IntValue::I16(_) => DataClass::Int16,
        IntValue::I32(_) => DataClass::Int32,
        IntValue::I64(_) => DataClass::Int64,
        IntValue::U8(_) => DataClass::UInt8,
        IntValue::U16(_) => DataClass::UInt16,
        IntValue::U32(_) => DataClass::UInt32,
        IntValue::U64(_) => DataClass::UInt64,
    }
}

fn data_class_from_numeric_dtype(dtype: NumericDType) -> DataClass {
    match dtype {
        NumericDType::F64 => DataClass::Double,
        NumericDType::F32 => DataClass::Single,
        NumericDType::I8 => DataClass::Int8,
        NumericDType::I16 => DataClass::Int16,
        NumericDType::I32 => DataClass::Int32,
        NumericDType::I64 => DataClass::Int64,
        NumericDType::U8 => DataClass::UInt8,
        NumericDType::U16 => DataClass::UInt16,
        NumericDType::U32 => DataClass::UInt32,
        NumericDType::U64 => DataClass::UInt64,
    }
}

fn integer_prototype(dtype: DataClass) -> Option<IntegerStorage> {
    Some(match dtype {
        DataClass::Int8 => IntegerStorage::I8(Vec::new()),
        DataClass::Int16 => IntegerStorage::I16(Vec::new()),
        DataClass::Int32 => IntegerStorage::I32(Vec::new()),
        DataClass::Int64 => IntegerStorage::I64(Vec::new()),
        DataClass::UInt8 => IntegerStorage::U8(Vec::new()),
        DataClass::UInt16 => IntegerStorage::U16(Vec::new()),
        DataClass::UInt32 => IntegerStorage::U32(Vec::new()),
        DataClass::UInt64 => IntegerStorage::U64(Vec::new()),
        _ => return None,
    })
}

fn infer_dtype_from_prototype(proto: &Value) -> BuiltinResult<DataClass> {
    match proto {
        Value::GpuTensor(handle) => dtype_from_gpu_handle(handle),
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
        Value::Tensor(tensor) => Ok(data_class_from_numeric_dtype(tensor.numeric_dtype())),
        Value::Num(_) => Ok(DataClass::Double),
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

fn dtype_from_gpu_handle(handle: &GpuTensorHandle) -> BuiltinResult<DataClass> {
    if runmat_accelerate_api::handle_is_logical(handle) {
        return Ok(DataClass::Logical);
    }
    if let Some(class_name) = runmat_accelerate_api::handle_class_name(handle) {
        if let Some(dtype) = DataClass::from_tag(class_name.trim().to_ascii_lowercase().as_str()) {
            return Ok(dtype);
        }
    }
    let precision = runmat_accelerate_api::handle_precision(handle).or_else(|| {
        runmat_accelerate_api::provider_for_handle(handle).map(|provider| provider.precision())
    });
    Ok(match precision {
        Some(ProviderPrecision::F32) => DataClass::Single,
        Some(ProviderPrecision::F64) | None => DataClass::Double,
    })
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
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    logical: bool,
    storage: runmat_accelerate_api::GpuTensorStorage,
    owns_handle: bool,
}

fn upload_host_value(value: Value, dtype: DataClass) -> BuiltinResult<PreparedHandle> {
    if matches!(&value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some()) {
        return Err(gpu_array_error(&GPUARRAY_ERROR_TYPED_COMPLEX_INTEGER));
    }

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
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: Value,
    dtype: DataClass,
) -> BuiltinResult<PreparedHandle> {
    if dtype.is_integer() {
        let (storage, shape) = match value {
            Value::Int(value) => {
                let source = IntegerStorage::from_scalar(value);
                (cast_integer_storage(&source, dtype)?, vec![1, 1])
            }
            Value::Tensor(tensor) => {
                let shape = tensor.shape.clone();
                let storage = if let Some(storage) = tensor.integer_storage().cloned() {
                    cast_integer_storage(&storage, dtype)?
                } else {
                    let values = tensor::tensor_into_values_f64(tensor);
                    Tensor::new_with_dtype(values, shape.clone(), dtype.numeric_dtype())
                        .map_err(|err| {
                            gpu_array_error_with_message(
                                format!("gpuArray: {err}"),
                                &GPUARRAY_ERROR_CONVERSION,
                            )
                        })?
                        .integer_storage()
                        .expect("integer dtype constructs integer storage")
                        .clone()
                };
                (storage, shape)
            }
            other => {
                let tensor = coerce_host_value(other)?;
                let shape = tensor.shape.clone();
                let values = tensor::tensor_into_values_f64(tensor);
                let storage = Tensor::new_with_dtype(values, shape.clone(), dtype.numeric_dtype())
                    .map_err(|err| {
                        gpu_array_error_with_message(
                            format!("gpuArray: {err}"),
                            &GPUARRAY_ERROR_CONVERSION,
                        )
                    })?
                    .integer_storage()
                    .expect("integer dtype constructs integer storage")
                    .clone();
                (storage, shape)
            }
        };
        let view = integer_tensor_view(&storage, &shape);
        let handle = provider.upload_integer(&view).map_err(|err| {
            gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_PROVIDER_IO)
        })?;
        return Ok(PreparedHandle {
            handle,
            provider,
            logical: false,
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            owns_handle: true,
        });
    }
    let tensor = coerce_host_value(value)?;
    let (tensor, logical) = cast_tensor(tensor, dtype)?;

    let values = tensor::tensor_values_f64_cow(&tensor);
    let view = HostTensorView {
        data: &values,
        shape: &tensor.shape,
    };
    let new_handle = provider.upload(&view).map_err(|err| {
        gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_PROVIDER_IO)
    })?;

    Ok(PreparedHandle {
        handle: new_handle,
        provider,
        logical,
        storage: runmat_accelerate_api::GpuTensorStorage::Real,
        owns_handle: true,
    })
}

fn cast_integer_storage(
    source: &IntegerStorage,
    dtype: DataClass,
) -> BuiltinResult<IntegerStorage> {
    let Some(prototype) = integer_prototype(dtype) else {
        return Err(gpu_array_error(&GPUARRAY_ERROR_TYPED_INTEGER));
    };
    let values = source
        .exact_values()
        .into_iter()
        .map(|value| prototype.cast_exact_assignment(&value))
        .collect();
    prototype.from_same_class_values(values).map_err(|err| {
        gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_CONVERSION)
    })
}

fn integer_tensor_view<'a>(
    storage: &'a IntegerStorage,
    shape: &'a [usize],
) -> HostIntegerTensorView<'a> {
    let data = match storage {
        IntegerStorage::I8(values) => HostIntegerDataView::I8(values),
        IntegerStorage::I16(values) => HostIntegerDataView::I16(values),
        IntegerStorage::I32(values) => HostIntegerDataView::I32(values),
        IntegerStorage::I64(values) => HostIntegerDataView::I64(values),
        IntegerStorage::U8(values) => HostIntegerDataView::U8(values),
        IntegerStorage::U16(values) => HostIntegerDataView::U16(values),
        IntegerStorage::U32(values) => HostIntegerDataView::U32(values),
        IntegerStorage::U64(values) => HostIntegerDataView::U64(values),
    };
    HostIntegerTensorView { data, shape }
}

fn upload_complex_host_value(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: ComplexTensor,
    dtype: DataClass,
) -> BuiltinResult<PreparedHandle> {
    if tensor.integer_storage().is_some() {
        return Err(gpu_array_error(&GPUARRAY_ERROR_TYPED_COMPLEX_INTEGER));
    }

    let shape = tensor.shape.clone();
    let tensor = match dtype {
        DataClass::Double => ComplexTensor::new(tensor.materialize_f64(), shape),
        DataClass::Single => ComplexTensor::from_f32(
            tensor
                .materialize_f64()
                .into_iter()
                .map(|(real, imag)| (real as f32, imag as f32))
                .collect(),
            shape,
        ),
        _ => {
            return Err(gpu_array_error_with_message(
                "gpuArray: complex inputs can only be uploaded as double or single precision",
                &GPUARRAY_ERROR_INPUT_TYPE,
            ));
        }
    }
    .map_err(|error| gpu_array_error_with_message(error, &GPUARRAY_ERROR_INPUT_TYPE))?;

    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).map_err(|err| {
        gpu_array_error_with_message(err.to_string(), &GPUARRAY_ERROR_PROVIDER_IO)
    })?;
    Ok(PreparedHandle {
        handle,
        provider,
        logical: false,
        storage: runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
        owns_handle: true,
    })
}

async fn convert_device_value(
    handle: GpuTensorHandle,
    dtype: DataClass,
) -> BuiltinResult<PreparedHandle> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle)
        .ok_or_else(|| gpu_array_error(&GPUARRAY_ERROR_NO_PROVIDER))?;
    let was_logical = runmat_accelerate_api::handle_is_logical(&handle);
    let was_complex = runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
    let current_precision = runmat_accelerate_api::handle_precision(&handle);
    let integer_type = runmat_accelerate_api::handle_integer_type(&handle);
    match dtype {
        DataClass::Double => {
            if !was_logical
                && integer_type.is_none()
                && current_precision == Some(runmat_accelerate_api::ProviderPrecision::F64)
            {
                return Ok(PreparedHandle {
                    handle,
                    provider,
                    logical: false,
                    storage: if was_complex {
                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                    } else {
                        runmat_accelerate_api::GpuTensorStorage::Real
                    },
                    owns_handle: false,
                });
            }
        }
        DataClass::Logical => {
            if was_logical {
                return Ok(PreparedHandle {
                    handle,
                    provider,
                    logical: true,
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                    owns_handle: false,
                });
            }
        }
        dtype if dtype.is_integer() => {
            if integer_type.is_some()
                && dtype_from_gpu_handle(&handle)
                    .ok()
                    .is_some_and(|current| current == dtype)
            {
                return Ok(PreparedHandle {
                    handle,
                    provider,
                    logical: false,
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                    owns_handle: false,
                });
            }
        }
        _ => {}
    }

    if dtype == DataClass::Double && provider.precision() != ProviderPrecision::F64 {
        return Err(gpu_array_error_with_message(
            "gpuArray: owning provider cannot preserve requested double precision",
            &GPUARRAY_ERROR_PROVIDER_IO,
        ));
    }
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
        return Ok(prepared);
    }

    if integer_type.is_some() && dtype.is_integer() {
        let downloaded = provider.download_integer(&handle).await.map_err(|err| {
            gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_PROVIDER_IO)
        })?;
        if downloaded.shape != handle.shape
            || downloaded.data.element_type() != integer_type.expect("guarded integer handle")
        {
            return Err(gpu_array_error_with_message(
                "gpuArray: provider returned contradictory integer payload metadata",
                &GPUARRAY_ERROR_PROVIDER_IO,
            ));
        }
        let storage = integer_storage_from_owned(downloaded.data);
        let cast = cast_integer_storage(&storage, dtype)?;
        let view = integer_tensor_view(&cast, &downloaded.shape);
        let new_handle = provider.upload_integer(&view).map_err(|err| {
            gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_PROVIDER_IO)
        })?;
        return Ok(PreparedHandle {
            handle: new_handle,
            provider,
            logical: false,
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            owns_handle: true,
        });
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|err| {
            gpu_array_error_with_message(err.to_string(), &GPUARRAY_ERROR_PROVIDER_IO)
        })?;
    let (tensor, logical) = cast_tensor(tensor, dtype)?;

    let values = tensor::tensor_values_f64_cow(&tensor);
    let view = HostTensorView {
        data: &values,
        shape: &tensor.shape,
    };
    let new_handle = provider.upload(&view).map_err(|err| {
        gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_PROVIDER_IO)
    })?;

    Ok(PreparedHandle {
        handle: new_handle,
        provider,
        logical,
        storage: runmat_accelerate_api::GpuTensorStorage::Real,
        owns_handle: true,
    })
}

fn validate_prepared_handle(
    handle: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_shape: &[usize],
    expected_storage: runmat_accelerate_api::GpuTensorStorage,
    dtype: DataClass,
    logical: bool,
) -> BuiltinResult<()> {
    let owner_matches = runmat_accelerate_api::provider_for_handle(handle)
        .is_some_and(|owner| std::ptr::eq(owner, provider));
    if !owner_matches
        || handle.device_id != provider.device_id()
        || handle.shape != expected_shape
        || runmat_accelerate_api::handle_storage(handle) != expected_storage
    {
        return Err(gpu_array_error_with_message(
            "gpuArray: provider returned a handle with the wrong owner or shape",
            &GPUARRAY_ERROR_PROVIDER_IO,
        ));
    }
    let expected_integer = integer_prototype(dtype).map(|storage| match storage {
        IntegerStorage::I8(_) => runmat_accelerate_api::IntegerElementType::I8,
        IntegerStorage::I16(_) => runmat_accelerate_api::IntegerElementType::I16,
        IntegerStorage::I32(_) => runmat_accelerate_api::IntegerElementType::I32,
        IntegerStorage::I64(_) => runmat_accelerate_api::IntegerElementType::I64,
        IntegerStorage::U8(_) => runmat_accelerate_api::IntegerElementType::U8,
        IntegerStorage::U16(_) => runmat_accelerate_api::IntegerElementType::U16,
        IntegerStorage::U32(_) => runmat_accelerate_api::IntegerElementType::U32,
        IntegerStorage::U64(_) => runmat_accelerate_api::IntegerElementType::U64,
    });
    let existing_class = runmat_accelerate_api::handle_class_name(handle);
    if runmat_accelerate_api::handle_integer_type(handle) != expected_integer
        || (expected_integer.is_some() && runmat_accelerate_api::handle_precision(handle).is_some())
        || (runmat_accelerate_api::handle_is_logical(handle) && !logical)
        || existing_class
            .as_deref()
            .is_some_and(|class_name| class_name != dtype.class_name())
    {
        return Err(gpu_array_error_with_message(
            "gpuArray: provider returned contradictory class metadata",
            &GPUARRAY_ERROR_PROVIDER_IO,
        ));
    }
    if expected_integer.is_none() {
        let expected_precision = match dtype {
            DataClass::Single => ProviderPrecision::F32,
            DataClass::Double => ProviderPrecision::F64,
            DataClass::Logical => provider.precision(),
            _ => unreachable!("integer classes were handled above"),
        };
        let effective_precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        if effective_precision != expected_precision {
            return Err(gpu_array_error_with_message(
                "gpuArray: provider cannot preserve the requested floating precision",
                &GPUARRAY_ERROR_PROVIDER_IO,
            ));
        }
    }
    Ok(())
}

fn integer_storage_from_owned(data: HostIntegerDataOwned) -> IntegerStorage {
    match data {
        HostIntegerDataOwned::I8(values) => IntegerStorage::I8(values),
        HostIntegerDataOwned::I16(values) => IntegerStorage::I16(values),
        HostIntegerDataOwned::I32(values) => IntegerStorage::I32(values),
        HostIntegerDataOwned::I64(values) => IntegerStorage::I64(values),
        HostIntegerDataOwned::U8(values) => IntegerStorage::U8(values),
        HostIntegerDataOwned::U16(values) => IntegerStorage::U16(values),
        HostIntegerDataOwned::U32(values) => IntegerStorage::U32(values),
        HostIntegerDataOwned::U64(values) => IntegerStorage::U64(values),
    }
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

fn cast_tensor(tensor: Tensor, dtype: DataClass) -> BuiltinResult<(Tensor, bool)> {
    // This function is reached for non-integer destinations only.  Native integer
    // inputs take the upload_integer path above unless the caller explicitly
    // requested a floating or logical class, in which case this is the deliberate
    // MATLAB-style cast boundary.
    let shape = tensor.shape.clone();
    let storage = tensor.into_numeric_storage().map_err(|err| {
        gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_CONVERSION)
    })?;
    let tensor = match dtype {
        DataClass::Logical => {
            let mut values = numeric_storage_into_f64(storage);
            convert_to_logical(&mut values)?;
            Tensor::new(values, shape).map_err(|err| {
                gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_CONVERSION)
            })?
        }
        DataClass::Single => {
            let storage = match storage {
                NumericStorage::F32(values) => NumericStorage::F32(values),
                storage => {
                    let mut values = numeric_storage_into_f64(storage);
                    convert_to_single(&mut values);
                    NumericStorage::F32(values.into_iter().map(|value| value as f32).collect())
                }
            };
            Tensor::from_numeric_storage(storage, shape).map_err(|err| {
                gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_CONVERSION)
            })?
        }
        DataClass::Double => {
            let values = numeric_storage_into_f64(storage);
            Tensor::new(values, shape).map_err(|err| {
                gpu_array_error_with_message(format!("gpuArray: {err}"), &GPUARRAY_ERROR_CONVERSION)
            })?
        }
        dtype => {
            debug_assert!(dtype.is_integer());
            return Err(gpu_array_error_with_message(
                format!(
                    "gpuArray: internal integer destination {} reached floating upload path",
                    dtype.class_name()
                ),
                &GPUARRAY_ERROR_INTERNAL,
            ));
        }
    };

    Ok((tensor, dtype == DataClass::Logical))
}

fn numeric_storage_into_f64(storage: NumericStorage) -> Vec<f64> {
    match storage {
        NumericStorage::F64(values) => values,
        NumericStorage::F32(values) => values.into_iter().map(f64::from).collect(),
        NumericStorage::I8(values) => values.into_iter().map(|value| value as f64).collect(),
        NumericStorage::I16(values) => values.into_iter().map(|value| value as f64).collect(),
        NumericStorage::I32(values) => values.into_iter().map(|value| value as f64).collect(),
        NumericStorage::I64(values) => values.into_iter().map(|value| value as f64).collect(),
        NumericStorage::U8(values) => values.into_iter().map(|value| value as f64).collect(),
        NumericStorage::U16(values) => values.into_iter().map(|value| value as f64).collect(),
        NumericStorage::U32(values) => values.into_iter().map(|value| value as f64).collect(),
        NumericStorage::U64(values) => values.into_iter().map(|value| value as f64).collect(),
    }
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
    use runmat_builtins::{
        ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, ResolveContext, Type,
    };

    fn call(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(gpu_array_builtin(value, rest))
    }

    fn call_with_mode(
        value: Value,
        rest: Vec<Value>,
        extensions_enabled: bool,
    ) -> crate::BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(extensions_enabled);
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

    #[test]
    fn gpu_array_rejects_handle_mislabeled_as_another_registered_provider() {
        use runmat_accelerate_api::AccelProvider as _;

        let _guard = test_support::accel_test_lock();
        let producer = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let mislabeled_owner = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        unsafe {
            runmat_accelerate_api::register_provider(producer);
            runmat_accelerate_api::register_provider(mislabeled_owner);
        }
        let handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: mislabeled_owner.device_id(),
            buffer_id: 991,
        };

        let error = validate_prepared_handle(
            &handle,
            producer,
            &[1, 1],
            GpuTensorStorage::Real,
            DataClass::Double,
            false,
        )
        .expect_err("a result must be owned by the producing provider");

        assert_eq!(error.identifier(), Some("RunMat:gpuArray:ProviderIO"));
    }

    #[test]
    fn gpu_array_rejects_integer_handle_with_floating_precision_metadata() {
        test_support::with_test_provider(|provider| {
            let values = [1_i32, 2_i32];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I32(&values),
                    shape: &[1, 2],
                })
                .expect("integer upload");
            runmat_accelerate_api::set_handle_precision(&handle, ProviderPrecision::F64);
            let error = validate_prepared_handle(
                &handle,
                provider,
                &[1, 2],
                GpuTensorStorage::Real,
                DataClass::Int32,
                false,
            )
            .expect_err("integer output cannot carry floating precision metadata");
            assert_eq!(error.identifier(), Some("RunMat:gpuArray:ProviderIO"));
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn gpu_array_extra_construction_forms_follow_compatibility_mode() {
        test_support::with_test_provider(|_| {
            call_with_mode(Value::Num(1.0), Vec::new(), false)
                .expect("MATLAB mode accepts documented gpuArray(X)");
        });
        for (rest, identifier) in [
            (
                vec![Value::from(1i32), Value::from(1i32)],
                "RunMat:compatibility:GpuArraySizeExtension",
            ),
            (
                vec![Value::from("uint8")],
                "RunMat:compatibility:GpuArrayDtypeExtension",
            ),
            (
                vec![Value::from("like"), Value::Num(0.0)],
                "RunMat:compatibility:GpuArrayLikeExtension",
            ),
        ] {
            let error = call_with_mode(Value::Num(1.0), rest, false)
                .expect_err("MATLAB mode rejects extra gpuArray construction form");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn gpu_array_text_upload_is_gated() {
        test_support::with_test_provider(|_| {
            let chars = CharArray::new("ab".chars().collect(), 1, 2).expect("chars");
            let error = call_with_mode(Value::CharArray(chars), Vec::new(), false)
                .expect_err("char upload is a RunMat extension");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:GpuArrayTextUploadExtension")
            );

            let error = call_with_mode(Value::String("ab".into()), Vec::new(), false)
                .expect_err("string upload is a RunMat extension");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:GpuArrayTextUploadExtension")
            );
        });
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
            assert!(runmat_accelerate_api::handle_is_explicit(&handle));
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather values");
            assert_eq!(gathered.shape, tensor.shape);
            assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_infers_and_preserves_native_single_tensor_class() {
        test_support::with_f32_test_provider(|_| {
            let tensor = Tensor::from_f32(vec![1.25, -3.5], vec![2, 1]).expect("single tensor");
            let result = call(Value::Tensor(tensor), Vec::new()).expect("gpuArray single upload");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_class_name(&handle).as_deref(),
                Some("single")
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&handle),
                Some(ProviderPrecision::F32)
            );

            let gathered =
                test_support::gather(Value::GpuTensor(handle)).expect("gather native single");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(
                gathered.into_numeric_storage().expect("single storage"),
                NumericStorage::F32(vec![1.25, -3.5])
            );

            let prototype =
                Tensor::from_f32(vec![0.0], vec![1, 1]).expect("single prototype tensor");
            let like_result = call(
                Value::Tensor(Tensor::new(vec![2.5], vec![1, 1]).expect("double input")),
                vec![Value::from("like"), Value::Tensor(prototype)],
            )
            .expect("gpuArray like single");
            let Value::GpuTensor(like_handle) = like_result else {
                panic!("expected like gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_class_name(&like_handle).as_deref(),
                Some("single")
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&like_handle),
                Some(ProviderPrecision::F32)
            );
        });
    }

    #[test]
    fn gpu_array_of_existing_single_handle_is_identity() {
        test_support::with_f32_test_provider(|_| {
            let uploaded = call(
                Value::Tensor(Tensor::from_f32(vec![1.25, -3.5], vec![2, 1]).unwrap()),
                Vec::new(),
            )
            .expect("single upload");
            let Value::GpuTensor(handle) = uploaded else {
                panic!("expected gpu tensor");
            };
            let identity =
                call(Value::GpuTensor(handle.clone()), Vec::new()).expect("gpuArray identity");
            let Value::GpuTensor(identity) = identity else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_identity(&identity),
                runmat_accelerate_api::handle_identity(&handle)
            );
            let gathered = test_support::gather(Value::GpuTensor(handle.clone()))
                .expect("source remains valid");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.materialize_f64(), vec![1.25, -3.5]);
        });
    }

    #[test]
    fn direct_gpu_builtin_results_preserve_explicit_gpuarray_intent() {
        test_support::with_test_provider(|_| {
            let input = call(
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
                Vec::new(),
            )
            .expect("explicit gpuArray");
            let output =
                crate::call_builtin("plus", &[input.clone(), input]).expect("direct provider plus");
            let Value::GpuTensor(handle) = &output else {
                panic!("expected resident plus output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(handle));
            assert_eq!(
                crate::call_builtin("isgpuarray", &[output]).expect("isgpuarray"),
                Value::Bool(true)
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_rewraps_integer_handles_without_double_roundtrip() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]),
                vec![1, 2],
            )
            .expect("integer tensor");
            let uploaded =
                call(Value::Tensor(tensor), Vec::new()).expect("gpuArray integer upload");
            let Value::GpuTensor(handle) = uploaded else {
                panic!("expected integer gpu tensor");
            };

            let rewrapped =
                call(Value::GpuTensor(handle.clone()), Vec::new()).expect("gpuArray rewrap");
            let Value::GpuTensor(rewrapped) = rewrapped else {
                panic!("expected rewrapped gpu tensor");
            };
            assert_eq!(rewrapped.buffer_id, handle.buffer_id);
            assert_eq!(
                runmat_accelerate_api::handle_class_name(&rewrapped).as_deref(),
                Some("uint64")
            );

            let gathered =
                test_support::gather(Value::GpuTensor(rewrapped)).expect("gather integer");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    9_223_372_036_854_775_808,
                    u64::MAX,
                ]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_noninteger_class_uploads_read_typed_integer_storage_exactly() {
        test_support::with_test_provider(|_| {
            for (class_name, expected, logical) in [
                ("double", vec![2.0, 4.0], false),
                ("logical", vec![1.0, 1.0], true),
            ] {
                let tensor = Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![1, 2])
                    .expect("integer tensor");

                let uploaded = call(Value::Tensor(tensor), vec![Value::from(class_name)])
                    .expect("gpuArray class upload");
                let Value::GpuTensor(handle) = uploaded else {
                    panic!("expected gpu tensor for {class_name}");
                };
                assert_eq!(runmat_accelerate_api::handle_is_logical(&handle), logical);
                assert!(runmat_accelerate_api::handle_integer_type(&handle).is_none());
                let gathered =
                    test_support::gather(Value::GpuTensor(handle)).expect("gather uploaded tensor");
                assert_eq!(gathered.shape, vec![1, 2]);
                assert_eq!(gathered.materialize_f64(), expected, "{class_name}");
            }
        });
        test_support::with_f32_test_provider(|_| {
            let tensor = Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![1, 2])
                .expect("integer tensor");
            let uploaded = call(Value::Tensor(tensor), vec![Value::from("single")])
                .expect("gpuArray single upload");
            let Value::GpuTensor(handle) = uploaded else {
                panic!("expected single gpu tensor");
            };
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather single");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.materialize_f64(), vec![2.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_integer_handle_conversion_uses_integer_buffers() {
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![0, 9_223_372_036_854_775_808, u64::MAX]),
                vec![3, 1],
            )
            .expect("integer tensor");
            let uploaded =
                call(Value::Tensor(tensor), Vec::new()).expect("gpuArray integer upload");
            let Value::GpuTensor(handle) = uploaded else {
                panic!("expected integer gpu tensor");
            };

            let converted = call(Value::GpuTensor(handle.clone()), vec![Value::from("int16")])
                .expect("gpuArray int16 conversion");
            let Value::GpuTensor(converted) = converted else {
                panic!("expected converted gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_class_name(&converted).as_deref(),
                Some("int16")
            );
            let gathered =
                test_support::gather(Value::GpuTensor(converted)).expect("gather converted");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::I16(vec![0, i16::MAX, i16::MAX]))
            );
            let original =
                test_support::gather(Value::GpuTensor(handle)).expect("source remains valid");
            assert_eq!(
                original.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    0,
                    9_223_372_036_854_775_808,
                    u64::MAX,
                ]))
            );
        });
    }

    #[test]
    fn gpu_array_dimension_preserves_representable_uint64_values() {
        let expected = usize::try_from(u64::MAX).ok();
        assert_eq!(int_to_dim(&IntValue::U64(u64::MAX)).ok(), expected);
        assert!(int_to_dim(&IntValue::I64(-1)).is_err());
    }

    #[test]
    fn gpu_array_tensor_dimensions_use_exact_integer_storage() {
        let expected = usize::try_from(u64::MAX).ok();
        let dims = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("integer size vector");
        assert_eq!(tensor_to_dims(&dims).ok(), expected.map(|dim| vec![dim]));

        let negative = Tensor::new_integer(IntegerStorage::I8(vec![-1]), vec![1, 1])
            .expect("integer size vector");
        assert!(tensor_to_dims(&negative).is_err());
    }

    #[test]
    fn gpu_array_float_dimensions_reject_unrepresentable_usize_boundary() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };

        assert!(float_to_dim(boundary).is_err());
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
            assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 1.0, 1.0]);
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
            assert_complex_close(&gathered.materialize_f64(), &complex.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_rejects_typed_complex_integer_tensor_without_uploading() {
        test_support::with_test_provider(|_| {
            let complex = ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                    IntegerStorage::U64(vec![1, 2]),
                )
                .unwrap(),
                vec![1, 2],
            )
            .unwrap();
            let err = call(Value::ComplexTensor(complex), Vec::new())
                .expect_err("typed complex integer gpuArray input must be rejected");
            assert_eq!(
                err.identifier.as_deref(),
                Some("RunMat:gpuArray:TypedComplexIntegerUnsupported")
            );
            assert!(err
                .to_string()
                .contains("typed complex integer arrays are not supported"));
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
            assert_eq!(gathered.materialize_f64(), vec![1.0]);
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
                    let code = gathered.materialize_f64()[idx];
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
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_passthrough_existing_handle() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
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
            assert_complex_close(&gathered.materialize_f64(), &complex.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_complex_gpu_to_single_rejects_when_owner_is_physically_double() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(
                vec![(1.234_567_89, -2.345_678_91), (3.456_789_12, 4.567_891_23)],
                vec![1, 2],
            )
            .unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
            let error = call(
                Value::GpuTensor(handle.clone()),
                vec![Value::from("single")],
            )
            .expect_err("double-only owner cannot preserve complex single storage");
            assert_eq!(error.identifier(), Some("RunMat:gpuArray:ProviderIO"));
            let gathered = gather_complex(Value::GpuTensor(handle));
            assert_complex_close(&gathered.materialize_f64(), &complex.materialize_f64());
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
            assert_eq!(gathered.materialize_f64(), vec![1.0, -4.0, 123456.0]);
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
            assert_eq!(gathered.materialize_f64(), vec![0.0, 13.0, 255.0, 255.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_single_precision_rounds() {
        test_support::with_f32_test_provider(|_| {
            let tensor = Tensor::new(vec![1.23456789, -9.87654321], vec![2, 1]).unwrap();
            let result =
                call(Value::Tensor(tensor), vec![Value::from("single")]).expect("gpuArray single");
            let Value::GpuTensor(handle) = result else {
                panic!("expected gpu tensor");
            };
            let gathered =
                test_support::gather(Value::GpuTensor(handle.clone())).expect("gather single");
            let expected = [1.234_567_9_f32 as f64, (-9.876_543_f32) as f64];
            for (observed, expected) in gathered.materialize_f64().iter().zip(expected.iter()) {
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
            assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, 1.0]);
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
                data: &tensor.materialize_f64(),
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
            assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 1.0]);
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
                data: &tensor.materialize_f64(),
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
                data: &tensor.materialize_f64(),
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
    fn gpu_array_wgpu_native_integer_roundtrip() {
        use runmat_accelerate_api::AccelProvider;

        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => {
                for storage in [
                    IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
                    IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
                    IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
                    IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                    IntegerStorage::U8(vec![0, u8::MAX]),
                    IntegerStorage::U16(vec![0, u16::MAX]),
                    IntegerStorage::U32(vec![0, u32::MAX]),
                    IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                ] {
                    let expected = storage.clone();
                    let result = call(
                        Value::Tensor(Tensor::new_integer(storage, vec![2, 1]).unwrap()),
                        Vec::new(),
                    )
                    .expect("wgpu integer upload");
                    let Value::GpuTensor(handle) = result else {
                        panic!("expected gpu tensor");
                    };
                    assert!(runmat_accelerate_api::handle_integer_type(&handle).is_some());
                    let gathered = test_support::gather(Value::GpuTensor(handle.clone()))
                        .expect("wgpu integer gather");
                    assert_eq!(gathered.shape, vec![2, 1]);
                    assert_eq!(gathered.integer_storage(), Some(&expected));
                    provider.free(&handle).expect("free integer gpu buffer");
                }
            }
            Err(err) => {
                tracing::warn!("Skipping gpu_array_wgpu_native_integer_roundtrip: {err}");
            }
        }
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn gpu_array_wgpu_u64_linear_gather_and_scatter_remain_exact() {
        use runmat_accelerate_api::AccelProvider;

        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let source = match call(
            Value::Tensor(
                Tensor::new_integer(
                    IntegerStorage::U64(vec![0, 1_u64 << 63, u64::MAX]),
                    vec![1, 3],
                )
                .unwrap(),
            ),
            Vec::new(),
        )
        .expect("upload source")
        {
            Value::GpuTensor(handle) => handle,
            other => panic!("expected gpu tensor, got {other:?}"),
        };
        let selected = provider
            .gather_linear(&source, &[2, 1], &[1, 2])
            .expect("exact u64 gather");
        let gathered = test_support::gather(Value::GpuTensor(selected.clone())).expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
        );

        let target = match call(
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![0, 0, 0]), vec![1, 3]).unwrap(),
            ),
            Vec::new(),
        )
        .expect("upload target")
        {
            Value::GpuTensor(handle) => handle,
            other => panic!("expected gpu tensor, got {other:?}"),
        };
        provider
            .scatter_linear(&target, &[0, 2], &selected)
            .expect("exact u64 scatter");
        let gathered = test_support::gather(Value::GpuTensor(target.clone())).expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 0, 1_u64 << 63]))
        );
        for handle in [&source, &selected, &target] {
            provider.free(handle).expect("free gpu buffer");
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
                let result = call(Value::ComplexTensor(complex.clone()), Vec::new());
                if provider.precision() == ProviderPrecision::F32 {
                    let error = result.expect_err(
                        "an F32 provider cannot preserve an implicit complex-double gpuArray",
                    );
                    assert_eq!(error.identifier(), Some("RunMat:gpuArray:ProviderIO"));
                    runmat_accelerate::simple_provider::register_inprocess_provider();
                    return;
                }
                let result = result.expect("wgpu complex-double upload");
                let Value::GpuTensor(handle) = result else {
                    panic!("expected gpu tensor");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_storage(&handle),
                    GpuTensorStorage::ComplexInterleaved
                );
                let gathered = gather_complex(Value::GpuTensor(handle.clone()));
                assert_eq!(gathered.shape, vec![1, 2]);
                assert_complex_close(&gathered.materialize_f64(), &complex.materialize_f64());
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
    fn gpu_array_roundtrips_native_integer_inputs_and_class_requests() {
        test_support::with_test_provider(|_| {
            for (storage, element_type, class_name) in [
                (
                    IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
                    runmat_accelerate_api::IntegerElementType::I8,
                    "int8",
                ),
                (
                    IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
                    runmat_accelerate_api::IntegerElementType::I16,
                    "int16",
                ),
                (
                    IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
                    runmat_accelerate_api::IntegerElementType::I32,
                    "int32",
                ),
                (
                    IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                    runmat_accelerate_api::IntegerElementType::I64,
                    "int64",
                ),
                (
                    IntegerStorage::U8(vec![0, u8::MAX]),
                    runmat_accelerate_api::IntegerElementType::U8,
                    "uint8",
                ),
                (
                    IntegerStorage::U16(vec![0, u16::MAX]),
                    runmat_accelerate_api::IntegerElementType::U16,
                    "uint16",
                ),
                (
                    IntegerStorage::U32(vec![0, u32::MAX]),
                    runmat_accelerate_api::IntegerElementType::U32,
                    "uint32",
                ),
                (
                    IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                    runmat_accelerate_api::IntegerElementType::U64,
                    "uint64",
                ),
            ] {
                let expected = storage.clone();
                let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
                let handle = match call(value, Vec::new()).expect("integer gpuArray upload") {
                    Value::GpuTensor(handle) => handle,
                    other => panic!("expected gpu tensor, got {other:?}"),
                };
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&handle),
                    Some(element_type)
                );
                assert_eq!(
                    runmat_accelerate_api::handle_class_name(&handle).as_deref(),
                    Some(class_name)
                );
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                assert_eq!(gathered.integer_storage(), Some(&expected));
            }

            let handle = match call(Value::Num(1.0), vec![Value::from("uint64")])
                .expect("uint64 gpuArray conversion")
            {
                Value::GpuTensor(handle) => handle,
                other => panic!("expected gpu tensor, got {other:?}"),
            };
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![1]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_array_like_integer_tensor_prototype_preserves_native_class() {
        test_support::with_test_provider(|_| {
            let prototype = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                    .expect("prototype"),
            );
            let handle = match call(Value::Num(7.0), vec![Value::from("like"), prototype])
                .expect("gpuArray like integer prototype")
            {
                Value::GpuTensor(handle) => handle,
                other => panic!("expected gpu tensor, got {other:?}"),
            };

            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(
                runmat_accelerate_api::handle_class_name(&handle).as_deref(),
                Some("uint64")
            );
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![7]))
            );
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
