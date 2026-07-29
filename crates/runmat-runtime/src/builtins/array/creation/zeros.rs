//! MATLAB-compatible `zeros` builtin with GPU-aware semantics.

use runmat_accelerate_api::{
    GpuTensorHandle, HostIntegerDataView, HostIntegerTensorView, HostTensorView,
    IntegerElementType, ProviderPrecision,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, SparseTensor, Value,
};
use runmat_macros::runtime_builtin;
use std::sync::OnceLock;

use crate::build_runtime_error;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionExprContext,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{shape::normalize_scalar_shape, tensor};
use runmat_builtins::NumericDType;
use runmat_builtins::Type;

use crate::builtins::array::type_resolvers::tensor_type_from_rank;
use runmat_builtins::ResolveContext;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::zeros")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "zeros",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("zeros"),
        ProviderHook::Custom("zeros_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates device zeros when providers expose dedicated hooks; otherwise falls back to host upload.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("zeros").build()
}

fn zeros_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    zeros_error_with_message(error.message, error)
}

fn zeros_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    zeros_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn zeros_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("zeros");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn zeros_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    tensor_type_from_rank(args, ctx)
}

const ZEROS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output array.",
}];

const ZEROS_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const ZEROS_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const ZEROS_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const ZEROS_SIG_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype value when no numeric dimension arguments are provided.",
}];

const ZEROS_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const ZEROS_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description:
            "Class name override (double|single|logical|int8|int16|int32|int64|uint8|uint16|uint32|uint64|gpuArray).",
    },
];

const ZEROS_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype array used for class/device.",
    },
];

const ZEROS_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "A = zeros()",
        inputs: &ZEROS_SIG_EMPTY_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(n)",
        inputs: &ZEROS_SIG_N_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(size_vector)",
        inputs: &ZEROS_SIG_SIZE_VECTOR_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(m, n, ...)",
        inputs: &ZEROS_SIG_DIMS_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(prototype)",
        inputs: &ZEROS_SIG_PROTOTYPE_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(..., typename)",
        inputs: &ZEROS_SIG_CLASS_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = zeros(..., \"like\", prototype)",
        inputs: &ZEROS_SIG_LIKE_INPUTS,
        outputs: &ZEROS_OUTPUT,
    },
];

const ZEROS_ERROR_LIKE_EXPECTED_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZEROS.LIKE_EXPECTED_PROTOTYPE",
    identifier: None,
    when: "The 'like' keyword is provided without a prototype argument.",
    message: "zeros: expected prototype after 'like'",
};

const ZEROS_ERROR_CLASS_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZEROS.CLASS_CONFLICT",
    identifier: None,
    when: "A class keyword and a 'like' prototype are both provided.",
    message: "zeros: cannot combine 'like' with other class specifiers",
};

const ZEROS_ERROR_UNRECOGNIZED_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZEROS.UNRECOGNIZED_OPTION",
    identifier: None,
    when: "A trailing option string is not a supported class keyword.",
    message: "zeros: unrecognised option",
};

const ZEROS_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZEROS.LIKE_DUPLICATE",
    identifier: None,
    when: "The 'like' keyword is specified more than once.",
    message: "zeros: multiple 'like' specifications are not supported",
};

const ZEROS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ZEROS_ERROR_LIKE_EXPECTED_PROTOTYPE,
    ZEROS_ERROR_CLASS_CONFLICT,
    ZEROS_ERROR_UNRECOGNIZED_OPTION,
    ZEROS_ERROR_LIKE_DUPLICATE,
];

pub const ZEROS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZEROS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZEROS_ERRORS,
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::zeros")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "zeros",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                ScalarType::I32 => "0".to_string(),
                ScalarType::Bool => "false".to_string(),
            };
            Ok(zero)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner materialises zeros as literal constants; providers may substitute inexpensive fill kernels.",
};

#[runtime_builtin(
    name = "zeros",
    category = "array/creation",
    summary = "Create arrays filled with zero values.",
    keywords = "zeros,array,logical,gpu,like",
    accel = "array_construct",
    type_resolver(zeros_type),
    descriptor(crate::builtins::array::creation::zeros::ZEROS_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::zeros"
)]
async fn zeros_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedZeros::parse(rest).await?;
    build_output(parsed).await
}

struct ParsedZeros {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    /// Single-precision request. Host tensors are stored as f64 today; we
    /// treat 'single' as a request for a numeric zeros tensor and honour
    /// single precision when allocating on GPU via 'like' or provider hooks.
    Single,
    Logical,
    Integer(IntegerStorage),
    /// GPU-resident zeros array request via 'gpuArray' keyword or gpuArray.zeros() static method
    GpuArray,
    Like(Value),
}

impl ParsedZeros {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut implicit_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(zeros_error(&ZEROS_ERROR_LIKE_DUPLICATE));
                        }
                        if class_override.is_some() {
                            return Err(zeros_error(&ZEROS_ERROR_CLASS_CONFLICT));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(zeros_error(&ZEROS_ERROR_LIKE_EXPECTED_PROTOTYPE));
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto)?);
                        }
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err(zeros_error_with_detail(
                                &ZEROS_ERROR_CLASS_CONFLICT,
                                "logical class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(zeros_error_with_detail(
                                &ZEROS_ERROR_CLASS_CONFLICT,
                                "double class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if like_proto.is_some() {
                            return Err(zeros_error_with_detail(
                                &ZEROS_ERROR_CLASS_CONFLICT,
                                "single class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32"
                    | "uint64" => {
                        if like_proto.is_some() {
                            return Err(zeros_error_with_detail(
                                &ZEROS_ERROR_CLASS_CONFLICT,
                                format!("{keyword} class override"),
                            ));
                        }
                        class_override = Some(OutputTemplate::Integer(
                            integer_storage_prototype_from_keyword(keyword.as_str())
                                .expect("matched integer class keyword"),
                        ));
                        idx += 1;
                        continue;
                    }
                    "gpuArray" | "gpuarray" => {
                        if like_proto.is_some() {
                            return Err(zeros_error_with_detail(
                                &ZEROS_ERROR_CLASS_CONFLICT,
                                "gpuArray class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::GpuArray);
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(zeros_error_with_detail(
                            &ZEROS_ERROR_UNRECOGNIZED_OPTION,
                            format!("'{other}'"),
                        ));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg).await? {
                tracing::trace!("zeros: parsed dimension arguments {:?}", parsed_dims);
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            tracing::debug!(
                arg_type = value_tag(&arg),
                "zeros: argument did not parse as dimensions"
            );

            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg)?);
            }
            if implicit_proto.is_none() {
                implicit_proto = Some(arg.clone());
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = shape_source {
            tracing::warn!(
                shape = ?shape,
                "zeros: falling back to shape source; no dimension arguments parsed"
            );
            shape
        } else {
            vec![1, 1]
        };

        tracing::trace!(
            "zeros: resolved output shape {:?} (saw_dims_arg={})",
            shape,
            saw_dims_arg
        );

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            OutputTemplate::Like(proto)
        } else {
            OutputTemplate::Double
        };

        Ok(Self { shape, template })
    }
}

async fn build_output(parsed: ParsedZeros) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => zeros_double(&parsed.shape),
        OutputTemplate::Single => zeros_single(&parsed.shape),
        OutputTemplate::Logical => zeros_logical(&parsed.shape),
        OutputTemplate::Integer(storage) => zeros_integer_like(&storage, &parsed.shape),
        OutputTemplate::GpuArray => zeros_gpu(&parsed.shape).await,
        OutputTemplate::Like(proto) => zeros_like(&proto, &parsed.shape).await,
    }
}

fn integer_storage_prototype_from_keyword(keyword: &str) -> Option<IntegerStorage> {
    Some(match keyword {
        "int8" => IntegerStorage::I8(Vec::new()),
        "int16" => IntegerStorage::I16(Vec::new()),
        "int32" => IntegerStorage::I32(Vec::new()),
        "int64" => IntegerStorage::I64(Vec::new()),
        "uint8" => IntegerStorage::U8(Vec::new()),
        "uint16" => IntegerStorage::U16(Vec::new()),
        "uint32" => IntegerStorage::U32(Vec::new()),
        "uint64" => IntegerStorage::U64(Vec::new()),
        _ => return None,
    })
}

fn value_tag(value: &Value) -> &'static str {
    match value {
        Value::Num(_) => "Num",
        Value::Int(_) => "Int",
        Value::Bool(_) => "Bool",
        Value::Tensor(_) => "Tensor",
        Value::SparseTensor(_) => "SparseTensor",
        Value::LogicalArray(_) => "LogicalArray",
        Value::GpuTensor(_) => "GpuTensor",
        Value::Complex(_, _) => "Complex",
        Value::ComplexTensor(_) => "ComplexTensor",
        Value::String(_) => "String",
        Value::StringArray(_) => "StringArray",
        Value::CharArray(_) => "CharArray",
        Value::Symbolic(_) => "Symbolic",
        Value::Cell(_) => "Cell",
        Value::Struct(_) => "Struct",
        Value::Object(_) => "Object",
        Value::HandleObject(_) => "HandleObject",
        Value::Listener(_) => "Listener",
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. } => "FunctionHandle",
        Value::Closure(_) => "Closure",
        Value::ClassRef(_) => "ClassRef",
        Value::MException(_) => "MException",
        Value::OutputList(_) => "OutputList",
    }
}

fn zeros_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = zeros_gpu_alloc(shape, NumericDType::F64)? {
            return Ok(value);
        }
    }
    let tensor = tensor::zeros(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn zeros_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = zeros_gpu_alloc(shape, NumericDType::F32)? {
            return Ok(value);
        }
    }
    let tensor = tensor::zeros_with_dtype(shape, NumericDType::F32)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn force_host_allocation(shape: &[usize]) -> bool {
    tensor::element_count(shape) <= 1
}

fn zeros_logical(shape: &[usize]) -> crate::BuiltinResult<Value> {
    Ok(Value::LogicalArray(LogicalArray::zeros(shape.to_vec())))
}

/// Create a GPU-resident zeros array. Falls back to host tensor if no GPU provider.
async fn zeros_gpu(shape: &[usize]) -> crate::BuiltinResult<Value> {
    // Try to allocate on GPU with default precision (usually F32)
    if let Some(provider) = runmat_accelerate_api::provider() {
        let precision = provider.precision();
        let dtype = dtype_from_precision(precision);
        match provider.zeros(shape) {
            Ok(handle) => {
                runmat_accelerate_api::set_handle_precision(&handle, precision);
                return Ok(Value::GpuTensor(handle));
            }
            Err(err) => {
                log::debug!(
                    "zeros_gpu: provider.zeros failed ({err}); falling back to host upload"
                );
            }
        }
        // Fallback: build a host tensor and upload
        let host = tensor::zeros_with_dtype(shape, dtype)?;
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        }
    }
    // No GPU provider: fall back to host double tensor
    zeros_double(shape)
}

#[async_recursion::async_recursion(?Send)]
async fn zeros_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => zeros_logical(shape),
        Value::ComplexTensor(tensor) if tensor.integer_data.is_some() => {
            zeros_complex_integer_like(
                tensor
                    .integer_data
                    .as_ref()
                    .expect("guarded typed complex integer storage"),
                shape,
            )
        }
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            let tensor = ComplexTensor::zeros(shape.to_vec());
            Ok(Value::ComplexTensor(tensor))
        }
        Value::GpuTensor(handle) => zeros_like_gpu(handle, shape).await,
        Value::SparseTensor(sparse) => zeros_sparse_like(sparse, shape),
        Value::Tensor(t) => match t.integer_storage() {
            Some(storage) => zeros_integer_like(storage, shape),
            None => match t.dtype {
                NumericDType::F32 => zeros_single(shape),
                NumericDType::F64 => zeros_double(shape),
                NumericDType::I8
                | NumericDType::I16
                | NumericDType::I32
                | NumericDType::I64
                | NumericDType::U8
                | NumericDType::U16
                | NumericDType::U32
                | NumericDType::U64 => tensor::zeros_with_dtype(shape, t.dtype)
                    .map(Value::Tensor)
                    .map_err(|e| builtin_error(format!("zeros: {e}"))),
            },
        },
        Value::Int(value) => zeros_integer_like(&IntegerStorage::from_scalar(value.clone()), shape),
        Value::Num(_) => zeros_double(shape),
        Value::CharArray(_) | Value::Cell(_) => zeros_double(shape),
        _ => zeros_double(shape),
    }
}

fn zeros_integer_like(storage: &IntegerStorage, shape: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = runmat_builtins::Tensor::new_integer(
        storage.zeros_like(tensor::element_count(shape)),
        shape.to_vec(),
    )
    .map_err(|e| builtin_error(format!("zeros: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn zeros_complex_integer_like(
    storage: &IntegerComplexStorage,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let storage =
        IntegerComplexStorage::new(storage.real.zeros_like(len), storage.imag.zeros_like(len))
            .map_err(|e| builtin_error(format!("zeros: {e}")))?;
    ComplexTensor::new_integer(storage, shape.to_vec())
        .map(Value::ComplexTensor)
        .map_err(|e| builtin_error(format!("zeros: {e}")))
}

fn zeros_sparse_like(proto: &SparseTensor, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match shape {
        [rows, cols] => Ok(Value::SparseTensor(match proto.integer_storage() {
            Some(storage) => SparseTensor::zeros_with_integer_storage(*rows, *cols, storage),
            None => SparseTensor::zeros(*rows, *cols),
        })),
        other => Err(builtin_error(format!(
            "zeros: sparse 'like' output must be 2-D, got {} dimensions",
            other.len()
        ))),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn zeros_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if let Some(integer_type) = runmat_accelerate_api::handle_integer_type(handle) {
        let prototype = integer_storage_prototype_from_element_type(integer_type);
        let storage = prototype.zeros_like(tensor::element_count(shape));
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) {
            let view = integer_tensor_view(&storage, shape);
            if let Ok(gpu) = provider.upload_integer(&view) {
                return Ok(Value::GpuTensor(gpu));
            }
        }
        return zeros_integer_like(&prototype, shape);
    }

    if let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let dtype = dtype_from_precision(precision);
        let attempt = if handle.shape == shape {
            provider.zeros_like(handle)
        } else {
            provider.zeros(shape)
        };
        if let Ok(gpu) = attempt {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_zeros_fallback(shape, dtype, "provider-like-error");
        }
        // Fallback: build a host tensor with dtype matching provider precision and upload
        let host = tensor::zeros_with_dtype(shape, dtype)?;
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_zeros_fallback(shape, dtype, "upload-error");
        }
    } else {
        log_zeros_fallback(shape, NumericDType::F32, "no-provider-like");
    }

    let gathered = crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|e| format!("zeros: {e}"))?;
    log_zeros_fallback(shape, NumericDType::F32, "gather-fallback");
    zeros_like(&gathered, shape).await
}

fn integer_storage_prototype_from_element_type(element_type: IntegerElementType) -> IntegerStorage {
    match element_type {
        IntegerElementType::I8 => IntegerStorage::I8(Vec::new()),
        IntegerElementType::I16 => IntegerStorage::I16(Vec::new()),
        IntegerElementType::I32 => IntegerStorage::I32(Vec::new()),
        IntegerElementType::I64 => IntegerStorage::I64(Vec::new()),
        IntegerElementType::U8 => IntegerStorage::U8(Vec::new()),
        IntegerElementType::U16 => IntegerStorage::U16(Vec::new()),
        IntegerElementType::U32 => IntegerStorage::U32(Vec::new()),
        IntegerElementType::U64 => IntegerStorage::U64(Vec::new()),
    }
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

fn zeros_gpu_alloc(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        log_zeros_fallback(shape, dtype, "no-provider");
        return Ok(None);
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
        NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U8
        | NumericDType::U16
        | NumericDType::U32
        | NumericDType::U64 => {
            log_zeros_fallback(shape, dtype, "integer-dtype");
            return Ok(None);
        }
    };
    if provider.precision() != precision {
        log_zeros_fallback(shape, dtype, "precision-mismatch");
        return Ok(None);
    }
    match provider.zeros(shape) {
        Ok(handle) => {
            runmat_accelerate_api::set_handle_precision(&handle, precision);
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(err) => {
            log::warn!("zeros: provider zeros failed ({err}); falling back to host tensor path");
            log_zeros_fallback(shape, dtype, "provider-error");
            Ok(None)
        }
    }
}

fn zeros_fallback_debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("RUNMAT_DEBUG_ZEROS_FALLBACK"),
            Ok(value)
                if value == "1"
                    || value.eq_ignore_ascii_case("true")
                    || value.eq_ignore_ascii_case("yes")
        )
    })
}

fn log_zeros_fallback(shape: &[usize], dtype: NumericDType, reason: &str) {
    if !zeros_fallback_debug_enabled() {
        return;
    }
    let elems = tensor::element_count(shape);
    tracing::debug!(
        dtype = ?dtype,
        elems,
        shape = ?shape,
        reason,
        "[zeros_debug] fallback"
    );
}

fn dtype_from_precision(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

async fn extract_dims(value: &Value) -> crate::BuiltinResult<Option<Vec<usize>>> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    let gpu_scalar = match value {
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape) == 1,
        _ => false,
    };
    match tensor::dims_from_value_async(value).await {
        Ok(dims) => Ok(dims),
        Err(err) => {
            if matches!(value, Value::Tensor(_))
                || (matches!(value, Value::GpuTensor(_)) && !gpu_scalar)
            {
                Ok(None)
            } else {
                Err(builtin_error(format!("zeros: {err}")))
            }
        }
    }
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::SparseTensor(t) => Ok(t.shape()),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(normalize_scalar_shape(&h.shape)),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("zeros: unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerComplexStorage, IntegerStorage, SparseTensor, Tensor};

    fn clear_accel_provider_state() -> test_support::AccelTestGuard {
        test_support::accel_test_lock()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_default_scalar() {
        let _guard = clear_accel_provider_state();
        let result = block_on(zeros_builtin(Vec::new())).expect("zeros");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn zeros_type_defaults_to_num() {
        assert_eq!(zeros_type(&[], &ResolveContext::new(Vec::new())), Type::Num);
    }

    #[test]
    fn zeros_type_infers_rank_from_scalar_dim() {
        assert_eq!(
            zeros_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[test]
    fn zeros_type_infers_rank_from_size_vector() {
        let size_vec = Type::Tensor {
            shape: Some(vec![Some(1), Some(3)]),
        };
        assert_eq!(
            zeros_type(&[size_vec], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![None, None, None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_square_from_single_dimension() {
        let _guard = clear_accel_provider_state();
        let args = vec![Value::Num(3.0)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![3, 3]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_rectangular_from_dims() {
        let _guard = clear_accel_provider_state();
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.data.len(), 8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_from_size_vector() {
        let _guard = clear_accel_provider_state();
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_logical_output() {
        let _guard = clear_accel_provider_state();
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("logical")];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                assert!(logical.data.iter().all(|&x| x == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_tensor_infers_shape() {
        let _guard = clear_accel_provider_state();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_complex_scalar() {
        let _guard = clear_accel_provider_state();
        let args = vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&(re, im)| re == 0.0 && im == 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn zeros_like_typed_complex_uint64_keeps_exact_integer_storage() {
        let prototype = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                IntegerStorage::U64(vec![u64::MAX, 1]),
            )
            .expect("typed complex prototype"),
            vec![1, 2],
        )
        .expect("typed complex tensor");
        let result = block_on(zeros_builtin(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::ComplexTensor(prototype),
        ]))
        .expect("zeros like");
        let Value::ComplexTensor(output) = result else {
            panic!("expected typed complex output");
        };
        assert_eq!(
            output.integer_data,
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![0; 4]),
                    IntegerStorage::U64(vec![0; 4]),
                )
                .expect("typed complex zeros"),
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_uses_shape_argument_when_combined_with_like() {
        let _guard = clear_accel_provider_state();
        let shape_source = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let proto = Tensor::new(vec![7.0, 8.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(shape_source.clone()),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 3]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_without_explicit_shape_uses_prototype_shape() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn zeros_like_preserves_every_exact_integer_class() {
        let _guard = clear_accel_provider_state();
        let storages = vec![
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in storages {
            let prototype = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("prototype");
            let result = block_on(zeros_builtin(vec![
                Value::from("like"),
                Value::Tensor(prototype),
            ]))
            .expect("zeros like");
            let Value::Tensor(output) = result else {
                panic!("expected integer tensor");
            };
            assert_eq!(output.shape, vec![1, 2]);
            assert_eq!(output.integer_storage(), Some(&storage.zeros_like(2)));
        }

        let result = block_on(zeros_builtin(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::Int(IntValue::I64(i64::MAX)),
        ]))
        .expect("integer scalar prototype");
        let Value::Tensor(output) = result else {
            panic!("expected int64 tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I64(vec![0; 4]))
        );
    }

    #[test]
    fn zeros_class_strings_create_exact_integer_storage() {
        let _guard = clear_accel_provider_state();
        let cases = [
            ("int8", IntegerStorage::I8(vec![0; 6])),
            ("int16", IntegerStorage::I16(vec![0; 6])),
            ("int32", IntegerStorage::I32(vec![0; 6])),
            ("int64", IntegerStorage::I64(vec![0; 6])),
            ("uint8", IntegerStorage::U8(vec![0; 6])),
            ("uint16", IntegerStorage::U16(vec![0; 6])),
            ("uint32", IntegerStorage::U32(vec![0; 6])),
            ("uint64", IntegerStorage::U64(vec![0; 6])),
        ];

        for (class_name, expected) in cases {
            let result = block_on(zeros_builtin(vec![
                Value::Num(2.0),
                Value::Num(3.0),
                Value::from(class_name),
            ]))
            .expect("zeros integer class");
            let Value::Tensor(output) = result else {
                panic!("expected integer tensor for {class_name}");
            };
            assert_eq!(output.shape, vec![2, 3]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_sparse_preserves_sparse_storage_with_explicit_shape() {
        let _guard = clear_accel_provider_state();
        let proto = SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![10.0, 20.0]).unwrap();
        let args = vec![
            Value::Num(3.0),
            Value::Num(4.0),
            Value::from("like"),
            Value::SparseTensor(proto),
        ];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        match result {
            Value::SparseTensor(sparse) => {
                assert_eq!(sparse.rows, 3);
                assert_eq!(sparse.cols, 4);
                assert_eq!(sparse.col_ptrs, vec![0, 0, 0, 0, 0]);
                assert!(sparse.row_indices.is_empty());
                assert!(sparse.values.is_empty());
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_sparse_without_explicit_shape_uses_prototype_shape() {
        let _guard = clear_accel_provider_state();
        let proto = SparseTensor::zeros(2, 5);
        let args = vec![Value::from("like"), Value::SparseTensor(proto)];
        let result = block_on(zeros_builtin(args)).expect("zeros");
        match result {
            Value::SparseTensor(sparse) => {
                assert_eq!(sparse.shape(), vec![2, 5]);
                assert_eq!(sparse.col_ptrs, vec![0, 0, 0, 0, 0, 0]);
                assert!(sparse.row_indices.is_empty());
                assert!(sparse.values.is_empty());
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_empty_input_returns_empty_matrix() {
        let _guard = clear_accel_provider_state();
        let empty = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let result = block_on(zeros_builtin(vec![Value::Tensor(empty)])).expect("zeros");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_conflicting_like_and_logical_is_error() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Num(2.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        assert!(block_on(zeros_builtin(args)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(zeros_builtin(args)).expect("zeros");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.data.iter().all(|&x| x == 0.0));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_gpu_integer_like_preserves_exact_class_resident() {
        test_support::with_test_provider(|provider| {
            let prototype_values = [u64::MAX, 9_007_199_254_740_993];
            let prototype = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&prototype_values),
                    shape: &[1, 2],
                })
                .expect("upload uint64 prototype");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(prototype),
            ];
            let result = block_on(zeros_builtin(args)).expect("zeros integer gpu like");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpuArray");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            assert_eq!(handle.shape, vec![2, 2]);
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![0; 4]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn zeros_wgpu_single_allocates_gpu_without_like() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let value = zeros_single(&[2, 2]).expect("zeros single");
        match value {
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
                assert!(gathered.data.iter().all(|&x| x == 0.0));
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }
}
