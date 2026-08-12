//! MATLAB-compatible `times` builtin with GPU-aware semantics for RunMat.

use async_recursion::async_recursion;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, NumericStorage, Tensor,
    Value,
};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{
    reject_integer_logical_operands, try_integer_binary, IntegerBinaryOp,
};
use crate::builtins::math::elementwise::sparse::{try_sparse_binary, SparseBinaryOp};
use crate::builtins::math::elementwise::sparse_integer::try_typed_sparse_integer_binary;
use crate::builtins::math::symbolic::{symbolic_binary, SymbolicBinaryOp};
use crate::builtins::math::type_resolvers::numeric_binary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::times")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "times",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_mul",
            commutative: true,
        },
        ProviderHook::Custom("scalar_mul"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses elem_mul for shape-compatible gpuArrays, including complex-interleaved handles, attempts provider-side implicit expansion with repmat, and uses scalar_mul when one operand is a real scalar; falls back to host execution for unsupported operand kinds.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::times")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "times",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let lhs = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let rhs = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("({lhs} * {rhs})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a plain product; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "times";

const TIMES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise product result.",
}];

const TIMES_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left numeric/logical operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right numeric/logical operand.",
    },
];

const TIMES_INPUTS_A_B_LIKE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left numeric/logical operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right numeric/logical operand.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal string \"like\".",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Output class/device prototype.",
    },
];

const TIMES_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = times(A, B)",
        inputs: &TIMES_INPUTS_A_B,
        outputs: &TIMES_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = times(A, B, \"like\", prototype)",
        inputs: &TIMES_INPUTS_A_B_LIKE,
        outputs: &TIMES_OUTPUT,
    },
];

const TIMES_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.INVALID_ARGUMENT",
    identifier: Some("RunMat:times:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "times: invalid argument",
};

const TIMES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.INVALID_INPUT",
    identifier: Some("RunMat:times:InvalidInput"),
    when: "Operands or prototypes cannot be converted into supported numeric/logical forms.",
    message: "times: invalid input",
};

const TIMES_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SIZE_MISMATCH",
    identifier: Some("RunMat:times:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "times: array sizes are not compatible for broadcasting",
};

const TIMES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.INTERNAL",
    identifier: Some("RunMat:times:Internal"),
    when: "Provider interaction, gather/upload, or internal tensor construction failed.",
    message: "times: internal error",
};

const TIMES_ERROR_SPARSE_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SPARSE_SIZE_MISMATCH",
    identifier: Some("RunMat:times:SparseSizeMismatch"),
    when: "Sparse operands cannot be implicitly expanded to a compatible result shape.",
    message: "times: sparse operand sizes are not compatible",
};

const TIMES_ERROR_SPARSE_UNSUPPORTED_OPERAND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SPARSE_UNSUPPORTED_OPERAND",
    identifier: Some("RunMat:times:SparseUnsupportedOperand"),
    when: "Sparse arithmetic is requested with an unsupported operand class or residency.",
    message: "times: unsupported sparse arithmetic operand",
};

const TIMES_ERROR_SPARSE_DENSIFY_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SPARSE_DENSIFY_TOO_LARGE",
    identifier: Some("RunMat:times:SparseDensifyTooLarge"),
    when: "A sparse operation would have to materialize a dense or fully populated sparse result beyond the runtime limit.",
    message: "times: sparse arithmetic result is too large to materialize",
};

const TIMES_ERROR_SPARSE_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMES.SPARSE_INTERNAL",
    identifier: Some("RunMat:times:SparseInternal"),
    when: "Sparse arithmetic storage construction or conversion failed unexpectedly.",
    message: "times: sparse arithmetic internal error",
};

const TIMES_ERRORS: [BuiltinErrorDescriptor; 8] = [
    TIMES_ERROR_INVALID_ARGUMENT,
    TIMES_ERROR_INVALID_INPUT,
    TIMES_ERROR_SIZE_MISMATCH,
    TIMES_ERROR_INTERNAL,
    TIMES_ERROR_SPARSE_SIZE_MISMATCH,
    TIMES_ERROR_SPARSE_UNSUPPORTED_OPERAND,
    TIMES_ERROR_SPARSE_DENSIFY_TOO_LARGE,
    TIMES_ERROR_SPARSE_INTERNAL,
];

pub const TIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TIMES_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn times_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn times_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "times",
    category = "math/elementwise",
    summary = "Multiply arrays element-wise with MATLAB-compatible implicit expansion.",
    keywords = "times,element-wise multiply,gpu,.*",
    accel = "elementwise",
    type_resolver(numeric_binary_type),
    descriptor(crate::builtins::math::elementwise::times::TIMES_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::times"
)]
async fn times_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(&rhs)
        || rest
            .iter()
            .any(crate::builtins::common::validation::is_typed_complex_integer)
    {
        return Err(builtin_error("complex integer arithmetic is not supported"));
    }
    reject_integer_logical_operands(&lhs, &rhs, BUILTIN_NAME).map_err(builtin_error)?;
    let template = parse_output_template(&rest)?;
    let base = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => times_gpu_pair(la, lb).await,
        (Value::GpuTensor(la), rhs) => times_gpu_host_left(la, rhs).await,
        (lhs, Value::GpuTensor(rb)) => times_gpu_host_right(lhs, rb).await,
        (lhs, rhs) => Ok(times_host(lhs, rhs)?),
    }?;
    apply_output_template(base, &template).await
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    if args.is_empty() {
        return Ok(OutputTemplate::Default);
    }
    if args.len() == 1 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Err(times_error_with_detail(
                &TIMES_ERROR_INVALID_ARGUMENT,
                "expected prototype after 'like'",
            ));
        }
        return Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_ARGUMENT,
            "unsupported option; only 'like' is accepted",
        ));
    }
    if args.len() == 2 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Ok(OutputTemplate::Like(args[1].clone()));
        }
        return Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_ARGUMENT,
            "unsupported option; only 'like' is accepted",
        ));
    }
    Err(times_error_with_detail(
        &TIMES_ERROR_INVALID_ARGUMENT,
        "too many input arguments",
    ))
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => apply_like_template(value, proto).await,
    }
}

#[derive(Clone, Copy)]
enum PrototypeClass {
    Real,
    Complex,
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

struct LikeAnalysis {
    device: DevicePreference,
    class: PrototypeClass,
}

async fn apply_like_template(value: Value, prototype: &Value) -> BuiltinResult<Value> {
    let analysed = analyse_like_prototype(prototype).await?;
    match analysed.class {
        PrototypeClass::Real => match analysed.device {
            DevicePreference::Host => ensure_device(value, DevicePreference::Host).await,
            DevicePreference::Gpu => ensure_device(value, DevicePreference::Gpu).await,
        },
        PrototypeClass::Complex => {
            let host_value = ensure_device(value, DevicePreference::Host).await?;
            real_to_complex(host_value).await
        }
    }
}

async fn ensure_device(value: Value, device: DevicePreference) -> BuiltinResult<Value> {
    match device {
        DevicePreference::Host => convert_to_host_like(value).await,
        DevicePreference::Gpu => convert_to_gpu(value),
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = value {
        let temp = Value::GpuTensor(handle);
        gpu_helpers::gather_value_async(&temp)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))
    } else {
        Ok(value)
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_ARGUMENT,
            "GPU output requested via 'like' but no acceleration provider is active",
        ));
    };
    match value {
        Value::GpuTensor(handle) => Ok(gpu_helpers::resident_gpu_value(handle)),
        Value::Tensor(tensor) => {
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|e| builtin_error(format!("times: failed to upload GPU result: {e}")))?;
            Ok(gpu_helpers::resident_gpu_value(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("times: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => {
            let tensor = Tensor::new_integer(integer_storage_from_scalar(i), vec![1, 1])
                .map_err(|e| builtin_error(format!("times: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("times: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_ARGUMENT,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        Value::String(_)
        | Value::StringArray(_)
        | Value::SparseTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_) => Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_ARGUMENT,
            "unsupported prototype conversion to GPU output",
        )),
        Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::OutputList(_) => Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_ARGUMENT,
            "unsupported prototype conversion to GPU output",
        )),
    }
}

#[async_recursion(?Send)]
async fn analyse_like_prototype(proto: &Value) -> BuiltinResult<LikeAnalysis> {
    match proto {
        Value::GpuTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Gpu,
            class: PrototypeClass::Real,
        }),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Real,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Complex,
        }),
        other => {
            let gathered = gather_like_prototype(other).await?;
            analyse_like_prototype(&gathered).await
        }
    }
}

async fn gather_like_prototype(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(value)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME)),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_) => Ok(value.clone()),
        _ => Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_ARGUMENT,
            format!("unsupported prototype for 'like' ({value:?})"),
        )),
    }
}

#[async_recursion(?Send)]
async fn real_to_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(t) => {
            let shape = t.shape.clone();
            let storage = t
                .into_numeric_storage()
                .map_err(|e| builtin_error(format!("times: {e}")))?;
            let storage = match storage {
                NumericStorage::F64(values) => {
                    ComplexStorage::F64(values.into_iter().map(|value| (value, 0.0)).collect())
                }
                NumericStorage::F32(values) => {
                    ComplexStorage::F32(values.into_iter().map(|value| (value, 0.0)).collect())
                }
                storage => promote_integer_real_storage_to_complex(storage),
            };
            let tensor = ComplexTensor::from_complex_storage(storage, shape)
                .map_err(|e| builtin_error(format!("times: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("times: {e}")))?;
            real_to_complex(Value::Tensor(tensor)).await
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            real_to_complex(Value::Tensor(tensor)).await
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            real_to_complex(gathered).await
        }
        other => Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_INPUT,
            format!("cannot convert value {other:?} to complex output"),
        )),
    }
}

fn promote_integer_real_storage_to_complex(storage: NumericStorage) -> ComplexStorage {
    ComplexStorage::F64(
        storage
            .materialize_f64()
            .into_iter()
            .map(|value| (value, 0.0))
            .collect(),
    )
}

async fn times_gpu_pair(lhs: GpuTensorHandle, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if lhs.shape == rhs.shape {
            if let Ok(handle) = provider.elem_mul(&lhs, &rhs).await {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
        // Attempt N-D broadcast via repmat on device
        if let Some((out_shape, reps_l, reps_r)) = broadcast_reps(&lhs.shape, &rhs.shape) {
            let made_left = reps_l.iter().any(|&r| r != 1);
            let made_right = reps_r.iter().any(|&r| r != 1);
            let left_expanded = if made_left {
                provider
                    .repmat(&lhs, &reps_l)
                    .map_err(|e| builtin_error(format!("times: {e}")))?
            } else {
                lhs.clone()
            };
            let right_expanded = if made_right {
                provider
                    .repmat(&rhs, &reps_r)
                    .map_err(|e| builtin_error(format!("times: {e}")))?
            } else {
                rhs.clone()
            };
            let result = provider
                .elem_mul(&left_expanded, &right_expanded)
                .await
                .map_err(|e| builtin_error(format!("times: {e}")));
            if made_left {
                let _ = provider.free(&left_expanded);
            }
            if made_right {
                let _ = provider.free(&right_expanded);
            }
            if let Ok(handle) = result {
                if handle.shape == out_shape {
                    return Ok(gpu_helpers::resident_gpu_value(handle));
                } else {
                    let _ = provider.free(&handle);
                }
            }
        }
        if is_scalar_shape(&lhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&lhs).await? {
                if let Ok(handle) = provider.scalar_mul(&rhs, scalar) {
                    return Ok(gpu_helpers::resident_gpu_value(handle));
                }
            }
        }
        if is_scalar_shape(&rhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&rhs).await? {
                if let Ok(handle) = provider.scalar_mul(&lhs, scalar) {
                    return Ok(gpu_helpers::resident_gpu_value(handle));
                }
            }
        }
    }
    let left = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    times_host(left, right)
}

fn broadcast_reps(a: &[usize], b: &[usize]) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rank = a.len().max(b.len()).max(1);
    let mut out = vec![1usize; rank];
    let aa = crate::builtins::common::broadcast::align_shape(a, rank);
    let bb = crate::builtins::common::broadcast::align_shape(b, rank);
    for i in 0..rank {
        let (ad, bd) = (aa[i], bb[i]);
        if ad == bd {
            out[i] = ad;
        } else if ad == 1 {
            out[i] = bd;
        } else if bd == 1 {
            out[i] = ad;
        } else {
            return None;
        }
    }
    let reps_a: Vec<usize> = (0..rank)
        .map(|i| if aa[i] == out[i] { 1 } else { out[i] })
        .collect();
    let reps_b: Vec<usize> = (0..rank)
        .map(|i| if bb[i] == out[i] { 1 } else { out[i] })
        .collect();
    Some((out, reps_a, reps_b))
}

async fn times_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&rhs)? {
            if let Ok(handle) = provider.scalar_mul(&lhs, scalar) {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
    }
    let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    times_host(host_lhs, rhs)
}

async fn times_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&lhs)? {
            if let Ok(handle) = provider.scalar_mul(&rhs, scalar) {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
    }
    let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    times_host(lhs, host_rhs)
}

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Some(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            Some(ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        _ => None,
    }
}

fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(re, im) => Some((*re, *im)),
        Value::ComplexTensor(ct) if tensor::is_scalar_complex_tensor(ct) => {
            let value = tensor::complex_tensor_value_complex64(ct, 0);
            Some((value.re, value.im))
        }
        _ => None,
    }
}

fn scalar_times_value(lhs: &Value, rhs: &Value) -> Option<Value> {
    if matches!(lhs, Value::Tensor(_) | Value::ComplexTensor(_))
        || matches!(rhs, Value::Tensor(_) | Value::ComplexTensor(_))
    {
        return None;
    }
    let left = scalar_complex_value(lhs).or_else(|| scalar_real_value(lhs).map(|v| (v, 0.0)))?;
    let right = scalar_complex_value(rhs).or_else(|| scalar_real_value(rhs).map(|v| (v, 0.0)))?;
    let (ar, ai) = left;
    let (br, bi) = right;
    if ai != 0.0 || bi != 0.0 {
        return Some(Value::Complex(ar * br - ai * bi, ar * bi + ai * br));
    }
    Some(Value::Num(ar * br))
}

fn times_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let Some(result) = symbolic_binary(&lhs, &rhs, SymbolicBinaryOp::Mul) {
        return Ok(result);
    }
    if let Some(result) =
        try_typed_sparse_integer_binary(&lhs, &rhs, SparseBinaryOp::Mul, BUILTIN_NAME)
    {
        return result;
    }
    if let Some(result) = try_sparse_binary(&lhs, &rhs, SparseBinaryOp::Mul, BUILTIN_NAME) {
        return result;
    }
    if let Some(result) = try_integer_binary(&lhs, &rhs, IntegerBinaryOp::Multiply, BUILTIN_NAME)
        .map_err(builtin_error)?
    {
        return Ok(result);
    }
    if let Some(result) = scalar_times_value(&lhs, &rhs) {
        return Ok(result);
    }
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (TimesOperand::Real(a), TimesOperand::Real(b)) => times_real_real(a, b),
        (TimesOperand::Complex(a), TimesOperand::Complex(b)) => times_complex_complex(&a, &b),
        (TimesOperand::Complex(a), TimesOperand::Real(b)) => times_complex_real(&a, &b),
        (TimesOperand::Real(a), TimesOperand::Complex(b)) => times_real_complex(&a, &b),
    }
}

fn times_real_real(lhs: Tensor, rhs: Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| times_error_with_detail(&TIMES_ERROR_SIZE_MISMATCH, &err))?;
    let output_shape = plan.output_shape().to_vec();
    let lhs = lhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    let rhs = rhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    let output = match (lhs, rhs) {
        (NumericStorage::F32(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] * rhs[rhs_index];
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f64; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] * rhs[rhs_index];
            }
            NumericStorage::F64(output)
        }
        (NumericStorage::F32(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (f64::from(lhs[lhs_index]) * rhs[rhs_index]) as f32;
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (lhs[lhs_index] * f64::from(rhs[rhs_index])) as f32;
            }
            NumericStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "times: integer operands did not use the exact integer arithmetic path",
            ))
        }
    };
    let tensor = Tensor::from_numeric_storage(output, output_shape)
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn times_complex_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| times_error_with_detail(&TIMES_ERROR_SIZE_MISMATCH, &err))?;
    let output = match (lhs.complex_storage(), rhs.complex_storage()) {
        (ComplexStorage::F64(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = multiply_complex_f64(lhs[lhs_index], rhs[rhs_index]);
            }
            ComplexStorage::F64(output)
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = multiply_complex_f32(lhs[lhs_index], rhs[rhs_index]);
            }
            ComplexStorage::F32(output)
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let lhs = (f64::from(lhs[lhs_index].0), f64::from(lhs[lhs_index].1));
                let value = multiply_complex_f64(lhs, rhs[rhs_index]);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output)
        }
        (ComplexStorage::F64(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let rhs = (f64::from(rhs[rhs_index].0), f64::from(rhs[rhs_index].1));
                let value = multiply_complex_f64(lhs[lhs_index], rhs);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "times: complex integer arithmetic is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn times_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| times_error_with_detail(&TIMES_ERROR_SIZE_MISMATCH, &err))?;
    let rhs = rhs
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    let output = multiply_complex_real_storage(lhs.complex_storage(), &rhs, &plan, true)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn times_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| times_error_with_detail(&TIMES_ERROR_SIZE_MISMATCH, &err))?;
    let lhs = lhs
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    let output = multiply_complex_real_storage(rhs.complex_storage(), &lhs, &plan, false)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn multiply_complex_real_storage(
    complex: &ComplexStorage,
    real: &NumericStorage,
    plan: &BroadcastPlan,
    complex_is_left: bool,
) -> BuiltinResult<ComplexStorage> {
    let indices = |lhs_index: usize, rhs_index: usize| {
        if complex_is_left {
            (lhs_index, rhs_index)
        } else {
            (rhs_index, lhs_index)
        }
    };
    Ok(match (complex, real) {
        (ComplexStorage::F64(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (value.0 * scalar, value.1 * scalar);
            }
            ComplexStorage::F64(output)
        }
        (ComplexStorage::F32(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (value.0 * scalar, value.1 * scalar);
            }
            ComplexStorage::F32(output)
        }
        (ComplexStorage::F32(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (
                    (f64::from(value.0) * scalar) as f32,
                    (f64::from(value.1) * scalar) as f32,
                );
            }
            ComplexStorage::F32(output)
        }
        (ComplexStorage::F64(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let value = complex[complex_index];
                let scalar = f64::from(real[real_index]);
                output[output_index] = ((value.0 * scalar) as f32, (value.1 * scalar) as f32);
            }
            ComplexStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "times: integer operands did not use the exact integer arithmetic path",
            ))
        }
    })
}

fn multiply_complex_f64(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
}

fn multiply_complex_f32(lhs: (f32, f32), rhs: (f32, f32)) -> (f32, f32) {
    (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
}

enum TimesOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> BuiltinResult<TimesOperand> {
    match value {
        Value::Tensor(t) => Ok(TimesOperand::Real(t)),
        Value::Num(n) => Ok(TimesOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("times: {e}")))?,
        )),
        Value::Bool(b) => Ok(TimesOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("times: {e}")))?,
        )),
        Value::LogicalArray(logical) => Ok(TimesOperand::Real(
            tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("times: {e}")))?,
        )),
        Value::CharArray(chars) => Ok(TimesOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(TimesOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("times: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(TimesOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(times_error(&TIMES_ERROR_INTERNAL)),
        other => Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_INPUT,
            format!(
                "unsupported operand type {:?}; expected numeric or logical data",
                other
            ),
        )),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| builtin_error(format!("times: {e}")))
}

fn extract_scalar_f64(value: &Value) -> BuiltinResult<Option<f64>> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(provider_scalar_from_integer(i))),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(Some(tensor::tensor_value_f64(t, 0))),
        Value::LogicalArray(l) if l.data.len() == 1 => {
            Ok(Some(if l.data[0] != 0 { 1.0 } else { 0.0 }))
        }
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => Ok(Some(
            ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0),
        )),
        _ => Ok(None),
    }
}

fn provider_scalar_from_integer(value: &IntValue) -> f64 {
    value.to_f64()
}

fn integer_storage_from_scalar(value: IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![value]),
    }
}

fn is_scalar_shape(shape: &[usize]) -> bool {
    shape.iter().copied().product::<usize>() <= 1
}

async fn gpu_scalar_value(handle: &GpuTensorHandle) -> BuiltinResult<Option<f64>> {
    if !is_scalar_shape(&handle.shape) {
        return Ok(None);
    }
    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
    Ok(tensor::tensor_values_f64(&tensor).first().copied())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{
        CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, SparseTensor, Tensor,
    };

    const EPS: f64 = 1e-12;

    fn times_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::times_builtin(lhs, rhs, rest))
    }

    #[test]
    fn scalar_extractors_read_typed_integer_tensor_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("integer tensor");
        let value = Value::Tensor(tensor);

        assert_eq!(
            scalar_real_value(&value),
            Some(9_007_199_254_740_993_u64 as f64)
        );
        assert_eq!(
            extract_scalar_f64(&value).expect("scalar"),
            Some(9_007_199_254_740_993_u64 as f64)
        );

        let storage = runmat_value::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![8]),
            IntegerStorage::I16(vec![-3]),
        )
        .expect("complex integer storage");
        let complex = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex tensor");
        assert_eq!(
            scalar_complex_value(&Value::ComplexTensor(complex)),
            Some((8.0, -3.0))
        );
    }

    #[test]
    fn times_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TIMES_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"C = times(A, B)"));
        assert!(labels.contains(&"C = times(A, B, \"like\", prototype)"));
    }

    #[test]
    fn times_parser_error_has_stable_identifier() {
        let err = times_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
            .expect_err("expected parser error");
        assert_eq!(err.identifier(), TIMES_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn times_type_preserves_tensor_shape() {
        let out = numeric_binary_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn times_type_scalar_returns_num() {
        let out = numeric_binary_type(&[Type::Num, Type::Int], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_scalar_numbers() {
        let result = times_builtin(Value::Num(2.0), Value::Num(3.5), Vec::new()).expect("times");
        match result {
            Value::Num(v) => assert!((v - 7.0).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_matrix_scalar() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            times_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("times");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![2.0, 4.0, 6.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn times_preserves_single_for_single_and_mixed_floating_inputs() {
        let single_lhs = Tensor::from_f32(vec![1.5, -2.0], vec![2, 1]).unwrap();
        let single_rhs = Tensor::from_f32(vec![2.0, 4.0], vec![2, 1]).unwrap();
        let result = times_builtin(
            Value::Tensor(single_lhs.clone()),
            Value::Tensor(single_rhs),
            Vec::new(),
        )
        .expect("single times single");
        let Value::Tensor(result) = result else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![3.0, -8.0])
        );

        for (lhs, rhs) in [
            (
                Value::Tensor(single_lhs.clone()),
                Value::Tensor(Tensor::new(vec![0.2, 0.5], vec![2, 1]).unwrap()),
            ),
            (
                Value::Tensor(Tensor::new(vec![0.2, 0.5], vec![2, 1]).unwrap()),
                Value::Tensor(single_lhs.clone()),
            ),
        ] {
            let result = times_builtin(lhs, rhs, Vec::new()).expect("mixed floating times");
            let Value::Tensor(result) = result else {
                panic!("expected tensor result");
            };
            assert_eq!(
                result.into_numeric_storage().unwrap(),
                NumericStorage::F32(vec![0.3, -1.0])
            );
        }
    }

    #[test]
    fn times_like_complex_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).unwrap();

        let result =
            block_on(super::real_to_complex(Value::Tensor(tensor))).expect("complex conversion");

        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.materialize_f64(), vec![(2.0, 0.0), (3.0, 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn times_typed_sparse_uint64_uses_exact_sparse_route() {
        let lhs = SparseTensor::new_integer(
            2,
            1,
            vec![0, 2],
            vec![0, 1],
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        )
        .unwrap();
        let rhs = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![2, 1]).unwrap();
        let Value::SparseTensor(result) =
            times_builtin(Value::SparseTensor(lhs), Value::Tensor(rhs), Vec::new()).expect("times")
        else {
            panic!("expected typed sparse result");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![18_014_398_509_481_986, u64::MAX]))
        );
    }

    #[test]
    fn times_dense_integer_arrays_preserve_exact_storage_without_mirror() {
        let lhs = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 63) + 1, u64::MAX]),
            vec![2, 1],
        )
        .expect("lhs");
        let rhs = Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 0]), vec![1, 3]).expect("rhs");

        let result = times_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new())
            .expect("integer times");
        let Value::Tensor(result) = result else {
            panic!("expected integer tensor");
        };
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                (1_u64 << 63) + 1,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                0,
                0
            ]))
        );

        let scalar_tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![i16::MAX]), vec![1, 1]).expect("scalar");
        assert_eq!(
            times_builtin(Value::Tensor(scalar_tensor), Value::Num(2.0), Vec::new())
                .expect("scalar times"),
            Value::Int(IntValue::I16(i16::MAX))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_row_column_broadcast() {
        let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
        let result = times_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
            .expect("broadcast times");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0];
                assert_eq!(t.materialize_f64(), expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_complex_inputs() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
        let result = times_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex times");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(4.0, 3.0), (1.0, 7.0)];
                for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS && (got.1 - exp.1).abs() < EPS);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn times_preserves_native_complex_single_storage() {
        let lhs = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::from_f32(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
        let result = times_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex times");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(result.as_f32_slice(), Some(&[(4.0, 3.0), (1.0, 7.0)][..]));
    }

    #[test]
    fn times_mixed_complex_floating_inputs_return_single() {
        let single = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let double = ComplexTensor::new(vec![(2.0, -1.0)], vec![1, 1]).unwrap();
        for (lhs, rhs) in [
            (single.clone(), double.clone()),
            (double.clone(), single.clone()),
        ] {
            let result = times_builtin(
                Value::ComplexTensor(lhs),
                Value::ComplexTensor(rhs),
                Vec::new(),
            )
            .expect("complex times");
            let Value::ComplexTensor(result) = result else {
                panic!("expected complex single tensor");
            };
            assert_eq!(result.as_f32_slice(), Some(&[(4.0, 3.0)][..]));
        }
    }

    #[test]
    fn times_mixed_real_complex_single_paths_preserve_single() {
        let complex = ComplexTensor::from_f32(vec![(1.0, 2.0), (-3.0, 1.0)], vec![1, 2]).unwrap();
        let real = Tensor::new(vec![2.0, 0.5], vec![1, 2]).unwrap();
        for (lhs, rhs) in [
            (
                Value::ComplexTensor(complex.clone()),
                Value::Tensor(real.clone()),
            ),
            (
                Value::Tensor(real.clone()),
                Value::ComplexTensor(complex.clone()),
            ),
        ] {
            let result = times_builtin(lhs, rhs, Vec::new()).expect("complex-real times");
            let Value::ComplexTensor(result) = result else {
                panic!("expected complex single tensor");
            };
            assert_eq!(result.as_f32_slice(), Some(&[(2.0, 4.0), (-1.5, 0.5)][..]));
        }
    }

    #[test]
    fn times_preserves_empty_complex_single_class() {
        let lhs = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
        let rhs = ComplexTensor::new(Vec::new(), vec![0, 2]).unwrap();
        let result = times_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex times");
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor");
        };
        assert_eq!(result.shape, vec![0, 2]);
        assert_eq!(result.as_f32_slice(), Some(&[][..]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_char_input() {
        let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
        let result = times_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new())
            .expect("char times");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.materialize_f64(), vec![130.0, 132.0, 134.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_logical_input_promotes_to_double() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![2.0, 2.0, 3.0, 3.0], vec![2, 2]).unwrap();
        let result = times_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![2.0, 0.0, 3.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_dimension_mismatch_errors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = times_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).unwrap_err();
        assert!(err.message().contains("times"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let ha = provider.upload(&view).expect("upload");
            let hb = provider.upload(&view).expect("upload");
            let result = times_builtin(
                Value::GpuTensor(ha.clone()),
                Value::GpuTensor(hb.clone()),
                Vec::new(),
            )
            .expect("gpu times");
            let gathered = test_support::gather(result).expect("gather");
            let expected = tensor
                .materialize_f64()
                .iter()
                .zip(tensor.materialize_f64().iter())
                .map(|(x, y)| x * y)
                .collect::<Vec<_>>();
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_gpu_scalar_right() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = times_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
                .expect("gpu scalar times");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![2.0, 4.0, 6.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_gpu_scalar_left() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = times_builtin(Value::Num(3.0), Value::GpuTensor(handle), Vec::new())
                .expect("gpu scalar times");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![6.0, 12.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_like_gpu_prototype_keeps_residency() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = times_builtin(
                Value::Tensor(lhs.clone()),
                Value::Tensor(rhs.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("times like gpu");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 1]);
                    assert_eq!(gathered.materialize_f64(), vec![3.0, 8.0]);
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_like_gpu_prototype_uploads_typed_integer_storage_exactly() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new_integer(IntegerStorage::U16(vec![10, 20]), vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");

            let result = times_builtin(
                Value::Tensor(lhs),
                Value::Num(3.0),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("times like gpu");

            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.materialize_f64(), vec![30.0, 60.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_like_host_gathers_gpu_value() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
            let view_l = HostTensorView {
                data: &lhs.materialize_f64(),
                shape: &lhs.shape,
            };
            let view_r = HostTensorView {
                data: &rhs.materialize_f64(),
                shape: &rhs.shape,
            };
            let ha = provider.upload(&view_l).expect("upload lhs");
            let hb = provider.upload(&view_r).expect("upload rhs");
            let result = times_builtin(
                Value::GpuTensor(ha),
                Value::GpuTensor(hb),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("times like host");
            let Value::Tensor(t) = result else {
                panic!("expected tensor result after host gather");
            };
            assert_eq!(t.shape, vec![2, 1]);
            assert_eq!(t.materialize_f64(), vec![5.0, 12.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_like_complex_prototype_yields_complex() {
        let lhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let result = times_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("times like complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                let expected = [(8.0, 0.0), (15.0, 0.0)];
                for (got, exp) in ct.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS);
                    assert!((got.1 - exp.1).abs() < EPS);
                }
            }
            Value::Complex(re, im) => {
                assert!((re - 8.0).abs() < EPS && im.abs() < EPS);
            }
            other => panic!("expected complex output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_like_missing_prototype_errors() {
        let lhs = Value::Num(2.0);
        let rhs = Value::Num(4.0);
        let err = times_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
        assert!(
            err.message().contains("prototype"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_like_keyword_char_array() {
        test_support::with_test_provider(|provider| {
            let keyword = CharArray::new_row("LIKE");
            let lhs = Value::Num(2.0);
            let rhs = Value::Num(5.0);
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = times_builtin(
                lhs,
                rhs,
                vec![Value::CharArray(keyword), Value::GpuTensor(proto)],
            )
            .expect("times like char");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.materialize_f64(), vec![10.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn times_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let cpu = times_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
        let view_l = HostTensorView {
            data: &lhs.materialize_f64(),
            shape: &lhs.shape,
        };
        let view_r = HostTensorView {
            data: &rhs.materialize_f64(),
            shape: &rhs.shape,
        };
        let ha = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view_l)
            .unwrap();
        let hb = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view_r)
            .unwrap();
        let gpu = block_on(times_gpu_pair(ha, hb)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(t) => assert_eq!(gathered.materialize_f64(), t.materialize_f64()),
            Value::Num(n) => assert_eq!(gathered.materialize_f64(), vec![n]),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_same_class_integer_inputs_preserve_class() {
        let lhs = Value::Int(IntValue::I32(3));
        let rhs = Value::Int(IntValue::I32(5));
        let result = times_builtin(lhs, rhs, Vec::new()).expect("times");
        assert_eq!(result, Value::Int(IntValue::I32(15)));
    }
}
