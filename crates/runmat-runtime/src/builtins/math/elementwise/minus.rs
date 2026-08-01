//! MATLAB-compatible `minus` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, IntegerStorage, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{try_integer_binary, IntegerBinaryOp};
use crate::builtins::math::elementwise::sparse::{try_sparse_binary, SparseBinaryOp};
use crate::builtins::math::elementwise::sparse_integer::try_typed_sparse_integer_binary;
use crate::builtins::math::symbolic::{symbolic_binary, SymbolicBinaryOp};
use crate::builtins::math::type_resolvers::numeric_binary_type;
use crate::{build_runtime_error, dispatcher::download_handle_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::minus")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "minus",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_sub",
            commutative: false,
        },
        ProviderHook::Custom("scalar_sub"),
        ProviderHook::Custom("scalar_rsub"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses elem_sub for equal-shape gpuArrays, including complex-interleaved handles, attempts provider-side implicit expansion with repmat, and uses scalar_sub / scalar_rsub hooks for real scalar broadcast cases; unsupported shapes fall back to host execution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::minus")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "minus",
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
            Ok(format!("({lhs} - {rhs})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a straightforward difference; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "minus";

const MINUS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise difference result.",
}];

const MINUS_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
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

const MINUS_INPUTS_A_B_LIKE: [BuiltinParamDescriptor; 4] = [
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

const MINUS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = minus(A, B)",
        inputs: &MINUS_INPUTS_A_B,
        outputs: &MINUS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = minus(A, B, \"like\", prototype)",
        inputs: &MINUS_INPUTS_A_B_LIKE,
        outputs: &MINUS_OUTPUT,
    },
];

const MINUS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.INVALID_ARGUMENT",
    identifier: Some("RunMat:minus:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "minus: invalid argument",
};

const MINUS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.INVALID_INPUT",
    identifier: Some("RunMat:minus:InvalidInput"),
    when: "Operands or prototypes cannot be converted into supported numeric/logical forms.",
    message: "minus: invalid input",
};

const MINUS_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SIZE_MISMATCH",
    identifier: Some("RunMat:minus:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "minus: array sizes are not compatible for broadcasting",
};

const MINUS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.INTERNAL",
    identifier: Some("RunMat:minus:Internal"),
    when: "Provider interaction, gather/upload, or internal tensor construction failed.",
    message: "minus: internal error",
};

const MINUS_ERROR_SPARSE_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SPARSE_SIZE_MISMATCH",
    identifier: Some("RunMat:minus:SparseSizeMismatch"),
    when: "Sparse operands cannot be implicitly expanded to a compatible result shape.",
    message: "minus: sparse operand sizes are not compatible",
};

const MINUS_ERROR_SPARSE_UNSUPPORTED_OPERAND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SPARSE_UNSUPPORTED_OPERAND",
    identifier: Some("RunMat:minus:SparseUnsupportedOperand"),
    when: "Sparse arithmetic is requested with an unsupported operand class or residency.",
    message: "minus: unsupported sparse arithmetic operand",
};

const MINUS_ERROR_SPARSE_DENSIFY_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SPARSE_DENSIFY_TOO_LARGE",
    identifier: Some("RunMat:minus:SparseDensifyTooLarge"),
    when: "A sparse operation would have to materialize a dense or fully populated sparse result beyond the runtime limit.",
    message: "minus: sparse arithmetic result is too large to materialize",
};

const MINUS_ERROR_SPARSE_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MINUS.SPARSE_INTERNAL",
    identifier: Some("RunMat:minus:SparseInternal"),
    when: "Sparse arithmetic storage construction or conversion failed unexpectedly.",
    message: "minus: sparse arithmetic internal error",
};

const MINUS_ERRORS: [BuiltinErrorDescriptor; 8] = [
    MINUS_ERROR_INVALID_ARGUMENT,
    MINUS_ERROR_INVALID_INPUT,
    MINUS_ERROR_SIZE_MISMATCH,
    MINUS_ERROR_INTERNAL,
    MINUS_ERROR_SPARSE_SIZE_MISMATCH,
    MINUS_ERROR_SPARSE_UNSUPPORTED_OPERAND,
    MINUS_ERROR_SPARSE_DENSIFY_TOO_LARGE,
    MINUS_ERROR_SPARSE_INTERNAL,
];

pub const MINUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MINUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MINUS_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn minus_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn minus_error_with_detail(
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
    name = "minus",
    category = "math/elementwise",
    summary = "Element-wise subtraction with MATLAB-compatible implicit expansion.",
    keywords = "minus,element-wise subtraction,gpu,-",
    accel = "elementwise",
    type_resolver(numeric_binary_type),
    descriptor(crate::builtins::math::elementwise::minus::MINUS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::minus"
)]
async fn minus_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::is_typed_complex_integer(&lhs)
        || crate::builtins::common::validation::is_typed_complex_integer(&rhs)
        || rest
            .iter()
            .any(crate::builtins::common::validation::is_typed_complex_integer)
    {
        return Err(builtin_error("complex integer arithmetic is not supported"));
    }
    let template = parse_output_template(&rest)?;
    let base = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => minus_gpu_pair(la, lb).await,
        (Value::GpuTensor(la), rhs) => minus_gpu_host_left(la, rhs).await,
        (lhs, Value::GpuTensor(rb)) => minus_gpu_host_right(lhs, rb).await,
        (lhs, rhs) => minus_host(lhs, rhs),
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
            return Err(minus_error_with_detail(
                &MINUS_ERROR_INVALID_ARGUMENT,
                "expected prototype after 'like'",
            ));
        }
        return Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_ARGUMENT,
            "unsupported option; only 'like' is accepted",
        ));
    }
    if args.len() == 2 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Ok(OutputTemplate::Like(args[1].clone()));
        }
        return Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_ARGUMENT,
            "unsupported option; only 'like' is accepted",
        ));
    }
    Err(minus_error_with_detail(
        &MINUS_ERROR_INVALID_ARGUMENT,
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
        gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await
    } else {
        Ok(value)
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_ARGUMENT,
            "GPU output requested via 'like' but no acceleration provider is active",
        ));
    };
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|e| builtin_error(format!("minus: failed to upload GPU result: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("minus: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => {
            let tensor = Tensor::new_integer(IntegerStorage::from_scalar(i), vec![1, 1])
                .map_err(|e| builtin_error(format!("minus: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("minus: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_ARGUMENT,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        Value::String(_)
        | Value::StringArray(_)
        | Value::SparseTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::Symbolic(_) => Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_ARGUMENT,
            "unsupported prototype conversion to GPU output",
        )),
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_ARGUMENT,
            "unsupported prototype conversion to GPU output",
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
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
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(value).await,
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_) => Ok(value.clone()),
        _ => Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_ARGUMENT,
            format!("unsupported prototype for 'like' ({value:?})"),
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn real_to_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(t) => {
            // Floating ComplexTensor storage is currently double-only, so this
            // output-template conversion is an explicit complex f64 boundary.
            let data: Vec<(f64, f64)> = t
                .materialize_f64()
                .into_iter()
                .map(|value| (value, 0.0))
                .collect();
            let tensor = ComplexTensor::new(data, t.shape.clone())
                .map_err(|e| builtin_error(format!("minus: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("minus: {e}")))?;
            real_to_complex(Value::Tensor(tensor)).await
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            real_to_complex(Value::Tensor(tensor)).await
        }
        Value::GpuTensor(handle) => {
            let gathered =
                gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
            real_to_complex(gathered).await
        }
        other => Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_INPUT,
            format!("cannot convert value {other:?} to complex output"),
        )),
    }
}

async fn minus_gpu_pair(lhs: GpuTensorHandle, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if lhs.shape == rhs.shape {
            if let Ok(handle) = provider.elem_sub(&lhs, &rhs).await {
                return Ok(Value::GpuTensor(handle));
            }
        }
        // Attempt N-D broadcast via repmat on device
        if let Some((out_shape, reps_l, reps_r)) = broadcast_reps(&lhs.shape, &rhs.shape) {
            let made_left = reps_l.iter().any(|&r| r != 1);
            let made_right = reps_r.iter().any(|&r| r != 1);
            let left_expanded = if made_left {
                provider
                    .repmat(&lhs, &reps_l)
                    .map_err(|e| builtin_error(format!("minus: {e}")))?
            } else {
                lhs.clone()
            };
            let right_expanded = if made_right {
                provider
                    .repmat(&rhs, &reps_r)
                    .map_err(|e| builtin_error(format!("minus: {e}")))?
            } else {
                rhs.clone()
            };
            let result = provider
                .elem_sub(&left_expanded, &right_expanded)
                .await
                .map_err(|e| builtin_error(format!("minus: {e}")));
            if made_left {
                let _ = provider.free(&left_expanded);
            }
            if made_right {
                let _ = provider.free(&right_expanded);
            }
            if let Ok(handle) = result {
                if handle.shape == out_shape {
                    return Ok(Value::GpuTensor(handle));
                } else {
                    let _ = provider.free(&handle);
                }
            }
        }
        if is_scalar_shape(&lhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&lhs).await? {
                if let Ok(handle) = provider.scalar_rsub(&rhs, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
        if is_scalar_shape(&rhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&rhs).await? {
                if let Ok(handle) = provider.scalar_sub(&lhs, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
    }
    let left = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs)).await?;
    let right = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs)).await?;
    minus_host(left, right)
}

fn broadcast_reps(a: &[usize], b: &[usize]) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rank = a.len().max(b.len()).max(1);
    let mut out = vec![1usize; rank];
    let mut aa = vec![1usize; rank];
    let mut bb = vec![1usize; rank];
    for i in 0..rank {
        aa[i] = *a.get(i).unwrap_or(&1);
        bb[i] = *b.get(i).unwrap_or(&1);
    }
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

async fn minus_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&rhs)? {
            if let Some(uploaded) =
                gpu_helpers::upload_exact_integer_scalar_like(provider, &lhs, scalar)
            {
                let result = provider.elem_sub(&lhs, &uploaded).await;
                let _ = provider.free(&uploaded);
                if let Ok(handle) = result {
                    return Ok(Value::GpuTensor(handle));
                }
            }
            if let Ok(handle) = provider.scalar_sub(&lhs, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs)).await?;
    minus_host(host_lhs, rhs)
}

async fn minus_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&lhs)? {
            if let Some(uploaded) =
                gpu_helpers::upload_exact_integer_scalar_like(provider, &rhs, scalar)
            {
                let result = provider.elem_sub(&uploaded, &rhs).await;
                let _ = provider.free(&uploaded);
                if let Ok(handle) = result {
                    return Ok(Value::GpuTensor(handle));
                }
            }
            if let Ok(handle) = provider.scalar_rsub(&rhs, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs)).await?;
    minus_host(lhs, host_rhs)
}

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
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

fn scalar_minus_value(lhs: &Value, rhs: &Value) -> Option<Value> {
    let left = scalar_complex_value(lhs).or_else(|| scalar_real_value(lhs).map(|v| (v, 0.0)))?;
    let right = scalar_complex_value(rhs).or_else(|| scalar_real_value(rhs).map(|v| (v, 0.0)))?;
    let (ar, ai) = left;
    let (br, bi) = right;
    if ai != 0.0 || bi != 0.0 {
        return Some(Value::Complex(ar - br, ai - bi));
    }
    Some(Value::Num(ar - br))
}

fn minus_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let Some(result) = symbolic_binary(&lhs, &rhs, SymbolicBinaryOp::Sub) {
        return Ok(result);
    }
    if let Some(result) =
        try_typed_sparse_integer_binary(&lhs, &rhs, SparseBinaryOp::Sub, BUILTIN_NAME)
    {
        return result;
    }
    if let Some(result) = try_sparse_binary(&lhs, &rhs, SparseBinaryOp::Sub, BUILTIN_NAME) {
        return result;
    }
    if (is_real_integer_operand(&lhs) && is_complex_operand(&rhs))
        || (is_complex_operand(&lhs) && is_real_integer_operand(&rhs))
    {
        return Err(builtin_error("complex integer arithmetic is not supported"));
    }
    if let Some(result) = try_integer_binary(&lhs, &rhs, IntegerBinaryOp::Subtract, BUILTIN_NAME)
        .map_err(builtin_error)?
    {
        return Ok(result);
    }
    if let Some(result) = scalar_minus_value(&lhs, &rhs) {
        return Ok(result);
    }
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (MinusOperand::Real(a), MinusOperand::Real(b)) => minus_real_real(a, b),
        (MinusOperand::Complex(a), MinusOperand::Complex(b)) => minus_complex_complex(&a, &b),
        (MinusOperand::Complex(a), MinusOperand::Real(b)) => minus_complex_real(&a, &b),
        (MinusOperand::Real(a), MinusOperand::Complex(b)) => minus_real_complex(&a, &b),
    }
}

fn is_real_integer_operand(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn is_complex_operand(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

enum MinusOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> BuiltinResult<MinusOperand> {
    match value {
        Value::Tensor(t) => Ok(MinusOperand::Real(t)),
        Value::Num(n) => Ok(MinusOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("minus: {e}")))?,
        )),
        Value::Int(i) => Ok(MinusOperand::Real(
            Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| builtin_error(format!("minus: {e}")))?,
        )),
        Value::Bool(b) => Ok(MinusOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("minus: {e}")))?,
        )),
        Value::LogicalArray(logical) => Ok(MinusOperand::Real(
            tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("minus: {e}")))?,
        )),
        Value::CharArray(chars) => Ok(MinusOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(MinusOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("minus: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(MinusOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(minus_error(&MINUS_ERROR_INTERNAL)),
        other => Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_INPUT,
            format!(
                "unsupported operand type {:?}; expected numeric or logical data",
                other
            ),
        )),
    }
}

fn minus_real_real(lhs: Tensor, rhs: Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| minus_error_with_detail(&MINUS_ERROR_SIZE_MISMATCH, &err))?;
    let output_shape = plan.output_shape().to_vec();
    let lhs = lhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    let rhs = rhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    let output = match (lhs, rhs) {
        (NumericStorage::F32(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] - rhs[rhs_index];
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f64; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] - rhs[rhs_index];
            }
            NumericStorage::F64(output)
        }
        (NumericStorage::F32(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (f64::from(lhs[lhs_index]) - rhs[rhs_index]) as f32;
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (lhs[lhs_index] - f64::from(rhs[rhs_index])) as f32;
            }
            NumericStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "minus: integer operands did not use the exact integer arithmetic path",
            ))
        }
    };
    let tensor = Tensor::from_numeric_storage(output, output_shape)
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn minus_complex_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| minus_error_with_detail(&MINUS_ERROR_SIZE_MISMATCH, &err))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("minus: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        out[out_idx] = (ar - br, ai - bi);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn minus_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| minus_error_with_detail(&MINUS_ERROR_SIZE_MISMATCH, &err))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("minus: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    // Floating ComplexTensor storage is currently double-only, so real
    // operands enter this explicitly floating complex-arithmetic boundary.
    let rhs_values = rhs.materialize_f64();
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let scalar = rhs_values[idx_rhs];
        out[out_idx] = (ar - scalar, ai);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn minus_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| minus_error_with_detail(&MINUS_ERROR_SIZE_MISMATCH, &err))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("minus: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    // Floating ComplexTensor storage is currently double-only, so real
    // operands enter this explicitly floating complex-arithmetic boundary.
    let lhs_values = lhs.materialize_f64();
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let scalar = lhs_values[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        out[out_idx] = (scalar - br, -bi);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| builtin_error(format!("minus: {e}")))
}

fn extract_scalar_f64(value: &Value) -> BuiltinResult<Option<f64>> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
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

fn is_scalar_shape(shape: &[usize]) -> bool {
    shape.iter().copied().product::<usize>() <= 1
}

async fn gpu_scalar_value(handle: &GpuTensorHandle) -> BuiltinResult<Option<f64>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if !is_scalar_shape(&handle.shape) {
        return Ok(None);
    }
    let host = download_handle_async(provider, handle)
        .await
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    if host.data.len() == 1 {
        Ok(Some(host.data[0]))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CharArray, ComplexTensor, IntegerStorage, LogicalArray, ResolveContext, SparseTensor,
        Tensor, Type,
    };

    const EPS: f64 = 1e-12;

    fn minus_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::minus_builtin(lhs, rhs, rest))
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

        let storage = runmat_builtins::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![8]),
            IntegerStorage::I16(vec![-3]),
        )
        .expect("complex integer storage");
        let mut complex = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex tensor");
        complex.data.clear();
        assert_eq!(
            scalar_complex_value(&Value::ComplexTensor(complex)),
            Some((8.0, -3.0))
        );
    }

    #[test]
    fn minus_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = MINUS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"C = minus(A, B)"));
        assert!(labels.contains(&"C = minus(A, B, \"like\", prototype)"));
    }

    #[test]
    fn minus_parser_error_has_stable_identifier() {
        let err = minus_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
            .expect_err("expected parser error");
        assert_eq!(err.identifier(), MINUS_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn minus_type_preserves_tensor_shape() {
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
    fn minus_type_scalar_returns_num() {
        let out = numeric_binary_type(&[Type::Num, Type::Int], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_scalar_numbers() {
        let result = minus_builtin(Value::Num(2.0), Value::Num(3.5), Vec::new()).expect("minus");
        match result {
            Value::Num(v) => assert!((v + 1.5).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_matrix_scalar() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            minus_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("minus");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(
                    t.as_f64_slice().expect("double result"),
                    &[-1.0, 0.0, 1.0, 2.0]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn minus_like_complex_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![-2, 3]), vec![1, 2]).unwrap();

        let result =
            block_on(super::real_to_complex(Value::Tensor(tensor))).expect("complex conversion");

        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![(-2.0, 0.0), (3.0, 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn minus_typed_sparse_int64_uses_exact_sparse_route() {
        let lhs = SparseTensor::new_integer(
            2,
            1,
            vec![0, 2],
            vec![0, 1],
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        )
        .unwrap();
        let rhs =
            SparseTensor::new_integer(2, 1, vec![0, 1], vec![0], IntegerStorage::I64(vec![1]))
                .unwrap();
        let Value::SparseTensor(result) = minus_builtin(
            Value::SparseTensor(lhs),
            Value::SparseTensor(rhs),
            Vec::new(),
        )
        .expect("minus") else {
            panic!("expected typed sparse result");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))
        );
    }

    #[test]
    fn minus_dense_integer_arrays_preserve_exact_storage() {
        let lhs = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1])
            .expect("lhs");
        let rhs = Tensor::new_integer(IntegerStorage::I64(vec![1, -7, i64::MIN]), vec![1, 3])
            .expect("rhs");

        let result = minus_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new())
            .expect("integer minus");
        let Value::Tensor(result) = result else {
            panic!("expected integer tensor");
        };
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::I64(vec![
                i64::MIN,
                i64::MAX - 1,
                i64::MIN + 7,
                i64::MAX,
                0,
                i64::MAX
            ]))
        );

        let scalar_tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).expect("scalar");
        assert_eq!(
            minus_builtin(Value::Tensor(scalar_tensor), Value::Num(1.0), Vec::new())
                .expect("scalar minus"),
            Value::Int(runmat_builtins::IntValue::U16(0))
        );
    }

    #[test]
    fn minus_float_arrays_preserve_native_single_class() {
        let lhs = Tensor::from_f32(vec![3.25, -4.0], vec![1, 2]).unwrap();
        let rhs = Tensor::from_f32(vec![2.0, 0.5], vec![1, 2]).unwrap();
        let Value::Tensor(result) =
            minus_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
        else {
            panic!("expected single tensor");
        };
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.25, -4.5])
        );

        let lhs = Tensor::new(vec![0.5, 0.2], vec![1, 2]).unwrap();
        let rhs = Tensor::from_f32(vec![0.2, 0.3], vec![1, 2]).unwrap();
        let Value::Tensor(result) =
            minus_builtin(Value::Tensor(lhs), Value::Tensor(rhs), Vec::new()).unwrap()
        else {
            panic!("expected mixed floating tensor");
        };
        assert_eq!(
            result.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![
                (0.5_f64 - f64::from(0.2_f32)) as f32,
                (0.2_f64 - f64::from(0.3_f32)) as f32,
            ])
        );
    }

    #[test]
    fn minus_rejects_real_integer_with_floating_complex() {
        let integer = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![(1_u64 << 63) + 1, u64::MAX]),
                vec![1, 2],
            )
            .unwrap(),
        );
        let complex = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap(),
        );
        for (lhs, rhs) in [
            (integer.clone(), complex.clone()),
            (complex.clone(), integer.clone()),
        ] {
            let error = minus_builtin(lhs, rhs, Vec::new()).unwrap_err();
            assert!(error
                .message()
                .contains("complex integer arithmetic is not supported"));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_row_column_broadcast() {
        let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
        let result = minus_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
            .expect("broadcast minus");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = vec![
                    -9.0, -8.0, -7.0, // column-first order
                    -19.0, -18.0, -17.0, -29.0, -28.0, -27.0,
                ];
                assert_eq!(t.as_f64_slice().expect("double result"), expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_complex_inputs() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
        let result = minus_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex minus");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(-1.0, 3.0), (4.0, -5.0)];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS && (got.1 - exp.1).abs() < EPS);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_char_input() {
        let chars = CharArray::new("DEF".chars().collect(), 1, 3).unwrap();
        let result = minus_builtin(Value::CharArray(chars), Value::Num(1.0), Vec::new())
            .expect("char minus");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(
                    t.as_f64_slice().expect("double result"),
                    &[67.0, 68.0, 69.0]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_logical_input_promotes_to_double() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![2.0, 2.0, 3.0, 3.0], vec![2, 2]).unwrap();
        let result = minus_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(
                    t.as_f64_slice().expect("double result"),
                    &[-1.0, -2.0, -2.0, -3.0]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_dimension_mismatch_errors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = minus_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).unwrap_err();
        assert!(
            err.message().contains("minus"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let ha = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let hb = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = minus_builtin(
                Value::GpuTensor(ha.clone()),
                Value::GpuTensor(hb.clone()),
                Vec::new(),
            )
            .expect("gpu minus");
            let gathered = test_support::gather(result).expect("gather");
            let expected = vec![0.0; tensor.len()];
            assert_eq!(gathered.as_f64_slice().expect("double result"), expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_gpu_scalar_right() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = minus_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
                .expect("gpu scalar minus");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(
                gathered.as_f64_slice().expect("double result"),
                &[-1.0, 0.0, 1.0]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_gpu_scalar_left() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let result = minus_builtin(Value::Num(3.0), Value::GpuTensor(handle), Vec::new())
                .expect("gpu scalar minus");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(
                gathered.as_f64_slice().expect("double result"),
                &[1.0, -1.0]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_like_gpu_prototype_keeps_residency() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = minus_builtin(
                Value::Tensor(lhs.clone()),
                Value::Tensor(rhs.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("minus like gpu");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 1]);
                    assert_eq!(
                        gathered.as_f64_slice().expect("double result"),
                        &[7.0, 16.0]
                    );
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_like_gpu_prototype_uploads_typed_integer_storage_exactly() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new_integer(IntegerStorage::I16(vec![10, 20]), vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");

            let result = minus_builtin(
                Value::Tensor(lhs),
                Value::Num(3.0),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("minus like gpu");

            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::I16(vec![7, 17]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_like_host_gathers_gpu_value() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
            let ha = gpu_helpers::upload_tensor(provider, &lhs).expect("upload lhs");
            let hb = gpu_helpers::upload_tensor(provider, &rhs).expect("upload rhs");
            let result = minus_builtin(
                Value::GpuTensor(ha),
                Value::GpuTensor(hb),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("minus like host");
            let Value::Tensor(t) = result else {
                panic!("expected tensor result after host gather");
            };
            assert_eq!(t.shape, vec![2, 1]);
            assert_eq!(t.as_f64_slice().expect("double result"), &[5.0, 14.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_like_complex_prototype_yields_complex() {
        let lhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let result = minus_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("minus like complex");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let expected = [(-2.0, 0.0), (-2.0, 0.0)];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS && (got.1 - exp.1).abs() < EPS);
                }
            }
            Value::Complex(re, im) => {
                assert!((re + 2.0).abs() < EPS && im.abs() < EPS);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_like_missing_prototype_errors() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = minus_builtin(
            Value::Tensor(lhs),
            Value::Num(1.0),
            vec![Value::from("like")],
        )
        .expect_err("expected error");
        assert!(err.message().contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_like_keyword_case_insensitive() {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = minus_builtin(
            Value::Tensor(tensor.clone()),
            Value::Num(1.0),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("minus like upper");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.as_f64_slice().expect("double result"), &[-1.0, 0.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = minus_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("minus like char");
        match result {
            Value::Num(v) => assert!((v + 1.0).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn minus_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = minus_host(Value::Tensor(t.clone()), Value::Tensor(t.clone())).unwrap();
        let h = gpu_helpers::upload_tensor(runmat_accelerate_api::provider().unwrap(), &t).unwrap();
        let gpu = block_on(minus_gpu_pair(h.clone(), h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                for (a, b) in gathered
                    .as_f64_slice()
                    .expect("gathered double")
                    .iter()
                    .zip(ct.as_f64_slice().expect("CPU double"))
                {
                    assert!((a - b).abs() < EPS);
                }
            }
            other => panic!("unexpected shapes {other:?}"),
        }
    }
}
