//! MATLAB-compatible `plus` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::symbolic::{symbolic_binary, SymbolicBinaryOp};
use crate::builtins::math::type_resolvers::numeric_binary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::plus")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "plus",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_add",
            commutative: true,
        },
        ProviderHook::Custom("scalar_add"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses elem_add for shape-compatible gpuArrays, including complex-interleaved handles, attempts provider-side implicit expansion with repmat, and uses scalar_add when one operand is a real scalar; falls back to host execution for unsupported operand kinds.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::plus")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "plus",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let lhs = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let rhs = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("({lhs} + {rhs})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion emits a plain sum; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "plus";

const PLUS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise sum result.",
}];

const PLUS_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
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

const PLUS_INPUTS_A_B_LIKE: [BuiltinParamDescriptor; 4] = [
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

const PLUS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = plus(A, B)",
        inputs: &PLUS_INPUTS_A_B,
        outputs: &PLUS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = plus(A, B, \"like\", prototype)",
        inputs: &PLUS_INPUTS_A_B_LIKE,
        outputs: &PLUS_OUTPUT,
    },
];

const PLUS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.INVALID_ARGUMENT",
    identifier: Some("RunMat:plus:InvalidArgument"),
    when: "Optional arguments are malformed or unsupported.",
    message: "plus: invalid argument",
};

const PLUS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.INVALID_INPUT",
    identifier: Some("RunMat:plus:InvalidInput"),
    when: "Operands or prototypes cannot be converted into supported numeric/logical forms.",
    message: "plus: invalid input",
};

const PLUS_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.SIZE_MISMATCH",
    identifier: Some("RunMat:plus:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "plus: array sizes are not compatible for broadcasting",
};

const PLUS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLUS.INTERNAL",
    identifier: Some("RunMat:plus:Internal"),
    when: "Provider interaction, gather/upload, or internal tensor construction failed.",
    message: "plus: internal error",
};

const PLUS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    PLUS_ERROR_INVALID_ARGUMENT,
    PLUS_ERROR_INVALID_INPUT,
    PLUS_ERROR_SIZE_MISMATCH,
    PLUS_ERROR_INTERNAL,
];

pub const PLUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PLUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PLUS_ERRORS,
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn plus_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn plus_error_with_detail(
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
    name = "plus",
    category = "math/elementwise",
    summary = "Compute element-wise addition.",
    keywords = "plus,element-wise addition,gpu,+",
    accel = "elementwise",
    type_resolver(numeric_binary_type),
    descriptor(crate::builtins::math::elementwise::plus::PLUS_DESCRIPTOR),
    builtin_path = "crate::builtins::math::elementwise::plus"
)]
async fn plus_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    let base = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => plus_gpu_pair(la, lb).await,
        (Value::GpuTensor(la), rhs) => plus_gpu_host_left(la, rhs).await,
        (lhs, Value::GpuTensor(rb)) => plus_gpu_host_right(lhs, rb).await,
        (lhs, rhs) => plus_host(lhs, rhs),
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
            return Err(plus_error_with_detail(
                &PLUS_ERROR_INVALID_ARGUMENT,
                "expected prototype after 'like'",
            ));
        }
        return Err(plus_error_with_detail(
            &PLUS_ERROR_INVALID_ARGUMENT,
            "unsupported option; only 'like' is accepted",
        ));
    }
    if args.len() == 2 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Ok(OutputTemplate::Like(args[1].clone()));
        }
        return Err(plus_error_with_detail(
            &PLUS_ERROR_INVALID_ARGUMENT,
            "unsupported option; only 'like' is accepted",
        ));
    }
    Err(plus_error_with_detail(
        &PLUS_ERROR_INVALID_ARGUMENT,
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
        return Err(plus_error_with_detail(
            &PLUS_ERROR_INVALID_ARGUMENT,
            "GPU output requested via 'like' but no acceleration provider is active",
        ));
    };
    match value {
        Value::GpuTensor(handle) => Ok(gpu_helpers::resident_gpu_value(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| builtin_error(format!("plus: failed to upload GPU result: {e}")))?;
            Ok(gpu_helpers::resident_gpu_value(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("plus: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("plus: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(plus_error_with_detail(
            &PLUS_ERROR_INVALID_ARGUMENT,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        Value::String(_)
        | Value::StringArray(_)
        | Value::SparseTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::Symbolic(_) => Err(plus_error_with_detail(
            &PLUS_ERROR_INVALID_ARGUMENT,
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
        | Value::OutputList(_) => Err(plus_error_with_detail(
            &PLUS_ERROR_INVALID_ARGUMENT,
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
        _ => Err(plus_error_with_detail(
            &PLUS_ERROR_INVALID_ARGUMENT,
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
            let data: Vec<(f64, f64)> = t.data.iter().map(|&v| (v, 0.0)).collect();
            let tensor = ComplexTensor::new(data, t.shape.clone())
                .map_err(|e| builtin_error(format!("plus: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("plus: {e}")))?;
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
        other => Err(plus_error_with_detail(
            &PLUS_ERROR_INVALID_INPUT,
            format!("cannot convert value {other:?} to complex output"),
        )),
    }
}

async fn plus_gpu_pair(lhs: GpuTensorHandle, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if lhs.shape == rhs.shape {
            if let Ok(handle) = provider.elem_add(&lhs, &rhs).await {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
        // Attempt N-D broadcast via repmat to keep computation on device
        if let Some((out_shape, reps_l, reps_r)) = broadcast_reps(&lhs.shape, &rhs.shape) {
            let made_left = reps_l.iter().any(|&r| r != 1);
            let made_right = reps_r.iter().any(|&r| r != 1);
            let left_expanded = if made_left {
                provider
                    .repmat(&lhs, &reps_l)
                    .map_err(|e| builtin_error(format!("plus: {e}")))?
            } else {
                lhs.clone()
            };
            let right_expanded = if made_right {
                provider
                    .repmat(&rhs, &reps_r)
                    .map_err(|e| builtin_error(format!("plus: {e}")))?
            } else {
                rhs.clone()
            };
            let result = provider
                .elem_add(&left_expanded, &right_expanded)
                .await
                .map_err(|e| builtin_error(format!("plus: {e}")));
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
                if let Ok(handle) = provider.scalar_add(&rhs, scalar) {
                    return Ok(gpu_helpers::resident_gpu_value(handle));
                }
            }
        }
        if is_scalar_shape(&rhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&rhs).await? {
                if let Ok(handle) = provider.scalar_add(&lhs, scalar) {
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
    plus_host(left, right)
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

async fn plus_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&rhs)? {
            if let Ok(handle) = provider.scalar_add(&lhs, scalar) {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
    }
    let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    plus_host(host_lhs, rhs)
}

async fn plus_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&lhs)? {
            if let Ok(handle) = provider.scalar_add(&rhs, scalar) {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
    }
    let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    plus_host(lhs, host_rhs)
}

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => t.data.first().copied(),
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
        Value::ComplexTensor(ct) if ct.data.len() == 1 => ct.data.first().copied(),
        _ => None,
    }
}

fn scalar_plus_value(lhs: &Value, rhs: &Value) -> Option<Value> {
    let left = scalar_complex_value(lhs).or_else(|| scalar_real_value(lhs).map(|v| (v, 0.0)))?;
    let right = scalar_complex_value(rhs).or_else(|| scalar_real_value(rhs).map(|v| (v, 0.0)))?;
    let (ar, ai) = left;
    let (br, bi) = right;
    if ai != 0.0 || bi != 0.0 {
        return Some(Value::Complex(ar + br, ai + bi));
    }
    Some(Value::Num(ar + br))
}

fn plus_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let Some(result) = symbolic_binary(&lhs, &rhs, SymbolicBinaryOp::Add) {
        return Ok(result);
    }
    if let Some(result) = scalar_plus_value(&lhs, &rhs) {
        return Ok(result);
    }
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (PlusOperand::Real(a), PlusOperand::Real(b)) => plus_real_real(&a, &b),
        (PlusOperand::Complex(a), PlusOperand::Complex(b)) => plus_complex_complex(&a, &b),
        (PlusOperand::Complex(a), PlusOperand::Real(b)) => plus_complex_real(&a, &b),
        (PlusOperand::Real(a), PlusOperand::Complex(b)) => plus_real_complex(&a, &b),
    }
}

fn plus_real_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| plus_error_with_detail(&PLUS_ERROR_SIZE_MISMATCH, &err))?;
    if plan.is_empty() {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("plus: {e}")))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut out = vec![0.0f64; plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        out[out_idx] = lhs.data[idx_lhs] + rhs.data[idx_rhs];
    }
    let tensor = Tensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("plus: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn plus_complex_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| plus_error_with_detail(&PLUS_ERROR_SIZE_MISMATCH, &err))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("plus: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        out[out_idx] = (ar + br, ai + bi);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("plus: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn plus_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| plus_error_with_detail(&PLUS_ERROR_SIZE_MISMATCH, &err))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("plus: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let scalar = rhs.data[idx_rhs];
        out[out_idx] = (ar + scalar, ai);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("plus: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn plus_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| plus_error_with_detail(&PLUS_ERROR_SIZE_MISMATCH, &err))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("plus: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let scalar = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        out[out_idx] = (scalar + br, bi);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("plus: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

enum PlusOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> BuiltinResult<PlusOperand> {
    match value {
        Value::Tensor(t) => Ok(PlusOperand::Real(t)),
        Value::Num(n) => Ok(PlusOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("plus: {e}")))?,
        )),
        Value::Int(i) => Ok(PlusOperand::Real(
            Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| builtin_error(format!("plus: {e}")))?,
        )),
        Value::Bool(b) => Ok(PlusOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("plus: {e}")))?,
        )),
        Value::LogicalArray(logical) => Ok(PlusOperand::Real(
            tensor::logical_to_tensor(&logical).map_err(|e| builtin_error(format!("plus: {e}")))?,
        )),
        Value::CharArray(chars) => Ok(PlusOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(PlusOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("plus: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(PlusOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(plus_error(&PLUS_ERROR_INTERNAL)),
        other => Err(plus_error_with_detail(
            &PLUS_ERROR_INVALID_INPUT,
            format!(
                "unsupported operand type {:?}; expected numeric or logical data",
                other
            ),
        )),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|e| builtin_error(format!("plus: {e}")))
}

fn extract_scalar_f64(value: &Value) -> BuiltinResult<Option<f64>> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(t) if t.data.len() == 1 => Ok(Some(t.data[0])),
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
    if !is_scalar_shape(&handle.shape) {
        return Ok(None);
    }
    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
    Ok(tensor.data.first().copied())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    #[cfg(feature = "wgpu")]
    fn register_wgpu_provider_available() -> bool {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_ok()
            && runmat_accelerate_api::provider().is_some()
    }
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CharArray, ComplexTensor, IntValue, LogicalArray, ResolveContext, Tensor, Type,
    };

    const EPS: f64 = 1e-12;

    fn plus_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::plus_builtin(lhs, rhs, rest))
    }

    #[test]
    fn plus_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = PLUS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"C = plus(A, B)"));
        assert!(labels.contains(&"C = plus(A, B, \"like\", prototype)"));
    }

    #[test]
    fn plus_parser_error_has_stable_identifier() {
        let err = plus_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
            .expect_err("expected parser error");
        assert_eq!(err.identifier(), PLUS_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn plus_type_preserves_tensor_shape() {
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
    fn plus_type_scalar_returns_num() {
        let out = numeric_binary_type(&[Type::Num, Type::Int], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_scalar_numbers() {
        let result = plus_builtin(Value::Num(2.0), Value::Num(3.5), Vec::new()).expect("plus");
        match result {
            Value::Num(v) => assert!((v - 5.5).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_matrix_scalar() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            plus_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("plus");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_row_column_broadcast() {
        let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap();
        let result = plus_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
            .expect("broadcast plus");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0];
                assert_eq!(t.data, expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_complex_inputs() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
        let result = plus_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex plus");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(3.0, 1.0), (2.0, -3.0)];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS && (got.1 - exp.1).abs() < EPS);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_char_input() {
        let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
        let result =
            plus_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new()).expect("char plus");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![67.0, 68.0, 69.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_logical_input_promotes_to_double() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![2.0, 2.0, 3.0, 3.0], vec![2, 2]).unwrap();
        let result = plus_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![3.0, 2.0, 4.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_dimension_mismatch_errors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = plus_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).unwrap_err();
        assert!(
            err.message().contains("plus"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let ha = provider.upload(&view).expect("upload");
            let hb = provider.upload(&view).expect("upload");
            let result = plus_builtin(
                Value::GpuTensor(ha.clone()),
                Value::GpuTensor(hb.clone()),
                Vec::new(),
            )
            .expect("gpu plus");
            let gathered = test_support::gather(result).expect("gather");
            let expected = tensor
                .data
                .iter()
                .zip(tensor.data.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>();
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_gpu_scalar_right() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = plus_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
                .expect("gpu scalar plus");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![3.0, 4.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_gpu_scalar_left() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = plus_builtin(Value::Num(3.0), Value::GpuTensor(handle), Vec::new())
                .expect("gpu scalar plus");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_like_gpu_prototype_keeps_residency() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = plus_builtin(
                Value::Tensor(lhs.clone()),
                Value::Tensor(rhs.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("plus like gpu");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 1]);
                    assert_eq!(gathered.data, vec![4.0, 6.0]);
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_like_host_gathers_gpu_value() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![5.0, 6.0], vec![2, 1]).unwrap();
            let view_l = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let view_r = HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            };
            let ha = provider.upload(&view_l).expect("upload lhs");
            let hb = provider.upload(&view_r).expect("upload rhs");
            let result = plus_builtin(
                Value::GpuTensor(ha),
                Value::GpuTensor(hb),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("plus like host");
            let Value::Tensor(t) = result else {
                panic!("expected tensor result after host gather");
            };
            assert_eq!(t.shape, vec![2, 1]);
            assert_eq!(t.data, vec![6.0, 8.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_like_complex_prototype_yields_complex() {
        let lhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let result = plus_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("plus like complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                let expected = [(6.0, 0.0), (8.0, 0.0)];
                for (got, exp) in ct.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS);
                    assert!((got.1 - exp.1).abs() < EPS);
                }
            }
            Value::Complex(re, im) => {
                assert!((re - 6.0).abs() < EPS && im.abs() < EPS);
            }
            other => panic!("expected complex output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_like_missing_prototype_errors() {
        let lhs = Value::Num(2.0);
        let rhs = Value::Num(4.0);
        let err = plus_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
        assert!(
            err.message().contains("prototype"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_like_keyword_char_array() {
        test_support::with_test_provider(|provider| {
            let keyword = CharArray::new_row("LIKE");
            let lhs = Value::Num(2.0);
            let rhs = Value::Num(5.0);
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = plus_builtin(
                lhs,
                rhs,
                vec![Value::CharArray(keyword), Value::GpuTensor(proto)],
            )
            .expect("plus like char");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.data, vec![7.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn plus_wgpu_matches_cpu_elementwise() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let cpu = plus_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
        let view_l = HostTensorView {
            data: &lhs.data,
            shape: &lhs.shape,
        };
        let view_r = HostTensorView {
            data: &rhs.data,
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
        let gpu = block_on(plus_gpu_pair(ha, hb)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(t) => assert_eq!(gathered.data, t.data),
            Value::Num(n) => assert_eq!(gathered.data, vec![n]),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn plus_wgpu_complex_gpu_stays_resident() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("provider");
        let shape = [2, 1];
        let real = provider
            .upload(&HostTensorView {
                data: &[1.0, -2.0],
                shape: &shape,
            })
            .expect("upload real");
        let imag = provider
            .upload(&HostTensorView {
                data: &[0.5, 4.0],
                shape: &shape,
            })
            .expect("upload imag");
        let complex = block_on(provider.complex_from_real_imag(&real, &imag))
            .expect("complex_from_real_imag");
        let offset = provider
            .upload(&HostTensorView {
                data: &[3.0, 7.0],
                shape: &shape,
            })
            .expect("upload offset");

        let result = plus_builtin(
            Value::GpuTensor(complex),
            Value::GpuTensor(offset),
            Vec::new(),
        )
        .expect("plus complex gpu");
        let handle = match result {
            Value::GpuTensor(handle) => handle,
            other => panic!("expected resident GPU result, got {other:?}"),
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(handle),
        ))
        .expect("gather complex gpu");
        match gathered {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                assert_eq!(ct.data, vec![(4.0, 0.5), (5.0, 4.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn plus_wgpu_complex_scalar_implicit_expansion_stays_resident() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("provider");
        let scalar_shape = [1, 1];
        let real = provider
            .upload(&HostTensorView {
                data: &[2.0],
                shape: &scalar_shape,
            })
            .expect("upload real");
        let imag = provider
            .upload(&HostTensorView {
                data: &[-3.0],
                shape: &scalar_shape,
            })
            .expect("upload imag");
        let complex_scalar =
            block_on(provider.complex_from_real_imag(&real, &imag)).expect("complex scalar");
        let vector_shape = [3, 1];
        let vector = provider
            .upload(&HostTensorView {
                data: &[10.0, 20.0, 30.0],
                shape: &vector_shape,
            })
            .expect("upload vector");

        let result = plus_builtin(
            Value::GpuTensor(complex_scalar),
            Value::GpuTensor(vector),
            Vec::new(),
        )
        .expect("plus implicit expansion");
        let handle = match result {
            Value::GpuTensor(handle) => handle,
            other => panic!("expected resident GPU result, got {other:?}"),
        };
        assert_eq!(handle.shape, vec![3, 1]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(handle),
        ))
        .expect("gather complex result");
        match gathered {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![3, 1]);
                assert_eq!(ct.data, vec![(12.0, -3.0), (22.0, -3.0), (32.0, -3.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn plus_wgpu_complex_gpu_host_complex_scalar_falls_back() {
        let _guard = test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("provider");
        let shape = [2, 1];
        let real = provider
            .upload(&HostTensorView {
                data: &[1.0, -2.0],
                shape: &shape,
            })
            .expect("upload real");
        let imag = provider
            .upload(&HostTensorView {
                data: &[0.5, 4.0],
                shape: &shape,
            })
            .expect("upload imag");
        let complex = block_on(provider.complex_from_real_imag(&real, &imag))
            .expect("complex_from_real_imag");

        let result = plus_builtin(
            Value::GpuTensor(complex),
            Value::Complex(10.0, -1.0),
            Vec::new(),
        )
        .expect("plus host complex scalar");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                assert_eq!(ct.data, vec![(11.0, -0.5), (8.0, 3.0)]);
            }
            other => panic!("expected host complex fallback, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn plus_int_inputs_promote_to_double() {
        let lhs = Value::Int(IntValue::I32(3));
        let rhs = Value::Int(IntValue::I32(5));
        let result = plus_builtin(lhs, rhs, Vec::new()).expect("plus");
        match result {
            Value::Num(v) => assert_eq!(v, 8.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }
}
