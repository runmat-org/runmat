//! MATLAB-compatible `ldivide` builtin with GPU-aware semantics for RunMat.

use async_recursion::async_recursion;
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::ldivide")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ldivide",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_div",
            commutative: false,
        },
        ProviderHook::Custom("scalar_div"),
        ProviderHook::Custom("scalar_rdiv"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses elem_div for B./A when shapes match, scalar_div for tensor ./ scalar cases (B ./ a), and scalar_rdiv for scalar ./ tensor cases (b ./ A); implicit expansion or unsupported operand kinds fall back to the CPU before 'like' prototypes are honoured.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::ldivide")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ldivide",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let divisor = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let numerator = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("({numerator} / {divisor})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a plain quotient; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "ldivide";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "ldivide",
    category = "math/elementwise",
    summary = "Element-wise left division (B ./ A) with MATLAB-compatible implicit expansion.",
    keywords = "ldivide,element-wise left division,gpu,.\\",
    accel = "elementwise",
    builtin_path = "crate::builtins::math::elementwise::ldivide"
)]
async fn ldivide_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    let base = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => ldivide_gpu_pair(la, lb).await,
        (Value::GpuTensor(la), rhs) => ldivide_gpu_host_left(la, rhs).await,
        (lhs, Value::GpuTensor(rb)) => ldivide_gpu_host_right(lhs, rb).await,
        (lhs, rhs) => Ok(ldivide_host(lhs, rhs)?),
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
            return Err(builtin_error("ldivide: expected prototype after 'like'"));
        }
        return Err(builtin_error(
            "ldivide: unsupported option; only 'like' is accepted",
        ));
    }
    if args.len() == 2 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Ok(OutputTemplate::Like(args[1].clone()));
        }
        return Err(builtin_error(
            "ldivide: unsupported option; only 'like' is accepted",
        ));
    }
    Err(builtin_error("ldivide: too many input arguments"))
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
            .map_err(|e| builtin_error(format!("ldivide: {e}")))
    } else {
        Ok(value)
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(builtin_error(
            "ldivide: GPU output requested via 'like' but no acceleration provider is active",
        ));
    };
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| builtin_error(format!("ldivide: failed to upload GPU result: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(builtin_error(
            "ldivide: GPU prototypes for 'like' only support real numeric outputs",
        )),
        Value::String(_) | Value::StringArray(_) | Value::Cell(_) | Value::Struct(_) => Err(
            builtin_error("ldivide: unsupported prototype conversion to GPU output"),
        ),
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(builtin_error(
            "ldivide: unsupported prototype conversion to GPU output",
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
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))
            .map_err(|e| builtin_error(format!("ldivide: {e}"))),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_) => Ok(value.clone()),
        _ => Err(builtin_error(format!(
            "ldivide: unsupported prototype for 'like' ({value:?})"
        ))),
    }
}

#[async_recursion(?Send)]
async fn real_to_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(t) => {
            let data: Vec<(f64, f64)> = t.data.iter().map(|&v| (v, 0.0)).collect();
            let tensor = ComplexTensor::new(data, t.shape.clone())
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
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
        other => Err(builtin_error(format!(
            "ldivide: cannot convert value {other:?} to complex output"
        ))),
    }
}

async fn ldivide_gpu_pair(
    divisor: GpuTensorHandle,
    numerator: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if divisor.shape == numerator.shape {
            if let Ok(handle) = provider.elem_div(&numerator, &divisor).await {
                return Ok(Value::GpuTensor(handle));
            }
        }
        // Try N-D broadcast on device using repmat + elem_div (B ./ A)
        if let Some((out_shape, reps_num, reps_div)) =
            broadcast_reps(&numerator.shape, &divisor.shape)
        {
            let made_num = reps_num.iter().any(|&r| r != 1);
            let made_div = reps_div.iter().any(|&r| r != 1);
            let num_expanded = if made_num {
                provider
                    .repmat(&numerator, &reps_num)
                    .map_err(|e| builtin_error(format!("ldivide: {e}")))?
            } else {
                numerator.clone()
            };
            let div_expanded = if made_div {
                provider
                    .repmat(&divisor, &reps_div)
                    .map_err(|e| builtin_error(format!("ldivide: {e}")))?
            } else {
                divisor.clone()
            };
            let result = provider
                .elem_div(&num_expanded, &div_expanded)
                .await
                .map_err(|e| builtin_error(format!("ldivide: {e}")));
            if made_num {
                let _ = provider.free(&num_expanded);
            }
            if made_div {
                let _ = provider.free(&div_expanded);
            }
            if let Ok(handle) = result {
                if handle.shape == out_shape {
                    return Ok(Value::GpuTensor(handle));
                } else {
                    let _ = provider.free(&handle);
                }
            }
        }
        if is_scalar_shape(&divisor.shape) {
            if let Some(scalar) = gpu_scalar_value(&divisor).await? {
                if let Ok(handle) = provider.scalar_div(&numerator, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
        if is_scalar_shape(&numerator.shape) {
            if let Some(scalar) = gpu_scalar_value(&numerator).await? {
                if let Ok(handle) = provider.scalar_rdiv(&divisor, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
    }
    let divisor_host = gpu_helpers::gather_tensor_async(&divisor)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let numerator_host = gpu_helpers::gather_tensor_async(&numerator)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    ldivide_host(Value::Tensor(divisor_host), Value::Tensor(numerator_host))
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

async fn ldivide_gpu_host_left(divisor: GpuTensorHandle, numerator: Value) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&numerator)? {
            if let Ok(handle) = provider.scalar_rdiv(&divisor, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let divisor_host = gpu_helpers::gather_tensor_async(&divisor)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    ldivide_host(Value::Tensor(divisor_host), numerator)
}

async fn ldivide_gpu_host_right(
    divisor: Value,
    numerator: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&divisor)? {
            if let Ok(handle) = provider.scalar_div(&numerator, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let numerator_host = gpu_helpers::gather_tensor_async(&numerator)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    ldivide_host(divisor, Value::Tensor(numerator_host))
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

fn scalar_ldivide_value(divisor: &Value, numerator: &Value) -> Option<Value> {
    let num = scalar_complex_value(numerator)
        .or_else(|| scalar_real_value(numerator).map(|v| (v, 0.0)))?;
    let div =
        scalar_complex_value(divisor).or_else(|| scalar_real_value(divisor).map(|v| (v, 0.0)))?;
    let (nr, ni) = num;
    let (dr, di) = div;
    if ni != 0.0 || di != 0.0 {
        let quotient = Complex64::new(nr, ni) / Complex64::new(dr, di);
        return Some(Value::Complex(quotient.re, quotient.im));
    }
    Some(Value::Num(nr / dr))
}

fn ldivide_host(divisor: Value, numerator: Value) -> BuiltinResult<Value> {
    if let Some(result) = scalar_ldivide_value(&divisor, &numerator) {
        return Ok(result);
    }
    match (classify_operand(divisor)?, classify_operand(numerator)?) {
        (LdivideOperand::Real(div), LdivideOperand::Real(num)) => ldivide_real_real(&div, &num),
        (LdivideOperand::Complex(div), LdivideOperand::Complex(num)) => {
            ldivide_complex_complex(&div, &num)
        }
        (LdivideOperand::Complex(div), LdivideOperand::Real(num)) => {
            ldivide_complex_real(&div, &num)
        }
        (LdivideOperand::Real(div), LdivideOperand::Complex(num)) => {
            ldivide_real_complex(&div, &num)
        }
    }
}

fn ldivide_real_real(divisor: &Tensor, numerator: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| builtin_error(format!("ldivide: {err}")))?;
    if plan.is_empty() {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut out = vec![0.0f64; plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        out[out_idx] = numerator.data[idx_lhs] / divisor.data[idx_rhs];
    }
    let tensor = Tensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ldivide_complex_complex(
    divisor: &ComplexTensor,
    numerator: &ComplexTensor,
) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| builtin_error(format!("ldivide: {err}")))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (nr, ni) = numerator.data[idx_lhs];
        let (dr, di) = divisor.data[idx_rhs];
        let quotient = Complex64::new(nr, ni) / Complex64::new(dr, di);
        out[out_idx] = (quotient.re, quotient.im);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn ldivide_complex_real(divisor: &ComplexTensor, numerator: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| builtin_error(format!("ldivide: {err}")))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let scalar = numerator.data[idx_lhs];
        let (dr, di) = divisor.data[idx_rhs];
        let quotient = Complex64::new(scalar, 0.0) / Complex64::new(dr, di);
        out[out_idx] = (quotient.re, quotient.im);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn ldivide_real_complex(divisor: &Tensor, numerator: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| builtin_error(format!("ldivide: {err}")))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (nr, ni) = numerator.data[idx_lhs];
        let scalar = divisor.data[idx_rhs];
        let quotient = Complex64::new(nr, ni) / Complex64::new(scalar, 0.0);
        out[out_idx] = (quotient.re, quotient.im);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

enum LdivideOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> BuiltinResult<LdivideOperand> {
    match value {
        Value::Tensor(t) => Ok(LdivideOperand::Real(t)),
        Value::Num(n) => Ok(LdivideOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("ldivide: {e}")))?,
        )),
        Value::Int(i) => Ok(LdivideOperand::Real(
            Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?,
        )),
        Value::Bool(b) => Ok(LdivideOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?,
        )),
        Value::LogicalArray(logical) => Ok(LdivideOperand::Real(
            tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?,
        )),
        Value::CharArray(chars) => Ok(LdivideOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(LdivideOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("ldivide: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(LdivideOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(builtin_error(
            "ldivide: internal error converting GPU value",
        )),
        other => Err(builtin_error(format!(
            "ldivide: unsupported operand type {:?}; expected numeric or logical data",
            other
        ))),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| builtin_error(format!("ldivide: {e}")))
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

async fn gpu_scalar_value(handle: &GpuTensorHandle) -> BuiltinResult<Option<f64>> {
    if !is_scalar_shape(&handle.shape) {
        return Ok(None);
    }
    let tensor = gpu_helpers::gather_tensor_async(handle)
        .await
        .map_err(|e| builtin_error(format!("ldivide: {e}")))?;
    Ok(tensor.data.first().copied())
}

fn is_scalar_shape(shape: &[usize]) -> bool {
    shape.iter().copied().product::<usize>() <= 1
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, Tensor};

    const EPS: f64 = 1e-12;
    const GPU_EPS: f64 = 1e-6;

    fn ldivide_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ldivide_builtin(lhs, rhs, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_scalar_numbers() {
        let result =
            ldivide_builtin(Value::Num(7.0), Value::Num(2.0), Vec::new()).expect("ldivide");
        match result {
            Value::Num(v) => assert!((v - (2.0 / 7.0)).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_matrix_scalar() {
        let tensor = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result =
            ldivide_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [1.0, 0.5, 0.3333333333333333, 0.25];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_row_column_broadcast() {
        let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0, 40.0], vec![1, 3]).unwrap();
        let result = ldivide_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
            .expect("broadcast ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = [
                    10.0,
                    5.0,
                    3.3333333333333335,
                    20.0,
                    10.0,
                    6.666666666666667,
                    40.0,
                    20.0,
                    13.333333333333334,
                ];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_complex_inputs() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
        let result = ldivide_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex ldivide");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(0.0, -1.0), (-0.28, -0.04)];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < 1e-10 && (got.1 - exp.1).abs() < 1e-10);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_division_by_zero() {
        let tensor = Tensor::new(vec![0.0, 1.0, -2.0], vec![3, 1]).unwrap();
        let result =
            ldivide_builtin(Value::Tensor(tensor), Value::Num(0.0), Vec::new()).expect("ldivide");
        match result {
            Value::Tensor(t) => {
                assert!(t.data[0].is_nan());
                assert_eq!(t.data[1], 0.0);
                assert_eq!(t.data[2], -0.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_logical_inputs_promote() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![2, 2]).unwrap();
        let result = ldivide_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("logical ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [1.0, f64::INFINITY, 4.0, 8.0];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    if exp.is_infinite() {
                        assert!(got.is_infinite());
                    } else {
                        assert!((got - exp).abs() < EPS);
                    }
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_char_array_promotes_to_double() {
        let chars = CharArray::new_row("AB");
        let result =
            ldivide_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new()).expect("ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - (2.0 / 65.0)).abs() < EPS);
                assert!((t.data[1] - (2.0 / 66.0)).abs() < EPS);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap();
            let rhs = Tensor::new(vec![2.0, 5.0, 10.0], vec![3, 1]).unwrap();
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
            let result = ldivide_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb), Vec::new())
                .expect("gpu ldivide");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected = [0.2, 0.25, 0.3333333333333333];
            for (got, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((got - exp).abs() < GPU_EPS);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_gpu_prototype_keeps_residency() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload proto");
            let result = ldivide_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("ldivide like gpu");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert!(gathered.data.iter().all(|v| (v - 0.5).abs() < GPU_EPS));
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_host_gathers_gpu_value() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![8.0, 18.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
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
            let result = ldivide_builtin(
                Value::GpuTensor(ha),
                Value::GpuTensor(hb),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("ldivide like host");
            let Value::Tensor(t) = result else {
                panic!("expected tensor result after host gather");
            };
            assert_eq!(t.shape, vec![2, 1]);
            let expected = [0.25, 1.0 / 6.0];
            for (got, exp) in t.data.iter().zip(expected.iter()) {
                assert!((got - exp).abs() < EPS);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_complex_prototype_yields_complex() {
        let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = ldivide_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("ldivide like complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                let expected = [(0.5, 0.0), (0.5, 0.0)];
                for (got, exp) in ct.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS);
                    assert!((got.1 - exp.1).abs() < EPS);
                }
            }
            Value::Complex(re, im) => {
                assert!((re - 0.5).abs() < EPS && im.abs() < EPS);
            }
            other => panic!("expected complex output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_missing_prototype_errors() {
        let lhs = Value::Num(2.0);
        let rhs = Value::Num(4.0);
        let err = ldivide_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
        assert!(
            err.message().contains("prototype"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_like_keyword_char_array() {
        test_support::with_test_provider(|provider| {
            let keyword = CharArray::new_row("LIKE");
            let lhs = Value::Num(2.0);
            let rhs = Value::Num(5.0);
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = ldivide_builtin(
                lhs,
                rhs,
                vec![Value::CharArray(keyword), Value::GpuTensor(proto)],
            )
            .expect("ldivide like char");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert!((gathered.data[0] - 2.5).abs() < GPU_EPS);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ldivide_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let cpu = ldivide_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
        let view_l = HostTensorView {
            data: &lhs.data,
            shape: &lhs.shape,
        };
        let view_r = HostTensorView {
            data: &rhs.data,
            shape: &rhs.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let ha = provider.upload(&view_l).unwrap();
        let hb = provider.upload(&view_r).unwrap();
        let gpu = block_on(ldivide_gpu_pair(ha, hb)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(t) => {
                assert_eq!(gathered.data.len(), t.data.len());
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (ga, ca) in gathered.data.iter().zip(t.data.iter()) {
                    assert!((ga - ca).abs() < tol);
                }
            }
            Value::Num(n) => assert_eq!(gathered.data, vec![n]),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ldivide_int_inputs_promote_to_double() {
        let lhs = Value::Int(IntValue::I32(6));
        let rhs = Value::Int(IntValue::I32(4));
        let result = ldivide_builtin(lhs, rhs, Vec::new()).expect("ldivide");
        match result {
            Value::Num(v) => assert!((v - (4.0 / 6.0)).abs() < EPS),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }
}
