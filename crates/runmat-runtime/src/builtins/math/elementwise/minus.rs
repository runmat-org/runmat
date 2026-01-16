//! MATLAB-compatible `minus` builtin with GPU-aware semantics for RunMat.

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
use crate::builtins::common::{gpu_helpers, tensor};

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
    notes: "Uses elem_sub for equal-shape gpuArrays and specialised scalar_sub / scalar_rsub hooks for scalar broadcast cases; unsupported shapes fall back to host execution.",
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

#[runtime_builtin(
    name = "minus",
    category = "math/elementwise",
    summary = "Element-wise subtraction with MATLAB-compatible implicit expansion.",
    keywords = "minus,element-wise subtraction,gpu,-",
    accel = "elementwise",
    builtin_path = "crate::builtins::math::elementwise::minus"
)]
fn minus_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> Result<Value, String> {
    let template = parse_output_template(&rest)?;
    let base = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => minus_gpu_pair(la, lb),
        (Value::GpuTensor(la), rhs) => minus_gpu_host_left(la, rhs),
        (lhs, Value::GpuTensor(rb)) => minus_gpu_host_right(lhs, rb),
        (lhs, rhs) => minus_host(lhs, rhs),
    }?;
    apply_output_template(base, &template)
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> Result<OutputTemplate, String> {
    if args.is_empty() {
        return Ok(OutputTemplate::Default);
    }
    if args.len() == 1 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Err("minus: expected prototype after 'like'".to_string());
        }
        return Err("minus: unsupported option; only 'like' is accepted".to_string());
    }
    if args.len() == 2 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Ok(OutputTemplate::Like(args[1].clone()));
        }
        return Err("minus: unsupported option; only 'like' is accepted".to_string());
    }
    Err("minus: too many input arguments".to_string())
}

fn apply_output_template(value: Value, template: &OutputTemplate) -> Result<Value, String> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => apply_like_template(value, proto),
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

fn apply_like_template(value: Value, prototype: &Value) -> Result<Value, String> {
    let analysed = analyse_like_prototype(prototype)?;
    match analysed.class {
        PrototypeClass::Real => match analysed.device {
            DevicePreference::Host => ensure_device(value, DevicePreference::Host),
            DevicePreference::Gpu => ensure_device(value, DevicePreference::Gpu),
        },
        PrototypeClass::Complex => {
            let host_value = ensure_device(value, DevicePreference::Host)?;
            real_to_complex(host_value)
        }
    }
}

fn ensure_device(value: Value, device: DevicePreference) -> Result<Value, String> {
    match device {
        DevicePreference::Host => convert_to_host_like(value),
        DevicePreference::Gpu => convert_to_gpu(value),
    }
}

fn convert_to_host_like(value: Value) -> Result<Value, String> {
    if let Value::GpuTensor(handle) = value {
        gpu_helpers::gather_value(&Value::GpuTensor(handle))
    } else {
        Ok(value)
    }
}

fn convert_to_gpu(value: Value) -> Result<Value, String> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(
            "minus: GPU output requested via 'like' but no acceleration provider is active"
                .to_string(),
        );
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
                .map_err(|e| format!("minus: failed to upload GPU result: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("minus: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|e| format!("minus: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("minus: GPU prototypes for 'like' only support real numeric outputs".to_string())
        }
        Value::String(_) | Value::StringArray(_) | Value::Cell(_) | Value::Struct(_) => {
            Err("minus: unsupported prototype conversion to GPU output".to_string())
        }
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => {
            Err("minus: unsupported prototype conversion to GPU output".to_string())
        }
    }
}

fn analyse_like_prototype(proto: &Value) -> Result<LikeAnalysis, String> {
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
            let gathered = gather_like_prototype(other)?;
            analyse_like_prototype(&gathered)
        }
    }
}

fn gather_like_prototype(value: &Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value(value),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_) => Ok(value.clone()),
        _ => Err(format!(
            "minus: unsupported prototype for 'like' ({value:?})"
        )),
    }
}

fn real_to_complex(value: Value) -> Result<Value, String> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(t) => {
            let data: Vec<(f64, f64)> = t.data.iter().map(|&v| (v, 0.0)).collect();
            let tensor =
                ComplexTensor::new(data, t.shape.clone()).map_err(|e| format!("minus: {e}"))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|e| format!("minus: {e}"))?;
            real_to_complex(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            real_to_complex(Value::Tensor(tensor))
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value(&Value::GpuTensor(handle.clone()))?;
            real_to_complex(gathered)
        }
        other => Err(format!(
            "minus: cannot convert value {other:?} to complex output"
        )),
    }
}

fn minus_gpu_pair(lhs: GpuTensorHandle, rhs: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if lhs.shape == rhs.shape {
            if let Ok(handle) = provider.elem_sub(&lhs, &rhs) {
                return Ok(Value::GpuTensor(handle));
            }
        }
        // Attempt N-D broadcast via repmat on device
        if let Some((out_shape, reps_l, reps_r)) = broadcast_reps(&lhs.shape, &rhs.shape) {
            let made_left = reps_l.iter().any(|&r| r != 1);
            let made_right = reps_r.iter().any(|&r| r != 1);
            let left_expanded = if made_left {
                provider.repmat(&lhs, &reps_l).map_err(|e| e.to_string())?
            } else {
                lhs.clone()
            };
            let right_expanded = if made_right {
                provider.repmat(&rhs, &reps_r).map_err(|e| e.to_string())?
            } else {
                rhs.clone()
            };
            let result = provider
                .elem_sub(&left_expanded, &right_expanded)
                .map_err(|e| e.to_string());
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
            if let Some(scalar) = gpu_scalar_value(&lhs)? {
                if let Ok(handle) = provider.scalar_rsub(&rhs, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
        if is_scalar_shape(&rhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&rhs)? {
                if let Ok(handle) = provider.scalar_sub(&lhs, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
    }
    let left = gpu_helpers::gather_tensor(&lhs)?;
    let right = gpu_helpers::gather_tensor(&rhs)?;
    minus_host(Value::Tensor(left), Value::Tensor(right))
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

fn minus_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&rhs)? {
            if let Ok(handle) = provider.scalar_sub(&lhs, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let host_lhs = gpu_helpers::gather_tensor(&lhs)?;
    minus_host(Value::Tensor(host_lhs), rhs)
}

fn minus_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&lhs)? {
            if let Ok(handle) = provider.scalar_rsub(&rhs, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let host_rhs = gpu_helpers::gather_tensor(&rhs)?;
    minus_host(lhs, Value::Tensor(host_rhs))
}

fn minus_host(lhs: Value, rhs: Value) -> Result<Value, String> {
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (MinusOperand::Real(a), MinusOperand::Real(b)) => minus_real_real(&a, &b),
        (MinusOperand::Complex(a), MinusOperand::Complex(b)) => minus_complex_complex(&a, &b),
        (MinusOperand::Complex(a), MinusOperand::Real(b)) => minus_complex_real(&a, &b),
        (MinusOperand::Real(a), MinusOperand::Complex(b)) => minus_real_complex(&a, &b),
    }
}

enum MinusOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> Result<MinusOperand, String> {
    match value {
        Value::Tensor(t) => Ok(MinusOperand::Real(t)),
        Value::Num(n) => Ok(MinusOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("minus: {e}"))?,
        )),
        Value::Int(i) => Ok(MinusOperand::Real(
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("minus: {e}"))?,
        )),
        Value::Bool(b) => Ok(MinusOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("minus: {e}"))?,
        )),
        Value::LogicalArray(logical) => Ok(MinusOperand::Real(
            tensor::logical_to_tensor(&logical).map_err(|e| format!("minus: {e}"))?,
        )),
        Value::CharArray(chars) => Ok(MinusOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(MinusOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("minus: {e}"))?,
        )),
        Value::ComplexTensor(ct) => Ok(MinusOperand::Complex(ct)),
        Value::GpuTensor(_) => Err("minus: internal error converting GPU value".to_string()),
        other => Err(format!(
            "minus: unsupported operand type {:?}; expected numeric or logical data",
            other
        )),
    }
}

fn minus_real_real(lhs: &Tensor, rhs: &Tensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("minus: {err}"))?;
    if plan.is_empty() {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("minus: {e}"))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut out = vec![0.0f64; plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        out[out_idx] = lhs.data[idx_lhs] - rhs.data[idx_rhs];
    }
    let tensor =
        Tensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("minus: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn minus_complex_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("minus: {err}"))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("minus: {e}"))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        out[out_idx] = (ar - br, ai - bi);
    }
    let tensor =
        ComplexTensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("minus: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn minus_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("minus: {err}"))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("minus: {e}"))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let scalar = rhs.data[idx_rhs];
        out[out_idx] = (ar - scalar, ai);
    }
    let tensor =
        ComplexTensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("minus: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn minus_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("minus: {err}"))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("minus: {e}"))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let scalar = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        out[out_idx] = (scalar - br, -bi);
    }
    let tensor =
        ComplexTensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("minus: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn char_array_to_tensor(chars: &CharArray) -> Result<Tensor, String> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|e| format!("minus: {e}"))
}

fn extract_scalar_f64(value: &Value) -> Result<Option<f64>, String> {
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

fn gpu_scalar_value(handle: &GpuTensorHandle) -> Result<Option<f64>, String> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if !is_scalar_shape(&handle.shape) {
        return Ok(None);
    }
    let host = provider
        .download(handle)
        .map_err(|e| format!("minus: {e}"))?;
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
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor};

    const EPS: f64 = 1e-12;

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
                assert_eq!(t.data, vec![-1.0, 0.0, 1.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
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
                assert_eq!(t.data, expected);
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
                assert_eq!(t.data, vec![67.0, 68.0, 69.0]);
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
                assert_eq!(t.data, vec![-1.0, -2.0, -2.0, -3.0]);
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
        assert!(err.contains("minus"), "unexpected error message: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let ha = provider.upload(&view).expect("upload");
            let hb = provider.upload(&view).expect("upload");
            let result = minus_builtin(
                Value::GpuTensor(ha.clone()),
                Value::GpuTensor(hb.clone()),
                Vec::new(),
            )
            .expect("gpu minus");
            let gathered = test_support::gather(result).expect("gather");
            let expected = vec![0.0; tensor.data.len()];
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_gpu_scalar_right() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = minus_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
                .expect("gpu scalar minus");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![-1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_gpu_scalar_left() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = minus_builtin(Value::Num(3.0), Value::GpuTensor(handle), Vec::new())
                .expect("gpu scalar minus");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![1.0, -1.0]);
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
                    assert_eq!(gathered.data, vec![7.0, 16.0]);
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minus_like_host_gathers_gpu_value() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
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
            assert_eq!(t.data, vec![5.0, 14.0]);
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
        assert!(err.contains("prototype"));
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
                assert_eq!(out.data, vec![-1.0, 0.0]);
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
        let view = HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = minus_gpu_pair(h.clone(), h).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                for (a, b) in gathered.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < EPS);
                }
            }
            other => panic!("unexpected shapes {other:?}"),
        }
    }
}
