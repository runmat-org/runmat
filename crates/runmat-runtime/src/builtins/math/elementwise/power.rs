//! MATLAB-compatible `power` builtin (element-wise exponentiation) with GPU-aware semantics.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    broadcast::BroadcastPlan, gpu_helpers, random_args::complex_tensor_into_value,
    random_args::keyword_of, tensor,
};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "power",
        builtin_path = "crate::builtins::math::elementwise::power"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "power"
category: "math/elementwise"
keywords: ["power", "element-wise power", "dot caret", "gpu", "broadcasting"]
summary: "Element-wise exponentiation A .^ B with MATLAB-compatible broadcasting, complex support, and GPU fallbacks."
references: ["https://www.mathworks.com/help/matlab/ref/power.html"]
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses provider elem_pow when both operands are gpuArrays of the same shape; gathers to the host for implicit expansion, complex inputs, or when elem_pow is unavailable."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::power::tests::power_scalar_numbers"
  integration: "builtins::math::elementwise::power::tests::power_matrix_broadcast"
  gpu: "builtins::math::elementwise::power::tests::power_gpu_pair_roundtrip"
  wgpu: "builtins::math::elementwise::power::tests::power_wgpu_matches_cpu_elementwise"
  like_gpu: "builtins::math::elementwise::power::tests::power_like_gpu_residency"
  like_complex: "builtins::math::elementwise::power::tests::power_like_complex_promotes_output"
  doc: "builtins::math::elementwise::power::tests::doc_examples_present"
---

# What does the `power` function do in MATLAB / RunMat?
`Y = power(A, B)` (or `A .^ B`) raises each element of `A` to the corresponding element of `B`
using MATLAB's implicit-expansion rules. Scalars broadcast automatically, and complex inputs
produce complex outputs that match MATLAB behaviour.

## How does the `power` function behave in MATLAB / RunMat?
- Supports scalars, vectors, matrices, and N-D tensors with the same shape or compatible singleton
  dimensions. Size mismatches raise the standard MATLAB error.
- Logical and character inputs are promoted to double precision before powering (`'A'.^2` uses the
  Unicode code points of the characters).
- Complex bases and/or exponents follow the analytic identity `z.^w = exp(w * log(z))`, so negative
  bases with fractional exponents return complex results instead of `NaN`.
- Empty tensors propagate emptiness; the result uses the broadcasted size.
- The optional `'like', prototype` arguments mirror the numeric flavour and residency
  (host vs gpuArray) of `prototype`, just like MATLAB.

## `power` Function GPU Execution Behaviour
- When both operands are gpuArrays with identical shapes, RunMat calls the provider's `elem_pow`
  hook. The WGPU backend uses a fused WGSL kernel, and the in-process provider executes on host data
  without leaving the GPU abstraction.
- If only one operand lives on the GPU and the other is a scalar, RunMat materialises a device
  buffer for the scalar and still uses `elem_pow`.
- Implicit expansion, complex operands, and providers that lack `elem_pow` automatically fall back
  to the host implementation. Results respect `'like'` residency hints, re-uploading to the GPU when
  requested.

## Examples of using the `power` function in MATLAB / RunMat

### Raise a scalar to a power

```matlab
y = power(2, 5);
```

Expected output:

```matlab
y = 32
```

### Compute element-wise powers of a matrix

```matlab
A = [1 2 3; 4 5 6];
B = power(A, 2);
```

Expected output:

```matlab
B =
     1     4     9
    16    25    36
```

### Broadcast exponents across rows

```matlab
base = (1:3)';
exponent = [1 2 3];
result = power(base, exponent);
```

Expected output:

```matlab
result =
     1     1     1
     2     4     8
     3     9    27
```

### Generate complex powers from negative bases

```matlab
values = power([-2 -1 0 1 2], 0.5);
```

Expected output (rounded):

```matlab
values = [0.0000 + 1.4142i, 0.0000 + 1.0000i, 0, 1, 1.4142]
```

### Keep GPU results with a `'like'` prototype

```matlab
proto = gpuArray.zeros(1, 1, 'single');
x = [1 2 3];
y = [2 3 4];
devicePowers = power(x, y, 'like', proto);
result = gather(devicePowers);
```

Expected output:

```matlab
devicePowers =
  1x3 gpuArray  single
     1     8    81
result =
     1     8    81
```

### Convert character codes before powering

```matlab
codes = power('ABC', 2);
```

Expected output:

```matlab
codes = [4225 4356 4489]
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Most workflows do **not** require manual `gpuArray` calls. RunMat's auto-offload and fusion planner
keep chains of element-wise operations on the GPU whenever the provider can satisfy them. When an
operation needs a fallback (implicit expansion, complex inputs, or unsupported kernels), RunMat
transparently gathers to the host, computes the MATLAB-accurate result, and honours any `'like'`
residency hints you supplied.

## FAQ

### Does `power` support MATLAB implicit expansion?
Yes. Singleton dimensions expand automatically, and size mismatches raise a dimension error with
the usual MATLAB wording.

### What numeric type does `power` return?
Real inputs produce doubles. Results promote to complex doubles whenever the exponentiation would
leave the real line (for example `(-2).^0.5` or complex exponents).

### Can I mix scalars and arrays?
Absolutely. Scalars broadcast to match the other operand. This includes scalar gpuArrays.

### What happens if only one operand is on the GPU?
If the other operand is a scalar, RunMat keeps everything on the GPU. Otherwise, it gathers the
device operand, performs the computation on the host, and returns a host tensor (unless `'like'`
instructs the runtime to re-upload the result).

### Does `power` modify the inputs in-place?
No. The builtin always allocates a fresh tensor (or complex tensor). Fusion can eliminate temporary
allocations when the expression continues with other element-wise operations.

### Can I force the result to stay on the GPU?
Yes—pass `'like', gpuArrayPrototype`. The runtime mirrors the residency of the prototype and
uploads the result when necessary.

## See Also
[.^ operator](https://www.mathworks.com/help/matlab/ref/power.html), [times](./times), [rdivide](./rdivide), [ldivide](./ldivide), [pow2](./pow2), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/elementwise/power.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/power.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::power")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "power",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_pow",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers execute element-wise pow when both operands reside on the device; host fallbacks cover implicit expansion and complex inputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::power")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "power",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let base = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let exp = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("pow({base}, {exp})"))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion planner lowers A.^B into WGSL pow() when both inputs are real; complex fallbacks execute on the host.",
};

#[runtime_builtin(
    name = "power",
    category = "math/elementwise",
    summary = "Element-wise power with MATLAB-compatible broadcasting and complex support.",
    keywords = "power,element-wise,.^,gpu,broadcast",
    accel = "elementwise",
    builtin_path = "crate::builtins::math::elementwise::power"
)]
fn power_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> Result<Value, String> {
    let template = parse_output_template(&rest)?;
    let base_result = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => power_gpu_pair(la, lb),
        (Value::GpuTensor(la), rhs) => power_gpu_host_left(la, rhs),
        (lhs, Value::GpuTensor(rb)) => power_gpu_host_right(lhs, rb),
        (lhs, rhs) => power_host(lhs, rhs),
    }?;
    apply_output_template(base_result, &template)
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> Result<OutputTemplate, String> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err("power: expected prototype after 'like'".to_string())
            } else {
                Err("power: unsupported option; only 'like' is accepted".to_string())
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err("power: unsupported option; only 'like' is accepted".to_string())
            }
        }
        _ => Err("power: too many input arguments".to_string()),
    }
}

fn apply_output_template(value: Value, template: &OutputTemplate) -> Result<Value, String> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => apply_like_template(value, proto),
    }
}

fn power_host(lhs: Value, rhs: Value) -> Result<Value, String> {
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (PowerOperand::Real(a), PowerOperand::Real(b)) => power_real_real(&a, &b),
        (PowerOperand::Complex(a), PowerOperand::Complex(b)) => power_complex_complex(&a, &b),
        (PowerOperand::Complex(a), PowerOperand::Real(b)) => power_complex_real(&a, &b),
        (PowerOperand::Real(a), PowerOperand::Complex(b)) => power_real_complex(&a, &b),
    }
}

fn power_real_real(lhs: &Tensor, rhs: &Tensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("power: {err}"))?;
    if plan.is_empty() {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("power: {e}"))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut out = Vec::with_capacity(plan.len());
    let mut all_im_zero = true;
    for (_, idx_lhs, idx_rhs) in plan.iter() {
        let base = lhs.data[idx_lhs];
        let exponent = rhs.data[idx_rhs];
        let pow = base.powf(exponent);
        if pow.is_nan() {
            let (re, im) = complex_pow_scalar(base, 0.0, exponent, 0.0);
            if im.abs() > 1e-12 {
                all_im_zero = false;
            }
            out.push((re, im));
        } else {
            out.push((pow, 0.0));
        }
    }
    if all_im_zero {
        let real_data: Vec<f64> = out.into_iter().map(|(re, _)| re).collect();
        let tensor = Tensor::new(real_data, plan.output_shape().to_vec())
            .map_err(|e| format!("power: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
            .map_err(|e| format!("power: {e}"))?;
        Ok(complex_tensor_into_value(tensor))
    }
}

fn power_complex_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("power: {err}"))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("power: {e}"))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        out[out_idx] = complex_pow_scalar(ar, ai, br, bi);
    }
    let tensor =
        ComplexTensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("power: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn power_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("power: {err}"))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("power: {e}"))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let exponent = rhs.data[idx_rhs];
        out[out_idx] = complex_pow_scalar(ar, ai, exponent, 0.0);
    }
    let tensor =
        ComplexTensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("power: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn power_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("power: {err}"))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("power: {e}"))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let base = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        out[out_idx] = complex_pow_scalar(base, 0.0, br, bi);
    }
    let tensor =
        ComplexTensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("power: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

enum PowerOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> Result<PowerOperand, String> {
    match value {
        Value::Tensor(t) => Ok(PowerOperand::Real(t)),
        Value::Num(n) => Ok(PowerOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("power: {e}"))?,
        )),
        Value::Int(i) => Ok(PowerOperand::Real(
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("power: {e}"))?,
        )),
        Value::Bool(b) => Ok(PowerOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("power: {e}"))?,
        )),
        Value::LogicalArray(logical) => Ok(PowerOperand::Real(
            tensor::logical_to_tensor(&logical).map_err(|e| format!("power: {e}"))?,
        )),
        Value::CharArray(chars) => Ok(PowerOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(PowerOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("power: {e}"))?,
        )),
        Value::ComplexTensor(ct) => Ok(PowerOperand::Complex(ct)),
        Value::GpuTensor(_) => Err("power: internal GPU operand escape".to_string()),
        other => Err(format!(
            "power: unsupported operand type {:?}; expected numeric, logical, or char data",
            other
        )),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> Result<Tensor, String> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|e| format!("power: {e}"))
}

fn power_gpu_pair(lhs: GpuTensorHandle, rhs: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if lhs.shape == rhs.shape {
            if let Ok(handle) = provider.elem_pow(&lhs, &rhs) {
                return Ok(Value::GpuTensor(handle));
            }
        }
        // Attempt N-D broadcast on device via repmat + elem_pow
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
                .elem_pow(&left_expanded, &right_expanded)
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
    }
    let host_lhs = gpu_helpers::gather_tensor(&lhs)?;
    let host_rhs = gpu_helpers::gather_tensor(&rhs)?;
    power_host(Value::Tensor(host_lhs), Value::Tensor(host_rhs))
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

fn power_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> Result<Value, String> {
    if is_complex_value(&rhs) {
        let host_rhs = gather_value(rhs)?;
        let host_lhs = gpu_helpers::gather_tensor(&lhs)?;
        return power_host(Value::Tensor(host_lhs), host_rhs);
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&rhs)? {
            if let Ok(filled) = provider.fill_like(&lhs, scalar) {
                if let Ok(handle) = provider.elem_pow(&lhs, &filled) {
                    let _ = provider.free(&filled);
                    return Ok(Value::GpuTensor(handle));
                }
                let _ = provider.free(&filled);
            }
        } else if let Some(tensor_rhs) = value_to_real_tensor_for_gpu(&rhs)? {
            if tensor_rhs.shape == lhs.shape {
                let view = HostTensorView {
                    data: &tensor_rhs.data,
                    shape: &tensor_rhs.shape,
                };
                if let Ok(uploaded) = provider.upload(&view) {
                    let result = provider.elem_pow(&lhs, &uploaded);
                    let _ = provider.free(&uploaded);
                    if let Ok(handle) = result {
                        return Ok(Value::GpuTensor(handle));
                    }
                }
            }
        }
    }
    let host_rhs = gather_value(rhs)?;
    let host_lhs = gpu_helpers::gather_tensor(&lhs)?;
    power_host(Value::Tensor(host_lhs), host_rhs)
}

fn power_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> Result<Value, String> {
    if is_complex_value(&lhs) {
        let host_lhs = gather_value(lhs)?;
        let host_rhs = gpu_helpers::gather_tensor(&rhs)?;
        return power_host(host_lhs, Value::Tensor(host_rhs));
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&lhs)? {
            if let Ok(filled) = provider.fill_like(&rhs, scalar) {
                if let Ok(handle) = provider.elem_pow(&filled, &rhs) {
                    let _ = provider.free(&filled);
                    return Ok(Value::GpuTensor(handle));
                }
                let _ = provider.free(&filled);
            }
        } else if let Some(tensor_lhs) = value_to_real_tensor_for_gpu(&lhs)? {
            if tensor_lhs.shape == rhs.shape {
                let view = HostTensorView {
                    data: &tensor_lhs.data,
                    shape: &tensor_lhs.shape,
                };
                if let Ok(uploaded) = provider.upload(&view) {
                    let result = provider.elem_pow(&uploaded, &rhs);
                    let _ = provider.free(&uploaded);
                    if let Ok(handle) = result {
                        return Ok(Value::GpuTensor(handle));
                    }
                }
            }
        }
    }
    let host_lhs = gather_value(lhs)?;
    let host_rhs = gpu_helpers::gather_tensor(&rhs)?;
    power_host(host_lhs, Value::Tensor(host_rhs))
}

fn gather_value(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            Ok(Value::Tensor(tensor))
        }
        other => Ok(other),
    }
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

fn value_to_real_tensor_for_gpu(value: &Value) -> Result<Option<Tensor>, String> {
    match value {
        Value::Tensor(t) => Ok(Some(t.clone())),
        Value::Num(n) => Ok(Some(
            Tensor::new(vec![*n], vec![1, 1]).map_err(|e| format!("power: {e}"))?,
        )),
        Value::Int(i) => Ok(Some(
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("power: {e}"))?,
        )),
        Value::Bool(b) => Ok(Some(
            Tensor::new(vec![if *b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("power: {e}"))?,
        )),
        Value::LogicalArray(l) => Ok(Some(
            tensor::logical_to_tensor(l).map_err(|e| format!("power: {e}"))?,
        )),
        Value::CharArray(chars) => Ok(Some(char_array_to_tensor(chars)?)),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(handle)?;
            Ok(Some(tensor))
        }
        _ => Ok(None),
    }
}

fn is_complex_value(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn complex_pow_scalar(base_re: f64, base_im: f64, exp_re: f64, exp_im: f64) -> (f64, f64) {
    if base_re == 0.0 && base_im == 0.0 {
        if exp_re == 0.0 && exp_im == 0.0 {
            return (1.0, 0.0);
        }
        if exp_im == 0.0 {
            if exp_re > 0.0 {
                return (0.0, 0.0);
            }
            if exp_re < 0.0 {
                return (f64::INFINITY, 0.0);
            }
            return (f64::NAN, f64::NAN);
        }
        if exp_re > 0.0 {
            return (0.0, 0.0);
        }
        if exp_re < 0.0 {
            return (f64::INFINITY, f64::NAN);
        }
        return (f64::NAN, f64::NAN);
    }

    let r = base_re.hypot(base_im);
    if r == 0.0 {
        return (0.0, 0.0);
    }
    let theta = base_im.atan2(base_re);
    let ln_r = r.ln();
    let a = exp_re * ln_r - exp_im * theta;
    let b = exp_re * theta + exp_im * ln_r;
    let mag = a.exp();
    (mag * b.cos(), mag * b.sin())
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
        DevicePreference::Host => match value {
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                Ok(Value::Tensor(tensor))
            }
            other => Ok(other),
        },
        DevicePreference::Gpu => match value {
            Value::GpuTensor(_) => Ok(value),
            Value::Tensor(t) => upload_tensor(t),
            Value::Num(n) => {
                upload_tensor(Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("power: {e}"))?)
            }
            Value::Int(i) => upload_tensor(
                Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("power: {e}"))?,
            ),
            Value::Bool(b) => upload_tensor(
                Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                    .map_err(|e| format!("power: {e}"))?,
            ),
            Value::LogicalArray(l) => {
                let tensor = tensor::logical_to_tensor(&l).map_err(|e| format!("power: {e}"))?;
                upload_tensor(tensor)
            }
            other => Err(format!(
                "power: cannot place result {:?} on the GPU via 'like'",
                other
            )),
        },
    }
}

fn upload_tensor(tensor: Tensor) -> Result<Value, String> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err("power: no acceleration provider available to honour GPU output".to_string());
    };
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let handle = provider
        .upload(&view)
        .map_err(|e| format!("power: failed to upload GPU result: {e}"))?;
    Ok(Value::GpuTensor(handle))
}

fn real_to_complex(value: Value) -> Result<Value, String> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(t) => {
            let data: Vec<(f64, f64)> = t.data.iter().map(|&v| (v, 0.0)).collect();
            let tensor =
                ComplexTensor::new(data, t.shape.clone()).map_err(|e| format!("power: {e}"))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(l) => {
            let tensor = tensor::logical_to_tensor(&l).map_err(|e| format!("power: {e}"))?;
            real_to_complex(Value::Tensor(tensor))
        }
        Value::Bool(b) => real_to_complex(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::Int(i) => real_to_complex(Value::Num(i.to_f64())),
        other => Err(format!(
            "power: cannot convert value {:?} to a complex result via 'like'",
            other
        )),
    }
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

#[derive(Clone, Copy)]
enum PrototypeClass {
    Real,
    Complex,
}

struct LikeAnalysis {
    device: DevicePreference,
    class: PrototypeClass,
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
            let gathered =
                crate::dispatcher::gather_if_needed(other).map_err(|e| format!("power: {e}"))?;
            analyse_like_prototype(&gathered)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_scalar_numbers() {
        let result = power_builtin(Value::Num(2.0), Value::Num(3.0), Vec::new()).expect("power");
        match result {
            Value::Num(v) => assert!((v - 8.0).abs() < 1e-12),
            other => panic!("expected scalar numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_matrix_broadcast() {
        let base = Tensor::new((1..=3).map(|v| v as f64).collect::<Vec<_>>(), vec![3, 1]).unwrap();
        let exp = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result =
            power_builtin(Value::Tensor(base), Value::Tensor(exp), Vec::new()).expect("power");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = [1.0, 2.0, 3.0, 1.0, 4.0, 9.0, 1.0, 8.0, 27.0];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_complex_scalar() {
        let result = power_builtin(
            Value::Complex(1.0, 2.0),
            Value::Complex(0.5, -1.0),
            Vec::new(),
        )
        .expect("power");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 4.382565059863358).abs() < 1e-9);
                assert!((im + 1.1243974773611554).abs() < 1e-9);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_char_array() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = power_builtin(
            Value::CharArray(chars),
            Value::Int(IntValue::I32(2)),
            Vec::new(),
        )
        .expect("power");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [4225.0, 8100.0];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-9);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_like_complex_promotes_output() {
        let base = Tensor::new(vec![-2.0], vec![1, 1]).unwrap();
        let result = power_builtin(
            Value::Tensor(base),
            Value::Num(0.5),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("power");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-8);
                assert!((im - std::f64::consts::SQRT_2).abs() < 1e-8);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_like_gpu_residency() {
        test_support::with_test_provider(|provider| {
            let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let exp = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = power_builtin(
                Value::Tensor(base.clone()),
                Value::Tensor(exp.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("power");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected = [1.0, 8.0, 81.0];
                    for (got, exp) in gathered.data.iter().zip(expected.iter()) {
                        assert!((got - exp).abs() < 1e-9);
                    }
                }
                other => panic!("expected gpu result, got {other:?}"),
            }
            let _ = provider.free(&proto);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let exp = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
            let base_view = HostTensorView {
                data: &base.data,
                shape: &base.shape,
            };
            let exp_view = HostTensorView {
                data: &exp.data,
                shape: &exp.shape,
            };
            let hb = provider.upload(&base_view).expect("upload");
            let he = provider.upload(&exp_view).expect("upload");
            let result = power_builtin(Value::GpuTensor(hb), Value::GpuTensor(he), Vec::new())
                .expect("power");
            let gathered = test_support::gather(result).expect("gather");
            let expected = [1.0, 8.0, 81.0];
            for (got, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((got - exp).abs() < 1e-9);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn power_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let exp = Tensor::new(vec![2.0, 0.5, -1.0], vec![1, 3]).unwrap();
        let cpu = power_host(Value::Tensor(base.clone()), Value::Tensor(exp.clone())).unwrap();
        let provider = runmat_accelerate_api::provider().unwrap();
        let base_view = HostTensorView {
            data: &base.data,
            shape: &base.shape,
        };
        let exp_view = HostTensorView {
            data: &exp.data,
            shape: &exp.shape,
        };
        let hb = provider.upload(&base_view).unwrap();
        let he = provider.upload(&exp_view).unwrap();
        let gpu = power_gpu_pair(hb.clone(), he.clone()).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu result {other:?}"),
        };
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered.data.iter().zip(cpu_tensor.data.iter()) {
            assert!((a - b).abs() < tol);
        }
        let _ = provider.free(&hb);
        let _ = provider.free(&he);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_like_missing_prototype_errors() {
        let err = power_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::from("like")])
            .expect_err("expected error");
        assert!(err.contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_like_extra_arguments_error() {
        let err = power_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            vec![Value::from("like"), Value::Num(1.0), Value::Num(2.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("too many"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_zero_negative_exponent_infinite() {
        let result = power_builtin(Value::Num(0.0), Value::Num(-2.0), Vec::new()).expect("power");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_positive()),
            other => panic!("expected scalar infinity, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_zero_complex_positive_real_part() {
        let result =
            power_builtin(Value::Num(0.0), Value::Complex(1.0, 2.0), Vec::new()).expect("power");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected zero complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn power_zero_complex_negative_real_part() {
        let result =
            power_builtin(Value::Num(0.0), Value::Complex(-1.0, 1.0), Vec::new()).expect("power");
        match result {
            Value::Complex(re, im) => {
                assert!(re.is_infinite());
                assert!(im.is_nan());
            }
            other => panic!("expected complex infinity, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
