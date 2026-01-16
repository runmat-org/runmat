//! MATLAB-compatible `times` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "times",
        builtin_path = "crate::builtins::math::elementwise::times"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "times"
category: "math/elementwise"
keywords: ["times", "element-wise multiply", ".*", "gpu", "implicit expansion"]
summary: "Element-wise multiplication A .* B with MATLAB-compatible implicit expansion, complex support, and GPU fallbacks."
references: ["https://www.mathworks.com/help/matlab/ref/times.html"]
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses provider elem_mul when both operands share a shape, scalar_mul when exactly one operand is a scalar, and gathers to host for implicit expansion or unsupported operand types."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::times::tests::times_scalar_numbers"
  integration: "builtins::math::elementwise::times::tests::times_row_column_broadcast"
  gpu: "builtins::math::elementwise::times::tests::times_gpu_pair_roundtrip"
  wgpu: "builtins::math::elementwise::times::tests::times_wgpu_matches_cpu_elementwise"
  doc: "builtins::math::elementwise::times::tests::doc_examples_present"
  like_gpu: "builtins::math::elementwise::times::tests::times_like_gpu_prototype_keeps_residency"
  like_host: "builtins::math::elementwise::times::tests::times_like_host_gathers_gpu_value"
  like_complex: "builtins::math::elementwise::times::tests::times_like_complex_prototype_yields_complex"
---

# What does the `times` function do in MATLAB / RunMat?
`times(A, B)` (or the operator form `A .* B`) multiplies corresponding elements of `A` and `B`, honouring MATLAB's implicit expansion rules so that scalars and singleton dimensions broadcast automatically.

## How does the `times` function behave in MATLAB / RunMat?
- Supports real, complex, logical, and character inputs; logical and character data are promoted to double precision before multiplication.
- Implicit expansion works across any dimension, provided the non-singleton extents match. Size mismatches raise the standard MATLAB-compatible error.
- Complex operands follow the analytic rule `(a + ib) .* (c + id) = (ac - bd) + i(ad + bc)`.
- Empty dimensions propagate naturally—if either operand has a zero-sized dimension after broadcasting, the result is empty with the broadcasted shape.
- Integer inputs currently promote to double precision, mirroring the behaviour of other RunMat arithmetic builtins.
- The optional `'like'` prototype makes the output adopt the residency (host or GPU) and complexity characteristics of the prototype, which is particularly useful for keeping implicit-expansion expressions on the device.

## `times` Function GPU Execution Behaviour
When a gpuArray provider is active:

1. If both operands are gpuArrays with identical shapes, RunMat dispatches to the provider's `elem_mul` hook.
2. If one operand is a scalar (host or device) and the other is a gpuArray, the runtime calls `scalar_mul` to keep the result on the device.
3. The fusion planner treats `times` as a fusible elementwise node, so adjacent elementwise producers/consumers can execute inside a single WGSL kernel or provider-optimised pipeline, avoiding spurious host↔device transfers.
4. Implicit-expansion workloads (e.g., mixing row and column vectors) or unsupported operand kinds gather transparently to host memory, compute the result with full MATLAB semantics, and return a host tensor. The documentation callouts below flag this fallback behaviour explicitly.

## Examples of using the `times` function in MATLAB / RunMat

### Multiply two matrices element-wise

```matlab
A = [1 2 3; 4 5 6];
B = [7 8 9; 1 2 3];
P = times(A, B);
```

Expected output:

```matlab
P =
    7   16   27
    4   10   18
```

### Scale a matrix by a scalar

```matlab
A = magic(3);
scaled = times(A, 0.5);
```

Expected output:

```matlab
scaled =
    4.5    0.5    3.5
    1.5    5.0    9.0
    8.0    6.5    2.0
```

### Use implicit expansion between a column and row vector

```matlab
col = (1:3)';
row = [10 20 30];
m = times(col, row);
```

Expected output:

```matlab
m =
    10    20    30
    20    40    60
    30    60    90
```

### Multiply complex inputs element-wise

```matlab
z1 = [1+2i, 3-4i];
z2 = [2-1i, -1+1i];
prod = times(z1, z2);
```

Expected output:

```matlab
prod =
    4 + 3i   1 + 7i
```

### Multiply character codes by a numeric scalar

```matlab
letters = 'ABC';
codes = times(letters, 2);
```

Expected output:

```matlab
codes = [130 132 134]
```

### Execute `times` directly on gpuArray inputs

```matlab
G1 = gpuArray([1 2 3]);
G2 = gpuArray([4 5 6]);
deviceProd = times(G1, G2);
result = gather(deviceProd);
```

Expected output:

```matlab
deviceProd =
  1x3 gpuArray
     4     10     18
result =
     4    10    18
```

### Keep the result on the GPU with a `'like'` prototype

```matlab
proto = gpuArray.zeros(1, 1);
A = [1 2 3];
B = [4 5 6];
C = times(A, B, 'like', proto);  % stays on the GPU for downstream work
```

Expected output:

```matlab
C =
  1x3 gpuArray
      4     10     18
```

## GPU residency in RunMat (Do I need `gpuArray`?)
RunMat's auto-offload planner keeps tensors on the GPU whenever fused expressions benefit from device execution. Explicit `gpuArray` / `gather` calls are still supported for MATLAB code that manages residency manually. When the active provider lacks the kernels needed for a particular call (for example, implicit expansion between gpuArrays of different shapes), RunMat gathers back to the host, computes the MATLAB-accurate result, and resumes execution seamlessly.

## FAQ

### Does `times` support MATLAB implicit expansion?
Yes. Any singleton dimensions expand automatically. If a dimension has incompatible non-singleton extents, `times` raises the standard size-mismatch error.

### What numeric type does `times` return?
Results are double precision for real inputs and complex double when either operand is complex. Logical and character inputs are promoted to double before multiplication.

### Can I multiply gpuArrays and host scalars?
Yes. RunMat keeps the computation on the GPU when the scalar is numeric. For other host operand types, the runtime gathers the gpuArray and computes on the CPU.

### Does `times` preserve gpuArray residency after a fallback?
When a fallback occurs (for example, implicit expansion that the provider does not implement), the current result remains on the host. Subsequent operations may move it back to the GPU when auto-offload decides it is profitable.

### How can I force the result to stay on the GPU?
Provide a `'like'` prototype: `times(A, B, 'like', gpuArray.zeros(1, 1))` keeps the result on the device even if one of the inputs originated on the host.

### How are empty arrays handled?
Empty dimensions propagate. If either operand has an extent of zero in the broadcasted shape, the result is empty with the broadcasted dimensions.

### Are integer inputs supported?
Yes. Integers promote to double precision during the multiplication, matching other RunMat arithmetic builtins.

### Can I mix complex and real operands?
Absolutely. The result is complex, with broadcasting rules identical to MATLAB.

### What about string arrays?
String arrays are not numeric and therefore raise an error when passed to `times`.

## See Also
[mtimes](./mtimes), [rdivide](./rdivide), [ldivide](./ldivide), [plus](./plus), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/math/elementwise/times.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/times.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

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
        "Uses elem_mul for shape-compatible gpuArrays and scalar_mul when one operand is a scalar; falls back to host execution for implicit expansion or unsupported operand kinds.",
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

#[runtime_builtin(
    name = "times",
    category = "math/elementwise",
    summary = "Element-wise multiplication with MATLAB-compatible implicit expansion.",
    keywords = "times,element-wise multiply,gpu,.*",
    accel = "elementwise",
    builtin_path = "crate::builtins::math::elementwise::times"
)]
fn times_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    let base = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => (times_gpu_pair(la, lb)).map_err(Into::into),
        (Value::GpuTensor(la), rhs) => (times_gpu_host_left(la, rhs)).map_err(Into::into),
        (lhs, Value::GpuTensor(rb)) => (times_gpu_host_right(lhs, rb)).map_err(Into::into),
        (lhs, rhs) => (times_host(lhs, rhs)).map_err(Into::into),
    }?;
    apply_output_template(base, &template).map_err(Into::into)
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
            return Err("times: expected prototype after 'like'".to_string());
        }
        return Err("times: unsupported option; only 'like' is accepted".to_string());
    }
    if args.len() == 2 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Ok(OutputTemplate::Like(args[1].clone()));
        }
        return Err("times: unsupported option; only 'like' is accepted".to_string());
    }
    Err("times: too many input arguments".to_string())
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
        let temp = Value::GpuTensor(handle);
        gpu_helpers::gather_value(&temp)
    } else {
        Ok(value)
    }
}

fn convert_to_gpu(value: Value) -> Result<Value, String> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(
            "times: GPU output requested via 'like' but no acceleration provider is active"
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
                .map_err(|e| format!("times: failed to upload GPU result: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("times: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|e| format!("times: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("times: GPU prototypes for 'like' only support real numeric outputs".to_string())
        }
        Value::String(_) | Value::StringArray(_) | Value::Cell(_) | Value::Struct(_) => {
            Err("times: unsupported prototype conversion to GPU output".to_string())
        }
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => {
            Err("times: unsupported prototype conversion to GPU output".to_string())
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
            "times: unsupported prototype for 'like' ({value:?})"
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
                ComplexTensor::new(data, t.shape.clone()).map_err(|e| format!("times: {e}"))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|e| format!("times: {e}"))?;
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
            "times: cannot convert value {other:?} to complex output"
        )),
    }
}

fn times_gpu_pair(lhs: GpuTensorHandle, rhs: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if lhs.shape == rhs.shape {
            if let Ok(handle) = provider.elem_mul(&lhs, &rhs) {
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
                .elem_mul(&left_expanded, &right_expanded)
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
                if let Ok(handle) = provider.scalar_mul(&rhs, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
        if is_scalar_shape(&rhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&rhs)? {
                if let Ok(handle) = provider.scalar_mul(&lhs, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
    }
    let left = gpu_helpers::gather_tensor(&lhs)?;
    let right = gpu_helpers::gather_tensor(&rhs)?;
    times_host(Value::Tensor(left), Value::Tensor(right))
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

fn times_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&rhs)? {
            if let Ok(handle) = provider.scalar_mul(&lhs, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let host_lhs = gpu_helpers::gather_tensor(&lhs)?;
    times_host(Value::Tensor(host_lhs), rhs)
}

fn times_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&lhs)? {
            if let Ok(handle) = provider.scalar_mul(&rhs, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let host_rhs = gpu_helpers::gather_tensor(&rhs)?;
    times_host(lhs, Value::Tensor(host_rhs))
}

fn times_host(lhs: Value, rhs: Value) -> Result<Value, String> {
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (TimesOperand::Real(a), TimesOperand::Real(b)) => times_real_real(&a, &b),
        (TimesOperand::Complex(a), TimesOperand::Complex(b)) => times_complex_complex(&a, &b),
        (TimesOperand::Complex(a), TimesOperand::Real(b)) => times_complex_real(&a, &b),
        (TimesOperand::Real(a), TimesOperand::Complex(b)) => times_real_complex(&a, &b),
    }
}

fn times_real_real(lhs: &Tensor, rhs: &Tensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("times: {err}"))?;
    let dtype = real_result_dtype(lhs, rhs);
    if plan.is_empty() {
        let mut tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("times: {e}"))?;
        apply_result_dtype(&mut tensor, dtype);
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut out = vec![0.0f64; plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        out[out_idx] = lhs.data[idx_lhs] * rhs.data[idx_rhs];
    }
    let mut tensor =
        Tensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("times: {e}"))?;
    apply_result_dtype(&mut tensor, dtype);
    Ok(tensor::tensor_into_value(tensor))
}

fn times_complex_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("times: {err}"))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("times: {e}"))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        let re = ar * br - ai * bi;
        let im = ar * bi + ai * br;
        out[out_idx] = (re, im);
    }
    let tensor =
        ComplexTensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("times: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn times_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("times: {err}"))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("times: {e}"))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let scalar = rhs.data[idx_rhs];
        out[out_idx] = (ar * scalar, ai * scalar);
    }
    let tensor =
        ComplexTensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("times: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn real_result_dtype(lhs: &Tensor, rhs: &Tensor) -> NumericDType {
    if lhs.dtype == NumericDType::F32 && rhs.dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn apply_result_dtype(tensor: &mut Tensor, dtype: NumericDType) {
    match dtype {
        NumericDType::F64 => {
            tensor.dtype = NumericDType::F64;
        }
        NumericDType::F32 => {
            for value in &mut tensor.data {
                *value = (*value as f32) as f64;
            }
            tensor.dtype = NumericDType::F32;
        }
    }
}

fn times_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("times: {err}"))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("times: {e}"))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let scalar = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        out[out_idx] = (scalar * br, scalar * bi);
    }
    let tensor =
        ComplexTensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("times: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

enum TimesOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> Result<TimesOperand, String> {
    match value {
        Value::Tensor(t) => Ok(TimesOperand::Real(t)),
        Value::Num(n) => Ok(TimesOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("times: {e}"))?,
        )),
        Value::Int(i) => Ok(TimesOperand::Real(
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("times: {e}"))?,
        )),
        Value::Bool(b) => Ok(TimesOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("times: {e}"))?,
        )),
        Value::LogicalArray(logical) => Ok(TimesOperand::Real(
            tensor::logical_to_tensor(&logical).map_err(|e| format!("times: {e}"))?,
        )),
        Value::CharArray(chars) => Ok(TimesOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(TimesOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("times: {e}"))?,
        )),
        Value::ComplexTensor(ct) => Ok(TimesOperand::Complex(ct)),
        Value::GpuTensor(_) => Err("times: internal error converting GPU value".to_string()),
        other => Err(format!(
            "times: unsupported operand type {:?}; expected numeric or logical data",
            other
        )),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> Result<Tensor, String> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|e| format!("times: {e}"))
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
    if !is_scalar_shape(&handle.shape) {
        return Ok(None);
    }
    let tensor = gpu_helpers::gather_tensor(handle)?;
    Ok(tensor.data.first().copied())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, Tensor};

    const EPS: f64 = 1e-12;

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
                assert_eq!(t.data, vec![2.0, 4.0, 6.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
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
                assert_eq!(t.data, expected);
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
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS && (got.1 - exp.1).abs() < EPS);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
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
                assert_eq!(t.data, vec![130.0, 132.0, 134.0]);
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
                assert_eq!(t.data, vec![2.0, 0.0, 3.0, 0.0]);
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
        assert!(err.contains("times"), "unexpected error message: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
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
                .data
                .iter()
                .zip(tensor.data.iter())
                .map(|(x, y)| x * y)
                .collect::<Vec<_>>();
            assert_eq!(gathered.data, expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_gpu_scalar_right() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = times_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
                .expect("gpu scalar times");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![2.0, 4.0, 6.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_gpu_scalar_left() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = times_builtin(Value::Num(3.0), Value::GpuTensor(handle), Vec::new())
                .expect("gpu scalar times");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![6.0, 12.0]);
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
                    assert_eq!(gathered.data, vec![3.0, 8.0]);
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_like_host_gathers_gpu_value() {
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
            assert_eq!(t.data, vec![5.0, 12.0]);
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
                for (got, exp) in ct.data.iter().zip(expected.iter()) {
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
        assert!(err.contains("prototype"), "unexpected error: {err}");
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
                    assert_eq!(gathered.data, vec![10.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
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
        let gpu = times_gpu_pair(ha, hb).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(t) => assert_eq!(gathered.data, t.data),
            Value::Num(n) => assert_eq!(gathered.data, vec![n]),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn times_int_inputs_promote_to_double() {
        let lhs = Value::Int(IntValue::I32(3));
        let rhs = Value::Int(IntValue::I32(5));
        let result = times_builtin(lhs, rhs, Vec::new()).expect("times");
        match result {
            Value::Num(v) => assert_eq!(v, 15.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }
}
