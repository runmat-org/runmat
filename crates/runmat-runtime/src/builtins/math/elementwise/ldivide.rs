//! MATLAB-compatible `ldivide` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "ldivide"
category: "math/elementwise"
keywords: ["ldivide", "element-wise left division", ".\\", "gpu", "implicit expansion"]
summary: "Element-wise left division A .\\ B (computes B ./ A) with MATLAB-compatible broadcasting, complex support, and GPU fallbacks."
references: ["https://www.mathworks.com/help/matlab/ref/ldivide.html"]
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers provider elem_div/scalar_div/scalar_rdiv hooks; gathers to host when shapes require implicit expansion or operands are unsupported."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::ldivide::tests::ldivide_scalar_numbers"
  integration: "builtins::math::elementwise::ldivide::tests::ldivide_row_column_broadcast"
  gpu: "builtins::math::elementwise::ldivide::tests::ldivide_gpu_pair_roundtrip"
  wgpu: "builtins::math::elementwise::ldivide::tests::ldivide_wgpu_matches_cpu_elementwise"
  doc: "builtins::math::elementwise::ldivide::tests::doc_examples_present"
  like_gpu: "builtins::math::elementwise::ldivide::tests::ldivide_like_gpu_prototype_keeps_residency"
  like_host: "builtins::math::elementwise::ldivide::tests::ldivide_like_host_gathers_gpu_value"
  like_complex: "builtins::math::elementwise::ldivide::tests::ldivide_like_complex_prototype_yields_complex"
---

# What does the `ldivide` function do in MATLAB / RunMat?
`ldivide(A, B)` (operator form `A .\ B`) divides each element of `B` by the corresponding element of `A`, delivering MATLAB-compatible left-division semantics. It is equivalent to `B ./ A` but keeps argument order consistent with MATLAB source code and operator precedence.

## How does the `ldivide` function behave in MATLAB / RunMat?
- Supports real, complex, logical, and character inputs; logical and character data are promoted to double precision before division.
- Implicit expansion follows MATLAB rules: singleton dimensions expand automatically, while mismatched non-singleton extents raise MATLAB-compatible size errors.
- Complex operands use the analytic continuation `B ./ A`, propagating `NaN` and `Inf` exactly as MATLAB does.
- Empty shapes propagate cleanly—if the broadcasted output has a zero dimension, the result is empty with the expected shape.
- Integer inputs promote to double precision, mirroring MATLAB’s numeric tower.
- The optional `'like'` prototype makes the result adopt the residency (host or GPU) and numeric flavour of the prototype. Complex prototypes are honoured on the host today; real gpuArray prototypes keep the result on the device.

## `ldivide` Function GPU Execution Behaviour
When a gpuArray provider is active:

1. If both operands are gpuArrays with identical shapes, RunMat calls the provider’s `elem_div` hook with `(B, A)` so the division runs entirely on the GPU.
2. If the divisor `A` is scalar (host or device) and the numerator `B` is a gpuArray, the runtime uses `scalar_div` to evaluate `B ./ a` on device memory.
3. If the numerator `B` is scalar and the divisor `A` is a gpuArray, `scalar_rdiv` performs `b ./ A` without leaving the GPU.
4. When shapes require implicit expansion—or the provider lacks the necessary kernels—RunMat gathers to the host, computes the MATLAB-accurate result, then reapplies `'like'` residency rules (including re-uploading to a gpuArray when requested).
5. The fusion planner treats `ldivide` as a fusible elementwise node, so adjacent elementwise producers and consumers can execute inside a single GPU pipeline or WGSL kernel, minimising redundant host↔device transfers.

## Examples of using the `ldivide` function in MATLAB / RunMat

### Left-dividing a vector by a scalar

```matlab
A = 2;
B = [4 6 8];
Q = ldivide(A, B);
```

Expected output:

```matlab
Q = [2 3 4]
```

### Broadcasting between column divisors and row numerators

```matlab
A = (1:3)';         % column of divisors
B = [10 20 40];     % row of numerators
M = ldivide(A, B);  % implicit expansion
```

Expected output:

```matlab
M =
   10.0000   20.0000   40.0000
    5.0000   10.0000   20.0000
    3.3333    6.6667   13.3333
```

### Element-wise left division of complex values

```matlab
A = [1+2i, 3-4i];
B = [2-1i, -1+1i];
Z = ldivide(A, B);
```

Expected output:

```matlab
Z =
   0.0000 - 1.0000i   -0.2800 - 0.0400i
```

### Dividing character codes by a scalar

```matlab
A = 'ABC';
B = 2;
codes = ldivide(A, B);
```

Expected output:

```matlab
codes = [0.0308 0.0303 0.0301]
```

### Computing reciprocals with `ldivide`

```matlab
A = [1 2 4 8];
B = 1;
R = ldivide(A, B);   % equivalent to 1 ./ A
```

Expected output:

```matlab
R = [1 0.5 0.25 0.125]
```

### Keeping results on the GPU with `'like'`

```matlab
proto = gpuArray.zeros(1, 1);
A = gpuArray([2 4 8 16]);
B = gpuArray([4 8 16 32]);
deviceResult = ldivide(A, B, 'like', proto);
hostCheck = gather(deviceResult);
```

Expected output:

```matlab
deviceResult =
  1x4 gpuArray
    2  2  2  2
hostCheck = [2 2 2 2]
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do not need to call `gpuArray` manually. RunMat’s auto-offload planner keeps tensors on the GPU whenever provider kernels cover the operation. Explicit `gpuArray` / `gather` calls remain available for MATLAB compatibility; when a provider fallback happens, the runtime gathers to host, computes the MATLAB-accurate answer, and reapplies `'like'` residency requests automatically.

## FAQ

### Does `ldivide` support MATLAB implicit expansion?
Yes. Singleton dimensions expand automatically; otherwise incompatible shapes raise MATLAB-style errors.

### What numeric type does `ldivide` return?
Real inputs return doubles; mixed or complex inputs return complex doubles. Logical and character inputs promote to double before division.

### How does `ldivide` handle division by zero?
`finite ./ 0` yields signed infinities, and `0 ./ 0` becomes `NaN`, matching MATLAB and IEEE-754 behaviour.

### Can I divide gpuArrays by host scalars?
Yes. Numeric scalars stay on device through `scalar_div`/`scalar_rdiv`. Non-numeric host scalars trigger a gather-then-divide fallback.

### Does `ldivide` preserve gpuArray residency after a fallback?
If the runtime gathers to host (for example, due to implicit expansion), the intermediate stays on the host. Later computations may move it back when auto-offload deems it profitable, or you can request GPU residency explicitly with `'like'`.

### How do I keep the result on the GPU?
Provide a real gpuArray prototype: `ldivide(A, B, 'like', gpuArray.zeros(1,1))`. The runtime re-uploads the host result when necessary.

### How are empty arrays handled?
Empty operands propagate cleanly—the output shape is the broadcasted shape, and the data vector is empty.

### Are integers and logicals supported?
Yes. Both promote to double precision before division so you get MATLAB-compatible numeric results (including `Inf` when dividing by zero).

### Can I mix real and complex operands?
Absolutely. Mixed cases return complex doubles with full MATLAB semantics.

## See Also
[times](./times), [rdivide](./rdivide), [mldivide](../linalg/ops/mldivide), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/math/elementwise/ldivide.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/ldivide.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

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

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ldivide",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let divisor = ctx.inputs.get(0).ok_or(FusionError::MissingInput(0))?;
            let numerator = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("({numerator} / {divisor})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a plain quotient; providers can override with specialised kernels when desirable.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("ldivide", DOC_MD);

#[runtime_builtin(
    name = "ldivide",
    category = "math/elementwise",
    summary = "Element-wise left division (B ./ A) with MATLAB-compatible implicit expansion.",
    keywords = "ldivide,element-wise left division,gpu,.\\",
    accel = "elementwise"
)]
fn ldivide_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> Result<Value, String> {
    let template = parse_output_template(&rest)?;
    let base = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => ldivide_gpu_pair(la, lb),
        (Value::GpuTensor(la), rhs) => ldivide_gpu_host_left(la, rhs),
        (lhs, Value::GpuTensor(rb)) => ldivide_gpu_host_right(lhs, rb),
        (lhs, rhs) => ldivide_host(lhs, rhs),
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
            return Err("ldivide: expected prototype after 'like'".to_string());
        }
        return Err("ldivide: unsupported option; only 'like' is accepted".to_string());
    }
    if args.len() == 2 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Ok(OutputTemplate::Like(args[1].clone()));
        }
        return Err("ldivide: unsupported option; only 'like' is accepted".to_string());
    }
    Err("ldivide: too many input arguments".to_string())
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
            "ldivide: GPU output requested via 'like' but no acceleration provider is active"
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
                .map_err(|e| format!("ldivide: failed to upload GPU result: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("ldivide: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor =
                tensor::logical_to_tensor(&logical).map_err(|e| format!("ldivide: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("ldivide: GPU prototypes for 'like' only support real numeric outputs".to_string())
        }
        Value::String(_) | Value::StringArray(_) | Value::Cell(_) | Value::Struct(_) => {
            Err("ldivide: unsupported prototype conversion to GPU output".to_string())
        }
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => {
            Err("ldivide: unsupported prototype conversion to GPU output".to_string())
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
            "ldivide: unsupported prototype for 'like' ({value:?})"
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
                ComplexTensor::new(data, t.shape.clone()).map_err(|e| format!("ldivide: {e}"))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor =
                tensor::logical_to_tensor(&logical).map_err(|e| format!("ldivide: {e}"))?;
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
            "ldivide: cannot convert value {other:?} to complex output"
        )),
    }
}

fn ldivide_gpu_pair(divisor: GpuTensorHandle, numerator: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if divisor.shape == numerator.shape {
            if let Ok(handle) = provider.elem_div(&numerator, &divisor) {
                return Ok(Value::GpuTensor(handle));
            }
        }
        if is_scalar_shape(&divisor.shape) {
            if let Some(scalar) = gpu_scalar_value(&divisor)? {
                if let Ok(handle) = provider.scalar_div(&numerator, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
        if is_scalar_shape(&numerator.shape) {
            if let Some(scalar) = gpu_scalar_value(&numerator)? {
                if let Ok(handle) = provider.scalar_rdiv(&divisor, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
    }
    let divisor_host = gpu_helpers::gather_tensor(&divisor)?;
    let numerator_host = gpu_helpers::gather_tensor(&numerator)?;
    ldivide_host(Value::Tensor(divisor_host), Value::Tensor(numerator_host))
}

fn ldivide_gpu_host_left(divisor: GpuTensorHandle, numerator: Value) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&numerator)? {
            if let Ok(handle) = provider.scalar_rdiv(&divisor, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let divisor_host = gpu_helpers::gather_tensor(&divisor)?;
    ldivide_host(Value::Tensor(divisor_host), numerator)
}

fn ldivide_gpu_host_right(divisor: Value, numerator: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&divisor)? {
            if let Ok(handle) = provider.scalar_div(&numerator, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let numerator_host = gpu_helpers::gather_tensor(&numerator)?;
    ldivide_host(divisor, Value::Tensor(numerator_host))
}

fn ldivide_host(divisor: Value, numerator: Value) -> Result<Value, String> {
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

fn ldivide_real_real(divisor: &Tensor, numerator: &Tensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| format!("ldivide: {err}"))?;
    if plan.len() == 0 {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("ldivide: {e}"))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut out = vec![0.0f64; plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        out[out_idx] = numerator.data[idx_lhs] / divisor.data[idx_rhs];
    }
    let tensor =
        Tensor::new(out, plan.output_shape().to_vec()).map_err(|e| format!("ldivide: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ldivide_complex_complex(
    divisor: &ComplexTensor,
    numerator: &ComplexTensor,
) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| format!("ldivide: {err}"))?;
    if plan.len() == 0 {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("ldivide: {e}"))?;
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
        .map_err(|e| format!("ldivide: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn ldivide_complex_real(divisor: &ComplexTensor, numerator: &Tensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| format!("ldivide: {err}"))?;
    if plan.len() == 0 {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("ldivide: {e}"))?;
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
        .map_err(|e| format!("ldivide: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn ldivide_real_complex(divisor: &Tensor, numerator: &ComplexTensor) -> Result<Value, String> {
    let plan = BroadcastPlan::new(&numerator.shape, &divisor.shape)
        .map_err(|err| format!("ldivide: {err}"))?;
    if plan.len() == 0 {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| format!("ldivide: {e}"))?;
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
        .map_err(|e| format!("ldivide: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

enum LdivideOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> Result<LdivideOperand, String> {
    match value {
        Value::Tensor(t) => Ok(LdivideOperand::Real(t)),
        Value::Num(n) => Ok(LdivideOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("ldivide: {e}"))?,
        )),
        Value::Int(i) => Ok(LdivideOperand::Real(
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("ldivide: {e}"))?,
        )),
        Value::Bool(b) => Ok(LdivideOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("ldivide: {e}"))?,
        )),
        Value::LogicalArray(logical) => Ok(LdivideOperand::Real(
            tensor::logical_to_tensor(&logical).map_err(|e| format!("ldivide: {e}"))?,
        )),
        Value::CharArray(chars) => Ok(LdivideOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(LdivideOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("ldivide: {e}"))?,
        )),
        Value::ComplexTensor(ct) => Ok(LdivideOperand::Complex(ct)),
        Value::GpuTensor(_) => Err("ldivide: internal error converting GPU value".to_string()),
        other => Err(format!(
            "ldivide: unsupported operand type {:?}; expected numeric or logical data",
            other
        )),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> Result<Tensor, String> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|e| format!("ldivide: {e}"))
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
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, Tensor};

    const EPS: f64 = 1e-12;
    const GPU_EPS: f64 = 1e-6;

    #[test]
    fn ldivide_scalar_numbers() {
        let result =
            ldivide_builtin(Value::Num(7.0), Value::Num(2.0), Vec::new()).expect("ldivide");
        match result {
            Value::Num(v) => assert!((v - (2.0 / 7.0)).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn ldivide_matrix_scalar() {
        let tensor = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result =
            ldivide_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = vec![1.0, 0.5, 0.3333333333333333, 0.25];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn ldivide_row_column_broadcast() {
        let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0, 40.0], vec![1, 3]).unwrap();
        let result = ldivide_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
            .expect("broadcast ldivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = vec![
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
                let expected = vec![(0.0, -1.0), (-0.28, -0.04)];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < 1e-10 && (got.1 - exp.1).abs() < 1e-10);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

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
                let expected = vec![1.0, f64::INFINITY, 4.0, 8.0];
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
            let expected = vec![0.2, 0.25, 0.3333333333333333];
            for (got, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((got - exp).abs() < GPU_EPS);
            }
        });
    }

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
            let expected = vec![0.25, 1.0 / 6.0];
            for (got, exp) in t.data.iter().zip(expected.iter()) {
                assert!((got - exp).abs() < EPS);
            }
        });
    }

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
                let expected = vec![(0.5, 0.0), (0.5, 0.0)];
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

    #[test]
    fn ldivide_like_missing_prototype_errors() {
        let lhs = Value::Num(2.0);
        let rhs = Value::Num(4.0);
        let err = ldivide_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
        assert!(err.contains("prototype"), "unexpected error: {err}");
    }

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

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

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
        let ha = provider
            .upload(&view_l)
            .unwrap();
        let hb = provider
            .upload(&view_r)
            .unwrap();
        let gpu = ldivide_gpu_pair(ha, hb).unwrap();
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
