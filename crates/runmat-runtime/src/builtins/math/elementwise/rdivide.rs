//! MATLAB-compatible `rdivide` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow};
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "rdivide",
        builtin_path = "crate::builtins::math::elementwise::rdivide"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "rdivide"
category: "math/elementwise"
keywords: ["rdivide", "element-wise division", "./", "gpu", "implicit expansion"]
summary: "Element-wise division A ./ B with MATLAB-compatible implicit expansion, complex support, and GPU fallbacks."
references: ["https://www.mathworks.com/help/matlab/ref/rdivide.html"]
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers provider elem_div/scalar_div/scalar_rdiv hooks; gathers to host for implicit expansion or unsupported operand types."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::rdivide::tests::rdivide_scalar_numbers"
  integration: "builtins::math::elementwise::rdivide::tests::rdivide_row_column_broadcast"
  gpu: "builtins::math::elementwise::rdivide::tests::rdivide_gpu_pair_roundtrip"
  wgpu: "builtins::math::elementwise::rdivide::tests::rdivide_wgpu_matches_cpu_elementwise"
  doc: "builtins::math::elementwise::rdivide::tests::doc_examples_present"
  like_gpu: "builtins::math::elementwise::rdivide::tests::rdivide_like_gpu_prototype_keeps_residency"
  like_host: "builtins::math::elementwise::rdivide::tests::rdivide_like_host_gathers_gpu_value"
  like_complex: "builtins::math::elementwise::rdivide::tests::rdivide_like_complex_prototype_yields_complex"
---

# What does the `rdivide` function do in MATLAB / RunMat?
`rdivide(A, B)` (or the operator form `A ./ B`) divides corresponding elements of `A` and `B`, honouring MATLAB's implicit expansion rules so that scalars and singleton dimensions broadcast automatically.

## How does the `rdivide` function behave in MATLAB / RunMat?
- Supports real, complex, logical, and character inputs; logical and character data are promoted to double precision before division.
- Implicit expansion works across any dimension, provided the non-singleton extents match. Size mismatches raise the standard MATLAB-compatible error.
- Complex operands follow the analytic rule `(a + ib) ./ (c + id) = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)`, matching MATLAB's behaviour for infinities and NaNs.
- Empty dimensions propagate naturally—if the broadcasted shape contains a zero extent, the result is empty with that shape.
- Integer inputs promote to double precision, mirroring the behaviour of other RunMat arithmetic builtins.
- The optional `'like'` prototype makes the output adopt the residency (host or GPU) and complexity characteristics of the prototype, letting you keep implicit-expansion expressions on the device without manual `gpuArray` calls. When the prototype is complex, the result remains a complex host array (complex gpuArray prototypes are not yet supported).

## `rdivide` Function GPU Execution Behaviour
When a gpuArray provider is active:

1. If both operands are gpuArrays with identical shapes, RunMat dispatches to the provider's `elem_div` hook so the entire computation stays on device memory.
2. If one operand is a scalar (host or device) and the other is a gpuArray, the runtime calls `scalar_div` (`tensor ./ scalar`) or `scalar_rdiv` (`scalar ./ tensor`) accordingly.
3. The fusion planner treats `rdivide` as a fusible elementwise node, so adjacent elementwise producers or consumers can execute inside a single WGSL kernel or provider-optimised pipeline, reducing host↔device transfers.
4. Implicit-expansion workloads (for example, combining a row vector with a column vector) or unsupported operand kinds gather transparently to host memory, compute the result with full MATLAB semantics, and return a host tensor. When you request `'like'` with a complex prototype, RunMat gathers to the host, performs the conversion, and returns a complex host value so downstream code still sees MATLAB-compatible types.

## Examples of using the `rdivide` function in MATLAB / RunMat

### Divide two matrices element-wise

```matlab
A = [8 12 18; 2 10 18];
B = [2 3 6; 2 5 9];
Q = rdivide(A, B);
```

Expected output:

```matlab
Q =
    4     4     3
    1     2     2
```

### Divide a matrix by a scalar

```matlab
A = magic(3);
scaled = rdivide(A, 2);
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
ratio = rdivide(col, row);
```

Expected output:

```matlab
ratio =
    0.1    0.05    0.0333
    0.2    0.10    0.0667
    0.3    0.15    0.1000
```

### Divide complex inputs element-wise

```matlab
z1 = [1+2i, 3-4i];
z2 = [2-1i, -1+1i];
quot = rdivide(z1, z2);
```

Expected output:

```matlab
quot =
    0.0 + 1.0i   -3.5 + 0.5i
```

### Divide character codes by a numeric scalar

```matlab
letters = 'ABC';
codes = rdivide(letters, 2);
```

Expected output:

```matlab
codes = [32.5 33 33.5]
```

### Execute `rdivide` directly on gpuArray inputs

```matlab
G1 = gpuArray([10 20 30]);
G2 = gpuArray([2 5 10]);
deviceQuot = rdivide(G1, G2);
result = gather(deviceQuot);
```

Expected output:

```matlab
deviceQuot =
  1x3 gpuArray
     5     4     3
result =
     5     4     3
```

### Keep the result on the GPU with a `'like'` prototype

```matlab
proto = gpuArray.zeros(1, 1);
A = [1 2 3];
B = [2 4 6];
C = rdivide(A, B, 'like', proto);  % stays on the GPU for downstream work
```

Expected output:

```matlab
C =
  1x3 gpuArray
      0.5    0.5    0.5
```

## GPU residency in RunMat (Do I need `gpuArray`?)
RunMat's auto-offload planner keeps tensors on the GPU whenever fused expressions benefit from device execution. Explicit `gpuArray` / `gather` calls are still supported for MATLAB code that manages residency manually. When the active provider lacks the kernels needed for a particular call (for example, implicit expansion between gpuArrays of different shapes), RunMat gathers back to the host, computes the MATLAB-accurate result, and resumes execution seamlessly.

## FAQ

### Does `rdivide` support MATLAB implicit expansion?
Yes. Any singleton dimensions expand automatically. If a dimension has incompatible non-singleton extents, `rdivide` raises the standard size-mismatch error.

### What numeric type does `rdivide` return?
Results are double precision for real inputs and complex double when either operand is complex. Logical and character inputs are promoted to double before division.

### How does `rdivide` handle division by zero?
RunMat follows IEEE rules: `finite ./ 0` produces signed infinity, while `0 ./ 0` yields `NaN`. Complex results follow MATLAB's analytic continuation rules.

### Can I divide gpuArrays by host scalars?
Yes. RunMat keeps the computation on the GPU when the scalar is numeric. For other host operand types, the runtime gathers the gpuArray and computes on the CPU.

### Does `rdivide` preserve gpuArray residency after a fallback?
When a fallback occurs (for example, implicit expansion that the provider does not implement), the current result remains on the host. Subsequent operations may move it back to the GPU when auto-offload decides it is profitable.

### How can I force the result to stay on the GPU?
Provide a `'like'` prototype: `rdivide(A, B, 'like', gpuArray.zeros(1, 1))` keeps the result on the device even if one of the inputs originated on the host. Complex prototypes are honoured on the host today; supply a real gpuArray prototype when you need the result to remain device-resident.

### How are empty arrays handled?
Empty dimensions propagate. If either operand has an extent of zero in the broadcasted shape, the result is empty with the broadcasted dimensions.

### Are integer inputs supported?
Yes. Integers promote to double precision during the division, matching other RunMat arithmetic builtins.

### Can I mix complex and real operands?
Absolutely. The result is complex, with broadcasting rules identical to MATLAB.

### What about string arrays?
String arrays are not numeric and therefore raise an error when passed to `rdivide`.

## See Also
[times](./times), [ldivide](./ldivide), [mrdivide](./mrdivide), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/math/elementwise/rdivide.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/rdivide.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::rdivide")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rdivide",
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
    notes: "Uses elem_div when shapes match, scalar_div for tensor ./ scalar, and scalar_rdiv for scalar ./ tensor; implicit expansion or unsupported operand kinds fall back to the CPU before 'like' prototypes are honoured.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::rdivide")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rdivide",
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
            Ok(format!("({lhs} / {rhs})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a plain quotient; providers can override with specialised kernels when desirable.",
};

const BUILTIN_NAME: &str = "rdivide";

fn builtin_error(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message).with_builtin(BUILTIN_NAME).build().into()
}

#[runtime_builtin(
    name = "rdivide",
    category = "math/elementwise",
    summary = "Element-wise division with MATLAB-compatible implicit expansion.",
    keywords = "rdivide,element-wise division,gpu,./",
    accel = "elementwise",
    builtin_path = "crate::builtins::math::elementwise::rdivide"
)]
fn rdivide_builtin(lhs: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    let base = match (lhs, rhs) {
        (Value::GpuTensor(la), Value::GpuTensor(lb)) => rdivide_gpu_pair(la, lb),
        (Value::GpuTensor(la), rhs) => rdivide_gpu_host_left(la, rhs),
        (lhs, Value::GpuTensor(rb)) => rdivide_gpu_host_right(lhs, rb),
        (lhs, rhs) => rdivide_host(lhs, rhs),
    }?;
    apply_output_template(base, &template)
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
            return Err(builtin_error("rdivide: expected prototype after 'like'"));
        }
        return Err(builtin_error(
            "rdivide: unsupported option; only 'like' is accepted",
        ));
    }
    if args.len() == 2 {
        if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
            return Ok(OutputTemplate::Like(args[1].clone()));
        }
        return Err(builtin_error(
            "rdivide: unsupported option; only 'like' is accepted",
        ));
    }
    Err(builtin_error("rdivide: too many input arguments"))
}

fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
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

fn apply_like_template(value: Value, prototype: &Value) -> BuiltinResult<Value> {
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

fn ensure_device(value: Value, device: DevicePreference) -> BuiltinResult<Value> {
    match device {
        DevicePreference::Host => convert_to_host_like(value),
        DevicePreference::Gpu => convert_to_gpu(value),
    }
}

fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = value {
        let temp = Value::GpuTensor(handle);
        gpu_helpers::gather_value(&temp)
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))
            .map_err(|e| builtin_error(format!("rdivide: {e}")))
    } else {
        Ok(value)
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err(builtin_error(
            "rdivide: GPU output requested via 'like' but no acceleration provider is active",
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
                .map_err(|e| builtin_error(format!("rdivide: failed to upload GPU result: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("rdivide: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor =
                tensor::logical_to_tensor(&logical).map_err(|e| builtin_error(format!("rdivide: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(builtin_error(
            "rdivide: GPU prototypes for 'like' only support real numeric outputs",
        )),
        Value::String(_) | Value::StringArray(_) | Value::Cell(_) | Value::Struct(_) => Err(
            builtin_error("rdivide: unsupported prototype conversion to GPU output"),
        ),
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(builtin_error(
            "rdivide: unsupported prototype conversion to GPU output",
        )),
    }
}

fn analyse_like_prototype(proto: &Value) -> BuiltinResult<LikeAnalysis> {
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

fn gather_like_prototype(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value(value)
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))
            .map_err(|e| builtin_error(format!("rdivide: {e}"))),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_) => Ok(value.clone()),
        _ => Err(builtin_error(format!(
            "rdivide: unsupported prototype for 'like' ({value:?})"
        ))),
    }
}

fn real_to_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(t) => {
            let data: Vec<(f64, f64)> = t.data.iter().map(|&v| (v, 0.0)).collect();
            let tensor =
                ComplexTensor::new(data, t.shape.clone()).map_err(|e| builtin_error(format!("rdivide: {e}")))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor =
                tensor::logical_to_tensor(&logical).map_err(|e| builtin_error(format!("rdivide: {e}")))?;
            real_to_complex(Value::Tensor(tensor))
        }
        Value::CharArray(chars) => {
            let tensor = char_array_to_tensor(&chars)?;
            real_to_complex(Value::Tensor(tensor))
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value(&Value::GpuTensor(handle.clone()))
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            real_to_complex(gathered)
        }
        other => Err(builtin_error(format!(
            "rdivide: cannot convert value {other:?} to complex output"
        ))),
    }
}

fn rdivide_gpu_pair(lhs: GpuTensorHandle, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if lhs.shape == rhs.shape {
            if let Ok(handle) = provider.elem_div(&lhs, &rhs) {
                return Ok(Value::GpuTensor(handle));
            }
        }
        // Try N-D implicit expansion on device using repmat + elem_div
        if let Some((out_shape, reps_l, reps_r)) = broadcast_reps(&lhs.shape, &rhs.shape) {
            let made_left = reps_l.iter().any(|&r| r != 1);
            let made_right = reps_r.iter().any(|&r| r != 1);
            let left_expanded = if made_left {
                provider
                    .repmat(&lhs, &reps_l)
                    .map_err(|e| builtin_error(format!("rdivide: {e}")))?
            } else {
                lhs.clone()
            };
            let right_expanded = if made_right {
                provider
                    .repmat(&rhs, &reps_r)
                    .map_err(|e| builtin_error(format!("rdivide: {e}")))?
            } else {
                rhs.clone()
            };
            let result = provider
                .elem_div(&left_expanded, &right_expanded)
                .map_err(|e| builtin_error(format!("rdivide: {e}")));
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
        if is_scalar_shape(&rhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&rhs)? {
                if let Ok(handle) = provider.scalar_div(&lhs, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
        if is_scalar_shape(&lhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&lhs)? {
                if let Ok(handle) = provider.scalar_rdiv(&rhs, scalar) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
        }
    }
    let left = gpu_helpers::gather_tensor(&lhs)
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::gather_tensor(&rhs)
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    rdivide_host(Value::Tensor(left), Value::Tensor(right))
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

fn rdivide_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&rhs)? {
            if let Ok(handle) = provider.scalar_div(&lhs, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let host_lhs = gpu_helpers::gather_tensor(&lhs)
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    rdivide_host(Value::Tensor(host_lhs), rhs)
}

fn rdivide_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&lhs)? {
            if let Ok(handle) = provider.scalar_rdiv(&rhs, scalar) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    let host_rhs = gpu_helpers::gather_tensor(&rhs)
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    rdivide_host(lhs, Value::Tensor(host_rhs))
}

fn rdivide_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (RdivideOperand::Real(a), RdivideOperand::Real(b)) => rdivide_real_real(&a, &b),
        (RdivideOperand::Complex(a), RdivideOperand::Complex(b)) => rdivide_complex_complex(&a, &b),
        (RdivideOperand::Complex(a), RdivideOperand::Real(b)) => rdivide_complex_real(&a, &b),
        (RdivideOperand::Real(a), RdivideOperand::Complex(b)) => rdivide_real_complex(&a, &b),
    }
}

fn rdivide_real_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Value> {
    let plan =
        BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| builtin_error(format!("rdivide: {err}")))?;
    if plan.is_empty() {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("rdivide: {e}")))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut out = vec![0.0f64; plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        out[out_idx] = lhs.data[idx_lhs] / rhs.data[idx_rhs];
    }
    let tensor =
        Tensor::new(out, plan.output_shape().to_vec()).map_err(|e| builtin_error(format!("rdivide: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn rdivide_complex_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan =
        BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| builtin_error(format!("rdivide: {err}")))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("rdivide: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        let quotient = Complex64::new(ar, ai) / Complex64::new(br, bi);
        out[out_idx] = (quotient.re, quotient.im);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("rdivide: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn rdivide_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> BuiltinResult<Value> {
    let plan =
        BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| builtin_error(format!("rdivide: {err}")))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("rdivide: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let (ar, ai) = lhs.data[idx_lhs];
        let scalar = rhs.data[idx_rhs];
        let quotient = Complex64::new(ar, ai) / Complex64::new(scalar, 0.0);
        out[out_idx] = (quotient.re, quotient.im);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("rdivide: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn rdivide_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan =
        BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| builtin_error(format!("rdivide: {err}")))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("rdivide: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut out = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_lhs, idx_rhs) in plan.iter() {
        let scalar = lhs.data[idx_lhs];
        let (br, bi) = rhs.data[idx_rhs];
        let quotient = Complex64::new(scalar, 0.0) / Complex64::new(br, bi);
        out[out_idx] = (quotient.re, quotient.im);
    }
    let tensor = ComplexTensor::new(out, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("rdivide: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

enum RdivideOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> BuiltinResult<RdivideOperand> {
    match value {
        Value::Tensor(t) => Ok(RdivideOperand::Real(t)),
        Value::Num(n) => Ok(RdivideOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("rdivide: {e}")))?,
        )),
        Value::Int(i) => Ok(RdivideOperand::Real(
            Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| builtin_error(format!("rdivide: {e}")))?,
        )),
        Value::Bool(b) => Ok(RdivideOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("rdivide: {e}")))?,
        )),
        Value::LogicalArray(logical) => Ok(RdivideOperand::Real(
            tensor::logical_to_tensor(&logical).map_err(|e| builtin_error(format!("rdivide: {e}")))?,
        )),
        Value::CharArray(chars) => Ok(RdivideOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(RdivideOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| builtin_error(format!("rdivide: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(RdivideOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(builtin_error("rdivide: internal error converting GPU value")),
        other => Err(builtin_error(format!(
            "rdivide: unsupported operand type {:?}; expected numeric or logical data",
            other
        ))),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|e| builtin_error(format!("rdivide: {e}")))
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

fn gpu_scalar_value(handle: &GpuTensorHandle) -> BuiltinResult<Option<f64>> {
    if !is_scalar_shape(&handle.shape) {
        return Ok(None);
    }
    let tensor = gpu_helpers::gather_tensor(handle)
        .map_err(|e| builtin_error(format!("rdivide: {e}")))?;
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
    fn rdivide_scalar_numbers() {
        let result =
            rdivide_builtin(Value::Num(7.0), Value::Num(2.0), Vec::new()).expect("rdivide");
        match result {
            Value::Num(v) => assert!((v - 3.5).abs() < EPS),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_matrix_scalar() {
        let tensor = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result =
            rdivide_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("rdivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_row_column_broadcast() {
        let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::new(vec![10.0, 20.0, 40.0], vec![1, 3]).unwrap();
        let result = rdivide_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
            .expect("broadcast rdivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = [0.1, 0.2, 0.3, 0.05, 0.10, 0.15, 0.025, 0.05, 0.075];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_complex_inputs() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
        let result = rdivide_builtin(
            Value::ComplexTensor(lhs),
            Value::ComplexTensor(rhs),
            Vec::new(),
        )
        .expect("complex rdivide");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(0.0, 1.0), (-3.5, 0.5)];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < 1e-10 && (got.1 - exp.1).abs() < 1e-10);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_division_by_zero() {
        let tensor = Tensor::new(vec![0.0, 1.0, -2.0], vec![3, 1]).unwrap();
        let result =
            rdivide_builtin(Value::Tensor(tensor), Value::Num(0.0), Vec::new()).expect("rdivide");
        match result {
            Value::Tensor(t) => {
                assert!(t.data[0].is_nan());
                assert!(t.data[1].is_infinite());
                assert!(t.data[2].is_infinite());
                assert!(t.data[2].is_sign_negative());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_logical_inputs_promote() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![2, 2]).unwrap();
        let result = rdivide_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("logical rdivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [1.0, 0.0, 0.25, 0.125];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < EPS);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_char_array_promotes_to_double() {
        let chars = CharArray::new_row("AB");
        let result =
            rdivide_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new()).expect("rdivide");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - 32.5).abs() < EPS);
                assert!((t.data[1] - 33.0).abs() < EPS);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_gpu_pair_roundtrip() {
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
            let result = rdivide_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb), Vec::new())
                .expect("gpu rdivide");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            assert_eq!(gathered.data, vec![5.0, 4.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_like_gpu_prototype_keeps_residency() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload proto");
            let result = rdivide_builtin(
                Value::Tensor(lhs),
                Value::Tensor(rhs),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("rdivide like gpu");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.data, vec![2.0, 2.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_like_host_gathers_gpu_value() {
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
            let result = rdivide_builtin(
                Value::GpuTensor(ha),
                Value::GpuTensor(hb),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("rdivide like host");
            let Value::Tensor(t) = result else {
                panic!("expected tensor result after host gather");
            };
            assert_eq!(t.shape, vec![2, 1]);
            assert_eq!(t.data, vec![4.0, 6.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_like_complex_prototype_yields_complex() {
        let lhs = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = rdivide_builtin(
            Value::Tensor(lhs),
            Value::Tensor(rhs),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("rdivide like complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                let expected = [(2.0, 0.0), (2.0, 0.0)];
                for (got, exp) in ct.data.iter().zip(expected.iter()) {
                    assert!((got.0 - exp.0).abs() < EPS);
                    assert!((got.1 - exp.1).abs() < EPS);
                }
            }
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < EPS && im.abs() < EPS);
            }
            other => panic!("expected complex output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_like_missing_prototype_errors() {
        let lhs = Value::Num(2.0);
        let rhs = Value::Num(4.0);
        let err = rdivide_builtin(lhs, rhs, vec![Value::from("like")]).unwrap_err();
        match err {
            RuntimeControlFlow::Error(err) => {
                assert!(err.message().contains("prototype"), "unexpected error: {err}");
            }
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspend"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_like_keyword_char_array() {
        test_support::with_test_provider(|provider| {
            let keyword = CharArray::new_row("LIKE");
            let lhs = Value::Num(2.0);
            let rhs = Value::Num(5.0);
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = rdivide_builtin(
                lhs,
                rhs,
                vec![Value::CharArray(keyword), Value::GpuTensor(proto)],
            )
            .expect("rdivide like char");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.data, vec![0.4]);
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
    fn rdivide_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let cpu = rdivide_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
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
        let gpu = rdivide_gpu_pair(ha, hb).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(t) => {
                assert_eq!(gathered.data.len(), t.data.len());
                for (ga, ca) in gathered.data.iter().zip(t.data.iter()) {
                    assert!((ga - ca).abs() < EPS);
                }
            }
            Value::Num(n) => assert_eq!(gathered.data, vec![n]),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rdivide_int_inputs_promote_to_double() {
        let lhs = Value::Int(IntValue::I32(6));
        let rhs = Value::Int(IntValue::I32(4));
        let result = rdivide_builtin(lhs, rhs, Vec::new()).expect("rdivide");
        match result {
            Value::Num(v) => assert!((v - 1.5).abs() < EPS),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }
}
