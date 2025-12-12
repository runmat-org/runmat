//! MATLAB-compatible `any` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorOwned};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "any",
        builtin_path = "crate::builtins::math::reduction::any"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "any"
category: "math/reduction"
keywords: ["any", "logical reduction", "omitnan", "all", "gpu"]
summary: "Test whether any element of an array is nonzero with MATLAB-compatible options."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "RunMat uses provider hooks (`reduce_any_dim`, `reduce_any`) when available; otherwise the runtime gathers to the host and evaluates there."
fusion:
  elementwise: false
  reduction: true
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::any::tests"
  integration: "builtins::math::reduction::any::tests::any_gpu_provider_roundtrip"
---

# What does the `any` function do in MATLAB / RunMat?
`any(X)` returns logical `true` wherever at least one element of the requested slice of `X` is nonzero.
When you omit the dimension, the reduction runs along the first non-singleton axis, mirroring MATLAB.

## How does the `any` function behave in MATLAB / RunMat?
- Works with logical, numeric, complex, and character arrays; other types raise a descriptive error.
- Accepts `any(X, dim)` to reduce along a single dimension or `any(X, vecdim)` to collapse multiple axes at once.
- `any(X, 'all')` flattens the entire array into a single logical scalar.
- `any(___, 'omitnan')` ignores `NaN` values (including complex parts) when deciding whether a slice contains nonzero content.
- `any(___, 'includenan')` (default) treats `NaN` as logical `true`, matching MATLAB behaviour.
- Empty dimensions yield logical zeros with MATLAB-compatible shapes; empty arrays reduced with `'all'` return `false`.
- Results are always host-resident logical scalars or logical arrays, even when the input tensor lives on the GPU, because the runtime copies the compact output back to the CPU.

## `any` Function GPU Execution Behaviour
RunMat Accelerate keeps inputs resident on the GPU whenever possible. Providers that expose
`reduce_any_dim` (and optionally `reduce_any`) perform the OR-reduction on device buffers, and
the runtime then downloads the tiny logical result back to the CPU (every `any` call returns a
host logical array). When those hooks are missing, RunMat gathers the input tensor and evaluates
the reduction on the host instead, preserving MATLAB behaviour in all cases.

## Examples of using the `any` function in MATLAB / RunMat

### Checking if any column in a matrix is nonzero

```matlab
A = [0 2 0; 0 0 0];
colHasData = any(A);
```

Expected output:

```matlab
colHasData = [0 1 0];
```

### Detecting whether any row contains a nonzero entry

```matlab
B = [0 4 0; 1 0 0; 0 0 0];
rowHasData = any(B, 2);
```

Expected output:

```matlab
rowHasData = [1; 1; 0];
```

### Reducing across multiple dimensions with `vecdim`

```matlab
C = reshape(1:24, [3 4 2]);
hasValues = any(C > 20, [1 2]);
```

Expected output:

```matlab
hasValues = [0 1];
```

### Checking all elements with the `'all'` option

```matlab
D = [0 0; 0 5];
anyNonZero = any(D, 'all');
```

Expected output:

```matlab
anyNonZero = true;
```

### Ignoring `NaN` values when probing slices

```matlab
E = [NaN 0 0; 0 0 0];
withNaN = any(E);              % returns [1 0 0]
ignoringNaN = any(E, 'omitnan'); % returns [0 0 0]
```

Expected output:

```matlab
withNaN = [1 0 0];
ignoringNaN = [0 0 0];
```

### Running `any` on GPU arrays with automatic fallback

```matlab
G = gpuArray([0 1 0; 0 0 0]);
gpuResult = any(G, 2);
hostResult = gather(gpuResult);
```

Expected output:

```matlab
hostResult = [1; 0];
```

### Evaluating `any` on character data

```matlab
chars = ['a' 0 'c'];
hasPrintable = any(chars);
```

Expected output:

```matlab
hasPrintable = [1 0 1];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do **not** need to call `gpuArray` manually. The fusion planner keeps GPU-resident
inputs on the device and only gathers the small logical results that `any` produces. If your
workload already uses explicit `gpuArray`/`gather` calls for MATLAB compatibility, RunMat honours
them and still produces correct logical outputs.

## FAQ

### When should I use the `any` function?
Use `any` whenever you need to know if any element of an array, row, column, or sub-array is nonzero or logical `true`.

### Does `any` always return logical values?
Yes. Results are `logical` scalars or logical arrays even when the computation involves GPU inputs.

### How do I test a specific dimension?
Pass the dimension as the second argument (for example, `any(X, 2)` reduces each row). Provide a vector such as `[1 3]` to collapse multiple axes.

### What does `any(X, 'all')` compute?
It reduces across every dimension of `X` and returns a single logical scalar indicating whether any element of the entire array is nonzero.

### How are `NaN` values handled?
By default they count as nonzero (`'includenan'`). Add `'omitnan'` to ignore them; if every element in a slice is `NaN`, the result becomes `false`.

### Does `any` work with complex numbers?
Yes. Complex values are considered nonzero when either the real or imaginary component is nonzero. Complex values containing `NaN` obey the `'omitnan'`/`'includenan'` rules.

### Can I apply `any` to character arrays?
Yes. Characters compare against their Unicode code points; zero-valued code points are treated as `false`, and everything else is `true`.

### What happens with empty inputs?
Empty reductions follow MATLAB semantics: dimensions of length zero produce logical zeros, while `any(X, 'all')` over an empty array evaluates to `false`.

### How do GPU backends accelerate `any`?
Providers may expose specialised OR-reduction kernels (`reduce_any_dim`, `reduce_any`) or use `fused_reduction` to remain on the device. When such hooks are absent, RunMat downloads the small output and computes on the host.

## See Also
[sum](./sum), [prod](./prod), [mean](./mean), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `any` function is available at: [`crates/runmat-runtime/src/builtins/math/reduction/any.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/any.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::any")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "any",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        crate::builtins::common::spec::ProviderHook::Reduction {
            name: "reduce_any_dim",
        },
        crate::builtins::common::spec::ProviderHook::Reduction {
            name: "reduce_any",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers may execute device-side OR reductions; runtimes gather to host when hooks are unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::any")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "any",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!(
                "accumulator = max(accumulator, select(0.0, 1.0, ({input} != 0.0) || ({input} != {input})));"
            ))
        },
    }),
    emits_nan: false,
    notes: "Fusion reductions short-circuit on the first nonzero (or NaN) element; providers can substitute native kernels.",
};

#[runtime_builtin(
    name = "any",
    category = "math/reduction",
    summary = "Test whether any element of an array is nonzero with MATLAB-compatible options.",
    keywords = "any,logical,reduction,omitnan,all,gpu",
    accel = "reduction",
    builtin_path = "crate::builtins::math::reduction::any"
)]
fn any_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (spec, nan_mode) = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => any_gpu(handle, spec, nan_mode),
        other => any_host(other, spec, nan_mode),
    }
}

fn any_host(value: Value, spec: ReductionSpec, nan_mode: ReductionNaN) -> Result<Value, String> {
    let truth = TruthTensor::from_value("any", value)?;
    let reduced = apply_reduction(truth, spec, nan_mode)?;
    reduced.into_value()
}

fn any_gpu(
    handle: GpuTensorHandle,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return gpu_fallback(handle, spec, nan_mode),
    };
    if matches!(nan_mode, ReductionNaN::Omit) {
        // Logical reductions return host results; gather to ensure exact omitnan semantics
        return gpu_fallback(handle, spec, nan_mode);
    }
    match try_any_gpu(provider, &handle, &spec, nan_mode)? {
        Some(host) => logical_from_host(host),
        None => gpu_fallback(handle, spec, nan_mode),
    }
}

fn gpu_fallback(
    handle: GpuTensorHandle,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> Result<Value, String> {
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    any_host(Value::Tensor(tensor), spec, nan_mode)
}

fn try_any_gpu(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    spec: &ReductionSpec,
    nan_mode: ReductionNaN,
) -> Result<Option<HostTensorOwned>, String> {
    let omit_nan = matches!(nan_mode, ReductionNaN::Omit);

    // For omitnan, prefer explicit truth-mask path to ensure semantics match host exactly
    if !omit_nan {
        if let ReductionSpec::All = spec {
            if let Ok(tmp) = provider.reduce_any(handle, omit_nan) {
                let host = provider.download(&tmp).map_err(|e| e.to_string())?;
                let _ = provider.free(&tmp);
                return Ok(Some(host));
            }
        }
    }

    reduce_dims_gpu(provider, handle, spec, omit_nan)
}

fn reduce_dims_gpu(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    spec: &ReductionSpec,
    omit_nan: bool,
) -> Result<Option<HostTensorOwned>, String> {
    let mut dims = dims_from_spec(spec, &handle.shape);
    if dims.is_empty() {
        return Ok(None);
    }
    dims.sort_unstable();
    dims.dedup();

    let mut current = handle.clone();
    let mut current_owned = false;
    let mut intermediates: Vec<GpuTensorHandle> = Vec::new();

    for dim in dims {
        if dim == 0 {
            if current_owned {
                let _ = provider.free(&current);
            }
            for owned in intermediates {
                let _ = provider.free(&owned);
            }
            return Ok(None);
        }
        let axis = dim - 1;
        if axis >= current.shape.len() {
            if current_owned {
                let _ = provider.free(&current);
            }
            for owned in intermediates {
                let _ = provider.free(&owned);
            }
            return Ok(None);
        }
        let next = if omit_nan {
            // Build truth mask: (!isnan(current)) && (current != 0)
            let zeros = provider.zeros_like(&current).map_err(|e| e.to_string())?;
            let ne_zero = provider
                .elem_ne(&current, &zeros)
                .map_err(|e| e.to_string())?;
            let _ = provider.free(&zeros);
            let is_nan = provider
                .logical_isnan(&current)
                .map_err(|e| e.to_string())?;
            let not_nan = provider.logical_not(&is_nan).map_err(|e| e.to_string())?;
            let _ = provider.free(&is_nan);
            let truth = provider
                .logical_and(&ne_zero, &not_nan)
                .map_err(|e| e.to_string())?;
            let _ = provider.free(&ne_zero);
            let _ = provider.free(&not_nan);

            // Sum along axis to count any true; then threshold > 0
            let summed = provider
                .reduce_sum_dim(&truth, axis)
                .map_err(|e| e.to_string())?;
            let _ = provider.free(&truth);
            let zeros_out = provider.zeros_like(&summed).map_err(|e| e.to_string())?;
            let gt_zero = provider
                .elem_gt(&summed, &zeros_out)
                .map_err(|e| e.to_string())?;
            let _ = provider.free(&summed);
            let _ = provider.free(&zeros_out);
            Ok(gt_zero)
        } else {
            provider
                .reduce_any_dim(&current, axis, omit_nan)
                .map_err(|e| e.to_string())
        };
        match next {
            Ok(new_handle) => {
                if current_owned {
                    intermediates.push(current.clone());
                }
                current = new_handle;
                current_owned = true;
            }
            Err(_) => {
                if current_owned {
                    let _ = provider.free(&current);
                }
                for owned in intermediates {
                    let _ = provider.free(&owned);
                }
                return Ok(None);
            }
        }
    }

    if !current_owned {
        return Ok(None);
    }

    let host = provider.download(&current).map_err(|e| e.to_string())?;
    let _ = provider.free(&current);
    for owned in intermediates {
        let _ = provider.free(&owned);
    }
    Ok(Some(host))
}

fn logical_from_host(host: HostTensorOwned) -> Result<Value, String> {
    if host.data.len() == 1 {
        return Ok(Value::Bool(host.data[0] != 0.0));
    }
    let shape = if host.shape.is_empty() {
        if host.data.is_empty() {
            Vec::new()
        } else {
            vec![host.data.len()]
        }
    } else {
        host.shape.clone()
    };
    let logical_data: Vec<u8> = host
        .data
        .into_iter()
        .map(|v| if v != 0.0 { 1 } else { 0 })
        .collect();
    LogicalArray::new(logical_data, shape)
        .map(Value::LogicalArray)
        .map_err(|e| format!("any: {e}"))
}

fn dims_from_spec(spec: &ReductionSpec, shape: &[usize]) -> Vec<usize> {
    match spec {
        ReductionSpec::Default => vec![default_dimension_from_shape(shape)],
        ReductionSpec::Dim(dim) => vec![*dim],
        ReductionSpec::VecDim(dims) => {
            let mut sorted = dims.clone();
            sorted.sort_unstable();
            sorted.dedup();
            sorted
        }
        ReductionSpec::All => {
            if shape.is_empty() {
                vec![1]
            } else {
                (1..=shape.len()).collect()
            }
        }
    }
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

#[derive(Debug, Clone)]
enum ReductionSpec {
    Default,
    Dim(usize),
    VecDim(Vec<usize>),
    All,
}

#[derive(Clone)]
struct TruthTensor {
    shape: Vec<usize>,
    data: Vec<TruthValue>,
}

#[derive(Clone, Copy)]
struct TruthValue {
    truthy: bool,
    has_nan: bool,
}

impl TruthValue {
    fn from_bool(truthy: bool) -> Self {
        Self {
            truthy,
            has_nan: false,
        }
    }
}

impl TruthTensor {
    fn from_value(name: &str, value: Value) -> Result<Self, String> {
        match value {
            Value::Tensor(t) => Ok(Self::from_tensor(t)),
            Value::LogicalArray(logical) => Ok(Self::from_logical(logical)),
            Value::Num(n) => Ok(Self::from_tensor(
                Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("{name}: {e}"))?,
            )),
            Value::Int(i) => Ok(Self::from_tensor(
                Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("{name}: {e}"))?,
            )),
            Value::Bool(b) => Ok(Self {
                shape: vec![1, 1],
                data: vec![TruthValue {
                    truthy: b,
                    has_nan: false,
                }],
            }),
            Value::Complex(re, im) => Ok(Self {
                shape: vec![1, 1],
                data: vec![TruthValue {
                    truthy: if re.is_nan() || im.is_nan() {
                        true
                    } else {
                        re != 0.0 || im != 0.0
                    },
                    has_nan: re.is_nan() || im.is_nan(),
                }],
            }),
            Value::ComplexTensor(ct) => Ok(Self::from_complex_tensor(ct)),
            Value::CharArray(ca) => Ok(Self::from_char_array(ca)),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                Ok(Self::from_tensor(tensor))
            }
            other => Err(format!(
                "{name}: unsupported input type {:?}; expected numeric, logical, complex, or char data",
                other
            )),
        }
    }

    fn from_tensor(tensor: Tensor) -> Self {
        let shape = if tensor.shape.is_empty() {
            if tensor.data.is_empty() {
                Vec::new()
            } else {
                vec![tensor.data.len()]
            }
        } else {
            tensor.shape.clone()
        };
        let data = tensor
            .data
            .iter()
            .map(|&v| TruthValue {
                truthy: if v.is_nan() { true } else { v != 0.0 },
                has_nan: v.is_nan(),
            })
            .collect();
        TruthTensor { shape, data }
    }

    fn from_logical(logical: LogicalArray) -> Self {
        let data = logical
            .data
            .iter()
            .map(|&b| TruthValue {
                truthy: b != 0,
                has_nan: false,
            })
            .collect();
        TruthTensor {
            shape: logical.shape.clone(),
            data,
        }
    }

    fn from_complex_tensor(ct: ComplexTensor) -> Self {
        let data = ct
            .data
            .iter()
            .map(|&(re, im)| TruthValue {
                truthy: if re.is_nan() || im.is_nan() {
                    true
                } else {
                    re != 0.0 || im != 0.0
                },
                has_nan: re.is_nan() || im.is_nan(),
            })
            .collect();
        TruthTensor {
            shape: ct.shape.clone(),
            data,
        }
    }

    fn from_char_array(ca: CharArray) -> Self {
        let data = ca
            .data
            .iter()
            .map(|&ch| TruthValue {
                truthy: (ch as u32) != 0,
                has_nan: false,
            })
            .collect();
        TruthTensor {
            shape: vec![ca.rows, ca.cols],
            data,
        }
    }

    fn reduce_dim(&self, dim: usize, nan_mode: ReductionNaN) -> Result<Self, String> {
        if dim == 0 {
            return Err("any: dimension must be >= 1".to_string());
        }
        if self.shape.is_empty() {
            let truth = self.data.first().copied().unwrap_or(TruthValue {
                truthy: false,
                has_nan: false,
            });
            let truthy = match nan_mode {
                ReductionNaN::Include => truth.truthy || truth.has_nan,
                ReductionNaN::Omit => truth.truthy && !truth.has_nan,
            };
            return Ok(TruthTensor {
                shape: vec![1, 1],
                data: vec![TruthValue::from_bool(truthy)],
            });
        }
        if dim > self.shape.len() {
            return Ok(self.clone());
        }
        let axis = dim - 1;
        let reduce_len = self.shape[axis];
        let stride_before = product(&self.shape[..axis]);
        let stride_after = product(&self.shape[axis + 1..]);
        let mut out_shape = self.shape.clone();
        out_shape[axis] = 1;
        let mut out = Vec::with_capacity(stride_before.saturating_mul(stride_after));

        if stride_before == 0 || stride_after == 0 {
            return Ok(TruthTensor {
                shape: out_shape,
                data: out,
            });
        }

        for after in 0..stride_after {
            for before in 0..stride_before {
                let mut any_true = false;
                for k in 0..reduce_len {
                    let idx = before + k * stride_before + after * stride_before * reduce_len;
                    if let Some(value) = self.data.get(idx) {
                        match nan_mode {
                            ReductionNaN::Include => {
                                if value.has_nan || value.truthy {
                                    any_true = true;
                                    break;
                                }
                            }
                            ReductionNaN::Omit => {
                                if value.truthy && !value.has_nan {
                                    any_true = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                out.push(TruthValue::from_bool(any_true));
            }
        }

        Ok(TruthTensor {
            shape: out_shape,
            data: out,
        })
    }

    fn into_value(self) -> Result<Value, String> {
        if self.data.len() == 1 {
            return Ok(Value::Bool(self.data[0].truthy));
        }
        let shape = if self.shape.is_empty() {
            if self.data.is_empty() {
                Vec::new()
            } else {
                vec![self.data.len()]
            }
        } else {
            self.shape
        };
        let logical_data: Vec<u8> = self
            .data
            .into_iter()
            .map(|value| if value.truthy { 1 } else { 0 })
            .collect();
        LogicalArray::new(logical_data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| format!("any: {e}"))
    }
}

fn apply_reduction(
    tensor: TruthTensor,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> Result<TruthTensor, String> {
    match spec {
        ReductionSpec::Default => {
            tensor.reduce_dim(default_dimension_from_shape(&tensor.shape), nan_mode)
        }
        ReductionSpec::Dim(dim) => tensor.reduce_dim(dim, nan_mode),
        ReductionSpec::VecDim(mut dims) => {
            dims.sort_unstable();
            dims.dedup();
            let mut current = tensor;
            for dim in dims {
                current = current.reduce_dim(dim, nan_mode)?;
            }
            Ok(current)
        }
        ReductionSpec::All => {
            let mut current = tensor;
            if current.shape.is_empty() {
                current = current.reduce_dim(1, nan_mode)?;
            } else {
                for dim in 1..=current.shape.len() {
                    current = current.reduce_dim(dim, nan_mode)?;
                }
            }
            Ok(current)
        }
    }
}

fn parse_arguments(args: &[Value]) -> Result<(ReductionSpec, ReductionNaN), String> {
    let mut spec = ReductionSpec::Default;
    let mut nan_mode = ReductionNaN::Include;

    for arg in args {
        if is_all_token(arg) {
            if !matches!(spec, ReductionSpec::Default) {
                return Err("any: 'all' cannot be combined with dimension arguments".to_string());
            }
            spec = ReductionSpec::All;
            continue;
        }
        if let Some(mode) = parse_nan_mode(arg)? {
            if !matches!(nan_mode, ReductionNaN::Include) {
                return Err("any: multiple NaN handling options specified".to_string());
            }
            nan_mode = mode;
            continue;
        }
        let dims = parse_dimensions(arg)?;
        if dims.is_empty() {
            return Err("any: dimension vector must contain at least one entry".to_string());
        }
        if dims.len() == 1 {
            if matches!(spec, ReductionSpec::Default) {
                spec = ReductionSpec::Dim(dims[0]);
            } else {
                return Err("any: multiple dimension specifications are not supported".to_string());
            }
        } else if matches!(spec, ReductionSpec::Default) {
            spec = ReductionSpec::VecDim(dims);
        } else {
            return Err("any: multiple dimension specifications are not supported".to_string());
        }
    }

    Ok((spec, nan_mode))
}

fn parse_nan_mode(value: &Value) -> Result<Option<ReductionNaN>, String> {
    let Some(text) = extract_text_token(value) else {
        return Ok(None);
    };
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "omitnan" => Ok(Some(ReductionNaN::Omit)),
        "includenan" => Ok(Some(ReductionNaN::Include)),
        _ => Err(format!("any: unknown option '{}'", text.trim())),
    }
}

fn is_all_token(value: &Value) -> bool {
    extract_text_token(value)
        .map(|s| s.trim().eq_ignore_ascii_case("all"))
        .unwrap_or(false)
}

fn parse_dimensions(value: &Value) -> Result<Vec<usize>, String> {
    let tensor = tensor::value_to_tensor(value)?;
    if tensor.data.is_empty() {
        return Ok(Vec::new());
    }
    let mut dims = Vec::new();
    for raw in tensor.data {
        if !raw.is_finite() {
            return Err("any: dimension values must be finite".to_string());
        }
        let rounded = raw.round();
        if (rounded - raw).abs() > f64::EPSILON {
            return Err("any: dimension values must be integers".to_string());
        }
        if rounded < 1.0 {
            return Err("any: dimension values must be >= 1".to_string());
        }
        if rounded > (usize::MAX as f64) {
            return Err("any: dimension value is too large".to_string());
        }
        let dim = rounded as usize;
        if !dims.contains(&dim) {
            dims.push(dim);
        }
    }
    Ok(dims)
}

fn extract_text_token(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, v| acc.saturating_mul(v))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue};

    #[test]
    fn any_matrix_default_dimension() {
        let tensor = Tensor::new(vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let result = any_builtin(Value::Tensor(tensor), Vec::new()).expect("any");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_zero_column_matrix_returns_empty_row() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![2, 0]).unwrap();
        let result = any_builtin(Value::Tensor(tensor), Vec::new()).expect("any");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 0]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_row_dimension() {
        let tensor = Tensor::new(vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = any_builtin(Value::Tensor(tensor), args).expect("any");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_zero_row_matrix_dim_two() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = any_builtin(Value::Tensor(tensor), args).expect("any");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![0, 1]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_vecdim_multiple_axes() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let vecdim = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = any_builtin(Value::Tensor(tensor), vec![Value::Tensor(vecdim)]).expect("any");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1, 2]);
                assert_eq!(out.data, vec![1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_all_option_returns_scalar() {
        let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = any_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("any");
        match result {
            Value::Bool(flag) => assert!(!flag),
            other => panic!("expected logical scalar, got {other:?}"),
        }
    }

    #[test]
    fn any_all_on_empty_returns_false() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = any_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("any");
        match result {
            Value::Bool(flag) => assert!(!flag),
            other => panic!("expected logical scalar, got {other:?}"),
        }
    }

    #[test]
    fn any_handles_nan_modes() {
        let tensor = Tensor::new(vec![f64::NAN, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let includenan = any_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("any");
        match includenan {
            Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0]),
            other => panic!("expected logical array, got {other:?}"),
        }

        let omit =
            any_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("any omit");
        match omit {
            Value::LogicalArray(out) => assert_eq!(out.data, vec![0, 0]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_char_array_support() {
        let chars = CharArray::new("a\0c".chars().collect(), 1, 3).unwrap();
        let result =
            any_builtin(Value::CharArray(chars), vec![Value::Int(IntValue::I32(1))]).expect("any");
        match result {
            Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0, 1]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_includenan_keyword_allowed() {
        let tensor = Tensor::new(vec![f64::NAN, 0.0], vec![2, 1]).unwrap();
        let result =
            any_builtin(Value::Tensor(tensor), vec![Value::from("includenan")]).expect("any");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1]);
                assert_eq!(out.data, vec![1]);
            }
            Value::Bool(flag) => assert!(flag),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_complex_tensor_with_omitnan() {
        let complex = ComplexTensor::new(vec![(0.0, 0.0), (f64::NAN, 0.0)], vec![2, 1]).unwrap();
        let tensor_value = Value::ComplexTensor(complex);
        let omit = any_builtin(tensor_value.clone(), vec![Value::from("omitnan")]).expect("any");
        match omit {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1]);
                assert_eq!(out.data, vec![0]);
            }
            Value::Bool(flag) => assert!(!flag),
            other => panic!("expected logical array, got {other:?}"),
        }
        let include = any_builtin(tensor_value, Vec::new()).expect("any include");
        match include {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1]);
                assert_eq!(out.data, vec![1]);
            }
            Value::Bool(flag) => assert!(flag),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_vecdim_with_omitnan() {
        let mut data = vec![0.0; 8];
        data[7] = f64::NAN;
        let tensor = Tensor::new(data, vec![2, 2, 2]).unwrap();
        let vecdim = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(vecdim), Value::from("omitnan")];
        let result = any_builtin(Value::Tensor(tensor), args).expect("any vecdim omitnan");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1, 2]);
                assert_eq!(out.data, vec![0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn any_all_with_dim_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap();
        let args = vec![Value::from("all"), Value::Int(IntValue::I32(1))];
        let err = any_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert!(err.contains("dimension"), "unexpected error message: {err}");
    }

    #[test]
    fn any_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = any_builtin(Value::GpuTensor(handle), Vec::new()).expect("any");
            match result {
                Value::LogicalArray(out) => {
                    assert_eq!(out.shape, vec![1, 2]);
                    assert_eq!(out.data, vec![1, 0]);
                }
                other => panic!("expected logical array, got {other:?}"),
            }
        });
    }

    #[test]
    fn any_gpu_provider_omitnan_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, 0.0, f64::NAN, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                any_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("any");
            match result {
                Value::LogicalArray(out) => {
                    assert_eq!(out.shape, vec![1, 2]);
                    assert_eq!(out.data, vec![0, 0]);
                }
                other => panic!("expected logical array, got {other:?}"),
            }
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn any_wgpu_default_matches_cpu() {
        let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            )
        }));
        let Ok(reg_result) = init else {
            eprintln!("skipping any_wgpu_default_matches_cpu: wgpu provider panicked during init");
            return;
        };
        if reg_result.is_err() {
            eprintln!("skipping any_wgpu_default_matches_cpu: wgpu provider unavailable");
            return;
        }
        let tensor = Tensor::new(vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let cpu = any_host(
            Value::Tensor(tensor.clone()),
            ReductionSpec::Default,
            ReductionNaN::Include,
        )
        .unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = match runmat_accelerate_api::provider() {
            Some(p) => p,
            None => {
                eprintln!("skipping any_wgpu_default_matches_cpu: provider not registered");
                return;
            }
        };
        let handle = match provider.upload(&view) {
            Ok(h) => h,
            Err(err) => {
                eprintln!("skipping any_wgpu_default_matches_cpu: upload failed: {err}");
                return;
            }
        };
        let gpu = any_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        match (cpu, gpu) {
            (Value::LogicalArray(expected), Value::LogicalArray(actual)) => {
                assert_eq!(expected.shape, actual.shape);
                assert_eq!(expected.data, actual.data);
            }
            _ => panic!("unexpected shapes"),
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn any_wgpu_omitnan_matches_cpu() {
        let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            )
        }));
        let Ok(reg_result) = init else {
            eprintln!("skipping any_wgpu_omitnan_matches_cpu: wgpu provider panicked during init");
            return;
        };
        if reg_result.is_err() {
            eprintln!("skipping any_wgpu_omitnan_matches_cpu: wgpu provider unavailable");
            return;
        }
        let tensor = Tensor::new(vec![f64::NAN, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let cpu = any_host(
            Value::Tensor(tensor.clone()),
            ReductionSpec::Default,
            ReductionNaN::Omit,
        )
        .unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = match runmat_accelerate_api::provider() {
            Some(p) => p,
            None => {
                eprintln!("skipping any_wgpu_omitnan_matches_cpu: provider not registered");
                return;
            }
        };
        let handle = match provider.upload(&view) {
            Ok(h) => h,
            Err(err) => {
                eprintln!("skipping any_wgpu_omitnan_matches_cpu: upload failed: {err}");
                return;
            }
        };
        let gpu = any_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).unwrap();
        match (cpu, gpu) {
            (Value::LogicalArray(expected), Value::LogicalArray(actual)) => {
                assert_eq!(expected.shape, actual.shape);
                assert_eq!(expected.data, actual.data);
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
