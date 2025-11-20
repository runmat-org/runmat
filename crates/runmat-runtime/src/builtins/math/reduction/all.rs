//! MATLAB-compatible `all` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorOwned};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "all"
category: "math/reduction"
keywords: ["all", "logical reduction", "omitnan", "gpu", "vectorization"]
summary: "Check whether every element of an array slice is nonzero with MATLAB-compatible semantics."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "RunMat uses provider hooks (`reduce_all_dim`, `reduce_all`) when available; otherwise the runtime gathers to the host and evaluates there."
fusion:
  elementwise: false
  reduction: true
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::all::tests"
  integration: "builtins::math::reduction::all::tests::all_gpu_provider_roundtrip"
---

# What does the `all` function do in MATLAB / RunMat?
`all(X)` returns logical `true` when every element of the requested slice of `X` is nonzero. When you omit the dimension, the reduction runs along the first non-singleton axis, mirroring MATLAB.

## How does the `all` function behave in MATLAB / RunMat?
- Works with logical, numeric, complex, and character arrays; other types raise a descriptive error.
- Accepts `all(X, dim)` to reduce along a single dimension or `all(X, vecdim)` to collapse multiple axes at once.
- `all(X, 'all')` flattens the entire array into a single logical scalar.
- `all(___, 'omitnan')` ignores `NaN` values (including complex parts) when deciding whether a slice contains nonzero content; empty or all-`NaN` slices evaluate to `true`.
- `all(___, 'includenan')` (default) treats `NaN` as logical `true`, matching MATLAB behaviour.
- Empty dimensions yield logical ones with MATLAB-compatible shapes; empty arrays reduced with `'all'` return `true`.
- Results are always host-resident logical scalars or logical arrays, even when the input tensor lives on the GPU, because the runtime copies the compact output back to the CPU.

## `all` Function GPU Execution Behaviour
RunMat Accelerate keeps inputs resident on the GPU whenever possible. Providers that expose
`reduce_all_dim` (and optionally `reduce_all`) perform the AND-reduction on device buffers, and
the runtime then downloads the tiny logical result back to the CPU. When those hooks are missing,
RunMat gathers the input tensor and evaluates the reduction on the host instead, preserving MATLAB
behaviour in all cases.

## Examples of using the `all` function in MATLAB / RunMat

### Checking if every column is nonzero
```matlab
A = [1 2 3; 4 5 6];
colAllNonZero = all(A);
```
Expected output:
```matlab
colAllNonZero = [1 1 1];
```

### Verifying that each row contains only nonzero values
```matlab
B = [1 0 3; 4 5 6; 0 7 8];
rowAllNonZero = all(B, 2);
```
Expected output:
```matlab
rowAllNonZero = [0; 1; 0];
```

### Collapsing multiple dimensions with `vecdim`
```matlab
C = reshape(1:24, [3 4 2]);
allAlongDims = squeeze(all(C > 0, [1 2]));
```
Expected output:
```matlab
allAlongDims = [1 1];
```

### Reducing all elements to a single logical scalar
```matlab
D = [2 4; 6 8];
everythingNonZero = all(D, 'all');
```
Expected output:
```matlab
everythingNonZero = true;
```

### Ignoring `NaN` values while testing slices
```matlab
E = [NaN 1 2; NaN 0 3];
withNaN = all(E);               % returns [1 0 1]
ignoringNaN = all(E, 'omitnan'); % returns [1 0 1]
```
Expected output:
```matlab
withNaN = [1 0 1];
ignoringNaN = [1 0 1];
```

### Evaluating `all` on GPU arrays
```matlab
G = gpuArray([1 1 1; 1 1 1]);
gpuResult = all(G, 2);     % RunMat returns a host logical array
```
Expected output:
```matlab
gpuResult = [1; 1];
```

### Testing `all` with character arrays
```matlab
chars = ['a' 0 'c'];
allPrintable = all(chars);
```
Expected output:
```matlab
allPrintable = [1 0 1];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` manually. The fusion planner keeps GPU-resident
inputs on the device and only gathers the small logical results that `all` produces. If your
workload already uses explicit `gpuArray`/`gather` calls for MATLAB compatibility, RunMat honours
them and still produces correct logical outputs.

## FAQ
### When should I use the `all` function?
Use `all` whenever you need to confirm that every element of an array, row, column, or sub-array is nonzero or logical `true`.

### Does `all` always return logical values?
Yes. Results are `logical` scalars or logical arrays even when the computation involves GPU inputs.

### How do I test a specific dimension?
Pass the dimension as the second argument (for example, `all(X, 2)` reduces each row). Provide a vector such as `[1 3]` to collapse multiple axes.

### What does `all(X, 'all')` compute?
It reduces across every dimension of `X` and returns a single logical scalar indicating whether every element of the entire array is nonzero.

### How are `NaN` values handled?
By default they count as nonzero (`'includenan'`). Add `'omitnan'` to ignore them; if every element in a slice is `NaN`, the result becomes `true`.

### Does `all` work with complex numbers?
Yes. Complex values are considered nonzero when either the real or imaginary component is nonzero. Complex values containing `NaN` obey the `'omitnan'`/`'includenan'` rules.

### Can I apply `all` to character arrays?
Yes. Characters compare against their Unicode code points; zero-valued code points are treated as `false`, and everything else is `true`.

### What happens with empty inputs?
Empty reductions follow MATLAB semantics: dimensions of length zero produce logical ones, while `all(X, 'all')` over an empty array evaluates to `true`.

### How do GPU backends accelerate `all`?
Providers may expose specialised AND-reduction kernels (`reduce_all_dim`, `reduce_all`) or use `fused_reduction` to remain on the device. When such hooks are absent, RunMat downloads the small output and computes on the host.

## See Also
[any](./any), [prod](./prod), [sum](./sum), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `all` function is available at: [`crates/runmat-runtime/src/builtins/math/reduction/all.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/all.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "all",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        crate::builtins::common::spec::ProviderHook::Reduction {
            name: "reduce_all_dim",
        },
        crate::builtins::common::spec::ProviderHook::Reduction {
            name: "reduce_all",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers may execute device-side AND reductions; runtimes gather to host when hooks are unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "all",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!(
                "accumulator *= select(0.0, 1.0, ({input} != 0.0) || ({input} != {input}));"
            ))
        },
    }),
    emits_nan: false,
    notes: "Fusion reductions treat NaNs as true; providers can substitute native kernels when profitable.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("all", DOC_MD);

#[runtime_builtin(
    name = "all",
    category = "math/reduction",
    summary = "Test whether every element of an array is nonzero with MATLAB-compatible options.",
    keywords = "all,logical,reduction,omitnan,gpu",
    accel = "reduction"
)]
fn all_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (spec, nan_mode) = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => all_gpu(handle, spec, nan_mode),
        other => all_host(other, spec, nan_mode),
    }
}

fn all_host(value: Value, spec: ReductionSpec, nan_mode: ReductionNaN) -> Result<Value, String> {
    let truth = TruthTensor::from_value("all", value)?;
    let reduced = apply_reduction(truth, spec, nan_mode)?;
    reduced.into_value()
}

fn all_gpu(
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
    match try_all_gpu(provider, &handle, &spec, nan_mode)? {
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
    all_host(Value::Tensor(tensor), spec, nan_mode)
}

fn try_all_gpu(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    spec: &ReductionSpec,
    nan_mode: ReductionNaN,
) -> Result<Option<HostTensorOwned>, String> {
    let omit_nan = matches!(nan_mode, ReductionNaN::Omit);

    if let ReductionSpec::All = spec {
        if let Ok(tmp) = provider.reduce_all(handle, omit_nan) {
            let host = provider.download(&tmp).map_err(|e| e.to_string())?;
            let _ = provider.free(&tmp);
            return Ok(Some(host));
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
        let next = provider
            .reduce_all_dim(&current, axis, omit_nan)
            .map_err(|e| e.to_string());
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
        .map_err(|e| format!("all: {e}"))
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
            return Err("all: dimension must be >= 1".to_string());
        }
        if self.shape.is_empty() {
            let truth = self.data.first().copied().unwrap_or(TruthValue {
                truthy: true,
                has_nan: false,
            });
            let truthy = match nan_mode {
                ReductionNaN::Include => truth.truthy,
                ReductionNaN::Omit => {
                    if truth.has_nan {
                        true
                    } else {
                        truth.truthy
                    }
                }
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
                let mut all_true = true;
                let mut saw_value = false;
                for k in 0..reduce_len {
                    let idx = before + k * stride_before + after * stride_before * reduce_len;
                    if let Some(value) = self.data.get(idx) {
                        match nan_mode {
                            ReductionNaN::Include => {
                                if value.has_nan {
                                    continue;
                                }
                                saw_value = true;
                                if !value.truthy {
                                    all_true = false;
                                    break;
                                }
                            }
                            ReductionNaN::Omit => {
                                if value.has_nan {
                                    continue;
                                }
                                saw_value = true;
                                if !value.truthy {
                                    all_true = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                if !saw_value {
                    all_true = true;
                }
                out.push(TruthValue::from_bool(all_true));
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
            .map_err(|e| format!("all: {e}"))
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
                return Err("all: 'all' cannot be combined with dimension arguments".to_string());
            }
            spec = ReductionSpec::All;
            continue;
        }
        if let Some(mode) = parse_nan_mode(arg)? {
            if !matches!(nan_mode, ReductionNaN::Include) {
                return Err("all: multiple NaN handling options specified".to_string());
            }
            nan_mode = mode;
            continue;
        }
        let dims = parse_dimensions(arg)?;
        if dims.is_empty() {
            return Err("all: dimension vector must contain at least one entry".to_string());
        }
        if dims.len() == 1 {
            if matches!(spec, ReductionSpec::Default) {
                spec = ReductionSpec::Dim(dims[0]);
            } else {
                return Err("all: multiple dimension specifications are not supported".to_string());
            }
        } else if matches!(spec, ReductionSpec::Default) {
            spec = ReductionSpec::VecDim(dims);
        } else {
            return Err("all: multiple dimension specifications are not supported".to_string());
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
        _ => Err(format!("all: unknown option '{}'", text.trim())),
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
            return Err("all: dimension values must be finite".to_string());
        }
        let rounded = raw.round();
        if (rounded - raw).abs() > f64::EPSILON {
            return Err("all: dimension values must be integers".to_string());
        }
        if rounded < 1.0 {
            return Err("all: dimension values must be >= 1".to_string());
        }
        if rounded > (usize::MAX as f64) {
            return Err("all: dimension value is too large".to_string());
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
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue};

    #[test]
    fn all_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 1.0, 4.0, 5.0, 0.0, 6.0], vec![2, 3]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), Vec::new()).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn all_zero_column_matrix_returns_empty_row() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![2, 0]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), Vec::new()).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 0]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn all_row_dimension() {
        let tensor = Tensor::new(vec![1.0, 1.0, 4.0, 5.0, 0.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = all_builtin(Value::Tensor(tensor), args).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn all_zero_row_matrix_dim_two() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = all_builtin(Value::Tensor(tensor), args).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![0, 1]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn all_vecdim_multiple_axes() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let vecdim = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), vec![Value::Tensor(vecdim)]).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1, 2]);
                assert_eq!(out.data, vec![1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn all_all_option_returns_scalar() {
        let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("all");
        match result {
            Value::Bool(flag) => assert!(!flag),
            other => panic!("expected logical scalar, got {other:?}"),
        }
    }

    #[test]
    fn all_all_on_empty_returns_true() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("all");
        match result {
            Value::Bool(flag) => assert!(flag),
            other => panic!("expected logical scalar, got {other:?}"),
        }
    }

    #[test]
    fn all_handles_nan_modes() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN, 1.0, 0.0], vec![2, 2]).unwrap();
        let includenan = all_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("all");
        match includenan {
            Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0]),
            other => panic!("expected logical array, got {other:?}"),
        }

        let omit =
            all_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("all omit");
        match omit {
            Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn all_char_array_support() {
        let chars = CharArray::new("a\0c".chars().collect(), 1, 3).unwrap();
        let result =
            all_builtin(Value::CharArray(chars), vec![Value::Int(IntValue::I32(1))]).expect("all");
        match result {
            Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0, 1]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn all_includenan_keyword_allowed() {
        let tensor = Tensor::new(vec![f64::NAN, 1.0], vec![2, 1]).unwrap();
        let result =
            all_builtin(Value::Tensor(tensor), vec![Value::from("includenan")]).expect("all");
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
    fn all_complex_tensor_with_omitnan() {
        let complex = ComplexTensor::new(vec![(f64::NAN, 0.0), (1.0, 0.0)], vec![2, 1]).unwrap();
        let tensor_value = Value::ComplexTensor(complex);
        let omit = all_builtin(tensor_value.clone(), vec![Value::from("omitnan")]).expect("all");
        match omit {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1]);
                assert_eq!(out.data, vec![1]);
            }
            Value::Bool(flag) => assert!(flag),
            other => panic!("expected logical array, got {other:?}"),
        }
        let include = all_builtin(tensor_value, Vec::new()).expect("all include");
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
    fn all_vecdim_with_omitnan() {
        let mut data = vec![0.0; 8];
        data[7] = f64::NAN;
        let tensor = Tensor::new(data, vec![2, 2, 2]).unwrap();
        let vecdim = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(vecdim), Value::from("omitnan")];
        let result = all_builtin(Value::Tensor(tensor), args).expect("all vecdim omitnan");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1, 2]);
                assert_eq!(out.data, vec![0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn all_all_with_dim_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap();
        let args = vec![Value::from("all"), Value::Int(IntValue::I32(1))];
        let err = all_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert!(err.contains("dimension"), "unexpected error message: {err}");
    }

    #[test]
    fn all_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = all_builtin(Value::GpuTensor(handle), Vec::new()).expect("all");
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
    fn all_gpu_provider_omitnan_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, f64::NAN, 1.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                all_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("all");
            match result {
                Value::LogicalArray(out) => {
                    assert_eq!(out.shape, vec![1, 2]);
                    assert_eq!(out.data, vec![1, 0]);
                }
                other => panic!("expected logical array, got {other:?}"),
            }
        });
    }

    #[cfg(feature = "doc_export")]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn all_wgpu_default_matches_cpu() {
        let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            )
        }));
        let Ok(reg_result) = init else {
            eprintln!("skipping all_wgpu_default_matches_cpu: wgpu provider panicked during init");
            return;
        };
        if reg_result.is_err() {
            eprintln!("skipping all_wgpu_default_matches_cpu: wgpu provider unavailable");
            return;
        }
        let tensor = Tensor::new(vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let cpu = all_host(
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
                eprintln!("skipping all_wgpu_default_matches_cpu: provider not registered");
                return;
            }
        };
        let handle = match provider.upload(&view) {
            Ok(h) => h,
            Err(err) => {
                eprintln!("skipping all_wgpu_default_matches_cpu: upload failed: {err}");
                return;
            }
        };
        let gpu = all_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
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
    fn all_wgpu_omitnan_matches_cpu() {
        let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            )
        }));
        let Ok(reg_result) = init else {
            eprintln!("skipping all_wgpu_omitnan_matches_cpu: wgpu provider panicked during init");
            return;
        };
        if reg_result.is_err() {
            eprintln!("skipping all_wgpu_omitnan_matches_cpu: wgpu provider unavailable");
            return;
        }
        let tensor = Tensor::new(vec![f64::NAN, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let cpu = all_host(
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
                eprintln!("skipping all_wgpu_omitnan_matches_cpu: provider not registered");
                return;
            }
        };
        let handle = match provider.upload(&view) {
            Ok(h) => h,
            Err(err) => {
                eprintln!("skipping all_wgpu_omitnan_matches_cpu: upload failed: {err}");
                return;
            }
        };
        let gpu = all_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).unwrap();
        match (cpu, gpu) {
            (Value::LogicalArray(expected), Value::LogicalArray(actual)) => {
                assert_eq!(expected.shape, actual.shape);
                assert_eq!(expected.data, actual.data);
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
