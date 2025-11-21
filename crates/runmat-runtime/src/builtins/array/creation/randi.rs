//! MATLAB-compatible `randi` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{random, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "randi"
category: "array/creation"
keywords: ["randi", "random", "integer", "gpu", "like"]
summary: "Uniform random integers with inclusive bounds."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider integer RNG hooks when available; otherwise generates samples on the host and uploads them to preserve GPU residency."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "none"
requires_feature: null
tested:
  unit: "builtins::array::creation::randi::tests"
  integration: "builtins::array::creation::randi::tests::randi_gpu_like_roundtrip"
---

# What does the `randi` function do in MATLAB / RunMat?
`randi` draws uniformly distributed random integers from an inclusive range. RunMat mirrors MATLAB calling forms, supporting scalar upper bounds, two-element range vectors, explicit dimension lists, size vectors, and `'like'` prototypes that keep type and residency consistent with an existing array.

## How does the `randi` function behave in MATLAB / RunMat?
- `randi(imax)` returns a scalar double selected uniformly from `1:imax`.
- `randi(imax, m, n, ...)` creates dense arrays whose entries are in `1:imax`.
- `randi([imin imax], sz)` accepts a two-element range vector plus either a size vector or explicit dimensions.
- `randi(___, 'like', A)` matches both the shape and device residency of `A`, including GPU tensors when acceleration is enabled.
- Negative lower bounds are supported so long as `imin <= imax`.
- `randi(___, 'double')` is accepted for MATLAB compatibility and keeps the default double output.
- `randi(___, 'logical')` emits a logical array when the inclusive range stays within `[0, 1]`.
- RunMat diagnoses unsupported class specifiers (e.g., `'single'`, `'uint8'`) with descriptive errors until native representations land.

## `randi` Function GPU Execution Behaviour
When the target output or `'like'` prototype resides on the GPU, RunMat first asks the active acceleration provider for device-side generation via `random_integer_like` (shape reuse) or `random_integer_range` (explicit shape). Providers that do not expose these hooks fall back to host-side sampling followed by a single upload, ensuring correctness while still keeping the resulting tensor on the GPU.

## Examples of using the `randi` function in MATLAB / RunMat

### Drawing a single die roll

```matlab
rng(0);
roll = randi(6);
```

Expected output:

```matlab
roll = 1
```

### Creating a matrix of random indices

```matlab
rng(0);
idx = randi(10, 2, 3);
```

Expected output:

```matlab
idx =
     1     9     7
    10     2     1
```

### Generating bounded integers with a size vector

```matlab
rng(0);
shape = [3 4 2];
tiles = randi([5 20], shape);
```

Expected output:

```matlab
tiles(:, :, 1) =
     6     7     5    15
    19    14     9    16
    18     5    10    13

tiles(:, :, 2) =
    20     5    17     8
    20    14     5     9
    10     9    18     5
```

### Matching an existing `gpuArray`

```matlab
rng(0);
G = gpuArray.zeros(4, 4);
labels = randi([0 3], 'like', G);
peek = gather(labels);
```

Expected output:

```matlab
peek =
     0     2     1     3
     3     0     2     3
     3     0     2     1
     0     1     2     0

isa(labels, 'gpuArray')
ans =
     logical
     1
```

### Building a random logical mask

```matlab
rng(0);
mask = randi([0 1], 4, 4, 'logical');
```

Expected output:

```matlab
mask =
     0     1     0     1
     1     0     1     1
     1     0     1     0
     0     0     1     0
```

### Reproducible integer tensors inside tests

```matlab
rng(0);
p = randi([1 4], 1, 5);
```

Expected output:

```matlab
p = [1 4 4 1 3]
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, in the examples above, the result of `randi` remains on the GPU whenever the planner determines that downstream work benefits from staying on device.

To preserve backwards compatibility with MathWorks MATLAB—and when you want to be explicit about residency—you can call `gpuArray` yourself to seed GPU execution.

Because MathWorks MATLAB lacks a fusion planner and ships GPU support as a separate toolbox, MATLAB users must move data manually. RunMat automates residency to streamline accelerated workflows.

## FAQ

### What range does `randi(imax)` use?
The call `randi(imax)` produces integers in the inclusive range `1:imax`. Use the two-element form `randi([imin imax], ...)` when you need a custom minimum.

### Can the lower bound be negative?
Yes. `randi([imin imax], ...)` accepts negative bounds as long as `imin <= imax` and both bounds are integers.

### Does `randi` return doubles or integer arrays?
RunMat matches MATLAB by storing results in double-precision tensors whose values are integers. Future releases will add direct integer array classes.

### Can I request a logical result directly?
Yes. Pass `'logical'` as the final argument, and ensure the inclusive range stays within `[0, 1]`. Any other range produces an error because logical arrays can only store 0/1 values.

### How do I control the array shape?
Pass either explicit dimensions (`randi(9, 4, 2)`) or a size vector (`randi(9, [4 2])`). Providing a `'like'` prototype also copies the prototype's shape automatically.

### How does `randi` interact with `rng`?
`randi` consumes RunMat's global RNG stream. Use the MATLAB-compatible `rng` builtin to seed or restore the generator for reproducible simulations.

### What happens if the provider lacks integer RNG hooks?
RunMat falls back to host generation followed by a single upload. The resulting array still lives on the GPU; only the initial samples are produced on the CPU.

### Does `randi` support `'single'` or integer output classes?
Not yet. The `randi` builtin currently supports doubles only. Supplying `'single'` or integer class names raises a descriptive error.

## See Also
[rand](./rand), [randn](./randn), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `randi` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/randi.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/randi.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "randi",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("random_integer_range"),
        ProviderHook::Custom("random_integer_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may offer integer RNG kernels via random_integer_range / random_integer_like; the runtime falls back to host sampling and upload when unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "randi",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random integer generation is treated as a sink and excluded from fusion planning.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("randi", DOC_MD);

#[runtime_builtin(
    name = "randi",
    category = "array/creation",
    summary = "Uniform random integers with inclusive bounds.",
    keywords = "randi,random,integer,gpu,like",
    accel = "array_construct"
)]
fn randi_builtin(args: Vec<Value>) -> Result<Value, String> {
    let parsed = ParsedRandi::parse(args)?;
    build_output(parsed)
}

struct ParsedRandi {
    bounds: Bounds,
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Logical,
    Like(Value),
}

#[derive(Clone, Copy)]
struct Bounds {
    lower: i64,
    upper: i64,
    span: u64,
}

impl Bounds {
    fn new(lower: i64, upper: i64) -> Result<Self, String> {
        if lower > upper {
            return Err("randi: lower bound must be <= upper bound".to_string());
        }
        let span = (upper as i128)
            .checked_sub(lower as i128)
            .and_then(|delta| delta.checked_add(1))
            .ok_or_else(|| "randi: range width overflows 64-bit arithmetic".to_string())?;
        if span <= 0 {
            return Err("randi: invalid bounds".to_string());
        }
        if span > (1u64 << 53) as i128 {
            return Err("randi: range width exceeds RNG precision (2^53)".to_string());
        }
        Ok(Self {
            lower,
            upper,
            span: span as u64,
        })
    }
}

impl ParsedRandi {
    fn parse(args: Vec<Value>) -> Result<Self, String> {
        if args.is_empty() {
            return Err("randi: requires at least one input argument".to_string());
        }

        let mut iter = args.into_iter();
        let bounds_value = iter.next().unwrap();
        let bounds = parse_bounds(bounds_value)?;

        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut implicit_proto: Option<Value> = None;

        let rest: Vec<Value> = iter.collect();
        let mut idx = 0;
        while idx < rest.len() {
            let arg = rest[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err("randi: multiple 'like' specifications are not supported"
                                .to_string());
                        }
                        if let Some(spec) = &class_override {
                            let keyword = match spec {
                                OutputTemplate::Logical => "'logical'",
                                OutputTemplate::Double => "'double'",
                                OutputTemplate::Like(_) => "another class specifier",
                            };
                            return Err(format!("randi: cannot combine 'like' with {keyword}"));
                        }
                        let Some(proto) = rest.get(idx + 1).cloned() else {
                            return Err("randi: expected prototype after 'like'".to_string());
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto)?);
                        }
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err("randi: cannot combine 'like' with 'double'".to_string());
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err("randi: cannot combine 'like' with 'logical'".to_string());
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err(
                            "randi: single precision output is not implemented yet".to_string()
                        );
                    }
                    "int8" | "uint8" | "int16" | "uint16" | "int32" | "uint32" | "int64"
                    | "uint64" => {
                        return Err(format!(
                            "randi: output class '{}' is not implemented yet",
                            keyword
                        ));
                    }
                    other => {
                        return Err(format!("randi: unrecognised option '{other}'"));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg)? {
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg)?);
            }
            if implicit_proto.is_none() {
                implicit_proto = Some(arg.clone());
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = shape_source {
            shape
        } else {
            vec![1, 1]
        };

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            OutputTemplate::Like(proto)
        } else {
            OutputTemplate::Double
        };

        Ok(Self {
            bounds,
            shape,
            template,
        })
    }
}

fn build_output(parsed: ParsedRandi) -> Result<Value, String> {
    match parsed.template {
        OutputTemplate::Double => randi_double(&parsed.bounds, &parsed.shape),
        OutputTemplate::Logical => randi_logical(&parsed.bounds, &parsed.shape),
        OutputTemplate::Like(proto) => randi_like(&proto, &parsed.bounds, &parsed.shape),
    }
}

fn randi_double(bounds: &Bounds, shape: &[usize]) -> Result<Value, String> {
    let tensor = integer_tensor(bounds, shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn randi_logical(bounds: &Bounds, shape: &[usize]) -> Result<Value, String> {
    if bounds.lower < 0 || bounds.upper > 1 {
        return Err(
            "randi: logical output requires bounds contained within the inclusive range [0, 1]"
                .to_string(),
        );
    }

    let len = tensor::element_count(shape);
    let mut data: Vec<u8> = Vec::with_capacity(len);
    if len == 0 {
        let logical = LogicalArray::new(data, shape.to_vec()).map_err(|e| format!("randi: {e}"))?;
        return Ok(Value::LogicalArray(logical));
    }

    if bounds.span == 1 {
        let byte = if bounds.lower == 0 { 0u8 } else { 1u8 };
        data.resize(len, byte);
    } else {
        let samples = generate_integer_data(bounds, len)?;
        data = samples
            .into_iter()
            .map(|value| if value != 0.0 { 1u8 } else { 0u8 })
            .collect();
    }

    let logical = LogicalArray::new(data, shape.to_vec()).map_err(|e| format!("randi: {e}"))?;
    Ok(Value::LogicalArray(logical))
}

fn randi_like(proto: &Value, bounds: &Bounds, shape: &[usize]) -> Result<Value, String> {
    match proto {
        Value::GpuTensor(handle) => randi_like_gpu(handle, bounds, shape),
        Value::LogicalArray(_) | Value::Bool(_) => randi_logical(bounds, shape),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => randi_double(bounds, shape),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            randi_double(bounds, shape)
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(
            "randi: complex prototypes are not supported; expected real-valued arrays".to_string(),
        ),
        Value::Cell(_) => Err("randi: cell prototypes are not supported".to_string()),
        other => Err(format!("randi: unsupported prototype {other:?}")),
    }
}

fn randi_like_gpu(
    handle: &GpuTensorHandle,
    bounds: &Bounds,
    shape: &[usize],
) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let attempt = if handle.shape == shape {
            provider.random_integer_like(handle, bounds.lower, bounds.upper)
        } else {
            provider.random_integer_range(bounds.lower, bounds.upper, shape)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        let tensor = integer_tensor(bounds, shape)?;
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            return Ok(Value::GpuTensor(gpu));
        }
        return Ok(tensor::tensor_into_value(tensor));
    }

    let gathered = crate::dispatcher::gather_if_needed(&Value::GpuTensor(handle.clone()))
        .map_err(|e| format!("randi: {e}"))?;
    randi_like(&gathered, bounds, shape)
}

fn integer_tensor(bounds: &Bounds, shape: &[usize]) -> Result<Tensor, String> {
    let len = tensor::element_count(shape);
    let data = generate_integer_data(bounds, len)?;
    Tensor::new(data, shape.to_vec()).map_err(|e| format!("randi: {e}"))
}

fn generate_integer_data(bounds: &Bounds, len: usize) -> Result<Vec<f64>, String> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if bounds.span == 1 {
        return Ok(vec![bounds.lower as f64; len]);
    }

    let uniforms = random::generate_uniform(len, "randi")?;
    let span = bounds.span as f64;
    let lower = bounds.lower as i128;
    let upper = bounds.upper as i128;
    let mut out = Vec::with_capacity(len);
    for u in uniforms {
        let mut offset = (u * span).floor() as u64;
        if offset >= bounds.span {
            offset = bounds.span - 1;
        }
        let mut value = lower
            .checked_add(offset as i128)
            .ok_or_else(|| "randi: integer overflow while sampling".to_string())?;
        if value > upper {
            value = upper;
        }
        out.push(value as f64);
    }
    Ok(out)
}

fn parse_bounds(value: Value) -> Result<Bounds, String> {
    match value {
        Value::Int(i) => parse_upper_scalar(i.to_i64()),
        Value::Num(n) => parse_upper_num(n),
        Value::Tensor(t) => parse_bounds_tensor(&t),
        Value::LogicalArray(_) | Value::Bool(_) => {
            Err("randi: bounds must be numeric scalars or vectors".to_string())
        }
        Value::GpuTensor(_) => Err("randi: bounds must be specified on the host".to_string()),
        Value::String(s) => Err(format!("randi: unexpected option '{s}' in first argument")),
        Value::StringArray(_) => {
            Err("randi: unexpected string array in first argument".to_string())
        }
        Value::CharArray(_) => Err("randi: string bounds are not supported".to_string()),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("randi: complex bounds are not supported".to_string())
        }
        other => Err(format!("randi: unsupported bounds argument {other:?}")),
    }
}

fn parse_upper_scalar(upper: i64) -> Result<Bounds, String> {
    if upper < 1 {
        return Err("randi: upper bound must be >= 1".to_string());
    }
    Bounds::new(1, upper)
}

fn parse_upper_num(n: f64) -> Result<Bounds, String> {
    if !n.is_finite() {
        return Err("randi: bounds must be finite".to_string());
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err("randi: bounds must be integers".to_string());
    }
    let upper = rounded as i64;
    parse_upper_scalar(upper)
}

fn parse_bounds_tensor(tensor: &Tensor) -> Result<Bounds, String> {
    let len = tensor.data.len();
    if len == 0 {
        return Err("randi: empty bound vector is not allowed".to_string());
    }
    if len == 1 {
        return parse_upper_num(tensor.data[0]);
    }
    if len == 2 && is_vector_like(tensor) {
        let lower = parse_integer_component(tensor.data[0])?;
        let upper = parse_integer_component(tensor.data[1])?;
        Bounds::new(lower, upper)
    } else {
        Err("randi: bound vector must contain exactly two elements".to_string())
    }
}

fn parse_integer_component(value: f64) -> Result<i64, String> {
    if !value.is_finite() {
        return Err("randi: bounds must be finite".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("randi: bounds must be integers".to_string());
    }
    Ok(rounded as i64)
}

fn is_vector_like(tensor: &Tensor) -> bool {
    tensor.rows() == 1 || tensor.cols() == 1 || tensor.shape.len() == 1
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

fn extract_dims(value: &Value) -> Result<Option<Vec<usize>>, String> {
    match value {
        Value::Int(i) => {
            let dim = i.to_i64();
            if dim < 0 {
                return Err("randi: matrix dimensions must be non-negative".to_string());
            }
            Ok(Some(vec![dim as usize]))
        }
        Value::Num(n) => parse_numeric_dimension(*n).map(|d| Some(vec![d])),
        Value::Tensor(t) => dims_from_tensor(t),
        Value::LogicalArray(_) => Ok(None),
        _ => Ok(None),
    }
}

fn parse_numeric_dimension(n: f64) -> Result<usize, String> {
    if !n.is_finite() {
        return Err("randi: dimensions must be finite".to_string());
    }
    if n < 0.0 {
        return Err("randi: matrix dimensions must be non-negative".to_string());
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err("randi: dimensions must be integers".to_string());
    }
    Ok(rounded as usize)
}

fn dims_from_tensor(tensor: &Tensor) -> Result<Option<Vec<usize>>, String> {
    let is_row = tensor.rows() == 1;
    let is_col = tensor.cols() == 1;
    let is_scalar = tensor.data.len() == 1;
    if !(is_row || is_col || is_scalar || tensor.shape.len() == 1) {
        return Ok(None);
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &v in &tensor.data {
        match parse_numeric_dimension(v) {
            Ok(dim) => dims.push(dim),
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(dims))
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(_) => Err("randi: complex prototypes are not supported".to_string()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(h.shape.clone()),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(vec![1, 1]),
        other => Err(format!("randi: unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};
    use runmat_builtins::LogicalArray;

    fn reset_rng_clean() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    fn expected_sequence(bounds: &Bounds, count: usize) -> Vec<i64> {
        let uniforms = random::expected_uniform_sequence(count);
        let span = bounds.span as f64;
        uniforms
            .into_iter()
            .map(|u| {
                let mut offset = (u * span).floor() as u64;
                if offset >= bounds.span {
                    offset = bounds.span - 1;
                }
                bounds.lower + offset as i64
            })
            .collect()
    }

    #[test]
    fn randi_default_scalar() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let result = randi_builtin(vec![Value::Num(6.0)]).expect("randi");
        let expected = expected_sequence(&Bounds::new(1, 6).unwrap(), 1)[0] as f64;
        match result {
            Value::Num(v) => {
                assert!((1.0..=6.0).contains(&v));
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar double, got {other:?}"),
        }
    }

    #[test]
    fn randi_range_with_dims() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let bounds = Tensor::new(vec![3.0, 8.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(bounds), Value::Num(2.0), Value::Num(3.0)];
        let result = randi_builtin(args).expect("randi");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                let expected = expected_sequence(&Bounds::new(3, 8).unwrap(), 6);
                for (observed, exp) in t.data.iter().zip(expected.iter().map(|v| *v as f64)) {
                    assert!((*observed - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn randi_like_tensor() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let proto = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let args = vec![Value::Num(5.0), Value::from("like"), Value::Tensor(proto)];
        let result = randi_builtin(args).expect("randi");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                for v in &t.data {
                    assert!((1.0..=5.0).contains(v));
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn randi_logical_output() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let bounds = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(bounds),
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("logical"),
        ];
        let result = randi_builtin(args).expect("randi logical");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                let expected = expected_sequence(&Bounds::new(0, 1).unwrap(), 4);
                for (idx, &byte) in logical.data.iter().enumerate() {
                    assert!(byte <= 1);
                    assert_eq!(byte, if expected[idx] == 0 { 0 } else { 1 });
                }
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn randi_logical_requires_binary_bounds() {
        let err = randi_builtin(vec![Value::Num(3.0), Value::from("logical")]).unwrap_err();
        assert!(err.contains("logical output requires"));
    }

    #[test]
    fn randi_like_logical_prototype() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let proto = LogicalArray::zeros(vec![2, 3]);
        let bounds = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(bounds),
            Value::from("like"),
            Value::LogicalArray(proto),
        ];
        let result = randi_builtin(args).expect("randi logical like");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 3]);
                assert!(logical.data.iter().all(|&b| b <= 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn randi_like_requires_prototype() {
        let err = randi_builtin(vec![Value::Num(5.0), Value::from("like")]).unwrap_err();
        assert!(err.contains("expected prototype"));
    }

    #[test]
    fn randi_duplicate_like_is_error() {
        let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let args = vec![
            Value::Num(5.0),
            Value::from("like"),
            Value::Tensor(proto.clone()),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let err = randi_builtin(args).unwrap_err();
        assert!(err.contains("multiple 'like' specifications"));
    }

    #[test]
    fn randi_like_logical_conflict_is_error() {
        let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let args = vec![
            Value::Num(1.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let err = randi_builtin(args).unwrap_err();
        assert!(err.contains("cannot combine 'like' with 'logical'"));
    }

    #[test]
    fn randi_gpu_like_roundtrip() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(4.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = randi_builtin(args).expect("randi");
            match result {
                Value::GpuTensor(gpu) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(gpu)).expect("gather to host");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for value in gathered.data {
                        assert!((1.0..=4.0).contains(&value));
                    }
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn randi_gpu_like_shape_override() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let proto = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &proto.data,
                shape: &proto.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let bounds = Tensor::new(vec![1.0, 4.0], vec![1, 2]).unwrap();
            let args = vec![
                Value::Tensor(bounds),
                Value::Num(3.0),
                Value::Num(1.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = randi_builtin(args).expect("randi gpu override");
            match result {
                Value::GpuTensor(gpu) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(gpu)).expect("gather override");
                    assert_eq!(gathered.shape, vec![3, 1]);
                    for value in gathered.data {
                        assert!((1.0..=4.0).contains(&value));
                    }
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn randi_invalid_upper_errors() {
        let err = randi_builtin(vec![Value::Num(0.0)]).unwrap_err();
        assert!(err.contains("upper bound"));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn randi_wgpu_like_produces_in_range_values() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let provider = match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(_) => runmat_accelerate_api::provider().expect("wgpu provider registered"),
            Err(err) => {
                eprintln!("randi_wgpu_like_produces_in_range_values skipped: {err}");
                return;
            }
        };

        let proto = Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &proto.data,
            shape: &proto.shape,
        };
        let handle = provider.upload(&view).expect("upload prototype");
        let bounds = Tensor::new(vec![1.0, 8.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(bounds),
            Value::from("like"),
            Value::GpuTensor(handle),
        ];

        let result = randi_builtin(args).expect("randi");
        match result {
            Value::GpuTensor(gpu) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(gpu)).expect("gather gpu result");
                assert_eq!(gathered.shape, vec![2, 3]);
                for value in gathered.data {
                    assert!(
                        (1.0..=8.0).contains(&value),
                        "expected value within [1, 8], got {value}"
                    );
                }
            }
            other => panic!("expected GPU tensor result, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
