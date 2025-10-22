//! MATLAB-compatible `rand` builtin with GPU-aware semantics.
//!
//! Implements uniform random number generation over (0, 1) with support for
//! MATLAB's scalar, vector, matrix, N-D, and `'like'` invocation forms. When a
//! GPU prototype is supplied, the builtin dispatches to acceleration provider
//! hooks and transparently falls back to host sampling if the provider lacks a
//! dedicated implementation.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::sync::{Mutex, OnceLock};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "rand"
category: "array/creation"
keywords: ["rand", "random", "uniform", "gpu", "like"]
summary: "Uniform random numbers on (0, 1) with MATLAB-compatible semantics."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider random_uniform hooks when available; otherwise uploads host-generated samples to keep GPU residency."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "none"
requires_feature: null
tested:
  unit: "builtins::array::rand::tests"
  integration: "builtins::array::rand::tests::rand_gpu_like_uniform"
---

# What does the `rand` function do in MATLAB / RunMat?
`rand` produces uniformly distributed pseudorandom numbers over the open
interval (0, 1). RunMat mirrors MATLAB semantics across scalar, vector, matrix,
and N-D invocations, including `'like'` prototypes that control data type and
device residency.

## How does the `rand` function behave in MATLAB / RunMat?
- `rand()` returns a scalar double drawn from `U(0, 1)`.
- `rand(n)` returns an `n × n` double matrix.
- `rand(m, n, ...)` returns a dense double array of the requested dimensions.
- `rand(sz)` accepts a size vector (row or column) and returns a tensor whose
  shape is `sz`.
- `rand(A)` or `rand(___, 'like', A)` matches the shape and residency of `A`,
  including GPU tensors when an acceleration provider is active.
- `rand(___, 'double')` leaves the output as double precision (default). `'single'`
  is reserved for future support and currently errors.

## `rand` Function GPU Execution Behaviour
When the prototype lives on the GPU, RunMat first asks the active acceleration
provider for a device-side random buffer via `random_uniform` / `random_uniform_like`.
If the provider lacks those hooks, RunMat generates samples on the host and
uploads them to maintain GPU residency. This guarantees MATLAB-compatible
behaviour while documenting the extra transfer cost.

## Examples of using the `rand` function in MATLAB / RunMat

### Creating a 3x3 matrix of random numbers

```matlab
R = rand(3);         % 3x3 doubles in (0, 1)
```

Expected output:

```matlab
R = [0.8147 0.9134 0.1270; 0.9058 0.6324 0.0975; 0.1270 0.0975 0.2785];
```

### Creating a 2x4x3 matrix of random numbers

```matlab
sz = [2 4 3];
T = rand(sz);
```

Expected output:

```matlab
T = [0.8147 0.9134 0.1270 0.9058 0.6324 0.0975 0.1270 0.0975 0.2785; 0.9134 0.6324 0.0975 0.2785 0.0975 0.1270 0.9058 0.6324 0.0975];
```

### Creating a 128x128 matrix of random numbers on a GPU

In RunMat:

```matlab
G = rand(128, 128);
```

In MathWorks MATLAB (supported in RunMat as well):

```matlab
G = gpuArray(rand(128, 128));

% OR:

H = gpuArray(rand(128, 128));
H = rand(128, 128, 'like', G);
```

Expected output:

```matlab
H = [0.8147 0.9134 0.1270 0.9058 0.6324 0.0975 0.1270 0.0975 0.2785; 0.9134 0.6324 0.0975 0.2785 0.0975 0.1270 0.9058 0.6324 0.0975];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, in the above example, the result of the `rand` call will already be on the GPU when the fusion planner has detected a net benefit to operating the fused expression it is part of on the GPU.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the core language, as their toolbox is a separate commercial product, MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.

## FAQ

### When should I use the `rand` function?

Use `rand` whenever you need to create arrays filled with random numbers over the open interval (0, 1). This is useful for Monte Carlo simulations, generating noise for testing, or creating random initial conditions for optimization.

### Does `rand` produce double arrays by default?

Yes, by default, `rand` creates dense double-precision arrays unless you explicitly specify a type such as `'single'` or use the `'like'` argument to match a prototype array.

### What does `rand(n)` return?

`rand(n)` returns an `n × n` dense double-precision matrix filled with random numbers over the open interval (0, 1). For example, `rand(3)` yields a 3-by-3 matrix of random numbers.

### How do I create a single precision array of random numbers?

Pass `'single'` as the last argument:
```matlab
S = rand(5, 5, 'single');
```
This produces a 5x5 single precision matrix of random numbers.

### How do I match the type and device residency of an existing array?

Use the `'like', prototype` syntax:
```matlab
A = gpuArray(rand(2,2));
B = rand(2, 2, 'like', A);
```
`B` will be a GPU array with the same type and shape as `A`.

### Can I create N-dimensional arrays with `rand`?

Yes! Pass more than two dimension arguments (or a size vector):
```matlab
T = rand(2, 3, 4);
```
This creates a 2×3×4 tensor of random numbers.

### How does `rand(A)` behave?

If you call `rand(A)`, where `A` is an array, the result is a new array of random numbers with the same shape as `A`.

### Is the output always dense?

Yes. `rand` always produces a dense array. For sparse matrices of random numbers, use `sparse` with appropriate arguments.

### What if I call `rand` with no arguments?

`rand()` returns a scalar double drawn from `U(0, 1)`.

## See Also
[randn](./randn), [randi](./randi), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `rand` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/rand.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/rand.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rand",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("random_uniform"),
        ProviderHook::Custom("random_uniform_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to provider random_uniform hooks; falls back to host sampling + upload when hooks are unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rand",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random generation is treated as a sink and is not eligible for fusion.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("rand", DOC_MD);

#[runtime_builtin(
    name = "rand",
    category = "array/creation",
    summary = "Uniform random numbers on (0, 1).",
    keywords = "rand,random,uniform,gpu,like",
    accel = "array_construct"
)]
fn rand_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let parsed = ParsedRand::parse(rest)?;
    build_output(parsed)
}

struct ParsedRand {
    shape: Vec<usize>,
    template: RandTemplate,
}

#[derive(Clone)]
enum RandTemplate {
    Double,
    Like(Value),
}

impl ParsedRand {
    fn parse(args: Vec<Value>) -> Result<Self, String> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut template: Option<RandTemplate> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();

            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("rand: expected prototype after 'like'".to_string());
                        };
                        template = Some(RandTemplate::Like(proto.clone()));
                        shape_source = Some(shape_from_value(&proto)?);
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        template = Some(RandTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err("rand: single precision output is not implemented yet".to_string());
                    }
                    other => {
                        return Err(format!("rand: unrecognised option '{other}'"));
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
            if template.is_none() {
                template = Some(RandTemplate::Like(arg.clone()));
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

        let template = template.unwrap_or(RandTemplate::Double);

        Ok(Self { shape, template })
    }
}

fn build_output(parsed: ParsedRand) -> Result<Value, String> {
    match parsed.template {
        RandTemplate::Double => rand_double(&parsed.shape),
        RandTemplate::Like(proto) => rand_like(&proto, &parsed.shape),
    }
}

fn rand_double(shape: &[usize]) -> Result<Value, String> {
    let len = tensor::element_count(shape);
    let data = generate_uniform(len)?;
    let tensor = Tensor::new(data, shape.to_vec()).map_err(|e| format!("rand: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn rand_like(proto: &Value, shape: &[usize]) -> Result<Value, String> {
    match proto {
        Value::GpuTensor(handle) => rand_like_gpu(handle, shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => rand_complex(shape),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => {
            rand_double(shape)
        }
        Value::CharArray(_) | Value::Cell(_) => rand_double(shape),
        other => Err(format!("rand: unsupported prototype {other:?}")),
    }
}

fn rand_complex(shape: &[usize]) -> Result<Value, String> {
    let len = tensor::element_count(shape);
    let data = generate_complex(len)?;
    let tensor = ComplexTensor::new(data, shape.to_vec()).map_err(|e| format!("rand: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn rand_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let attempt = if handle.shape == shape {
            provider.random_uniform_like(handle)
        } else {
            provider.random_uniform(shape)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        let len = tensor::element_count(shape);
        let data = generate_uniform(len)?;
        let tensor = Tensor::new(data, shape.to_vec()).map_err(|e| format!("rand: {e}"))?;
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            return Ok(Value::GpuTensor(gpu));
        }
    }

    let gathered = crate::dispatcher::gather_if_needed(&Value::GpuTensor(handle.clone()))
        .map_err(|e| format!("rand: {e}"))?;
    rand_like(&gathered, shape)
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
                return Err("rand: matrix dimensions must be non-negative".to_string());
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
        return Err("rand: dimensions must be finite".to_string());
    }
    if n < 0.0 {
        return Err("rand: matrix dimensions must be non-negative".to_string());
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err("rand: dimensions must be integers".to_string());
    }
    Ok(rounded as usize)
}

fn dims_from_tensor(tensor: &Tensor) -> Result<Option<Vec<usize>>, String> {
    let is_row = tensor.rows() == 1;
    let is_column = tensor.cols() == 1;
    let is_scalar = tensor.data.len() == 1;
    if !(is_row || is_column || is_scalar || tensor.shape.len() == 1) {
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
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(h.shape.clone()),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("rand: unsupported prototype {other:?}")),
    }
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

const DEFAULT_RNG_SEED: u64 = 0x9e3779b97f4a7c15;
const RNG_MULTIPLIER: u64 = 6364136223846793005;
const RNG_INCREMENT: u64 = 1;
const RNG_SHIFT: u32 = 11;
const RNG_SCALE: f64 = 1.0 / ((1u64 << 53) as f64);

static RNG_STATE: OnceLock<Mutex<u64>> = OnceLock::new();

fn rng_state() -> &'static Mutex<u64> {
    RNG_STATE.get_or_init(|| Mutex::new(DEFAULT_RNG_SEED))
}

fn generate_uniform(len: usize) -> Result<Vec<f64>, String> {
    let mut guard = rng_state()
        .lock()
        .map_err(|_| "rand: failed to acquire RNG lock".to_string())?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(next_uniform_state(&mut *guard));
    }
    Ok(out)
}

fn generate_complex(len: usize) -> Result<Vec<(f64, f64)>, String> {
    let mut guard = rng_state()
        .lock()
        .map_err(|_| "rand: failed to acquire RNG lock".to_string())?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let re = next_uniform_state(&mut *guard);
        let im = next_uniform_state(&mut *guard);
        out.push((re, im));
    }
    Ok(out)
}

fn next_uniform_state(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(RNG_MULTIPLIER)
        .wrapping_add(RNG_INCREMENT);
    let bits = *state >> RNG_SHIFT;
    (bits as f64) * RNG_SCALE
}

#[cfg(test)]
fn reset_rng() {
    if let Some(mutex) = RNG_STATE.get() {
        if let Ok(mut guard) = mutex.lock() {
            *guard = DEFAULT_RNG_SEED;
        }
    } else {
        let _ = RNG_STATE.set(Mutex::new(DEFAULT_RNG_SEED));
    }
}

#[cfg(test)]
fn expected_uniform_sequence(count: usize) -> Vec<f64> {
    let mut seed = DEFAULT_RNG_SEED;
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        seq.push(next_uniform_state(&mut seed));
    }
    seq
}

#[cfg(test)]
fn expected_complex_sequence(count: usize) -> Vec<(f64, f64)> {
    let mut seed = DEFAULT_RNG_SEED;
    let mut seq = Vec::with_capacity(count);
    for _ in 0..count {
        let re = next_uniform_state(&mut seed);
        let im = next_uniform_state(&mut seed);
        seq.push((re, im));
    }
    seq
}

#[cfg(test)]
static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

#[cfg(test)]
fn test_lock() -> &'static Mutex<()> {
    TEST_MUTEX.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn rand_default_scalar() {
        let _guard = test_lock().lock().unwrap();
        reset_rng();
        let result = rand_builtin(Vec::new()).expect("rand");
        let expected = expected_uniform_sequence(1)[0];
        match result {
            Value::Num(v) => {
                assert!(v >= 0.0 && v < 1.0);
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar double, got {other:?}"),
        }
    }

    #[test]
    fn rand_square_from_single_dimension() {
        let _guard = test_lock().lock().unwrap();
        reset_rng();
        let args = vec![Value::Num(3.0)];
        let result = rand_builtin(args).expect("rand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = expected_uniform_sequence(9);
                assert_eq!(t.data.len(), expected.len());
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn rand_like_tensor_infers_shape() {
        let _guard = test_lock().lock().unwrap();
        reset_rng();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = rand_builtin(args).expect("rand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = expected_uniform_sequence(4);
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn rand_like_complex_produces_complex_tensor() {
        let _guard = test_lock().lock().unwrap();
        reset_rng();
        let args = vec![
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("like"),
            Value::Complex(0.0, 1.0),
        ];
        let result = rand_builtin(args).expect("rand");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = expected_complex_sequence(4);
                for ((re, im), (eref, eim)) in t.data.iter().zip(expected.iter()) {
                    assert!((*re - *eref).abs() < 1e-12);
                    assert!((*im - *eim).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn rand_gpu_like_uniform() {
        let _guard = test_lock().lock().unwrap();
        reset_rng();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = rand_builtin(args).expect("rand");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered =
                        test_support::gather(Value::GpuTensor(gpu)).expect("gather to host");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for value in gathered.data {
                        assert!(value >= 0.0 && value < 1.0);
                    }
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
    fn rand_wgpu_like_uniform_and_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        // Create a GPU prototype and request rand like it
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView { data: &tensor.data, shape: &tensor.shape };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).expect("upload");
        let result = rand_like(&Value::GpuTensor(handle), &[2, 2]).expect("rand like gpu");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
                for v in gathered.data { assert!(v >= 0.0 && v < 1.0); }
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn rand_wgpu_fusion_then_sin_then_sum() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let r = rand_double(&[2, 2]).expect("rand");
        let s = crate::call_builtin("sin", &[r]).expect("sin");
        let summed = crate::call_builtin("sum", &[s, Value::Num(1.0)]).expect("sum");
        let gathered = test_support::gather(summed).expect("gather");
        assert_eq!(gathered.shape, vec![1, 2]);
    }
}
