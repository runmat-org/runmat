//! MATLAB-compatible `rand` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{ComplexTensor, NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::sync::OnceLock;

use crate::builtins::common::random;
use crate::builtins::common::random_args::{
    complex_tensor_into_value, extract_dims, keyword_of, shape_from_value,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "rand",
        builtin_path = "crate::builtins::array::creation::rand"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "rand"
category: "array/creation"
keywords: ["rand", "random", "uniform", "gpu", "like"]
summary: "Uniform random numbers on (0, 1) within the MATLAB language."
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
  returns single-precision results that mirror MATLAB's behaviour.

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
[randn](./randn), [randi](./randi), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `rand` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/rand.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/rand.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::rand")]
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::rand")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rand",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random generation is treated as a sink and is not eligible for fusion.",
};

#[runtime_builtin(
    name = "rand",
    category = "array/creation",
    summary = "Uniform random numbers on (0, 1).",
    keywords = "rand,random,uniform,gpu,like",
    accel = "array_construct",
    builtin_path = "crate::builtins::array::creation::rand"
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
    Single,
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
                        shape_source = Some(shape_from_value(&proto, "rand")?);
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        template = Some(RandTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        template = Some(RandTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(format!("rand: unrecognised option '{other}'"));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg, "rand")? {
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
                shape_source = Some(shape_from_value(&arg, "rand")?);
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
        RandTemplate::Single => rand_single(&parsed.shape),
        RandTemplate::Like(proto) => rand_like(&proto, &parsed.shape),
    }
}

fn rand_double(shape: &[usize]) -> Result<Value, String> {
    if let Some(value) = try_gpu_uniform(shape, NumericDType::F64)? {
        return Ok(value);
    }
    let len = tensor::element_count(shape);
    let data = random::generate_uniform(len, "rand")?;
    let tensor = Tensor::new(data, shape.to_vec()).map_err(|e| format!("rand: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn rand_like(proto: &Value, shape: &[usize]) -> Result<Value, String> {
    match proto {
        Value::GpuTensor(handle) => rand_like_gpu(handle, shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => rand_complex(shape),
        Value::Tensor(t) => match t.dtype {
            NumericDType::F32 => rand_single(shape),
            NumericDType::F64 => rand_double(shape),
        },
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => {
            rand_double(shape)
        }
        Value::CharArray(_) | Value::Cell(_) => rand_double(shape),
        other => Err(format!("rand: unsupported prototype {other:?}")),
    }
}

fn rand_single(shape: &[usize]) -> Result<Value, String> {
    if let Some(value) = try_gpu_uniform(shape, NumericDType::F32)? {
        return Ok(value);
    }
    let len = tensor::element_count(shape);
    let data = random::generate_uniform_single(len, "rand")?;
    let tensor = Tensor::new_with_dtype(data, shape.to_vec(), NumericDType::F32)
        .map_err(|e| format!("rand: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn rand_complex(shape: &[usize]) -> Result<Value, String> {
    let len = tensor::element_count(shape);
    let data = random::generate_complex(len, "rand")?;
    let tensor = ComplexTensor::new(data, shape.to_vec()).map_err(|e| format!("rand: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn rand_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let dtype = dtype_from_precision(precision);
        let attempt = if handle.shape == shape {
            provider.random_uniform_like(handle)
        } else {
            provider.random_uniform(shape)
        };
        if let Ok(gpu) = attempt {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            let len = tensor::element_count(shape);
            random::skip_uniform(len, "rand")?;
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_rand_fallback(shape, dtype, "provider-like-error");
        }

        let len = tensor::element_count(shape);
        let data = random::generate_uniform(len, "rand")?;
        let tensor = Tensor::new(data, shape.to_vec()).map_err(|e| format!("rand: {e}"))?;
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            runmat_accelerate_api::set_handle_precision(&gpu, precision);
            return Ok(Value::GpuTensor(gpu));
        } else {
            log_rand_fallback(shape, dtype, "upload-error");
        }
    } else {
        log_rand_fallback(shape, NumericDType::F32, "no-provider-like");
    }

    let gathered = crate::dispatcher::gather_if_needed(&Value::GpuTensor(handle.clone()))
        .map_err(|e| format!("rand: {e}"))?;
    log_rand_fallback(shape, NumericDType::F32, "gather-fallback");
    rand_like(&gathered, shape)
}

fn try_gpu_uniform(shape: &[usize], dtype: NumericDType) -> Result<Option<Value>, String> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        log_rand_fallback(shape, dtype, "no-provider");
        return Ok(None);
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
    };
    if provider.precision() != precision {
        log_rand_fallback(shape, dtype, "precision-mismatch");
        return Ok(None);
    }
    match provider.random_uniform(shape) {
        Ok(handle) => {
            runmat_accelerate_api::set_handle_precision(&handle, precision);
            let len = tensor::element_count(shape);
            random::skip_uniform(len, "rand")?;
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(err) => {
            log::warn!(
                "rand: provider random_uniform failed ({err}); falling back to host tensor path"
            );
            log_rand_fallback(shape, dtype, "provider-error");
            Ok(None)
        }
    }
}

fn rand_fallback_debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("RUNMAT_DEBUG_RAND_FALLBACK"),
            Ok(value) if value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
        )
    })
}

fn log_rand_fallback(shape: &[usize], dtype: NumericDType, reason: &str) {
    if !rand_fallback_debug_enabled() {
        return;
    }
    let elems = tensor::element_count(shape);
    tracing::debug!(
        dtype = ?dtype,
        elems,
        shape = ?shape,
        reason,
        "[rand_debug] fallback"
    );
}

fn dtype_from_precision(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};

    fn reset_rng_clean() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_default_scalar() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let result = rand_builtin(Vec::new()).expect("rand");
        let expected = random::expected_uniform_sequence(1)[0];
        match result {
            Value::Num(v) => {
                assert!((0.0..1.0).contains(&v));
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar double, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_square_from_single_dimension() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![Value::Num(3.0)];
        let result = rand_builtin(args).expect("rand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = random::expected_uniform_sequence(9);
                assert_eq!(t.data.len(), expected.len());
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_like_tensor_infers_shape() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = rand_builtin(args).expect("rand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = random::expected_uniform_sequence(4);
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_single_matrix_has_f32_dtype() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("single")];
        let result = rand_builtin(args).expect("rand single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.dtype, NumericDType::F32);
                let expected = random::expected_uniform_sequence(4)
                    .into_iter()
                    .map(|v| {
                        let val = v as f32;
                        val as f64
                    })
                    .collect::<Vec<f64>>();
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-7);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_like_complex_produces_complex_tensor() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
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
                let expected = random::expected_complex_sequence(4);
                for ((re, im), (eref, eim)) in t.data.iter().zip(expected.iter()) {
                    assert!((*re - *eref).abs() < 1e-12);
                    assert!((*im - *eim).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_gpu_like_uniform() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
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
                        assert!((0.0..1.0).contains(&value));
                    }
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
    fn rand_wgpu_like_uniform_and_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        // Create a GPU prototype and request rand like it
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).expect("upload");
        let result = rand_like(&Value::GpuTensor(handle), &[2, 2]).expect("rand like gpu");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
                for v in gathered.data {
                    assert!((0.0..1.0).contains(&v));
                }
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rand_wgpu_single_allocates_gpu_without_like() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let value = rand_single(&[2, 2]).expect("rand single");
        match value {
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }
}
