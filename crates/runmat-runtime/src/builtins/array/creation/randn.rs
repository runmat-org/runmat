//! MATLAB-compatible `randn` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random;
use crate::builtins::common::random_args::{
    complex_tensor_into_value, extract_dims, keyword_of, shape_from_value,
};
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
title: "randn"
category: "array/creation"
keywords: ["randn", "random", "normal", "gaussian", "gpu", "like"]
summary: "Standard normal random numbers that mirror MATLAB semantics."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider random_normal hooks when available; otherwise generates samples on the host and uploads them to keep GPU residency."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "none"
requires_feature: null
tested:
  unit: "builtins::array::creation::randn::tests"
  integration: "builtins::array::creation::randn::tests::randn_gpu_like_roundtrip"
---

# What does the `randn` function do in MATLAB / RunMat?
`randn` draws pseudorandom samples from the standard normal distribution (`μ = 0`, `σ = 1`). RunMat matches MATLAB call patterns for scalars, explicit dimension lists, size vectors, and `'like'` prototypes while honouring GPU residency whenever an acceleration provider is active.

## How does the `randn` function behave in MATLAB / RunMat?
- `randn()` returns a scalar double drawn from `𝒩(0, 1)`.
- `randn(n)` returns an `n × n` dense double matrix.
- `randn(m, n, ...)` accepts an arbitrary number of dimension arguments.
- `randn(sz)` accepts a size vector (row or column) and returns an array with shape `sz`.
- `randn(A)` or `randn(___, 'like', A)` matches both the shape and residency of `A`, including GPU tensors and complex prototypes.
- Complex prototypes yield complex Gaussian samples with independent `𝒩(0, 1)` real and imaginary parts.
- Class specifiers currently support `'double'`; other classes (e.g., `'single'`) emit descriptive errors until native representations land.

## `randn` Function GPU Execution Behaviour
When the output or `'like'` prototype lives on the GPU, RunMat calls into the active acceleration provider via `random_normal` / `random_normal_like`. Providers without these hooks fall back to host generation followed by a single upload, ensuring the resulting tensor still resides on device even if samples were produced on the CPU.

## Examples of using the `randn` function in MATLAB / RunMat

### Drawing a single standard normal variate

```matlab
rng(0);
z = randn();
```

Expected output:

```matlab
z = 1.8179
```

### Creating a matrix of Gaussian noise

```matlab
rng(0);
E = randn(2, 3);
```

Expected output:

```matlab
E =
    1.8179    0.3895    0.9838
   -1.1645    0.4175    0.1386
```

### Specifying dimensions with a size vector

```matlab
rng(0);
shape = [2 2 2];
T = randn(shape);
```

Expected output (pages shown along the third dimension):

```matlab
T(:, :, 1) =
    1.8179    0.3895
   -1.1645    0.4175

T(:, :, 2) =
    0.9838   -1.1226
    0.1386    2.7430
```

### Matching an existing `gpuArray` prototype

```matlab
rng(0);
G = gpuArray.zeros(512, 512);
noise = randn('like', G);
stats = [mean(gather(noise(:))) std(gather(noise(:)))];
```

Expected output:

```matlab
size(noise)
ans =
       512   512

stats =
   -0.0009    0.9986
```

### Generating complex Gaussian samples

```matlab
rng(0);
z = randn(3, 1, 'like', 1 + 1i);
```

Expected output:

```matlab
z =
   1.8179 - 1.1645i
   0.3895 + 0.4175i
   0.9838 + 0.1386i
```

### Producing reproducible noise for Monte Carlo tests

```matlab
rng(0);
samples = randn(1, 5);
```

Expected output:

```matlab
samples = [1.8179  -1.1645   0.3895   0.4175   0.9838]
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do **not** need to call `gpuArray` explicitly in RunMat. The fusion planner keeps results on the GPU when downstream work benefits from device residency. However, for MATLAB compatibility—and when you want deterministic control—you can still use `gpuArray` to seed GPU execution manually.

MathWorks MATLAB lacks an integrated fusion planner and ships GPU acceleration as a separate toolbox, so MATLAB users move data manually. RunMat automates this to streamline accelerated workflows.

## FAQ

### What distribution does `randn` use?
`randn` returns samples from the standard normal distribution with mean `0` and standard deviation `1`.

### How is `randn` different from `rand`?
`randn` draws from a Gaussian distribution, whereas `rand` draws from the uniform distribution over `(0, 1)`.

### How do I control reproducibility?
Use the MATLAB-compatible `rng` builtin before calling `randn` to seed the global generator.

### Does `randn(___, 'like', A)` work with complex prototypes?
Yes. When `A` is complex, RunMat emits complex Gaussian samples whose real and imaginary parts are independent `𝒩(0, 1)` variates.

### What happens if I request `'single'` precision?
RunMat currently supports double precision. Supplying `'single'` raises a descriptive error until native single-precision tensors land.

### How does `randn` behave on the GPU?
If the active acceleration provider implements normal RNG hooks, samples are generated directly on device. Otherwise RunMat produces them on the host, uploads once, and continues execution on the GPU.

### Can I request zero-sized dimensions?
Yes. Any dimension argument equal to zero yields an empty array, matching MATLAB semantics.

### Does `randn` fuse with other operations?
No. Random generation is treated as a sink operation and excluded from fusion planning to preserve statistical properties.

## See Also
[rand](./rand), [randi](./randi), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `randn` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/randn.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/randn.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "randn",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("random_normal"),
        ProviderHook::Custom("random_normal_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Leverages provider normal RNG hooks when available; otherwise falls back to host sampling followed by a single upload to preserve GPU residency.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "randn",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random generation is treated as a sink and excluded from fusion planning.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("randn", DOC_MD);

#[runtime_builtin(
    name = "randn",
    category = "array/creation",
    summary = "Standard normal random numbers.",
    keywords = "randn,random,normal,gaussian,gpu,like",
    accel = "array_construct"
)]
fn randn_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let parsed = ParsedRandn::parse(rest)?;
    build_output(parsed)
}

struct ParsedRandn {
    shape: Vec<usize>,
    template: RandnTemplate,
    dtype: NumericDType,
}

#[derive(Clone)]
enum RandnTemplate {
    Double,
    Like(Value),
}

impl ParsedRandn {
    fn parse(args: Vec<Value>) -> Result<Self, String> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut template: Option<RandnTemplate> = None;
        let mut dtype: NumericDType = NumericDType::F64;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();

            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("randn: expected prototype after 'like'".to_string());
                        };
                        template = Some(RandnTemplate::Like(proto.clone()));
                        shape_source = Some(shape_from_value(&proto, "randn")?);
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        template = Some(RandnTemplate::Double);
                        dtype = NumericDType::F64;
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        template = Some(RandnTemplate::Double);
                        dtype = NumericDType::F32;
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(format!("randn: unrecognised option '{other}'"));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg, "randn")? {
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
                shape_source = Some(shape_from_value(&arg, "randn")?);
            }
            if template.is_none() {
                template = Some(RandnTemplate::Like(arg.clone()));
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

        let template = template.unwrap_or(RandnTemplate::Double);

        Ok(Self { shape, template, dtype })
    }
}

fn build_output(parsed: ParsedRandn) -> Result<Value, String> {
    match parsed.template {
        RandnTemplate::Double => match parsed.dtype {
            NumericDType::F64 => randn_double(&parsed.shape),
            NumericDType::F32 => randn_single(&parsed.shape),
        },
        RandnTemplate::Like(proto) => randn_like(&proto, &parsed.shape),
    }
}

fn randn_double(shape: &[usize]) -> Result<Value, String> {
    let len = tensor::element_count(shape);
    let data = random::generate_normal(len, "randn")?;
    let tensor = Tensor::new(data, shape.to_vec()).map_err(|e| format!("randn: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn randn_single(shape: &[usize]) -> Result<Value, String> {
    let len = tensor::element_count(shape);
    let data = random::generate_normal(len, "randn")?;
    let tensor = Tensor::new_with_dtype(data, shape.to_vec(), NumericDType::F32)
        .map_err(|e| format!("randn: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn randn_like(proto: &Value, shape: &[usize]) -> Result<Value, String> {
    match proto {
        Value::GpuTensor(handle) => randn_like_gpu(handle, shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => randn_complex(shape),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_) => randn_double(shape),
        Value::CharArray(_) | Value::Cell(_) => randn_double(shape),
        other => Err(format!("randn: unsupported prototype {other:?}")),
    }
}

fn randn_complex(shape: &[usize]) -> Result<Value, String> {
    let len = tensor::element_count(shape);
    let data = random::generate_normal_complex(len, "randn")?;
    let tensor = ComplexTensor::new(data, shape.to_vec()).map_err(|e| format!("randn: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn randn_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let attempt = if handle.shape == shape {
            provider.random_normal_like(handle)
        } else {
            provider.random_normal(shape)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        let len = tensor::element_count(shape);
        let data = random::generate_normal(len, "randn")?;
        let tensor = Tensor::new(data, shape.to_vec()).map_err(|e| format!("randn: {e}"))?;
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            return Ok(Value::GpuTensor(gpu));
        }
    }

    let gathered = crate::dispatcher::gather_if_needed(&Value::GpuTensor(handle.clone()))
        .map_err(|e| format!("randn: {e}"))?;
    randn_like(&gathered, shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};

    #[test]
    fn randn_default_scalar() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let result = randn_builtin(Vec::new()).expect("randn");
        let expected = random::expected_normal_sequence(1)[0];
        match result {
            Value::Num(v) => assert!((v - expected).abs() < 1e-12),
            other => panic!("expected scalar double, got {other:?}"),
        }
    }

    #[test]
    fn randn_square_from_single_dimension() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let args = vec![Value::Num(2.0)];
        let result = randn_builtin(args).expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = random::expected_normal_sequence(4);
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn randn_size_vector_argument() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let size_vec = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let result = randn_builtin(vec![Value::Tensor(size_vec)]).expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3, 4]);
                let expected = random::expected_normal_sequence(24);
                assert_eq!(t.data.len(), expected.len());
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn randn_zero_dimension_returns_empty() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let result = randn_builtin(vec![Value::Num(0.0)]).expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn randn_single_precision_errors() {
        let result = randn_builtin(vec![Value::from("single")]);
        assert!(matches!(result, Err(message) if message.contains("single precision")));
    }

    #[test]
    fn randn_like_tensor_infers_shape() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = randn_builtin(args).expect("randn");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = random::expected_normal_sequence(4);
                for (observed, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn randn_like_complex_produces_complex_tensor() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let args = vec![
            Value::Num(2.0),
            Value::Num(1.0),
            Value::from("like"),
            Value::Complex(0.0, 1.0),
        ];
        let result = randn_builtin(args).expect("randn");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let expected = random::expected_complex_normal_sequence(2);
                for ((re, im), (eref, eim)) in t.data.iter().zip(expected.iter()) {
                    assert!((*re - *eref).abs() < 1e-12);
                    assert!((*im - *eim).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn randn_gpu_like_roundtrip() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::from("like"), Value::GpuTensor(handle)];
            let result = randn_builtin(args).expect("randn");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for value in gathered.data {
                        assert!(value.is_finite());
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
    fn randn_wgpu_like_and_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).expect("upload");
        let result = randn_like(&Value::GpuTensor(handle), &[2, 2]).expect("randn like gpu");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
                for v in gathered.data {
                    assert!(v.is_finite());
                }
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn randn_wgpu_provider_random_normal() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let handle = provider
            .random_normal(&[4, 4])
            .expect("wgpu random_normal hook");
        let gathered =
            test_support::gather(Value::GpuTensor(handle)).expect("gather random_normal output");
        assert_eq!(gathered.shape, vec![4, 4]);
        assert!(
            gathered
                .data
                .iter()
                .any(|value| value.is_finite() && value.abs() > 1.0e-6),
            "expected at least one non-trivial normal sample"
        );
    }
}
