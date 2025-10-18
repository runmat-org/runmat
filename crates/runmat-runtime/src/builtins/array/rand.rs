//! MATLAB-compatible `rand` builtin with GPU-aware semantics and rich documentation.
//!
//! Supports the common scalar, matrix, size-vector, and `'like'` forms while honouring
//! provider hooks for device-side random number generation and falling back to host
//! uploads when a backend does not expose dedicated RNG kernels.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

use std::sync::{Mutex, OnceLock};

#[cfg_attr(not(test), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "rand"
category: "array/creation"
keywords: ["rand", "random", "uniform", "gpu", "like"]
summary: "Uniform random numbers with MATLAB-compatible size and prototype semantics."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider random generation hooks when available; otherwise uploads host-generated uniforms."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::rand::tests"
  integration: "builtins::array::rand::tests::rand_gpu_like_upload"
---

# MATLAB / RunMat `rand` Function
`rand` draws independent samples from the continuous uniform distribution on the interval `[0, 1)`.
The builtin mirrors MATLAB semantics across scalar, matrix, N-D, and `'like'` forms while integrating
with RunMat Accelerate for device-resident tensors.

## Behaviour
- `rand()` returns a scalar double in `[0, 1)`.
- `rand(n)` returns an `n × n` double array.
- `rand(m, n, ...)` returns an array with the requested dimensions.
- `rand(sz)` accepts a size vector (row or column) and returns an array with `prod(sz)` elements.
- `rand(___, 'double')` is accepted for compatibility (doubles are the default).
- `rand(___, 'single')` is currently unsupported and raises a descriptive error.
- `rand(___, 'like', prototype)` matches the class and device residency of `prototype`.
  Complex prototypes receive complex samples where both the real and imaginary parts follow `U(0, 1)`.

## GPU Execution
When the prototype lives on a GPU (or `'like'` targets a GPU tensor), RunMat first tries the provider’s
dedicated RNG hooks. Providers that do not yet expose these hooks automatically fall back to generating
host data and uploading it once, guaranteeing correctness while clearly documenting the extra transfer.

## Examples

```matlab
v = rand(1, 5);
```

```matlab
G = gpuArray(rand(64, 64));
H = rand(64, 64, 'like', G);
```

```matlab
Z = complex(zeros(2));
R = rand(2, 2, 'like', Z);   % complex results with independent real/imag parts
```

## See Also
[`randn`], [`randi`], [`gpuArray`], [`gather`]
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rand",
    op_kind: GpuOpKind::Custom("random"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("rand_uniform"),
        ProviderHook::Custom("rand_uniform_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may generate uniform samples in-place; the runtime uploads host-generated data when hooks are unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rand",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random generation currently bypasses fusion; planners treat rand as a standalone allocation.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[runtime_builtin(
    name = "rand",
    category = "array/creation",
    summary = "Uniform random numbers in [0, 1).",
    keywords = "rand,random,uniform,gpu,like"
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
        let mut template: Option<RandTemplate> = None;
        let mut prototype_shape: Option<Vec<usize>> = None;

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
                        prototype_shape = Some(shape_from_value(&proto)?);
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        template = Some(RandTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err(
                            "rand: single precision output is not implemented yet".to_string()
                        );
                    }
                    other => {
                        return Err(format!("rand: unrecognised option '{other}'"));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg)? {
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            if template.is_none() {
                template = Some(RandTemplate::Like(arg.clone()));
            }
            if prototype_shape.is_none() {
                prototype_shape = Some(shape_from_value(&arg)?);
            }
            idx += 1;
        }

        let shape = if !dims.is_empty() {
            if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = prototype_shape {
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
    let tensor = rand_tensor(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn rand_tensor(shape: &[usize]) -> Result<Tensor, String> {
    let len = tensor::element_count(shape);
    let mut data = vec![0.0f64; len];
    fill_uniform(&mut data)?;
    Tensor::new(data, shape.to_vec()).map_err(|e| format!("rand: {e}"))
}

fn rand_complex(shape: &[usize]) -> Result<Value, String> {
    let len = tensor::element_count(shape);
    let mut data = vec![(0.0f64, 0.0f64); len];
    fill_uniform_complex(&mut data)?;
    ComplexTensor::new(data, shape.to_vec())
        .map(Value::ComplexTensor)
        .map_err(|e| format!("rand: {e}"))
}

fn rand_like(proto: &Value, shape: &[usize]) -> Result<Value, String> {
    match proto {
        Value::GpuTensor(handle) => rand_like_gpu(handle, shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => rand_complex(shape),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => rand_double(shape),
        Value::LogicalArray(_) | Value::Bool(_) | Value::CharArray(_) => rand_double(shape),
        Value::Struct(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::Cell(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Object(_)
        | Value::MException(_) => {
            Err("rand: prototype must be numeric or a GPU tensor".to_string())
        }
        _ => rand_double(shape),
    }
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

        if let Ok(tensor) = rand_tensor(shape) {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            if let Ok(gpu) = provider.upload(&view) {
                return Ok(Value::GpuTensor(gpu));
            }
            return Ok(tensor::tensor_into_value(tensor));
        }
    }

    rand_double(shape)
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
        Value::LogicalArray(l) => dims_from_logical(l),
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

fn dims_from_logical(
    _logical: &runmat_builtins::LogicalArray,
) -> Result<Option<Vec<usize>>, String> {
    Ok(None)
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

fn fill_uniform(data: &mut [f64]) -> Result<(), String> {
    if data.is_empty() {
        return Ok(());
    }
    let mut guard = rng_state()
        .lock()
        .map_err(|_| "rand: failed to acquire RNG lock".to_string())?;
    for slot in data.iter_mut() {
        *slot = next_uniform(&mut *guard);
    }
    Ok(())
}

fn fill_uniform_complex(data: &mut [(f64, f64)]) -> Result<(), String> {
    if data.is_empty() {
        return Ok(());
    }
    let mut guard = rng_state()
        .lock()
        .map_err(|_| "rand: failed to acquire RNG lock".to_string())?;
    for slot in data.iter_mut() {
        let re = next_uniform(&mut *guard);
        let im = next_uniform(&mut *guard);
        *slot = (re, im);
    }
    Ok(())
}

fn rng_state() -> &'static Mutex<u64> {
    static STATE: OnceLock<Mutex<u64>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(0x9e3779b97f4a7c15))
}

fn next_uniform(state: &mut u64) -> f64 {
    const MULTIPLIER: u64 = 6364136223846793005;
    const INCREMENT: u64 = 1;
    const SHIFT: u32 = 11;
    const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);

    *state = state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
    let bits = *state >> SHIFT;
    (bits as f64) * SCALE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    fn extract_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("expected tensor-compatible value, got {other:?}"),
        }
    }

    #[test]
    fn rand_default_scalar() {
        let value = rand_builtin(Vec::new()).expect("rand");
        match value {
            Value::Num(n) => assert!((0.0..1.0).contains(&n)),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn rand_square_from_single_dimension() {
        let args = vec![Value::Num(3.0)];
        let value = rand_builtin(args).expect("rand");
        let tensor = extract_tensor(value);
        assert_eq!(tensor.shape, vec![3, 3]);
        assert!(tensor.data.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn rand_rectangular_from_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let value = rand_builtin(args).expect("rand");
        let tensor = extract_tensor(value);
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.data.len(), 8);
        assert!(tensor.data.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn rand_from_size_vector() {
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let value = rand_builtin(args).expect("rand");
        let tensor = extract_tensor(value);
        assert_eq!(tensor.shape, vec![2, 3]);
    }

    #[test]
    fn rand_like_tensor_infers_shape() {
        let template = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(template)];
        let value = rand_builtin(args).expect("rand");
        let tensor = extract_tensor(value);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(tensor.data.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn rand_like_complex_returns_complex_tensor() {
        let args = vec![
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("like"),
            Value::Complex(0.0, 1.0),
        ];
        let value = rand_builtin(args).expect("rand");
        match value {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
                for &(re, im) in &ct.data {
                    assert!((0.0..1.0).contains(&re));
                    assert!((0.0..1.0).contains(&im));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn rand_gpu_like_upload() {
        test_support::with_test_provider(|provider| {
            let template = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &template.data,
                shape: &template.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let value = rand_builtin(args).expect("rand");
            match value {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.data.iter().all(|&x| (0.0..1.0).contains(&x)));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn rand_single_not_supported() {
        let args = vec![Value::Num(2.0), Value::from("single")];
        let err = rand_builtin(args).unwrap_err();
        assert!(err.contains("single precision"));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
