//! MATLAB-compatible `ones` builtin with GPU-aware semantics.
//!
//! Mirrors MATLAB's `ones` semantics across scalar, vector, matrix, and N-D
//! invocations, including `'like'` prototypes, logical outputs, and GPU residency.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionExprContext,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "ones"
category: "array/creation"
keywords: ["ones", "array", "logical", "gpu", "like"]
summary: "Create arrays filled with ones with MATLAB-compatible semantics."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider one-allocation hooks when available; otherwise fills via scalar add or uploads from the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::ones::tests"
  integration: "builtins::array::ones::tests::ones_gpu_like_alloc"
---

# MATLAB / RunMat `ones` Function
`ones` creates arrays filled with ones. It mirrors MATLAB semantics across the scalar,
vector, matrix, and N-D forms, including `'like'` and `'logical'` options.

## Behaviour
- `ones()` returns the scalar `1`.
- `ones(n)` returns an `n × n` double array.
- `ones(m, n, ...)` returns a dense double array with the requested dimensions.
- `ones(sz)` accepts a size vector (row or column) and returns an array with `prod(sz)` elements arranged using MATLAB column-major ordering.
- `ones(A)` returns an array of ones with the same size (and device residency) as `A`.
- `ones(___, 'logical')` returns a logical array instead of double precision (all elements set to `true`).
- `ones(___, 'like', prototype)` matches the device residency and numeric/logical flavour of `prototype`.

## GPU Execution
When the prototype or `'like'` argument is a GPU tensor, RunMat asks the active acceleration
provider to allocate a one-filled buffer. Acceleration providers that do not yet support the dedicated hooks
fall back to zero-allocation plus scalar fill or, as a last resort, upload a host tensor.
This guarantees correct behaviour while documenting the extra transfer cost.

## Examples

```matlab
B = ones(3);
```

```matlab
mask = ones(4, 1, 'logical');
```

```matlab
G = gpuArray(rand(64, 64));
H = ones(64, 64, 'like', G);
```

## See Also
[`zeros`], [`eye`], [`gpuArray`], [`gather`]
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ones",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("ones"),
        ProviderHook::Custom("ones_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates device ones when providers expose dedicated hooks; otherwise falls back to scalar fill or host upload.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ones",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let literal = match ctx.scalar_ty {
                ScalarType::F32 => "1.0".to_string(),
                ScalarType::F64 => "f64(1.0)".to_string(),
                ScalarType::I32 => "1".to_string(),
                ScalarType::Bool => "true".to_string(),
            };
            Ok(literal)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner materialises ones as inline literals; providers may substitute inexpensive fill kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("ones", DOC_MD);

#[runtime_builtin(
    name = "ones",
    category = "array/creation",
    summary = "Create arrays filled with ones.",
    keywords = "ones,array,logical,gpu,like",
    accel = "array_construct"
)]
fn ones_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let parsed = ParsedOnes::parse(rest)?;
    build_output(parsed)
}

struct ParsedOnes {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Logical,
    Like(Value),
}

impl ParsedOnes {
    fn parse(args: Vec<Value>) -> Result<Self, String> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut implicit_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err("ones: multiple 'like' specifications are not supported"
                                .to_string());
                        }
                        if class_override.is_some() {
                            return Err(
                                "ones: cannot combine 'like' with other class specifiers"
                                    .to_string(),
                            );
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("ones: expected prototype after 'like'".to_string());
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto)?);
                        }
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err("ones: cannot combine 'like' with 'logical'".to_string());
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err("ones: cannot combine 'like' with 'double'".to_string());
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err(
                            "ones: single precision output is not implemented yet".to_string()
                        );
                    }
                    other => {
                        return Err(format!("ones: unrecognised option '{other}'"));
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

        Ok(Self { shape, template })
    }
}

fn build_output(parsed: ParsedOnes) -> Result<Value, String> {
    match parsed.template {
        OutputTemplate::Double => ones_double(&parsed.shape),
        OutputTemplate::Logical => ones_logical(&parsed.shape),
        OutputTemplate::Like(proto) => ones_like(&proto, &parsed.shape),
    }
}

fn ones_double(shape: &[usize]) -> Result<Value, String> {
    let tensor = tensor::ones(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn ones_logical(shape: &[usize]) -> Result<Value, String> {
    let len = tensor::element_count(shape);
    LogicalArray::new(vec![1u8; len], shape.to_vec())
        .map(Value::LogicalArray)
        .map_err(|e| format!("ones: {e}"))
}

fn ones_like(proto: &Value, shape: &[usize]) -> Result<Value, String> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => ones_logical(shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            let len = tensor::element_count(shape);
            let data = vec![(1.0, 0.0); len];
            ComplexTensor::new(data, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|e| format!("ones: {e}"))
        }
        Value::GpuTensor(handle) => ones_like_gpu(handle, shape),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => ones_double(shape),
        Value::CharArray(_) | Value::Cell(_) => ones_double(shape),
        _ => ones_double(shape),
    }
}

fn ones_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let attempt = if handle.shape == shape {
            provider.ones_like(handle)
        } else {
            provider.ones(shape)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        if let Ok(zero_handle) = provider.zeros(shape) {
            let add_result = provider.scalar_add(&zero_handle, 1.0);
            let _ = provider.free(&zero_handle);
            if let Ok(filled) = add_result {
                return Ok(Value::GpuTensor(filled));
            }
        }

        if let Ok(host) = tensor::ones(shape) {
            let view = HostTensorView {
                data: &host.data,
                shape: &host.shape,
            };
            if let Ok(gpu) = provider.upload(&view) {
                return Ok(Value::GpuTensor(gpu));
            }
        }
    }

    let gathered = crate::dispatcher::gather_if_needed(&Value::GpuTensor(handle.clone()))
        .map_err(|e| format!("ones: {e}"))?;
    ones_like(&gathered, shape)
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
                return Err("ones: matrix dimensions must be non-negative".to_string());
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
        return Err("ones: dimensions must be finite".to_string());
    }
    if n < 0.0 {
        return Err("ones: matrix dimensions must be non-negative".to_string());
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err("ones: dimensions must be integers".to_string());
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

fn dims_from_logical(_logical: &LogicalArray) -> Result<Option<Vec<usize>>, String> {
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
        other => Err(format!("ones: unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn ones_default_scalar() {
        let result = ones_builtin(Vec::new()).expect("ones");
        assert_eq!(result, Value::Num(1.0));
    }

    #[test]
    fn ones_square_from_single_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = ones_builtin(args).expect("ones");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&x| x == 1.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn ones_rectangular_from_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = ones_builtin(args).expect("ones");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 4]);
                assert!(t.data.iter().all(|&x| x == 1.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn ones_from_size_vector() {
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = ones_builtin(args).expect("ones");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn ones_logical_output() {
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("logical")];
        let result = ones_builtin(args).expect("ones");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                assert!(logical.data.iter().all(|&x| x == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn ones_like_tensor_infers_shape() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = ones_builtin(args).expect("ones");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t.data.iter().all(|&x| x == 1.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn ones_like_complex_scalar() {
        let args = vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = ones_builtin(args).expect("ones");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&(re, im)| (re, im) == (1.0, 0.0)));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ones_like_logical_array() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let args = vec![Value::LogicalArray(logical)];
        let result = ones_builtin(args).expect("ones");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert!(out.data.iter().all(|&x| x == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn ones_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
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
            let result = ones_builtin(args).expect("ones");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.data.iter().all(|&x| x == 1.0));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
