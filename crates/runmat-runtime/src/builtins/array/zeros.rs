//! MATLAB-compatible `zeros` builtin with GPU-aware semantics.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionExprContext,
    FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg_attr(not(test), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "zeros"
category: "array/creation"
keywords: ["zeros", "array", "logical", "gpu", "like"]
summary: "Create arrays filled with zeros with MATLAB-compatible semantics."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider zero-allocation hooks when available; otherwise falls back to uploading a host tensor."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::zeros::tests"
  integration: "builtins::array::zeros::tests::zeros_gpu_like_alloc"
---

# MATLAB / RunMat `zeros` Function
`zeros` creates arrays filled with zeros. It mirrors MATLAB semantics across the scalar,
vector, matrix, and N-D forms, including `'like'` and `'logical'` options.

## Behaviour
- `zeros()` returns the scalar `0`.
- `zeros(n)` returns an `n × n` double array.
- `zeros(m, n, ...)` returns a dense double array with the requested dimensions.
- `zeros(sz)` accepts a size vector (row or column) and returns an array with `prod(sz)` elements arranged using MATLAB column-major ordering.
- `zeros(A)` returns an array of zeros with the same size (and device residency) as `A`.
- `zeros(___, 'logical')` returns a logical array instead of double precision.
- `zeros(___, 'like', prototype)` matches the device residency and numeric/logical flavour of `prototype`.

## GPU Execution
When the prototype or `'like'` argument is a GPU tensor, RunMat asks the active acceleration
provider to allocate a zero-filled buffer in-place. Providers that do not yet support the zero
allocation hooks automatically fall back to uploading a host zero tensor, guaranteeing correct
behaviour at the cost of an extra transfer.

## Examples

```matlab
A = zeros(2, 3);
```

```matlab
mask = zeros(4, 1, 'logical');
```

```matlab
G = gpuArray(rand(128, 128));
H = zeros(128, 128, 'like', G);
```

## See Also
[`ones`], [`eye`], [`gpuArray`], [`gather`]
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "zeros",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("zeros"),
        ProviderHook::Custom("zeros_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates device zeros when providers expose dedicated hooks; otherwise falls back to host upload.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "zeros",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                ScalarType::I32 => "0".to_string(),
                ScalarType::Bool => "false".to_string(),
            };
            Ok(zero)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner materialises zeros as literal constants; providers may substitute inexpensive fill kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[runtime_builtin(
    name = "zeros",
    category = "array/creation",
    summary = "Create arrays filled with zeros.",
    keywords = "zeros,array,logical,gpu,like"
)]
fn zeros_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let parsed = ParsedZeros::parse(rest)?;
    build_output(parsed)
}

struct ParsedZeros {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Logical,
    Like(Value),
}

impl ParsedZeros {
    fn parse(args: Vec<Value>) -> Result<Self, String> {
        let mut dims: Vec<usize> = Vec::new();
        let mut template: Option<OutputTemplate> = None;
        let mut prototype_shape: Option<Vec<usize>> = None;

        let mut i = 0;
        while i < args.len() {
            let arg = args[i].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        let Some(proto) = args.get(i + 1).cloned() else {
                            return Err("zeros: expected prototype after 'like'".to_string());
                        };
                        template = Some(OutputTemplate::Like(proto.clone()));
                        prototype_shape = Some(shape_from_value(&proto)?);
                        i += 2;
                        continue;
                    }
                    "logical" => {
                        template = Some(OutputTemplate::Logical);
                        i += 1;
                        continue;
                    }
                    "double" => {
                        template = Some(OutputTemplate::Double);
                        i += 1;
                        continue;
                    }
                    "single" => {
                        return Err(
                            "zeros: single precision output is not implemented yet".to_string()
                        );
                    }
                    other => {
                        return Err(format!("zeros: unrecognised option '{other}'"));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg)? {
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                i += 1;
                continue;
            }

            if template.is_none() {
                template = Some(OutputTemplate::Like(arg.clone()));
            }
            if prototype_shape.is_none() {
                prototype_shape = Some(shape_from_value(&arg)?);
            }
            i += 1;
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

        let template = template.unwrap_or(OutputTemplate::Double);
        Ok(Self { shape, template })
    }
}

fn build_output(parsed: ParsedZeros) -> Result<Value, String> {
    match parsed.template {
        OutputTemplate::Double => zeros_double(&parsed.shape),
        OutputTemplate::Logical => zeros_logical(&parsed.shape),
        OutputTemplate::Like(proto) => zeros_like(&proto, &parsed.shape),
    }
}

fn zeros_double(shape: &[usize]) -> Result<Value, String> {
    let tensor = tensor::zeros(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn zeros_logical(shape: &[usize]) -> Result<Value, String> {
    Ok(Value::LogicalArray(LogicalArray::zeros(shape.to_vec())))
}

fn zeros_like(proto: &Value, shape: &[usize]) -> Result<Value, String> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => zeros_logical(shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            let tensor = ComplexTensor::zeros(shape.to_vec());
            Ok(Value::ComplexTensor(tensor))
        }
        Value::GpuTensor(handle) => zeros_like_gpu(handle, shape),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => zeros_double(shape),
        Value::CharArray(_) | Value::Cell(_) => zeros_double(shape),
        _ => zeros_double(shape),
    }
}

fn zeros_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let attempt = if handle.shape == shape {
            provider.zeros_like(handle)
        } else {
            provider.zeros(shape)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        let host = tensor::zeros(shape)?;
        let view = HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            return Ok(Value::GpuTensor(gpu));
        }
    }

    let gathered = crate::dispatcher::gather_if_needed(&Value::GpuTensor(handle.clone()))
        .map_err(|e| format!("zeros: {e}"))?;
    zeros_like(&gathered, shape)
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
                return Err("zeros: matrix dimensions must be non-negative".to_string());
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
        return Err("zeros: dimensions must be finite".to_string());
    }
    if n < 0.0 {
        return Err("zeros: matrix dimensions must be non-negative".to_string());
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err("zeros: dimensions must be integers".to_string());
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

fn dims_from_logical(logical: &LogicalArray) -> Result<Option<Vec<usize>>, String> {
    let _ = logical;
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
        other => Err(format!("zeros: unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn zeros_default_scalar() {
        let result = zeros_builtin(Vec::new()).expect("zeros");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn zeros_square_from_single_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = zeros_builtin(args).expect("zeros");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&x| x == 0.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn zeros_rectangular_from_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = zeros_builtin(args).expect("zeros");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 4]);
                assert_eq!(t.data.len(), 8);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn zeros_from_size_vector() {
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = zeros_builtin(args).expect("zeros");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn zeros_logical_output() {
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("logical")];
        let result = zeros_builtin(args).expect("zeros");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                assert!(logical.data.iter().all(|&x| x == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn zeros_like_tensor_infers_shape() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = zeros_builtin(args).expect("zeros");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t.data.iter().all(|&x| x == 0.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn zeros_like_complex_scalar() {
        let args = vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = zeros_builtin(args).expect("zeros");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert!(t.data.iter().all(|&(re, im)| re == 0.0 && im == 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn zeros_gpu_like_alloc() {
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
            let result = zeros_builtin(args).expect("zeros");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.data.iter().all(|&x| x == 0.0));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
