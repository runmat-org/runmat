//! MATLAB-compatible `string.empty` builtin for RunMat.

use runmat_builtins::{StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{extract_dims, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

const LABEL: &str = "string.empty";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "string.empty",
        wasm_path = "crate::builtins::strings::core::string_empty"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "string.empty"
category: "strings/core"
keywords: ["string.empty", "empty string array", "preallocate text", "size vector", "0-by-N", "'like'"]
summary: "Construct empty string arrays with MATLAB-compatible dimension semantics and 'like' prototypes."
references:
  - https://www.mathworks.com/help/matlab/ref/string.empty.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Creates host string arrays; GPU tensors are neither read nor written."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::string_empty::tests"
  integration: "builtins::strings::core::string_empty::tests::doc_examples_present"
---

# What does the `string.empty` function do in MATLAB / RunMat?
`string.empty` constructs an empty string array. By default it returns a `0×0` array, and when you
specify additional dimensions they define the trailing extents while the leading dimension remains
zero, ensuring the total element count is zero. This mirrors MATLAB's static `string.empty` method.

## How does the `string.empty` function behave in MATLAB / RunMat?
- `string.empty` with no arguments yields a `0×0` string array.
- `string.empty(n)` produces a `0×n` array. The leading dimension is fixed at `0`, so the result is
  still empty even if `n > 0`.
- `string.empty(m, n, p, ...)` returns a `0×n×p×…` array. All trailing dimensions are honoured while the leading dimension remains zero.
- You can provide a single size vector such as `string.empty([0 5 3])`; the first entry is ignored
  beyond confirming it is non-negative, and the remaining entries set the trailing dimensions.
- `string.empty(___, 'like', prototype)` copies the trailing dimensions from `prototype` when you do
  not supply explicit sizes. Any dimensions you pass explicitly take precedence. GPU-resident
  prototypes are automatically gathered so their shape can be inspected.
- Size inputs must be finite, real, non-negative integers. Fractional or negative values produce a
  MATLAB-compatible error.
- The result always resides on the host; there is no GPU counterpart for string arrays.

## `string.empty` GPU Execution Behaviour
`string.empty` does not allocate or interact with GPU memory. It is a pure host constructor that
instantly returns the requested shape metadata and an empty data buffer. When the runtime is
executing under RunMat Accelerate, no provider hooks are invoked. `'like'` prototypes that happen to
live on the GPU are gathered to the host before their shape is examined.

## Examples of using the `string.empty` function in MATLAB / RunMat

### Creating a 0x0 string array
```matlab
S = string.empty;
```
Expected output:
```matlab
S =
  0x0 string array
```

### Building a 0xN string row vector
```matlab
row = string.empty(5);
```
Expected output:
```matlab
row =
  0x5 string array
```

### Creating a 0xN string array with extra dimensions
```matlab
cube = string.empty(0, 4, 3);
```
Expected output:
```matlab
cube =
  0x4x3 string array
```

### Using a size vector with string.empty
```matlab
sz = [0 2 5];
grid = string.empty(sz);
```
Expected output:
```matlab
grid =
  0x2x5 string array
```

### Resetting a preallocated string array to empty
```matlab
A = strings(3, 2);  % Some application-specific strings
A = string.empty(size(A));
```
Expected output:
```matlab
A =
  0x2 string array
```

### Preserving higher-dimensional layout while empty
```matlab
layout = string.empty([2 0 4 6]);
```
Expected output:
```matlab
layout =
  0x0x4x6 string array
```

### Reusing the shape of an existing array with `'like'`
```matlab
proto = strings(3, 2);
sameCols = string.empty('like', proto);
```
Expected output:
```matlab
sameCols =
  0x2 string array
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `string.empty` allocates metadata for an empty string array entirely on the host. Because the
result contains no elements and string scalars are host-only, there is nothing to transfer to or from
the GPU. Using `gpuArray` with `string.empty` has no effect and is unnecessary.

## FAQ

### Why is the first dimension always zero?
MATLAB defines `classname.empty` so that the leading dimension is zero, guaranteeing the result
contains no elements. RunMat mirrors this rule for perfect compatibility.

### Can I request negative or fractional dimensions?
No. Dimensions must be finite, non-negative integers. Any other input raises a descriptive error.

### Does `string.empty(n)` create space for `n` elements?
No. It returns a `0×n` array, which still has zero elements. Use `strings(n)` if you want an array of
string scalars that you can fill later.

### Can I combine scalars and size vectors?
Yes. Calls like `string.empty([0 3], 5)` flatten to `string.empty(0, 3, 5)` internally.

### What does the `'like'` option do?
`'like', prototype` copies the trailing dimensions from `prototype` when you omit explicit sizes.
The first dimension is still forced to `0`, so the result remains empty. The prototype is gathered
automatically if it resides on the GPU.

### Does the result share storage with existing arrays?
No. Every call returns a new handle. Because the array is empty, the data buffer is an empty vector
and consumes negligible memory.

### Is there a GPU-accelerated variant?
No. String arrays live on the host in RunMat, and this builtin never touches GPU memory.

### How do I obtain a 0x0 string array quickly?
Call `string.empty` with no arguments. It is equivalent to `strings(0)` but makes the intention
explicit.

### Can I use `size` output directly?
Yes. Expressions like `string.empty(size(existingArray))` are supported. The first element of the
size vector is ignored when constructing the new array so that the first dimension is zero.

### What happens if I pass an empty array as the size vector?
`string.empty([])` returns the canonical `0×0` string array, just like calling `string.empty` with no
arguments.

### Does `string.empty` ever throw away extra arguments?
Only when they cannot be interpreted as dimensions. In that case RunMat throws an error rather than
guessing.

## See Also
`string`, `strings`, `char`, `zeros`, `ones`
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::strings::core::string_empty")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "string.empty",
    op_kind: GpuOpKind::Custom("constructor"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only constructor that returns a new empty string array without contacting GPU providers.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::strings::core::string_empty")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "string.empty",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Pure constructor; fusion planner treats calls as non-fusable sinks.",
};

#[runtime_builtin(
    name = "string.empty",
    category = "strings/core",
    summary = "Construct an empty string array with MATLAB-compatible dimensions.",
    keywords = "string.empty,empty,string array,preallocate",
    accel = "none",
    wasm_path = "crate::builtins::strings::core::string_empty"
)]
fn string_empty_builtin(rest: Vec<Value>) -> Result<Value, String> {
    let shape = parse_shape(&rest)?;
    let total: usize = shape.iter().product();
    debug_assert_eq!(total, 0, "string.empty must produce an empty array");
    let data = Vec::<String>::new();
    let array = StringArray::new(data, shape).map_err(|e| format!("{LABEL}: {e}"))?;
    Ok(Value::StringArray(array))
}

fn parse_shape(args: &[Value]) -> Result<Vec<usize>, String> {
    if args.is_empty() {
        return Ok(vec![0, 0]);
    }

    let mut explicit_dims: Vec<usize> = Vec::new();
    let mut like_shape: Option<Vec<usize>> = None;
    let mut idx = 0;

    while idx < args.len() {
        let arg_host = gather_if_needed(&args[idx]).map_err(|e| format!("{LABEL}: {e}"))?;

        if let Some(keyword) = keyword_of(&arg_host) {
            if keyword.as_str() == "like" {
                if like_shape.is_some() {
                    return Err(format!(
                        "{LABEL}: multiple 'like' prototypes are not supported"
                    ));
                }
                let Some(proto_raw) = args.get(idx + 1) else {
                    return Err(format!("{LABEL}: expected prototype after 'like'"));
                };
                let proto = gather_if_needed(proto_raw).map_err(|e| format!("{LABEL}: {e}"))?;
                like_shape = Some(prototype_dims(&proto));
                idx += 2;
                continue;
            }
            // Unrecognized keywords are treated as non-keyword inputs and will
            // be validated under numeric size parsing below.
        }

        if let Some(parsed) = extract_dims(&arg_host, LABEL)? {
            if explicit_dims.is_empty() {
                explicit_dims = parsed;
            } else {
                explicit_dims.extend(parsed);
            }
            idx += 1;
            continue;
        }

        return Err(format!(
            "{LABEL}: size inputs must be numeric scalars or size vectors"
        ));
    }

    let shape = if !explicit_dims.is_empty() {
        shape_from_explicit_dims(&explicit_dims)
    } else if let Some(proto_shape) = like_shape {
        shape_from_like(&proto_shape)
    } else {
        vec![0, 0]
    };
    ensure_empty_shape(&shape)?;
    Ok(shape)
}

fn shape_from_explicit_dims(dims: &[usize]) -> Vec<usize> {
    match dims.len() {
        0 => vec![0, 0],
        1 => vec![0, dims[0]],
        _ => {
            let mut shape = Vec::with_capacity(dims.len());
            shape.push(0);
            shape.extend_from_slice(&dims[1..]);
            shape
        }
    }
}

fn shape_from_like(proto: &[usize]) -> Vec<usize> {
    if proto.is_empty() {
        return vec![0, 0];
    }
    if proto.len() == 1 {
        return vec![0, proto[0]];
    }
    let mut shape = Vec::with_capacity(proto.len());
    shape.push(0);
    shape.extend_from_slice(&proto[1..]);
    shape
}

fn ensure_empty_shape(shape: &[usize]) -> Result<(), String> {
    if shape.iter().product::<usize>() != 0 {
        return Err(format!(
            "{LABEL}: at least one dimension must be zero to construct an empty string array"
        ));
    }
    Ok(())
}

fn prototype_dims(proto: &Value) -> Vec<usize> {
    match proto {
        Value::StringArray(sa) => sa.shape.clone(),
        Value::CharArray(ca) => vec![ca.rows, ca.cols],
        Value::Tensor(t) => t.shape.clone(),
        Value::ComplexTensor(t) => t.shape.clone(),
        Value::LogicalArray(l) => l.shape.clone(),
        Value::Cell(cell) => cell.shape.clone(),
        Value::GpuTensor(handle) => handle.shape.clone(),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => vec![1, 1],
        Value::String(_) => vec![1, 1],
        _ => vec![1, 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{StringArray, Tensor, Value};

    #[test]
    fn default_is_zero_by_zero() {
        let result = string_empty_builtin(Vec::new()).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 0]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn single_dimension_creates_zero_by_n() {
        let result = string_empty_builtin(vec![Value::from(5)]).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 5]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn multiple_dimensions_respect_trailing_sizes() {
        let args = vec![Value::from(3), Value::from(4), Value::from(2)];
        let result = string_empty_builtin(args).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 4, 2]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn size_vector_argument_supported() {
        let tensor = Tensor::new(vec![0.0, 5.0, 3.0], vec![1, 3]).unwrap();
        let result = string_empty_builtin(vec![Value::Tensor(tensor)]).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 5, 3]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn size_vector_from_nonempty_array_drops_leading_extent() {
        let tensor = Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap();
        let result = string_empty_builtin(vec![Value::Tensor(tensor)]).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 2]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn accepts_zero_in_any_position() {
        let args = vec![Value::from(3), Value::from(4), Value::from(0)];
        let result = string_empty_builtin(args).expect("string.empty");
        match result {
            Value::StringArray(sa) => assert_eq!(sa.shape, vec![0, 4, 0]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn like_prototype_without_explicit_dims() {
        let proto = StringArray::new(vec!["alpha".to_string(); 6], vec![2, 3]).unwrap();
        let result = string_empty_builtin(vec![Value::from("like"), Value::StringArray(proto)])
            .expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 3]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn like_prototype_with_scalar_shape() {
        let proto = StringArray::new(vec!["foo".to_string()], vec![1, 1]).unwrap();
        let result = string_empty_builtin(vec![Value::from("like"), Value::StringArray(proto)])
            .expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 1]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn like_with_numeric_prototype() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = string_empty_builtin(vec![Value::from("like"), Value::Tensor(tensor)])
            .expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 1]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn like_with_explicit_dims_prefers_dimensions() {
        let proto = StringArray::new(Vec::new(), vec![0, 2]).unwrap();
        let args = vec![
            Value::from(0),
            Value::from(7),
            Value::from("like"),
            Value::StringArray(proto),
        ];
        let result = string_empty_builtin(args).expect("string.empty");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 7]);
                assert_eq!(sa.data.len(), 0);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn missing_like_prototype_errors() {
        let err = string_empty_builtin(vec![Value::from("like")]).expect_err("expected error");
        assert!(
            err.contains("expected prototype"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn duplicate_like_errors() {
        let proto = StringArray::new(Vec::new(), vec![0, 2]).unwrap();
        let err = string_empty_builtin(vec![
            Value::from("like"),
            Value::StringArray(proto.clone()),
            Value::from("like"),
            Value::StringArray(proto),
        ])
        .expect_err("expected error");
        assert!(err.contains("multiple 'like'"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_non_dimension_inputs() {
        let err =
            string_empty_builtin(vec![Value::String("oops".into())]).expect_err("expected error");
        assert!(
            err.contains("size inputs must be numeric"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn like_gathers_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=6).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                string_empty_builtin(vec![Value::from("like"), Value::GpuTensor(handle.clone())])
                    .expect("string.empty");
            match result {
                Value::StringArray(sa) => {
                    assert_eq!(sa.shape, vec![0, 3]);
                    assert_eq!(sa.data.len(), 0);
                }
                other => panic!("expected string array, got {other:?}"),
            }
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn gpu_dimension_arguments_are_gathered() {
        test_support::with_test_provider(|provider| {
            let dims = Tensor::new(vec![0.0, 5.0, 3.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &dims.data,
                shape: &dims.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                string_empty_builtin(vec![Value::GpuTensor(handle.clone())]).expect("string.empty");
            match result {
                Value::StringArray(sa) => {
                    assert_eq!(sa.shape, vec![0, 5, 3]);
                    assert_eq!(sa.data.len(), 0);
                }
                other => panic!("expected string array, got {other:?}"),
            }
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn rejects_negative_dimension() {
        let err = string_empty_builtin(vec![Value::from(-1.0)]).expect_err("expected error");
        assert!(
            err.contains("matrix dimensions must be non-negative"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
