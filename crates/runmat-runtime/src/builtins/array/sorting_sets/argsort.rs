//! MATLAB-compatible `argsort` builtin returning permutation indices.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::sort;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "argsort",
        builtin_path = "crate::builtins::array::sorting_sets::argsort"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "argsort"
category: "array/sorting_sets"
keywords: ["argsort", "sort", "indices", "permutation", "gpu"]
summary: "Return the permutation indices that would sort tensors along a dimension."
references:
  - https://www.mathworks.com/help/matlab/ref/sort.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses the same sort kernels as `sort`; falls back to host evaluation when the provider lacks `sort_dim`."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::sorting_sets::argsort::tests"
  integration: "builtins::array::sorting_sets::argsort::tests::argsort_gpu_roundtrip"
---

# What does the `argsort` function do in MATLAB / RunMat?
`argsort(X)` returns the permutation indices that order `X` the same way `sort(X)` would. It matches the indices produced by `[~, I] = sort(X, ...)` in MathWorks MATLAB and honours the same argument forms for dimensions, directions, and comparison methods.

## How does the `argsort` function behave in MATLAB / RunMat?
- Operates along the first non-singleton dimension by default. Pass a dimension argument to override.
- Accepts the same direction keywords as `sort`: `'ascend'` (default) or `'descend'`.
- Supports `'ComparisonMethod'` values `'auto'`, `'real'`, and `'abs'` for real and complex inputs.
- Returns indices as double-precision tensors using MATLAB's one-based indexing.
- Treats NaN values as missing: they appear at the end for ascending permutations and at the beginning for descending permutations.
- Acts as a residency sink. GPU tensors are gathered when the active provider does not expose a specialised sort kernel.

## GPU execution in RunMat
- `argsort` shares the `sort_dim` provider hook with the `sort` builtin. When implemented, indices are computed without leaving the device.
- If the provider lacks `sort_dim`, RunMat gathers tensors to host memory, evaluates the permutation, and returns host-resident indices.
- Outputs are always host-resident double tensors because permutation indices are consumed immediately by host-side logic (e.g., indexing).

## Examples of using `argsort` in MATLAB / RunMat

### Getting indices that sort a vector
```matlab
A = [4; 1; 3];
idx = argsort(A);
```
Expected output:
```matlab
idx =
     2
     3
     1
```

### Reordering data with the permutation indices
```matlab
A = [3 9 1 5];
idx = argsort(A);
sorted = A(idx);
```
Expected output:
```matlab
sorted =
     1     3     5     9
```

### Sorting along a specific dimension
```matlab
A = [1 6 4; 2 3 5];
idx = argsort(A, 2);
```
Expected output:
```matlab
idx =
     1     3     2
     1     2     3
```

### Descending order permutations
```matlab
A = [10 4 7 9];
idx = argsort(A, 'descend');
```
Expected output:
```matlab
idx =
     1     4     3     2
```

### Using `ComparisonMethod` to sort by magnitude
```matlab
A = [-8 -1 3 -2];
idx = argsort(A, 'ComparisonMethod', 'abs');
```
Expected output:
```matlab
idx =
     2     4     3     1
```

### Handling NaN values during permutation
```matlab
A = [NaN 4 1 2];
idx = argsort(A);
```
Expected output:
```matlab
idx =
     3     4     2     1
```

### Argsort on GPU tensors falls back gracefully
```matlab
G = gpuArray(randn(5, 1));
idx = argsort(G);
```
RunMat gathers `G` to the host when no device sort kernel is available, ensuring the returned indices match MATLAB exactly.

## FAQ

### How is `argsort` different from `sort`?
`argsort` returns only the permutation indices. It behaves like calling `[~, I] = sort(X, ...)` without materialising the sorted values.

### Are the indices one-based like MATLAB?
Yes. All indices follow MATLAB's one-based convention so they can be used directly with subsequent indexing operations.

### Does `argsort` support the same arguments as `sort`?
Yes. Dimension arguments, direction keywords, and `'ComparisonMethod'` behave exactly like they do for `sort`.

### How are NaN values ordered?
NaNs are treated as missing. They appear at the end for ascending permutations and at the beginning for descending permutations, matching MATLAB.

### Can I call `argsort` on GPU arrays?
Yes. When the active provider implements the `sort_dim` hook, permutations stay on the device. Otherwise tensors are gathered automatically and sorted on the host.

### Is the permutation stable?
Yes. Equal elements keep their relative order so that `argsort` remains consistent with MATLAB's stable sorting semantics.

### What type is returned?
A double-precision tensor (or scalar) with the same shape as the input, containing permutation indices.

### Does `argsort` mutate its input?
No. It only returns indices. Combine the result with indexing (`A(idx)`) to obtain reordered values when needed.

## See also
[sort](./sort), [sortrows](./sortrows), [randperm](../../array/creation/randperm), [max](../../math/reduction/max), [min](../../math/reduction/min)

## Source & Feedback
- Source code: [`crates/runmat-runtime/src/builtins/array/sorting_sets/argsort.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/argsort.rs)
- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::argsort")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "argsort",
    op_kind: GpuOpKind::Custom("sort"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("sort_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Shares provider hooks with `sort`; when unavailable tensors are gathered to host memory before computing indices.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::argsort"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "argsort",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "`argsort` breaks fusion chains and acts as a residency sink; upstream tensors are gathered when no GPU sort kernel is provided.",
};

#[runtime_builtin(
    name = "argsort",
    category = "array/sorting_sets",
    summary = "Return the permutation indices that would sort tensors along a dimension.",
    keywords = "argsort,sort,indices,permutation,gpu",
    accel = "sink",
    sink = true,
    builtin_path = "crate::builtins::array::sorting_sets::argsort"
)]
fn argsort_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let evaluation = sort::evaluate(value, &rest)?;
    Ok(evaluation.indices_value())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::sort;
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ComplexTensor, IntValue, Tensor, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_vector_default() {
        let tensor = Tensor::new(vec![4.0, 1.0, 3.0], vec![3, 1]).unwrap();
        let indices = argsort_builtin(Value::Tensor(tensor), Vec::new()).expect("argsort");
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![2.0, 3.0, 1.0]);
                assert_eq!(t.shape, vec![3, 1]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_descend_direction() {
        let tensor = Tensor::new(vec![10.0, 4.0, 7.0, 9.0], vec![4, 1]).unwrap();
        let indices =
            argsort_builtin(Value::Tensor(tensor), vec![Value::from("descend")]).expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 4.0, 3.0, 2.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 6.0, 4.0, 2.0, 3.0, 5.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let indices =
            argsort_builtin(Value::Tensor(tensor.clone()), args.clone()).expect("argsort");
        let expected = sort::evaluate(Value::Tensor(tensor), &args)
            .expect("sort evaluate")
            .indices_value();
        assert_eq!(indices, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_absolute_comparison() {
        let tensor = Tensor::new(vec![-8.0, -1.0, 3.0, -2.0], vec![4, 1]).unwrap();
        let indices = argsort_builtin(
            Value::Tensor(tensor),
            vec![Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 4.0, 3.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_handles_nan_like_sort() {
        let tensor = Tensor::new(vec![f64::NAN, 4.0, 1.0, 2.0], vec![4, 1]).unwrap();
        let indices = argsort_builtin(Value::Tensor(tensor), Vec::new()).expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 4.0, 2.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_dimension_placeholder_then_dim() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 2.0], vec![2, 2]).unwrap();
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = vec![
            Value::Tensor(placeholder),
            Value::Int(IntValue::I32(2)),
            Value::from("descend"),
        ];
        let indices =
            argsort_builtin(Value::Tensor(tensor.clone()), args.clone()).expect("argsort");
        let expected = sort::evaluate(Value::Tensor(tensor), &args)
            .expect("sort evaluate")
            .indices_value();
        assert_eq!(indices, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_dimension_greater_than_ndims_returns_ones() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0], vec![3, 1]).unwrap();
        let indices = argsort_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(5))])
            .expect("argsort");
        match indices {
            Value::Tensor(t) => assert!(t.data.iter().all(|v| (*v - 1.0).abs() < f64::EPSILON)),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_dimension_zero_errors() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err =
            argsort_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))]).unwrap_err();
        assert!(
            err.contains("dimension must be >= 1"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_invalid_argument_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = argsort_builtin(
            Value::Tensor(tensor),
            vec![Value::from("MissingPlacement"), Value::from("auto")],
        )
        .unwrap_err();
        assert!(
            err.contains("sort: the 'MissingPlacement' option is not supported"),
            "{err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_invalid_comparison_method_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = argsort_builtin(
            Value::Tensor(tensor),
            vec![Value::from("ComparisonMethod"), Value::from("unknown")],
        )
        .unwrap_err();
        assert!(
            err.contains("unsupported ComparisonMethod"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_invalid_comparison_method_value_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = argsort_builtin(
            Value::Tensor(tensor),
            vec![
                Value::from("ComparisonMethod"),
                Value::Int(IntValue::I32(1)),
            ],
        )
        .unwrap_err();
        assert!(
            err.contains("requires a string value"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_stable_with_duplicates() {
        let tensor = Tensor::new(vec![2.0, 2.0, 1.0, 2.0], vec![4, 1]).unwrap();
        let indices = argsort_builtin(Value::Tensor(tensor), Vec::new()).expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 1.0, 2.0, 4.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_complex_real_method() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.5), (1.0, -1.0)], vec![3, 1]).unwrap();
        let indices = argsort_builtin(
            Value::ComplexTensor(tensor),
            vec![Value::from("ComparisonMethod"), Value::from("real")],
        )
        .expect("argsort");
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argsort_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let indices = argsort_builtin(Value::GpuTensor(handle), Vec::new()).expect("argsort");
            match indices {
                Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0, 1.0]),
                other => panic!("expected tensor result, got {other:?}"),
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
    fn argsort_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 5.0, -1.0, 2.0], vec![4, 1]).unwrap();
        let cpu_indices = argsort_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let gpu_handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_indices = argsort_builtin(Value::GpuTensor(gpu_handle), Vec::new()).unwrap();

        let cpu_tensor = match cpu_indices {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        let gpu_tensor = match gpu_indices {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        };
        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        assert_eq!(gpu_tensor.data, cpu_tensor.data);
    }
}
