//! MATLAB-compatible `ipermute` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the inverse permutation primitive that undoes the action of
//! `permute`. It shares the same validation rules and GPU plumbing as `permute`, but
//! automatically computes the inverse order vector before delegating to the shared
//! permutation helpers.

use crate::builtins::array::shape::permute::{
    parse_order_argument, permute_char_array, permute_complex_tensor, permute_gpu,
    permute_logical_array, permute_string_array, permute_tensor, validate_rank,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "ipermute"
category: "array/shape"
keywords: ["ipermute", "inverse permute", "dimension reorder", "gpu"]
summary: "Reorder array dimensions using the inverse of a permutation vector."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64", "i32", "bool"]
  broadcasting: "none"
  notes: "Reuses the GPU permute hook; when unavailable the runtime gathers, applies the inverse permutation on the host, and re-uploads."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::ipermute::tests"
  integration: "builtins::array::shape::ipermute::tests::ipermute_gpu_roundtrip"
---

# What does the `ipermute` function do in MATLAB / RunMat?
`ipermute(A, order)` reverses the dimension reordering performed by `permute(A, order)`.
It is equivalent to `permute(A, invOrder)`, where `invOrder(order(i)) = i`.

## How does the `ipermute` function behave in MATLAB / RunMat?
- `order` must be a permutation of `1:n`, where `n = numel(order)`.
- The length of `order` must be at least `ndims(A)`; missing axes are treated as trailing singleton dimensions.
- Duplicate, zero, or negative indices raise an error.
- Works with numeric tensors, logical masks, complex tensors, string arrays, character arrays, and gpuArray values.
- Character arrays stay 2-D; only `[1 2]` (identity) and `[2 1]` (transpose) are valid orders.
- If the active acceleration provider lacks a dedicated permute kernel, RunMat gathers the data, applies the inverse permutation on the host, and re-uploads so the result remains a gpuArray.

## `ipermute` Function GPU Execution Behaviour
RunMat calls the provider's `permute` hook with the inverse order vector. Providers that support the
hook perform the reorder entirely on the device. Otherwise RunMat gathers to the host, permutes, and
uploads the result back to the GPU, matching MATLAB's gpuArray behaviour.

## Examples of using the `ipermute` function in MATLAB / RunMat

### Reversing a previous permute call
```matlab
A = reshape(1:24, [2 3 4]);
B = permute(A, [2 1 3]);
C = ipermute(B, [2 1 3]);
isequal(A, C)
```
Expected output:
```matlab
ans = logical 1
```

### Restoring matrix orientation after swapping dimensions
```matlab
M = magic(3);
P = permute(M, [2 1]);
R = ipermute(P, [2 1]);
size(R)
```
Expected output:
```matlab
ans = [3 3]
```

### Undoing a permutation with extra singleton dimensions
```matlab
row = 1:5;
P = permute(row, [2 1 3]);
R = ipermute(P, [2 1 3]);
size(R)
```
Expected output:
```matlab
ans = [1 5 1]
```

### Recovering logical masks after reordering axes
```matlab
mask = false(2, 1, 3);
mask(1, 1, 2) = true;
rot = permute(mask, [3 1 2]);
orig = ipermute(rot, [3 1 2]);
orig(1,1,2)
```
Expected output:
```matlab
ans = logical 1
```

### Working with character arrays
```matlab
chars = ['r','u','n'; 'm','a','t'];
T = permute(chars, [2 1]);
R = ipermute(T, [2 1]);
R
```
Expected output:
```matlab
R =
    'run'
    'mat'
```

### Restoring gpuArray tensors to host layout
```matlab
G = gpuArray(rand(4, 2, 3));
H = permute(G, [3 1 2]);
R = ipermute(H, [3 1 2]);
isequal(gather(G), gather(R))
```
Expected output:
```matlab
ans = logical 1
```

## FAQ
1. **Do I have to compute the inverse permutation myself?**  
   No. Pass the same `order` vector you used with `permute`; `ipermute` computes the inverse for you.
2. **What happens if I pass an invalid permutation?**  
   RunMat raises an error when indices repeat, fall outside `1:n`, or when the order vector is not a row or column vector.
3. **Can `ipermute` introduce new singleton dimensions?**  
   Yes. If `order` is longer than `ndims(A)`, trailing singleton dimensions are added just like MATLAB.
4. **Does `ipermute` modify the input array in-place?**  
   No. It returns a new array with reordered metadata while leaving the original untouched.
5. **How does `ipermute` behave for character arrays?**  
   Character arrays remain 2-D. Only `[1 2]` and `[2 1]` orders are accepted; other permutations raise an error.
6. **Is gpuArray behaviour identical to MATLAB?**  
   Yes. Results remain gpuArray values. When providers lack a device kernel, RunMat gathers, permutes on the host, and uploads the result before returning.
7. **Does `ipermute` preserve data ordering?**  
   Yes. Column-major ordering is preserved. Applying `permute` followed by `ipermute` returns the original array exactly.
8. **Can I use `ipermute` on scalar inputs?**  
   Absolutely. Scalars are treated as 0-D/1-D tensors so the result is the same scalar.
9. **What if the order vector is a gpuArray?**  
   MATLAB requires host numeric vectors for order specifications. RunMat enforces the same rule and raises an error.
10. **Is there a performance benefit on the GPU?**  
    Yes when the provider implements the permute hook. Otherwise the fallback path behaves like MATLAB's gather/permute/gpuArray workflow.

## See Also
- [`permute`](./permute)
- [`reshape`](./reshape)
- [`squeeze`](./squeeze)
- [`size`](../introspection/size)
- [`ndims`](../introspection/ndims)
- [`gpuArray`](../../acceleration/gpu/gpuArray)
- [`gather`](../../acceleration/gpu/gather)

## Source & Feedback
- The full source for `ipermute` lives at [`crates/runmat-runtime/src/builtins/array/shape/ipermute.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/ipermute.rs)
- Found a behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ipermute",
    op_kind: GpuOpKind::Custom("permute"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("permute")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses the same provider permute hook as `permute`; falls back to gather→permute→upload when unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ipermute",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a layout barrier in fusion graphs, mirroring the behaviour of `permute`.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("ipermute", DOC_MD);

#[runtime_builtin(
    name = "ipermute",
    category = "array/shape",
    summary = "Reorder array dimensions using the inverse of a permutation vector.",
    keywords = "ipermute,inverse permute,dimension reorder,gpu",
    accel = "custom"
)]
fn ipermute_builtin(value: Value, order: Value) -> Result<Value, String> {
    let order_vec = parse_order_argument(order).map_err(map_perm_error)?;
    let inverse = inverse_permutation(&order_vec);

    match value {
        Value::Tensor(t) => {
            validate_rank(&order_vec, t.shape.len())
                .map_err(map_perm_error)?;
            permute_tensor(t, &inverse)
                .map_err(map_perm_error)
                .map(tensor::tensor_into_value)
        }
        Value::LogicalArray(la) => {
            validate_rank(&order_vec, la.shape.len())
                .map_err(map_perm_error)?;
            permute_logical_array(la, &inverse)
                .map_err(map_perm_error)
                .map(Value::LogicalArray)
        }
        Value::ComplexTensor(ct) => {
            validate_rank(&order_vec, ct.shape.len())
                .map_err(map_perm_error)?;
            permute_complex_tensor(ct, &inverse)
                .map_err(map_perm_error)
                .map(Value::ComplexTensor)
        }
        Value::StringArray(sa) => {
            validate_rank(&order_vec, sa.shape.len())
                .map_err(map_perm_error)?;
            permute_string_array(sa, &inverse)
                .map_err(map_perm_error)
                .map(Value::StringArray)
        }
        Value::CharArray(ca) => {
            validate_rank(&order_vec, 2).map_err(map_perm_error)?;
            permute_char_array(ca, &inverse)
                .map_err(map_perm_error)
                .map(Value::CharArray)
        }
        Value::GpuTensor(handle) => {
            validate_rank(&order_vec, handle.shape.len())
                .map_err(map_perm_error)?;
            ipermute_gpu(handle, &inverse)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("ipermute", value)?;
            validate_rank(&order_vec, tensor.shape.len())
                .map_err(map_perm_error)?;
            permute_tensor(tensor, &inverse)
                .map_err(map_perm_error)
                .map(tensor::tensor_into_value)
        }
        other => Err(format!(
            "ipermute: unsupported input type {:?}; expected numeric, logical, complex, string, or gpuArray values",
            other
        )),
    }
}

fn ipermute_gpu(handle: GpuTensorHandle, inverse_order: &[usize]) -> Result<Value, String> {
    permute_gpu(handle, inverse_order).map_err(map_perm_error)
}

fn inverse_permutation(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0usize; order.len()];
    for (pos, &idx) in order.iter().enumerate() {
        let place = idx
            .checked_sub(1)
            .expect("parse_order_argument guarantees indices are >= 1");
        inverse[place] = pos + 1;
    }
    inverse
}

fn map_perm_error(err: String) -> String {
    if let Some(rest) = err.strip_prefix("permute: ") {
        format!("ipermute: {rest}")
    } else if err.starts_with("permute:") {
        // handle "permute:" without trailing space
        format!("ipermute{}", &err["permute".len()..])
    } else {
        format!("ipermute: {err}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::array::shape::permute::{
        parse_order_argument, permute_char_array, permute_gpu, permute_logical_array,
        permute_string_array, permute_tensor,
    };
    use crate::builtins::common::{tensor, test_support};
    use runmat_builtins::{CharArray, LogicalArray, StringArray, Tensor, Value};

    fn make_tensor(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[test]
    fn ipermute_inverts_permute() {
        let data: Vec<f64> = (1..=24).map(|n| n as f64).collect();
        let order = make_tensor(&[3.0, 1.0, 2.0], &[1, 3]);
        let order_vec = parse_order_argument(Value::Tensor(order.clone())).expect("parse order");
        let original_tensor = make_tensor(&data, &[2, 3, 4]);
        let permuted_tensor = permute_tensor(original_tensor.clone(), &order_vec).expect("permute");
        let permuted = tensor::tensor_into_value(permuted_tensor);
        let restored = ipermute_builtin(permuted, Value::Tensor(order)).expect("ipermute");
        match (Value::Tensor(original_tensor), restored) {
            (Value::Tensor(orig), Value::Tensor(rest)) => {
                assert_eq!(orig.shape, rest.shape);
                assert_eq!(orig.data, rest.data);
            }
            _ => panic!("expected tensor pair"),
        }
    }

    #[test]
    fn ipermute_rejects_invalid_order() {
        let order = make_tensor(&[1.0, 1.0], &[1, 2]);
        let err = ipermute_builtin(
            Value::Tensor(make_tensor(&[1.0], &[1, 1])),
            Value::Tensor(order),
        )
        .expect_err("should fail");
        assert!(err.contains("duplicate"), "unexpected error: {err}");
    }

    #[test]
    fn ipermute_requires_vector_order() {
        let order = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = ipermute_builtin(
            Value::Tensor(make_tensor(&[1.0, 2.0, 3.0, 4.0], &[4, 1])),
            Value::Tensor(order),
        )
        .expect_err("should fail");
        assert!(
            err.contains("row or column vector"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ipermute_char_array_roundtrip() {
        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let order = make_tensor(&[2.0, 1.0], &[1, 2]);
        let order_vec = parse_order_argument(Value::Tensor(order.clone())).expect("parse order");
        let permuted = permute_char_array(chars.clone(), &order_vec).expect("permute chars");
        let restored = ipermute_builtin(Value::CharArray(permuted), Value::Tensor(order))
            .expect("ipermute chars");
        match restored {
            Value::CharArray(out) => {
                assert_eq!(out.rows, chars.rows);
                assert_eq!(out.cols, chars.cols);
                assert_eq!(out.data, chars.data);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn ipermute_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let host = make_tensor(&(0..24).map(|n| n as f64).collect::<Vec<_>>(), &[2, 3, 4]);
            let order = make_tensor(&[3.0, 1.0, 2.0], &[1, 3]);
            let view = runmat_accelerate_api::HostTensorView {
                data: &host.data,
                shape: &host.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let order_vec =
                parse_order_argument(Value::Tensor(order.clone())).expect("parse order");
            let permuted = permute_gpu(handle, &order_vec).expect("permute gpu");
            let restored = ipermute_builtin(permuted, Value::Tensor(order)).expect("ipermute gpu");
            let gathered = test_support::gather(restored).expect("gather");
            assert_eq!(gathered.shape, host.shape);
            assert_eq!(gathered.data, host.data);
        });
    }

    #[test]
    fn ipermute_numeric_scalar() {
        let value = Value::Num(42.0);
        let order = make_tensor(&[1.0, 2.0], &[1, 2]);
        let result = ipermute_builtin(value.clone(), Value::Tensor(order)).expect("ipermute");
        assert_eq!(result, value);
    }

    #[test]
    fn ipermute_logical_array_roundtrip() {
        let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let order = make_tensor(&[2.0, 1.0], &[1, 2]);
        let order_vec = parse_order_argument(Value::Tensor(order.clone())).expect("parse order");
        let permuted = permute_logical_array(logical.clone(), &order_vec).expect("permute logical");
        let restored = ipermute_builtin(Value::LogicalArray(permuted), Value::Tensor(order))
            .expect("ipermute logical");
        match restored {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, logical.shape);
                assert_eq!(out.data, logical.data);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg(feature = "doc_export")]
    #[test]
    fn ipermute_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty(), "expected doc examples");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn ipermute_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let host = make_tensor(&(0..24).map(|n| n as f64).collect::<Vec<_>>(), &[2, 3, 4]);
        let order = make_tensor(&[3.0, 1.0, 2.0], &[1, 3]);
        let order_vec = parse_order_argument(Value::Tensor(order.clone())).expect("parse order");

        let permuted_tensor = permute_tensor(host.clone(), &order_vec).expect("permute host");
        let permuted = tensor::tensor_into_value(permuted_tensor);
        let cpu = ipermute_builtin(permuted, Value::Tensor(order.clone())).expect("cpu ipermute");

        let view = runmat_accelerate_api::HostTensorView {
            data: &host.data,
            shape: &host.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let permuted_gpu = permute_gpu(handle, &order_vec).expect("permute gpu");
        let gpu = ipermute_builtin(permuted_gpu, Value::Tensor(order)).expect("gpu ipermute");
        let gathered = test_support::gather(gpu).expect("gather");

        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(ct.shape, gathered.shape);
                assert_eq!(ct.data, gathered.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn ipermute_string_array_roundtrip() {
        let data = vec![
            "run".to_string(),
            "mat".to_string(),
            "fast".to_string(),
            "gpu".to_string(),
        ];
        let strings = StringArray::new(data.clone(), vec![2, 2]).unwrap();
        let order = make_tensor(&[2.0, 1.0], &[1, 2]);
        let order_vec = parse_order_argument(Value::Tensor(order.clone())).expect("parse order");
        let permuted =
            permute_string_array(strings.clone(), &order_vec).expect("permute string array");
        let restored =
            ipermute_builtin(Value::StringArray(permuted), Value::Tensor(order)).expect("ipermute");
        match restored {
            Value::StringArray(out) => {
                assert_eq!(out.shape, strings.shape);
                assert_eq!(out.data, data);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn ipermute_extends_missing_dimensions() {
        let row = make_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
        let order = make_tensor(&[2.0, 1.0, 3.0], &[1, 3]);
        let order_vec = parse_order_argument(Value::Tensor(order.clone())).expect("parse order");
        let permuted = permute_tensor(row.clone(), &order_vec).expect("permute");
        let restored = ipermute_builtin(tensor::tensor_into_value(permuted), Value::Tensor(order))
            .expect("ipermute");
        match restored {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 5, 1]);
                assert_eq!(out.data, row.data);
            }
            Value::Num(n) => {
                panic!("expected tensor result, got scalar {n}");
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn ipermute_errors_when_order_too_short() {
        let matrix = make_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let order = make_tensor(&[1.0], &[1, 1]);
        let err = ipermute_builtin(Value::Tensor(matrix), Value::Tensor(order)).unwrap_err();
        assert!(
            err.contains("order length"),
            "expected rank error, got {err}"
        );
    }
}
