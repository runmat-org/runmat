//! MATLAB-compatible `ndims` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_ndims;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "ndims")]
pub const DOC_MD: &str = r#"---
title: "ndims"
category: "array/introspection"
keywords: ["ndims", "number of dimensions", "array rank", "gpu metadata", "MATLAB compatibility"]
summary: "Return the number of dimensions of scalars, vectors, matrices, and N-D arrays."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Reads tensor metadata from handles; falls back to host when provider metadata is missing."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::introspection::ndims::tests"
  integration: "builtins::array::introspection::ndims::tests::ndims_gpu_tensor_reads_shape"
---

# What does the `ndims` function do in MATLAB / RunMat?
`ndims(A)` returns the number of dimensions (rank) of `A`, following MATLAB conventions. Scalars,
vectors, matrices, and N-D tensors always report at least two dimensions; higher-rank arrays report
their full dimension count. This behaviour matches MathWorks MATLAB for numeric, logical, char,
string, cell, struct, and GPU arrays.

## How does the `ndims` function behave in MATLAB / RunMat?
- Scalars and vectors return `2`, because MATLAB treats them as `1×1` or `1×N`.
- Matrices return `2` unless they have trailing dimensions beyond the second.
- N-D arrays return the length of their dimension vector after trailing singleton dimensions are
  preserved.
- Character arrays, string arrays, logical arrays, complex arrays, and cell arrays all follow their
  MATLAB array shape metadata.
- GPU tensors report their rank without gathering when the provider populates shape metadata on the
  handle. When metadata is missing, RunMat downloads the tensor once to preserve correctness.
- Objects, scalars, and other non-array values return `2`, consistent with MATLAB's scalar rules.

## `ndims` Function GPU Execution Behaviour
`ndims` is a metadata query. When the input resides on the GPU, RunMat inspects the shape stored in
`GpuTensorHandle.shape`. No kernels are launched, no buffers are allocated, and residency is not
altered. If a provider leaves the shape empty, the runtime downloads the tensor a single time to
reconstruct its dimensions before returning a host scalar. The builtin therefore preserves fusion
pipelines while still providing MATLAB-compatible answers.

## Examples of using the `ndims` function in MATLAB / RunMat

### Checking how many dimensions a scalar has

```matlab
n = ndims(42);
```

Expected output:

```matlab
n = 2;
```

### Counting the dimensions of a matrix

```matlab
A = rand(5, 3);
r = ndims(A);
```

Expected output:

```matlab
r = 2;
```

### Detecting 3-D array dimensionality

```matlab
V = rand(4, 5, 6);
r = ndims(V);
```

Expected output:

```matlab
r = 3;
```

### Finding ndims for gpuArray data without gathering

```matlab
G = gpuArray(ones(16, 32, 2));
r = ndims(G);
```

Expected output:

```matlab
r = 3;
```

### Understanding ndims with cell arrays

```matlab
C = {1, 2, 3; 4, 5, 6};
r = ndims(C);
```

Expected output:

```matlab
r = 2;
```

### Verifying ndims on string arrays

```matlab
S = ["alpha"; "beta"; "gamma"];
r = ndims(S);
```

Expected output:

```matlab
r = 2;
```

## FAQ

### Why does `ndims` return 2 for scalars and vectors?
MATLAB treats scalars and vectors as 2-D arrays (`1×1` or `1×N`). RunMat mirrors this behaviour, so
`ndims(5)` and `ndims([1 2 3])` both return `2`.

### Does `ndims` ignore trailing singleton dimensions?
No. MATLAB preserves trailing singletons when reporting the rank. If your tensor was explicitly
created with size `10×1×1×4`, `ndims` returns `4`.

### How does `ndims` behave for GPU tensors?
The runtime inspects shape metadata stored in the GPU tensor handle. When the provider populates
that field, no gathering occurs. If metadata is absent, RunMat downloads the tensor to keep the
answer correct.

### Can I call `ndims` on cell arrays and structs?
Yes. `ndims` inspects the MATLAB array shape of the container, so cell arrays and struct arrays
report their grid dimensions.

### What about objects or scalars like strings and logical values?
Objects and scalar values follow MATLAB scalar rules and return `2`. Use specialised functions (like
`isobject`) when you need more information about their type.

### Does `ndims` participate in GPU fusion?
No. `ndims` returns a host scalar immediately and performs no computation on the GPU. It is safe to
use inside expressions that also run on the GPU because it does not allocate device memory.

## See Also
[size](./size), [length](./length), [numel](./numel), [MathWorks ndims reference](https://www.mathworks.com/help/matlab/ref/ndims.html)
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ndims",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Metadata-only query; relies on tensor handle shapes and gathers only when provider metadata is unavailable.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ndims",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query; fusion planner bypasses this builtin and emits a host scalar.",
};

#[runtime_builtin(
    name = "ndims",
    category = "array/introspection",
    summary = "Return the number of dimensions of scalars, vectors, matrices, and N-D arrays.",
    keywords = "ndims,number of dimensions,array rank,gpu metadata,MATLAB compatibility",
    accel = "metadata"
)]
fn ndims_builtin(value: Value) -> Result<Value, String> {
    let rank = value_ndims(&value) as f64;
    Ok(Value::Num(rank))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{
        CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
    };

    #[test]
    fn ndims_scalar_returns_two() {
        let result = ndims_builtin(Value::Num(std::f64::consts::PI)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn ndims_row_vector_returns_two() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = ndims_builtin(Value::Tensor(tensor)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn ndims_three_dimensional_tensor_returns_three() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let result = ndims_builtin(Value::Tensor(tensor)).expect("ndims");
        assert_eq!(result, Value::Num(3.0));
    }

    #[test]
    fn ndims_trailing_singletons_preserved() {
        let tensor = Tensor::new(vec![0.0; 40], vec![5, 1, 1, 8]).unwrap();
        let result = ndims_builtin(Value::Tensor(tensor)).expect("ndims");
        assert_eq!(result, Value::Num(4.0));
    }

    #[test]
    fn ndims_cell_array_returns_two() {
        let cells = CellArray::new(
            vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
            ],
            2,
            2,
        )
        .unwrap();
        let result = ndims_builtin(Value::Cell(cells)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn ndims_string_array_returns_two() {
        let sa = StringArray::new(vec!["a".into(), "bb".into(), "ccc".into()], vec![3, 1]).unwrap();
        let result = ndims_builtin(Value::StringArray(sa)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn ndims_char_array_returns_two() {
        let chars = CharArray::new_row("RunMat");
        let result = ndims_builtin(Value::CharArray(chars)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn ndims_complex_tensor_uses_shape() {
        let complex = ComplexTensor::new(vec![(0.0, 0.0); 18], vec![3, 3, 2]).unwrap();
        let result = ndims_builtin(Value::ComplexTensor(complex)).expect("ndims");
        assert_eq!(result, Value::Num(3.0));
    }

    #[test]
    fn ndims_logical_array_returns_two() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let result = ndims_builtin(Value::LogicalArray(logical)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn ndims_gpu_tensor_reads_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..48).map(|x| x as f64).collect(), vec![4, 3, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = ndims_builtin(Value::GpuTensor(handle)).expect("ndims");
            assert_eq!(result, Value::Num(3.0));
        });
    }

    #[test]
    fn ndims_gpu_tensor_without_metadata_defaults_correctly() {
        // Simulate a provider that does not populate shape metadata.
        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![],
            device_id: 0,
            buffer_id: 42,
        };
        let result = ndims_builtin(Value::GpuTensor(handle)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn ndims_wgpu_tensor_reads_shape() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new((0..64).map(|x| x as f64).collect(), vec![4, 4, 4]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = ndims_builtin(Value::GpuTensor(handle)).expect("ndims");
        assert_eq!(result, Value::Num(3.0));
    }
}
