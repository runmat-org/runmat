//! MATLAB-compatible `squeeze` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "squeeze",
        builtin_path = "crate::builtins::array::shape::squeeze"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "squeeze"
category: "array/shape"
keywords: ["squeeze", "singleton dimensions", "array reshape", "gpu"]
summary: "Remove singleton (size-one) dimensions while preserving MATLAB vector semantics."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64", "i32", "bool"]
  broadcasting: "none"
  notes: "Backends reuse the reshape hook to update device metadata; runtimes fall back to host metadata updates when unavailable."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::squeeze::tests"
  integration: "builtins::array::shape::squeeze::tests::squeeze_gpu_roundtrip"
---

# What does the `squeeze` function do in MATLAB / RunMat?
`squeeze(A)` removes singleton (dimension size equal to one) dimensions from `A`. The result keeps
MATLAB's row/column vector semantics—2-D matrices stay 2-D, and higher-rank arrays collapse down
until only meaningful axes remain.

## How does the `squeeze` function behave in MATLAB / RunMat?
- Dimensions greater than two drop all size-one axes. If exactly one dimension remains, the output becomes an `N × 1` column vector.
- Two-dimensional inputs (row/column vectors and matrices) are returned unchanged.
- Zero-length dimensions are preserved; only length-one axes are removed.
- Scalar values remain scalars.
- GPU arrays stay resident on the device where possible; the runtime only gathers when it cannot infer the shape.

## `squeeze` Function GPU Execution Behaviour
When a GPU provider is active, RunMat calls the provider's `reshape` hook to update tensor metadata
without copying data. Providers that do not override `reshape` fall back to the in-process behaviour,
simply updating the handle shape locally. If a handle advertises no shape, RunMat downloads metadata
once to infer dimensions before applying `squeeze`.

## Examples of using the `squeeze` function in MATLAB / RunMat

### Removing Singleton Dimensions from a 3-D Matrix
```matlab
A = reshape(1:12, [1, 3, 4]);
B = squeeze(A);
size(B)
```
Expected output:
```matlab
ans = [3 4]
```

### Turning 1x1xN Slices into Column Vectors
```matlab
T = zeros(1, 1, 5);
T(:, :, 3) = 7;
vec = squeeze(T);
size(vec)
```
Expected output:
```matlab
ans = [5 1]
```

### Keeping Row Vectors Unchanged After Squeeze
```matlab
row = rand(1, 8);
out = squeeze(row);
size(out)
```
Expected output:
```matlab
ans = [1 8]
```

### Squeezing Logical Masks from Simulations
```matlab
mask = false(1, 10, 1, 1);
mask(1, 4, 1, 1) = true;
flat = squeeze(mask);
```
Expected output:
```matlab
flat = [0 0 0 1 0 0 0 0 0 0]'
```

### Using Squeeze on GPU-Resident Data
```matlab
G = gpuArray(rand(1, 64, 1));
H = squeeze(G);
isgpuarray(H)
```
Expected output:
```matlab
ans = logical 1
```

### Squeezing String Arrays While Preserving Layout
```matlab
S = strings(1, 1, 3);
S(1, 1, 1) = "run";
S(1, 1, 2) = "mat";
S(1, 1, 3) = "gpu";
T = squeeze(S);
size(T)
```
Expected output:
```matlab
ans = [3 1]
```

## FAQ
### Why does `squeeze` keep 2-D matrices unchanged?
MATLAB treats row and column vectors as 2-D objects. Preserving those dimensions avoids turning row vectors into column vectors or scalars unintentionally.

### What happens when every dimension is singleton?
The result becomes a `1 × 1` array. Scalars (e.g., plain doubles) remain scalars.

### Does `squeeze` affect zero-length dimensions?
No. Only size-one dimensions are removed. Zero-length dimensions are preserved exactly.

### Can I `squeeze` character or cell arrays?
Yes. They already store up to two dimensions in RunMat, so `squeeze` leaves them unchanged but still validates inputs.

### How does `squeeze` interact with GPU tensors?
The runtime updates GPU tensor metadata via the provider's reshape hook. Data stays on the device unless a provider cannot report shape information, in which case RunMat gathers once to infer dimensions.

### What if I need to remove a specific dimension?
Use `reshape`, `permute`, or indexing to reorganize axes manually before calling `squeeze`.

### Does `squeeze` change data ordering?
No. It only alters metadata. Element order in memory remains unchanged.

### Will `squeeze` ever increase the number of dimensions?
No. It either returns the same number or fewer. The only exception is padding with a trailing singleton to maintain MATLAB's column-vector convention when exactly one dimension remains.

### Is `squeeze` safe to run before GPU fusion?
Yes. Since it only updates shapes, it integrates cleanly with fusion and does not impede kernel generation.

### How do I undo a squeeze?
Use `reshape` with the original size vector (for example, captured via `size(A)`) to restore the previous dimension layout.

## See Also
- [`reshape`](./reshape)
- [`permute`](./permute)
- [`ndims`](./ndims)
- [`size`](./size)
- [`gpuArray`](./gpuarray)
- [`gather`](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::squeeze")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "squeeze",
    op_kind: GpuOpKind::Custom("squeeze"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("reshape")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses provider reshape hook to drop singleton metadata without moving device buffers.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::squeeze")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "squeeze",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Squeeze only mutates metadata; fusion planner treats it as a no-op for kernel generation.",
};

fn squeeze_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("squeeze").build()
}

#[runtime_builtin(
    name = "squeeze",
    category = "array/shape",
    summary = "Remove singleton dimensions while preserving MATLAB row/column semantics.",
    keywords = "squeeze,singleton dimensions,array reshape,gpu",
    accel = "shape",
    builtin_path = "crate::builtins::array::shape::squeeze"
)]
async fn squeeze_builtin(value: Value) -> crate::BuiltinResult<Value> {
    squeeze_value(value).await
}

async fn squeeze_value(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => squeeze_numeric_tensor(tensor).map(Value::Tensor),
        Value::ComplexTensor(ct) => squeeze_complex_tensor(ct).map(Value::ComplexTensor),
        Value::LogicalArray(logical) => squeeze_logical_array(logical).map(Value::LogicalArray),
        Value::StringArray(strings) => squeeze_string_array(strings).map(Value::StringArray),
        Value::GpuTensor(handle) => squeeze_gpu(handle).await,
        Value::String(_) | Value::CharArray(_) | Value::Cell(_) | Value::Struct(_) => Ok(value),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(value),
        other => Err(squeeze_error(format!(
            "squeeze: unsupported input type {}; expected numeric, logical, string, char, cell, or gpu array",
            value_kind(&other)
        ))),
    }
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Tensor(_) => "tensor",
        Value::ComplexTensor(_) => "complex tensor",
        Value::LogicalArray(_) => "logical array",
        Value::StringArray(_) => "string array",
        Value::CharArray(_) => "char array",
        Value::Cell(_) => "cell array",
        Value::GpuTensor(_) => "gpu array",
        Value::Num(_) => "double scalar",
        Value::Int(_) => "integer scalar",
        Value::Bool(_) => "logical scalar",
        Value::Complex(_, _) => "complex scalar",
        Value::String(_) => "string scalar",
        Value::Object(_) => "object",
        Value::HandleObject(_) => "handle object",
        Value::Listener(_) => "listener",
        Value::Struct(_) => "struct",
        Value::FunctionHandle(_) | Value::Closure(_) => "function handle",
        Value::ClassRef(_) => "class reference",
        Value::MException(_) => "exception",
    }
}

fn squeeze_numeric_tensor(tensor: Tensor) -> crate::BuiltinResult<Tensor> {
    let shape = squeeze_shape(&tensor.shape);
    if shape == tensor.shape {
        return Ok(tensor);
    }
    Tensor::new(tensor.data, shape).map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

fn squeeze_complex_tensor(ct: ComplexTensor) -> crate::BuiltinResult<ComplexTensor> {
    let shape = squeeze_shape(&ct.shape);
    if shape == ct.shape {
        return Ok(ct);
    }
    ComplexTensor::new(ct.data, shape).map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

fn squeeze_logical_array(logical: LogicalArray) -> crate::BuiltinResult<LogicalArray> {
    let shape = squeeze_shape(&logical.shape);
    if shape == logical.shape {
        return Ok(logical);
    }
    LogicalArray::new(logical.data, shape).map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

fn squeeze_string_array(strings: StringArray) -> crate::BuiltinResult<StringArray> {
    let shape = squeeze_shape(&strings.shape);
    if shape == strings.shape {
        return Ok(strings);
    }
    StringArray::new(strings.data, shape).map_err(|e| squeeze_error(format!("squeeze: {e}")))
}

async fn squeeze_gpu(handle: GpuTensorHandle) -> crate::BuiltinResult<Value> {
    let shape_source = if handle.shape.is_empty() {
        let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
        gathered.shape
    } else {
        handle.shape.clone()
    };

    let squeezed = squeeze_shape(&shape_source);
    if squeezed == handle.shape {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(updated) = provider.reshape(&handle, &squeezed) {
            return Ok(Value::GpuTensor(updated));
        }
    }

    let mut updated = handle;
    updated.shape = squeezed;
    Ok(Value::GpuTensor(updated))
}

fn squeeze_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() <= 2 {
        return shape.to_vec();
    }
    let mut squeezed: Vec<usize> = shape.iter().copied().filter(|&d| d != 1).collect();
    if squeezed.is_empty() {
        squeezed.push(1);
        squeezed.push(1);
    } else if squeezed.len() == 1 {
        let first = squeezed[0];
        squeezed = vec![first, 1];
    }
    squeezed
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn squeeze_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::squeeze_builtin(value))
    }
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_removes_middle_singletons() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
        let result = squeeze_builtin(Value::Tensor(tensor)).expect("squeeze ok");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_preserves_row_vector() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = squeeze_builtin(Value::Tensor(tensor)).expect("squeeze ok");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![1, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_single_dimension_becomes_column_vector() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4, 1]).unwrap();
        let result = squeeze_builtin(Value::Tensor(tensor)).expect("squeeze ok");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![4, 1]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_on_logical_array_respects_zero_dims() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![1, 4, 1]).unwrap();
        let result = squeeze_builtin(Value::LogicalArray(logical)).expect("squeeze ok");
        match result {
            Value::LogicalArray(arr) => assert_eq!(arr.shape, vec![4, 1]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_on_string_array() {
        let strings = StringArray::new(vec!["a".into(), "b".into()], vec![1, 1, 2]).unwrap();
        let result = squeeze_builtin(Value::StringArray(strings)).expect("squeeze ok");
        match result {
            Value::StringArray(sa) => assert_eq!(sa.shape, vec![2, 1]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_preserves_zero_length_dimensions() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![1, 0, 3]).unwrap();
        let result = squeeze_builtin(Value::Tensor(tensor)).expect("squeeze ok");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![0, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let value = squeeze_builtin(Value::GpuTensor(handle)).expect("squeeze ok");
            let gathered = test_support::gather(value).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn squeeze_scalar_inputs_passthrough() {
        let result = squeeze_builtin(Value::Int(IntValue::I32(42))).expect("squeeze ok for scalar");
        assert_eq!(result, Value::Int(IntValue::I32(42)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn squeeze_wgpu_updates_shape_metadata() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let tensor = Tensor::new(
            (0..12).map(|v| v as f64).collect::<Vec<_>>(),
            vec![1, 3, 4, 1],
        )
        .unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload source tensor");

        let squeezed =
            squeeze_builtin(Value::GpuTensor(handle.clone())).expect("squeeze gpu tensor");
        let gpu_handle = match squeezed {
            Value::GpuTensor(h) => h,
            other => panic!("expected gpu tensor, got {other:?}"),
        };
        assert_eq!(gpu_handle.shape, vec![3, 4]);

        let downloaded = provider
            .download(&gpu_handle)
            .expect("download squeezed tensor");
        assert_eq!(downloaded.shape.as_slice(), &[3, 4]);
        assert_eq!(downloaded.data.as_slice(), tensor.data.as_slice());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
