//! MATLAB-compatible `triu` builtin with GPU-aware semantics for RunMat.
//!
//! Mirrors MathWorks MATLAB behaviour across numeric, logical, and complex
//! tensors while preserving gpuArray residency whenever possible. The builtin
//! honours diagonal offsets and applies the mask independently to every matrix
//! page of higher-dimensional tensors.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "triu",
        builtin_path = "crate::builtins::array::shape::triu"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "triu"
category: "array/shape"
keywords: ["triu", "upper triangular", "matrix", "diagonal", "gpu"]
summary: "Keep the upper triangular portion of a matrix (optionally including sub-diagonals)."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider triu kernels when available; otherwise gathers once, masks on the host, and re-uploads so results remain gpu-resident."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::triu::tests"
  integration: "builtins::array::shape::triu::tests::triu_gpu_roundtrip"
---

# What does the `triu` function do in MATLAB / RunMat?
`triu(A)` keeps the elements on and above a selected diagonal of `A` and sets
everything below that diagonal to zero. The optional second argument `k`
controls which diagonal is retained:

- `k = 0` (default) keeps the main diagonal.
- `k > 0` keeps the diagonal `k` steps above the main diagonal (the main
  diagonal and everything below it become zero).
- `k < 0` includes additional sub-diagonals beneath the main diagonal.

Every matrix "page" in an N-D tensor is processed independently.

## How does the `triu` function behave in MATLAB / RunMat?
- Works on numeric, logical, and complex arrays.
- Operates on the first two dimensions; trailing dimensions act as
  independent pages.
- Preserves logical types and complex-valued elements.
- Scalars are treated as `1×1` matrices and honour diagonal offsets (for
  example `triu(5, 1)` returns `0`).
- gpuArray inputs stay on the device when an acceleration provider supplies a
  native `triu` hook; otherwise the runtime gathers, masks on the host, and
  uploads the result back to the GPU.

## `triu` Function GPU Execution Behaviour
- If the active acceleration provider implements a `triu` kernel the entire
  operation executes on the GPU.
- Without a provider hook, RunMat gathers the tensor to host memory once,
  applies the mask, uploads the result, and returns a gpuArray so residency is
  preserved for downstream kernels.
- Fallbacks never affect numerical results—only where the computation runs.

## Examples of using the `triu` function in MATLAB / RunMat

### Extracting the upper triangular part of a matrix
```matlab
A = [1 2 3; 4 5 6; 7 8 9];
U = triu(A);
```
Expected output:
```matlab
U =
     1     2     3
     0     5     6
     0     0     9
```

### Keeping one sub-diagonal beneath the main diagonal
```matlab
A = magic(4);
U = triu(A, -1);
```
Expected output:
```matlab
U =
    16     2     3     13
     5    11    10     8
     0     7     6    12
     0     0    15     1
```

### Dropping the main diagonal with a positive offset
```matlab
A = [1 2 3; 4 5 6; 7 8 9];
strict = triu(A, 1);
```
Expected output:
```matlab
strict =
     0     2     3
     0     0     6
     0     0     0
```

### Applying `triu` to every page of a 3-D array
```matlab
T = reshape(1:18, [3 3 2]);
U = triu(T);
```
Expected output:
```matlab
U(:, :, 1) =
     1     2     3
     0     5     6
     0     0     9

U(:, :, 2) =
    10    11    12
     0    14    15
     0     0    18
```

### Preserving gpuArray residency with `triu`
```matlab
G = gpuArray(rand(5));
U = triu(G, -2);
isa(U, 'gpuArray')
```
Expected output:
```matlab
ans =
  logical
   1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No additional steps are required. RunMat keeps gpuArray inputs on the device,
and the auto-offload planner promotes large CPU tensors to GPU residency when
profitable. Explicit `gpuArray` / `gather` calls remain available for MATLAB
compatibility or when coordinating with external libraries.

## FAQ
### What happens when `k` is smaller than the matrix size (large negative)?
The entire matrix is preserved; `triu` never removes elements above the chosen
diagonal.

### Does `triu` work with logical arrays?
Yes. Elements below the retained diagonal become `false`, while the rest keep
their logical values.

### How are complex numbers handled?
Each element retains its real and imaginary parts. Only elements below the
chosen diagonal become `0 + 0i`.

### What about empty matrices or zero-sized dimensions?
`triu` returns an array of the same shape, leaving all entries at zero. Trailing
dimensions with size zero are treated as empty batches.

### Does `triu` change the class of character arrays?
Character arrays are converted to their numeric codes (double precision) before
the triangular mask is applied, matching MATLAB behaviour.

## See Also
- [`tril`](./tril) *(lower triangular complement)*
- [`diag`](./diag)
- [`kron`](./kron)
- [`reshape`](./reshape)
- [`gpuArray`](./gpuarray)
- [`gather`](./gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/array/shape/triu.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/triu.rs)
- Issues & feedback: [github.com/runmat-org/runmat/issues/new/choose](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::triu")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "triu",
    op_kind: GpuOpKind::Custom("triu"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("triu")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement triu directly; the runtime falls back to gather→mask→upload when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::triu")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "triu",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Triangular masking is currently treated as a fusion boundary.",
};

fn triu_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("triu").build()
}

#[runtime_builtin(
    name = "triu",
    category = "array/shape",
    summary = "Upper triangular portion of a matrix or paged tensor.",
    keywords = "triu,upper triangular,matrix,diagonal,gpu",
    accel = "custom",
    builtin_path = "crate::builtins::array::shape::triu"
)]
async fn triu_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(triu_error("triu: too many input arguments"));
    }
    let offset = parse_diagonal_offset(&rest).await?;
    match value {
        Value::Tensor(tensor) => Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?),
        Value::LogicalArray(array) => {
            Ok(triu_logical_array(array, offset).map(Value::LogicalArray)?)
        }
        Value::ComplexTensor(tensor) => {
            Ok(triu_complex_tensor(tensor, offset).map(Value::ComplexTensor)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| triu_error(format!("triu: {e}")))?;
            Ok(triu_complex_tensor(tensor, offset).map(complex_tensor_into_value)?)
        }
        Value::Num(n) => {
            let tensor =
                tensor::value_into_tensor_for("triu", Value::Num(n)).map_err(|e| triu_error(e))?;
            Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("triu", Value::Int(i.clone()))
                .map_err(|e| triu_error(e))?;
            Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("triu", Value::Bool(flag))
                .map_err(|e| triu_error(e))?;
            Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::CharArray(chars) => {
            let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
            let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|e| triu_error(format!("triu: {e}")))?;
            Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => Ok(triu_gpu(handle, offset).await?),
        Value::String(_) | Value::StringArray(_) => {
            Err(triu_error("triu: string arrays are not supported"))
        }
        Value::Cell(_) => Err(triu_error("triu: cell arrays are not supported")),
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(triu_error("triu: unsupported input type")),
    }
}

async fn parse_diagonal_offset(args: &[Value]) -> crate::BuiltinResult<isize> {
    if args.is_empty() {
        return Ok(0);
    }
    let gathered = crate::dispatcher::gather_if_needed_async(&args[0])
        .await
        .map_err(|e| triu_error(format!("triu: {e}")))?;
    scalar_to_isize(&gathered, "triu")
}

fn scalar_to_isize(value: &Value, name: &str) -> crate::BuiltinResult<isize> {
    match value {
        Value::Int(i) => Ok(i.to_i64() as isize),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(triu_error(format!(
                    "{name}: diagonal offset must be finite"
                )));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(triu_error(format!(
                    "{name}: diagonal offset must be an integer"
                )));
            }
            Ok(rounded as isize)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            scalar_to_isize(&Value::Num(t.data[0]), name)
        }
        Value::Bool(flag) => Ok(if *flag { 1 } else { 0 }),
        other => Err(triu_error(format!(
            "{name}: diagonal offset must be a scalar numeric value, got {other:?}"
        ))),
    }
}

fn triu_tensor(mut tensor: Tensor, offset: isize) -> crate::BuiltinResult<Tensor> {
    apply_triu_inplace(&mut tensor.data, &tensor.shape, offset, 0.0)?;
    Ok(tensor)
}

fn triu_logical_array(
    mut array: LogicalArray,
    offset: isize,
) -> crate::BuiltinResult<LogicalArray> {
    apply_triu_inplace(&mut array.data, &array.shape, offset, 0u8)?;
    Ok(array)
}

fn triu_complex_tensor(
    mut tensor: ComplexTensor,
    offset: isize,
) -> crate::BuiltinResult<ComplexTensor> {
    apply_triu_inplace(&mut tensor.data, &tensor.shape, offset, (0.0, 0.0))?;
    Ok(tensor)
}

async fn triu_gpu(handle: GpuTensorHandle, offset: isize) -> crate::BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.triu(&handle, offset) {
            Ok(out) => return Ok(Value::GpuTensor(out)),
            Err(_) => {
                // Fall through to the gather path.
            }
        }
        let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        let result = triu_tensor(tensor, offset)?;
        let view = HostTensorView {
            data: &result.data,
            shape: &result.shape,
        };
        let uploaded = provider
            .upload(&view)
            .map_err(|e| triu_error(format!("triu: failed to upload fallback result: {e}")))?;
        Ok(Value::GpuTensor(uploaded))
    } else {
        let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?)
    }
}

fn apply_triu_inplace<T>(
    data: &mut [T],
    shape: &[usize],
    offset: isize,
    zero: T,
) -> crate::BuiltinResult<()>
where
    T: Clone,
{
    if data.is_empty() {
        return Ok(());
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let plane = rows.saturating_mul(cols);
    let pages = if shape.len() <= 2 {
        1usize
    } else {
        shape[2..].iter().product::<usize>()
    };
    if plane == 0 || pages == 0 {
        return Ok(());
    }
    let expected = plane
        .checked_mul(pages)
        .ok_or_else(|| triu_error("triu: dimension product overflow"))?;
    if expected != data.len() {
        return Err(triu_error("triu: tensor data length mismatch"));
    }

    let offset_i128 = offset as i128;
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            let col_i = col as i128;
            for row in 0..rows {
                let row_i = row as i128;
                let diff = col_i - row_i;
                if diff < offset_i128 {
                    data[col_base + row] = zero.clone();
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn triu_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::triu_builtin(value, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_main_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let value = triu_builtin(Value::Tensor(tensor), Vec::new()).expect("triu");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.shape, vec![2, 3]);
                assert_eq!(result.data, vec![1.0, 0.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_positive_offset_drops_main_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let offset = Value::Int(IntValue::I32(1));
        let value =
            triu_builtin(Value::Tensor(tensor), vec![offset]).expect("triu with positive offset");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.data, vec![0.0, 0.0, 2.0, 0.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_negative_offset_includes_sub_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let offset = Value::Int(IntValue::I32(-1));
        let value =
            triu_builtin(Value::Tensor(tensor), vec![offset]).expect("triu with negative offset");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_logical_array_preserves_type() {
        let logical =
            LogicalArray::new(vec![1, 0, 1, 1, 1, 1], vec![2, 3]).expect("logical creation");
        let value =
            triu_builtin(Value::LogicalArray(logical), Vec::new()).expect("triu logical array");
        match value {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data, vec![1, 0, 1, 1, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_complex_tensor_masks_values() {
        let data = vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let value = triu_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("triu complex");
        match value {
            Value::ComplexTensor(result) => {
                assert_eq!(result.shape, vec![2, 2]);
                assert_eq!(result.data[0], (1.0, 2.0));
                assert_eq!(result.data[1], (0.0, 0.0));
                assert_eq!(result.data[2], (5.0, 6.0));
                assert_eq!(result.data[3], (7.0, 8.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_scalar_with_positive_offset_returns_zero() {
        let value =
            triu_builtin(Value::Num(5.0), vec![Value::Int(IntValue::I32(1))]).expect("triu");
        match value {
            Value::Num(result) => assert_eq!(result, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let value = triu_builtin(Value::GpuTensor(handle), Vec::new()).expect("triu gpu");
            let gathered = test_support::gather(value).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 3.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_rejects_string_offset() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = triu_builtin(Value::Tensor(tensor), vec![Value::from("diagonal")]).unwrap_err();
        assert!(
            err.to_string()
                .contains("diagonal offset must be a scalar numeric value"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_rejects_extra_arguments() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = triu_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(1.0), Value::Num(2.0)],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("too many input arguments"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn triu_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new((1..=16).map(|v| v as f64).collect::<Vec<_>>(), vec![4, 4]).unwrap();
        let cpu = triu_tensor(tensor.clone(), 1).expect("cpu triu");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = block_on(super::triu_gpu(handle, 1)).expect("gpu triu");
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }
}
