//! MATLAB-compatible `tril` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the `tril` function, mirroring MathWorks MATLAB
//! behaviour across real, logical, and complex tensors, including paged
//! matrices. It honours diagonal offsets, keeps higher-dimensional slices
//! independent, and preserves gpuArray residency whenever an acceleration
//! provider is registered.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "tril",
        builtin_path = "crate::builtins::array::shape::tril"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "tril"
category: "array/shape"
keywords: ["tril", "lower triangular", "matrix", "diagonal", "gpu"]
summary: "Extract the lower triangular portion of a matrix (optionally including super-diagonals)."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider tril kernels when available; otherwise gathers once, computes on the host, and re-uploads to keep results gpu-resident."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::tril::tests"
  integration: "builtins::array::shape::tril::tests::tril_gpu_roundtrip"
---

# What does the `tril` function do in MATLAB / RunMat?
`tril(A)` keeps the elements on and below a selected diagonal of `A` and sets
everything above that diagonal to zero. The optional second argument `k`
controls which diagonal is retained:

- `k = 0` (default) keeps the main diagonal.
- `k > 0` includes super-diagonals above the main diagonal.
- `k < 0` drops the main diagonal and starts below it.

The operation applies independently to every matrix "page" of N-D tensors.

## How does the `tril` function behave in MATLAB / RunMat?
- Works on numeric, logical, and complex arrays.
- Operates on the first two dimensions; trailing dimensions are handled as
  independent pages.
- Accepts scalar, vector, matrix, or paged inputs of any size, including empty
  dimensions.
- Logical inputs remain logical, and complex values keep their real/imaginary
  components.
- Scalars are treated as `1×1` matrices and honour negative offsets (`k < 0`
  yields zero).
- gpuArray inputs stay on the device when the provider exposes a native `tril`
  hook; otherwise RunMat performs a gather → compute → upload cycle.

## `tril` Function GPU Execution Behaviour
- If the active acceleration provider implements the custom `tril` hook the
  entire operation runs on the GPU.
- When the hook is missing, RunMat gathers the data once, computes the result on
  the host, uploads the lower triangular tensor back to the device, and returns
  a gpuArray handle so residency is preserved for downstream kernels.
- Falling back to the host never changes numerical results; it only affects
  where the computation is carried out.

## Examples of using the `tril` function in MATLAB / RunMat

### Extracting the lower triangular part of a matrix
```matlab
A = [1 2 3; 4 5 6; 7 8 9];
L = tril(A);
```
Expected output:
```matlab
L =
     1     0     0
     4     5     0
     7     8     9
```

### Keeping one super-diagonal above the main diagonal
```matlab
A = magic(4);
L = tril(A, 1);
```
Expected output:
```matlab
L =
    16     2     0     0
     5    11    10     0
     9     7     6    12
     4    14    15     1
```

### Dropping the main diagonal with a negative offset
```matlab
A = [1 2 3; 4 5 6; 7 8 9];
strict = tril(A, -1);
```
Expected output:
```matlab
strict =
     0     0     0
     4     0     0
     7     8     0
```

### Applying `tril` to every page of a 3-D array
```matlab
T = reshape(1:18, [3 3 2]);
L = tril(T);
```
Expected output:
```matlab
L(:, :, 1) =
     1     0     0
     4     5     0
     7     8     9

L(:, :, 2) =
    10     0     0
    13    14     0
    16    17    18
```

### Preserving gpuArray residency with `tril`
```matlab
G = gpuArray(rand(5));
L = tril(G, -2);
isa(L, 'gpuArray')
```
Expected output:
```matlab
ans =
  logical
   1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No additional steps are required. RunMat keeps gpuArray inputs on the device,
and the auto-offload planner will promote large CPU tensors to GPU residency
when it detects a benefit. Explicit `gpuArray` / `gather` calls remain available
for MATLAB compatibility or to force a particular residency in workflows that
interact with other libraries.

## FAQ
### What happens when `k` is larger than the matrix size?
The entire matrix is preserved; `tril` never removes elements below the chosen
diagonal.

### Does `tril` work with logical arrays?
Yes. Elements above the retained diagonal become `false`, while the rest keep
their logical values.

### How are complex numbers handled?
Each element keeps its real and imaginary parts intact. Only the elements above
the chosen diagonal are zeroed out (`0 + 0i`).

### What about empty matrices or zero-sized dimensions?
`tril` returns an array of the same shape, leaving all entries at zero. Trailing
dimensions with size zero are treated as empty batches.

### Does `tril` change the class of character arrays?
Character arrays are converted to their numeric codes (double precision) before
the triangular mask is applied, matching MATLAB's behaviour.

## See Also
- [`triu`](./triu) *(upper triangular complement)*
- [`diag`](./diag)
- [`kron`](./kron)
- [`reshape`](./reshape)
- [`gpuArray`](./gpuarray)
- [`gather`](./gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/array/shape/tril.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/tril.rs)
- Issues & feedback: [github.com/runmat-org/runmat/issues/new/choose](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::tril")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tril",
    op_kind: GpuOpKind::Custom("tril"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("tril")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement tril directly; the runtime falls back to gather→compute→upload when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::tril")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tril",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Triangular masking is not currently fused; fusion planner treats tril nodes as boundaries.",
};

#[runtime_builtin(
    name = "tril",
    category = "array/shape",
    summary = "Lower triangular portion of a matrix or paged tensor.",
    keywords = "tril,lower triangular,matrix,diagonal,gpu",
    accel = "custom",
    builtin_path = "crate::builtins::array::shape::tril"
)]
fn tril_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err((("tril: too many input arguments".to_string())).into());
    }
    let offset = parse_diagonal_offset(&rest)?;
    match value {
        Value::Tensor(tensor) => (tril_tensor(tensor, offset).map(tensor::tensor_into_value)).map_err(Into::into),
        Value::LogicalArray(array) => (tril_logical_array(array, offset).map(Value::LogicalArray)).map_err(Into::into),
        Value::ComplexTensor(tensor) => {
            tril_complex_tensor(tensor, offset).map(Value::ComplexTensor)
        }
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("tril: {e}"))?;
            tril_complex_tensor(tensor, offset).map(complex_tensor_into_value)
        }
        Value::Num(n) => (tril_tensor(
            tensor::value_into_tensor_for("tril", Value::Num(n))?,
            offset,
        )
        .map(tensor::tensor_into_value)).map_err(Into::into),
        Value::Int(i) => (tril_tensor(
            tensor::value_into_tensor_for("tril", Value::Int(i.clone()))?,
            offset,
        )
        .map(tensor::tensor_into_value)).map_err(Into::into),
        Value::Bool(flag) => (tril_tensor(
            tensor::value_into_tensor_for("tril", Value::Bool(flag))?,
            offset,
        )
        .map(tensor::tensor_into_value)).map_err(Into::into),
        Value::CharArray(chars) => {
            let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
            let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|e| format!("tril: {e}"))?;
            tril_tensor(tensor, offset).map(tensor::tensor_into_value)
        }
        Value::GpuTensor(handle) => (tril_gpu(handle, offset)).map_err(Into::into),
        Value::String(_) | Value::StringArray(_) => {
            Err((("tril: string arrays are not supported".to_string())).into())
        }
        Value::Cell(_) => Err((("tril: cell arrays are not supported".to_string())).into()),
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err((("tril: unsupported input type".to_string())).into()),
    }
}

fn parse_diagonal_offset(args: &[Value]) -> Result<isize, String> {
    if args.is_empty() {
        return Ok(0);
    }
    let gathered =
        crate::dispatcher::gather_if_needed(&args[0]).map_err(|e| format!("tril: {e}"))?;
    scalar_to_isize(&gathered, "tril")
}

fn scalar_to_isize(value: &Value, name: &str) -> Result<isize, String> {
    match value {
        Value::Int(i) => Ok(i.to_i64() as isize),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format!("{name}: diagonal offset must be finite"));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(format!("{name}: diagonal offset must be an integer"));
            }
            Ok(rounded as isize)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            let val = t.data[0];
            scalar_to_isize(&Value::Num(val), name)
        }
        Value::Bool(flag) => Ok(if *flag { 1 } else { 0 }),
        other => Err(format!(
            "{name}: diagonal offset must be a scalar numeric value, got {other:?}"
        )),
    }
}

fn tril_tensor(mut tensor: Tensor, offset: isize) -> Result<Tensor, String> {
    apply_tril_inplace(&mut tensor.data, &tensor.shape, offset, 0.0)?;
    Ok(tensor)
}

fn tril_logical_array(mut array: LogicalArray, offset: isize) -> Result<LogicalArray, String> {
    apply_tril_inplace(&mut array.data, &array.shape, offset, 0u8)?;
    Ok(array)
}

fn tril_complex_tensor(mut tensor: ComplexTensor, offset: isize) -> Result<ComplexTensor, String> {
    apply_tril_inplace(&mut tensor.data, &tensor.shape, offset, (0.0, 0.0))?;
    Ok(tensor)
}

fn tril_gpu(handle: GpuTensorHandle, offset: isize) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.tril(&handle, offset) {
            Ok(out) => return Ok(Value::GpuTensor(out)),
            Err(_) => {
                // Fall through to gather path.
            }
        }
        let tensor = gpu_helpers::gather_tensor(&handle)?;
        let result = tril_tensor(tensor, offset)?;
        let view = HostTensorView {
            data: &result.data,
            shape: &result.shape,
        };
        let uploaded = provider
            .upload(&view)
            .map_err(|e| format!("tril: failed to upload fallback result: {e}"))?;
        Ok(Value::GpuTensor(uploaded))
    } else {
        let tensor = gpu_helpers::gather_tensor(&handle)?;
        tril_tensor(tensor, offset).map(tensor::tensor_into_value)
    }
}

fn apply_tril_inplace<T>(
    data: &mut [T],
    shape: &[usize],
    offset: isize,
    zero: T,
) -> Result<(), String>
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
        1
    } else {
        shape[2..].iter().product::<usize>()
    };
    if plane == 0 || pages == 0 {
        return Ok(());
    }
    let expected = plane
        .checked_mul(pages)
        .ok_or_else(|| "tril: dimension product overflow".to_string())?;
    if expected != data.len() {
        return Err("tril: tensor data length mismatch".to_string());
    }
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            for row in 0..rows {
                let row_idx = row as i128;
                let col_idx = col as i128;
                let offset_idx = offset as i128;
                if row_idx < col_idx - offset_idx {
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
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_main_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let value = tril_builtin(Value::Tensor(tensor), Vec::new()).expect("tril");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.shape, vec![2, 3]);
                assert_eq!(result.data, vec![1.0, 4.0, 0.0, 5.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_with_positive_offset_keeps_super_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let offset = Value::Int(IntValue::I32(1));
        let value =
            tril_builtin(Value::Tensor(tensor), vec![offset]).expect("tril with positive offset");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 0.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_negative_offset_drops_main_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let offset = Value::Int(IntValue::I32(-1));
        let value =
            tril_builtin(Value::Tensor(tensor), vec![offset]).expect("tril with negative offset");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.data, vec![0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_logical_array_preserves_type() {
        let logical =
            LogicalArray::new(vec![1, 0, 1, 1, 1, 1], vec![2, 3]).expect("logical creation");
        let value =
            tril_builtin(Value::LogicalArray(logical), Vec::new()).expect("tril logical array");
        match value {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data, vec![1, 0, 0, 1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_complex_tensor_masks_values() {
        let data = vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let value = tril_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("tril complex");
        match value {
            Value::ComplexTensor(result) => {
                assert_eq!(result.shape, vec![2, 2]);
                assert_eq!(result.data[0], (1.0, 2.0));
                assert_eq!(result.data[1], (3.0, 4.0));
                assert_eq!(result.data[2], (0.0, 0.0));
                assert_eq!(result.data[3], (7.0, 8.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_scalar_with_negative_offset_returns_zero() {
        let value =
            tril_builtin(Value::Num(5.0), vec![Value::Int(IntValue::I32(-1))]).expect("tril");
        match value {
            Value::Num(result) => assert_eq!(result, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let value = tril_builtin(Value::GpuTensor(handle), Vec::new()).expect("tril gpu");
            let gathered = test_support::gather(value).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 2.0, 0.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn tril_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new((1..=16).map(|v| v as f64).collect::<Vec<_>>(), vec![4, 4]).unwrap();
        let cpu = tril_tensor(tensor.clone(), -1).expect("cpu tril");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = tril_gpu(handle, -1).expect("gpu tril");
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }
}
