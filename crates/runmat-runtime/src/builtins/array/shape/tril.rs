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
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

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

fn preserve_matrix_type(args: &[Type], _context: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(shape) } => {
            let rows = shape.get(0).copied().unwrap_or(None);
            let cols = shape.get(1).copied().unwrap_or(None);
            Type::Tensor {
                shape: Some(vec![rows, cols]),
            }
        }
        Type::Logical { shape: Some(shape) } => {
            let rows = shape.get(0).copied().unwrap_or(None);
            let cols = shape.get(1).copied().unwrap_or(None);
            Type::Logical {
                shape: Some(vec![rows, cols]),
            }
        }
        Type::Tensor { shape: None } => Type::tensor(),
        Type::Logical { shape: None } => Type::logical(),
        Type::Num | Type::Int | Type::Bool => Type::tensor(),
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn tril_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("tril").build()
}

#[runtime_builtin(
    name = "tril",
    category = "array/shape",
    summary = "Lower triangular portion of a matrix or paged tensor.",
    keywords = "tril,lower triangular,matrix,diagonal,gpu",
    accel = "custom",
    type_resolver(preserve_matrix_type),
    builtin_path = "crate::builtins::array::shape::tril"
)]
async fn tril_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(tril_error("tril: too many input arguments"));
    }
    let offset = parse_diagonal_offset(&rest).await?;
    match value {
        Value::Tensor(tensor) => Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?),
        Value::LogicalArray(array) => {
            Ok(tril_logical_array(array, offset).map(Value::LogicalArray)?)
        }
        Value::ComplexTensor(tensor) => {
            Ok(tril_complex_tensor(tensor, offset).map(Value::ComplexTensor)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| tril_error(format!("tril: {e}")))?;
            Ok(tril_complex_tensor(tensor, offset).map(complex_tensor_into_value)?)
        }
        Value::Num(n) => {
            let tensor =
                tensor::value_into_tensor_for("tril", Value::Num(n)).map_err(|e| tril_error(e))?;
            Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("tril", Value::Int(i.clone()))
                .map_err(|e| tril_error(e))?;
            Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("tril", Value::Bool(flag))
                .map_err(|e| tril_error(e))?;
            Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::CharArray(chars) => {
            let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
            let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|e| tril_error(format!("tril: {e}")))?;
            Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => Ok(tril_gpu(handle, offset).await?),
        Value::String(_) | Value::StringArray(_) => {
            Err(tril_error("tril: string arrays are not supported"))
        }
        Value::Cell(_) => Err(tril_error("tril: cell arrays are not supported")),
        Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(tril_error("tril: unsupported input type")),
    }
}

async fn parse_diagonal_offset(args: &[Value]) -> crate::BuiltinResult<isize> {
    if args.is_empty() {
        return Ok(0);
    }
    let gathered = crate::dispatcher::gather_if_needed_async(&args[0])
        .await
        .map_err(|e| tril_error(format!("tril: {e}")))?;
    scalar_to_isize(&gathered, "tril")
}

fn scalar_to_isize(value: &Value, name: &str) -> crate::BuiltinResult<isize> {
    match value {
        Value::Int(i) => Ok(i.to_i64() as isize),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(tril_error(format!(
                    "{name}: diagonal offset must be finite"
                )));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(tril_error(format!(
                    "{name}: diagonal offset must be an integer"
                )));
            }
            Ok(rounded as isize)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            let val = t.data[0];
            scalar_to_isize(&Value::Num(val), name)
        }
        Value::Bool(flag) => Ok(if *flag { 1 } else { 0 }),
        other => Err(tril_error(format!(
            "{name}: diagonal offset must be a scalar numeric value, got {other:?}"
        ))),
    }
}

fn tril_tensor(mut tensor: Tensor, offset: isize) -> crate::BuiltinResult<Tensor> {
    apply_tril_inplace(&mut tensor.data, &tensor.shape, offset, 0.0)?;
    Ok(tensor)
}

fn tril_logical_array(
    mut array: LogicalArray,
    offset: isize,
) -> crate::BuiltinResult<LogicalArray> {
    apply_tril_inplace(&mut array.data, &array.shape, offset, 0u8)?;
    Ok(array)
}

fn tril_complex_tensor(
    mut tensor: ComplexTensor,
    offset: isize,
) -> crate::BuiltinResult<ComplexTensor> {
    apply_tril_inplace(&mut tensor.data, &tensor.shape, offset, (0.0, 0.0))?;
    Ok(tensor)
}

async fn tril_gpu(handle: GpuTensorHandle, offset: isize) -> crate::BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.tril(&handle, offset).await {
            Ok(out) => return Ok(Value::GpuTensor(out)),
            Err(_) => {
                // Fall through to gather path.
            }
        }
        let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        let result = tril_tensor(tensor, offset)?;
        let view = HostTensorView {
            data: &result.data,
            shape: &result.shape,
        };
        let uploaded = provider
            .upload(&view)
            .map_err(|e| tril_error(format!("tril: failed to upload fallback result: {e}")))?;
        Ok(Value::GpuTensor(uploaded))
    } else {
        let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?)
    }
}

fn apply_tril_inplace<T>(
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
        1
    } else {
        shape[2..].iter().product::<usize>()
    };
    if plane == 0 || pages == 0 {
        return Ok(());
    }
    let expected = plane
        .checked_mul(pages)
        .ok_or_else(|| tril_error("tril: dimension product overflow"))?;
    if expected != data.len() {
        return Err(tril_error("tril: tensor data length mismatch"));
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
    use futures::executor::block_on;

    fn tril_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::tril_builtin(value, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray, Type};

    #[test]
    fn tril_type_preserves_matrix_shape() {
        let out = preserve_matrix_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(2)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(2)])
            }
        );
    }

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
        let gpu = block_on(super::tril_gpu(handle, -1)).expect("gpu tril");
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }
}
