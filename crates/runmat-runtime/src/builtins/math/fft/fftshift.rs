//! MATLAB-compatible `fftshift` builtin with GPU-aware semantics for RunMat.
//!
//! `fftshift` recenters zero-frequency components for outputs produced by FFTs.

use super::common::{apply_shift, build_shift_plan, compute_shift_dims, ShiftKind};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::type_resolvers::fftshift_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::fftshift")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fftshift",
    op_kind: GpuOpKind::Custom("fftshift"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("circshift")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to provider circshift kernels when available; otherwise gathers once and shifts on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::fftshift")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fftshift",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not currently fused; treated as an explicit data shuffling operation.",
};

const BUILTIN_NAME: &str = "fftshift";

fn fftshift_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "fftshift",
    category = "math/fft",
    summary = "Shift zero-frequency components to the center of a spectrum.",
    keywords = "fftshift,fourier transform,frequency centering,spectrum,gpu",
    accel = "custom",
    type_resolver(fftshift_type),
    builtin_path = "crate::builtins::math::fft::fftshift"
)]
async fn fftshift_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(fftshift_error("fftshift: too many input arguments"));
    }
    let dims_arg = rest.first();

    match value {
        Value::Tensor(tensor) => {
            let dims = compute_shift_dims(&tensor.shape, dims_arg, BUILTIN_NAME)?;
            Ok(fftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::ComplexTensor(ct) => {
            let dims = compute_shift_dims(&ct.shape, dims_arg, BUILTIN_NAME)?;
            Ok(fftshift_complex_tensor(ct, &dims).map(Value::ComplexTensor)?)
        }
        Value::LogicalArray(array) => {
            let dims = compute_shift_dims(&array.shape, dims_arg, BUILTIN_NAME)?;
            Ok(fftshift_logical(array, &dims).map(Value::LogicalArray)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| fftshift_error(format!("fftshift: {e}")))?;
            let dims = compute_shift_dims(&tensor.shape, dims_arg, BUILTIN_NAME)?;
            Ok(fftshift_complex_tensor(tensor, &dims).map(|result| {
                if result.data.len() == 1 {
                    let (r, i) = result.data[0];
                    Value::Complex(r, i)
                } else {
                    Value::ComplexTensor(result)
                }
            })?)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
                .map_err(|e| fftshift_error(e))?;
            let dims = compute_shift_dims(&tensor.shape, dims_arg, BUILTIN_NAME)?;
            Ok(fftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => {
            let dims = compute_shift_dims(&handle.shape, dims_arg, BUILTIN_NAME)?;
            Ok(fftshift_gpu(handle, &dims).await?)
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => Err(
            fftshift_error("fftshift: expected numeric or logical input"),
        ),
        Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(fftshift_error("fftshift: unsupported input type")),
    }
}

fn fftshift_tensor(tensor: Tensor, dims: &[usize]) -> BuiltinResult<Tensor> {
    let Tensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return Tensor::new(data, shape).map_err(|e| fftshift_error(format!("fftshift: {e}")));
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    Tensor::new(rotated, shape).map_err(|e| fftshift_error(format!("fftshift: {e}")))
}

fn fftshift_complex_tensor(tensor: ComplexTensor, dims: &[usize]) -> BuiltinResult<ComplexTensor> {
    let ComplexTensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return ComplexTensor::new(data, shape)
            .map_err(|e| fftshift_error(format!("fftshift: {e}")));
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    ComplexTensor::new(rotated, shape).map_err(|e| fftshift_error(format!("fftshift: {e}")))
}

fn fftshift_logical(array: LogicalArray, dims: &[usize]) -> BuiltinResult<LogicalArray> {
    let LogicalArray { data, shape } = array;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return LogicalArray::new(data, shape)
            .map_err(|e| fftshift_error(format!("fftshift: {e}")));
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    LogicalArray::new(rotated, shape).map_err(|e| fftshift_error(format!("fftshift: {e}")))
}

async fn fftshift_gpu(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let plan = build_shift_plan(&handle.shape, dims, ShiftKind::Fft);
    if plan.is_noop() {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut working = handle.clone();
        if plan.ext_shape != working.shape {
            match provider.reshape(&working, &plan.ext_shape) {
                Ok(reshaped) => working = reshaped,
                Err(_) => return fftshift_gpu_fallback(handle, dims).await,
            }
        }
        if let Ok(mut out) = provider.circshift(&working, &plan.provider) {
            if plan.ext_shape != handle.shape {
                match provider.reshape(&out, &handle.shape) {
                    Ok(restored) => out = restored,
                    Err(_) => {
                        let mut coerced = out.clone();
                        coerced.shape = handle.shape.clone();
                        out = coerced;
                    }
                }
            }
            return Ok(Value::GpuTensor(out));
        }
    }

    fftshift_gpu_fallback(handle, dims).await
}

async fn fftshift_gpu_fallback(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let host_tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let shifted = fftshift_tensor(host_tensor, dims)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &shifted.data,
            shape: &shifted.shape,
        };
        return provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| fftshift_error(format!("fftshift: {e}")));
    }
    Ok(tensor::tensor_into_value(shifted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        ComplexTensor, IntValue, LogicalArray, ResolveContext, Tensor, Type,
    };

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    #[test]
    fn fftshift_type_preserves_tensor_shape() {
        let out = fftshift_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(5)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(5)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_even_length_vector() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![8, 1]);
                assert_eq!(out.data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_odd_length_vector() {
        let tensor = Tensor::new((1..=5).map(|v| v as f64).collect(), vec![5, 1]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![5, 1]);
                assert_eq!(out.data, vec![4.0, 5.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_rows_only() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))])
            .expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_columns_only_via_vector_dims() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let dims = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![3.0, 6.0, 1.0, 4.0, 2.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_rows_only_logical_mask() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mask = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), vec![Value::LogicalArray(mask)])
            .expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_matrix_all_dims() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![6.0, 3.0, 4.0, 1.0, 5.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_with_empty_dimension_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let original = tensor.clone();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, original.shape);
                assert_eq!(out.data, original.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_dimension_beyond_rank_is_ignored() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
        let dims = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let original = tensor.clone();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, original.shape);
                assert_eq!(out.data, original.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_logical_array_input_supported() {
        let logical = LogicalArray::new(vec![1, 0, 0, 0], vec![4, 1]).unwrap();
        let result = fftshift_builtin(Value::LogicalArray(logical), Vec::new()).expect("fftshift");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(out.data, vec![0, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_complex_tensor() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            vec![4, 1],
        )
        .unwrap();
        let result = fftshift_builtin(Value::ComplexTensor(tensor), Vec::new()).unwrap();
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(
                    out.data,
                    vec![(2.0, 2.0), (3.0, 3.0), (0.0, 0.0), (1.0, 1.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_complex_scalar_passthrough() {
        let result = fftshift_builtin(Value::Complex(1.0, -2.0), Vec::new()).expect("fftshift");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_zero_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
                .unwrap_err(),
        );
        assert!(
            err.contains("dimension indices must be >= 1"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_non_integer_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Num(1.5)]).unwrap_err(),
        );
        assert!(
            err.contains("dimensions must be integers"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_non_numeric_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(
            fftshift_builtin(Value::Tensor(tensor), vec![Value::from("invalid")]).unwrap_err(),
        );
        assert!(
            err.contains("dimension indices must be numeric"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_rejects_non_vector_dimension_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = error_message(
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).unwrap_err(),
        );
        assert!(
            err.contains("dimension vectors must be row or column vectors"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("fftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![8, 1]);
            assert_eq!(gathered.data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fftshift_gpu_with_explicit_dims() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let dims = Value::Int(IntValue::I32(1));
            let result = fftshift_builtin(Value::GpuTensor(handle), vec![dims]).expect("fftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
            assert_eq!(gathered.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fftshift_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let cpu =
            fftshift_tensor(tensor.clone(), &(0..tensor.shape.len()).collect::<Vec<_>>()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = fftshift_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }

    fn fftshift_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::fftshift_builtin(value, rest))
    }
}
