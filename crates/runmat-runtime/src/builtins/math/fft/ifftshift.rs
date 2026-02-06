//! MATLAB-compatible `ifftshift` builtin with GPU-aware semantics for RunMat.
//!
//! `ifftshift` moves the zero-frequency component back to the origin, undoing
//! the reordering performed by `fftshift` and preparing spectra for inverse FFTs.

use super::common::{apply_shift, build_shift_plan, compute_shift_dims, ShiftKind};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::type_resolvers::ifftshift_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifftshift")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifftshift",
    op_kind: GpuOpKind::Custom("ifftshift"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("circshift")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Delegates to provider circshift kernels; falls back to host when the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifftshift")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifftshift",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Explicit data movement; not fused with surrounding elementwise graphs.",
};

const BUILTIN_NAME: &str = "ifftshift";

fn ifftshift_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "ifftshift",
    category = "math/fft",
    summary = "Undo fftshift by moving the zero-frequency component back to the origin.",
    keywords = "ifftshift,inverse fft shift,frequency alignment,gpu",
    accel = "custom",
    type_resolver(ifftshift_type),
    builtin_path = "crate::builtins::math::fft::ifftshift"
)]
async fn ifftshift_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(ifftshift_error("ifftshift: too many input arguments"));
    }
    let dims_arg = rest.first();

    match value {
        Value::Tensor(tensor) => {
            let dims = compute_shift_dims(&tensor.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::ComplexTensor(ct) => {
            let dims = compute_shift_dims(&ct.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_complex_tensor(ct, &dims).map(Value::ComplexTensor)?)
        }
        Value::LogicalArray(array) => {
            let dims = compute_shift_dims(&array.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_logical(array, &dims).map(Value::LogicalArray)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| ifftshift_error(format!("ifftshift: {e}")))?;
            let dims = compute_shift_dims(&tensor.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_complex_tensor(tensor, &dims).map(|result| {
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
                .map_err(|e| ifftshift_error(e))?;
            let dims = compute_shift_dims(&tensor.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => {
            let dims = compute_shift_dims(&handle.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_gpu(handle, &dims).await?)
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => Err(
            ifftshift_error("ifftshift: expected numeric or logical input"),
        ),
        Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(ifftshift_error("ifftshift: unsupported input type")),
    }
}

fn ifftshift_tensor(tensor: Tensor, dims: &[usize]) -> BuiltinResult<Tensor> {
    let Tensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Ifft);
    if data.is_empty() || plan.is_noop() {
        return Tensor::new(data, shape).map_err(|e| ifftshift_error(format!("ifftshift: {e}")));
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    Tensor::new(rotated, shape).map_err(|e| ifftshift_error(format!("ifftshift: {e}")))
}

fn ifftshift_complex_tensor(tensor: ComplexTensor, dims: &[usize]) -> BuiltinResult<ComplexTensor> {
    let ComplexTensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Ifft);
    if data.is_empty() || plan.is_noop() {
        return ComplexTensor::new(data, shape)
            .map_err(|e| ifftshift_error(format!("ifftshift: {e}")));
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    ComplexTensor::new(rotated, shape).map_err(|e| ifftshift_error(format!("ifftshift: {e}")))
}

fn ifftshift_logical(array: LogicalArray, dims: &[usize]) -> BuiltinResult<LogicalArray> {
    let LogicalArray { data, shape } = array;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Ifft);
    if data.is_empty() || plan.is_noop() {
        return LogicalArray::new(data, shape)
            .map_err(|e| ifftshift_error(format!("ifftshift: {e}")));
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    LogicalArray::new(rotated, shape).map_err(|e| ifftshift_error(format!("ifftshift: {e}")))
}

async fn ifftshift_gpu(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let plan = build_shift_plan(&handle.shape, dims, ShiftKind::Ifft);
    if plan.is_noop() {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut working = handle.clone();
        if plan.ext_shape != working.shape {
            match provider.reshape(&working, &plan.ext_shape) {
                Ok(reshaped) => working = reshaped,
                Err(_) => return ifftshift_gpu_fallback(handle, dims).await,
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

    ifftshift_gpu_fallback(handle, dims).await
}

async fn ifftshift_gpu_fallback(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let host_tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let shifted = ifftshift_tensor(host_tensor, dims)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &shifted.data,
            shape: &shifted.shape,
        };
        return provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| ifftshift_error(format!("ifftshift: {e}")));
    }
    Ok(tensor::tensor_into_value(shifted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::common::{apply_shift, build_shift_plan, ShiftKind};
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ComplexTensor, IntValue, LogicalArray, ResolveContext, Tensor, Type};

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    #[test]
    fn ifftshift_type_preserves_tensor_shape() {
        let out = ifftshift_type(
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
    fn ifftshift_even_length_vector() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("ifftshift");
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
    fn ifftshift_odd_length_vector() {
        let tensor = Tensor::new((1..=5).map(|v| v as f64).collect(), vec![5, 1]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![5, 1]);
                assert_eq!(out.data, vec![3.0, 4.0, 5.0, 1.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_inverts_fftshift() {
        let tensor = Tensor::new((0..7).map(|v| v as f64).collect(), vec![7, 1]).unwrap();
        let fft_plan = build_shift_plan(&tensor.shape, &[0], ShiftKind::Fft);
        let fft_data = apply_shift(
            "ifftshift",
            &tensor.data,
            &fft_plan.ext_shape,
            &fft_plan.positive,
        )
        .expect("apply_shift");
        let fft_tensor = Tensor::new(fft_data, tensor.shape.clone()).unwrap();

        let restored =
            ifftshift_builtin(Value::Tensor(fft_tensor), Vec::new()).expect("ifftshift restore");
        match restored {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![7, 1]);
                assert_eq!(out.data, tensor.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_dimension_subset() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let dims = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result =
            ifftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![2.0, 5.0, 3.0, 6.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_logical_mask_dimensions() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mask = LogicalArray::new(vec![0, 1], vec![2, 1]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), vec![Value::LogicalArray(mask)])
            .expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![2.0, 5.0, 3.0, 6.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_empty_dimension_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let empty_dims = Tensor::new(vec![], vec![0, 1]).unwrap();
        let result = ifftshift_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(empty_dims)],
        )
        .expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                assert_eq!(out.data, tensor.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_dimension_zero_error() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(
            ifftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
                .unwrap_err(),
        );
        assert!(
            err.contains("dimension indices must be >= 1"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_logical_array_supported() {
        let logical = LogicalArray::new(vec![1, 0, 0, 0], vec![4, 1]).unwrap();
        let result =
            ifftshift_builtin(Value::LogicalArray(logical), Vec::new()).expect("ifftshift");
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
    fn ifftshift_rejects_non_numeric_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(
            ifftshift_builtin(Value::Tensor(tensor), vec![Value::from("invalid")]).unwrap_err(),
        );
        assert!(
            err.contains("dimension indices must be numeric"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                ifftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("ifftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![8, 1]);
            assert_eq!(gathered.data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_complex_tensor() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            vec![4, 1],
        )
        .unwrap();
        let result = ifftshift_builtin(Value::ComplexTensor(tensor), Vec::new()).unwrap();
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
    fn ifftshift_matrix_rows_only_via_int() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))])
            .expect("ifftshift");
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
    #[cfg(feature = "wgpu")]
    fn ifftshift_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let cpu = ifftshift_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let handle = provider.upload(&view).expect("upload");

        let gpu_value =
            ifftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("ifftshift gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");

        match cpu {
            Value::Tensor(host) => {
                assert_eq!(gathered.shape, host.shape);
                assert_eq!(gathered.data, host.data);
            }
            other => panic!("expected tensor cpu result, got {other:?}"),
        }
    }

    fn ifftshift_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ifftshift_builtin(value, rest))
    }
}
