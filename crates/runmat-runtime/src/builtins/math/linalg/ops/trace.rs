//! MATLAB-compatible `trace` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::linalg::type_resolvers::numeric_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "trace";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::trace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("diag_extract"),
        ProviderHook::Reduction {
            name: "reduce_sum",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes:
        "Uses provider diagonal extraction followed by a sum reduction when available; otherwise gathers once, computes on the host, and uploads a 1×1 scalar back to the device.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::trace")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Trace is treated as a scalar reduction boundary; fusion wrappers stop at trace so producers/consumers can fuse independently.",
};

#[runtime_builtin(
    name = "trace",
    category = "math/linalg/ops",
    summary = "Sum the diagonal elements of matrices and matrix-like tensors.",
    keywords = "trace,matrix trace,diagonal sum,gpu",
    accel = "reduction",
    type_resolver(numeric_scalar_type),
    builtin_path = "crate::builtins::math::linalg::ops::trace"
)]
async fn trace_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => trace_gpu(handle).await,
        Value::ComplexTensor(ct) => trace_complex_tensor(ct),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        Value::CharArray(ca) => trace_char_array(ca),
        other => trace_numeric(other),
    }
}

fn trace_numeric(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(NAME, value).map_err(builtin_error)?;
    ensure_matrix_shape(NAME, &tensor.shape)?;
    let sum = trace_tensor_sum(&tensor);
    Ok(Value::Num(sum))
}

fn trace_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    ensure_matrix_shape(NAME, &ct.shape)?;
    let rows = if ct.rows == 0 {
        ct.shape.first().copied().unwrap_or(0)
    } else {
        ct.rows
    };
    let cols = if ct.cols == 0 {
        if ct.shape.len() >= 2 {
            ct.shape[1]
        } else if ct.shape.len() == 1 {
            1
        } else {
            rows
        }
    } else {
        ct.cols
    };
    let diag_len = rows.min(cols);
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for idx in 0..diag_len {
        let linear = idx + idx * rows;
        let (re, im) = ct.data[linear];
        sum_re += re;
        sum_im += im;
    }
    Ok(Value::Complex(sum_re, sum_im))
}

fn trace_char_array(ca: CharArray) -> BuiltinResult<Value> {
    ensure_matrix_shape(NAME, &[ca.rows, ca.cols])?;
    let diag_len = ca.rows.min(ca.cols);
    let mut sum = 0.0;
    for idx in 0..diag_len {
        let linear = idx * ca.cols + idx;
        if let Some(ch) = ca.data.get(linear) {
            sum += *ch as u32 as f64;
        }
    }
    Ok(Value::Num(sum))
}

async fn trace_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    ensure_matrix_shape(NAME, &handle.shape)?;
    let (rows, cols) = matrix_extents_from_shape(&handle.shape);
    let diag_len = rows.min(cols);

    if diag_len == 0 {
        return trace_gpu_fallback(&handle, 0.0);
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(diagonal) = provider.diag_extract(&handle, 0) {
            let reduced = provider.reduce_sum(&diagonal).await;
            let _ = provider.free(&diagonal);
            if let Ok(result) = reduced {
                return Ok(Value::GpuTensor(result));
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(map_control_flow)?;
    let sum = trace_tensor_sum(&tensor);
    trace_gpu_fallback(&handle, sum)
}

fn trace_gpu_fallback(_handle: &GpuTensorHandle, sum: f64) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let data = vec![sum];
        let shape = [1usize, 1usize];
        if let Ok(h) = provider.upload(&HostTensorView {
            data: &data,
            shape: &shape,
        }) {
            return Ok(Value::GpuTensor(h));
        }
    }
    // If no provider is registered, return a host scalar
    Ok(Value::Num(sum))
}

fn trace_tensor_sum(tensor: &Tensor) -> f64 {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let diag_len = rows.min(cols);
    let mut sum = 0.0;
    for idx in 0..diag_len {
        let linear = idx + idx * rows;
        sum += tensor.data[linear];
    }
    sum
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> BuiltinResult<()> {
    if shape.len() > 2 && shape.iter().skip(2).any(|&d| d != 1) {
        Err(builtin_error(format!("{name}: input must be 2-D")))
    } else {
        Ok(())
    }
}

fn matrix_extents_from_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => (shape[0], shape[1]),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::dispatcher::download_handle_async;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, Type};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_scalar_num() {
        let result = trace_builtin(Value::Num(7.0)).expect("trace");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn trace_type_returns_scalar() {
        let out = numeric_scalar_type(&[Type::Tensor {
            shape: Some(vec![Some(2), Some(2)]),
        }]);
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_rectangular_matrix() {
        let tensor = Tensor::new(vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0], vec![3, 2]).unwrap();
        let result = trace_builtin(Value::Tensor(tensor)).expect("trace");
        assert_eq!(result, Value::Num(10.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_vector_returns_first_element() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = trace_builtin(Value::Tensor(tensor)).expect("trace");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_empty_matrix_returns_zero() {
        let tensor = Tensor::new(Vec::new(), vec![0, 5]).unwrap();
        let result = trace_builtin(Value::Tensor(tensor)).expect("trace");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_complex_matrix() {
        let data = vec![(1.0, 2.0), (3.0, -4.0), (5.0, 6.0), (7.0, 8.0)];
        let ct = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = trace_builtin(Value::ComplexTensor(ct)).expect("trace");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 8.0).abs() < 1e-12);
                assert!((im - 10.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_char_array_promotes_to_double() {
        let chars = CharArray::new("ab".chars().collect(), 1, 2).unwrap();
        let result = trace_builtin(Value::CharArray(chars)).expect("trace");
        match result {
            Value::Num(value) => assert!((value - ('a' as u32 as f64)).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_char_array_square_matrix_uses_diagonal() {
        let chars = CharArray::new("abcd".chars().collect(), 2, 2).unwrap();
        let result = trace_builtin(Value::CharArray(chars)).expect("trace");
        match result {
            Value::Num(value) => {
                let expected = ('a' as u32 as f64) + ('d' as u32 as f64);
                assert!((value - expected).abs() < 1e-12);
            }
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = trace_builtin(Value::GpuTensor(handle)).expect("trace");
            match result {
                Value::GpuTensor(out) => {
                    let host = block_on(download_handle_async(provider, &out)).expect("download");
                    assert_eq!(host.shape, vec![1, 1]);
                    assert_eq!(host.data.len(), 1);
                    assert!((host.data[0] - 6.0).abs() < 1e-12);
                    let _ = provider.free(&out);
                }
                other => panic!("expected gpu result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_gpu_fallback_uploads_scalar() {
        // Force gather path by using a zero-length diagonal
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = trace_builtin(Value::GpuTensor(handle)).expect("trace");
            match result {
                Value::GpuTensor(out) => {
                    let host = block_on(download_handle_async(provider, &out)).expect("download");
                    assert_eq!(host.data, vec![0.0]);
                    let _ = provider.free(&out);
                }
                other => panic!("expected gpu result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_integer_promotes_to_double() {
        let value = Value::Int(IntValue::I32(5));
        let result = trace_builtin(value).expect("trace");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_bool_promotes_to_double() {
        let result = trace_builtin(Value::Bool(true)).expect("trace");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_logical_array_matches_numeric() {
        let data = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let logical = LogicalArray::new(data, vec![3, 3]).expect("logical");
        let result = trace_builtin(Value::LogicalArray(logical)).expect("trace");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_complex_empty_matrix_returns_zero() {
        let complex = ComplexTensor::new(Vec::new(), vec![0, 5]).expect("complex");
        let result = trace_builtin(Value::ComplexTensor(complex)).expect("trace");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 0.0);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected complex zero, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn trace_rejects_higher_dimensional_inputs() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = unwrap_error(trace_builtin(Value::Tensor(tensor)).unwrap_err());
        assert_eq!(err.message(), "trace: input must be 2-D");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn trace_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 8.0, 3.0, 6.0], vec![3, 2]).unwrap();
        let cpu = trace_numeric(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = trace_builtin(Value::GpuTensor(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let expected = match cpu {
            Value::Num(n) => n,
            Value::Tensor(t) if !t.data.is_empty() => t.data[0],
            Value::Tensor(_) => 0.0,
            other => panic!("unexpected cpu comparison value {other:?}"),
        };
        assert_eq!(gathered.shape, vec![1, 1]);
        let actual = gathered
            .data
            .first()
            .copied()
            .expect("gathered tensor should contain one element");
        assert!((expected - actual).abs() < 1e-9);
    }

    fn trace_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::trace_builtin(value))
    }
}
