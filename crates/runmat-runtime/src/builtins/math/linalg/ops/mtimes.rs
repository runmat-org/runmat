//! MATLAB-compatible `mtimes` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{linalg, tensor};
use crate::builtins::math::linalg::type_resolvers::matmul_type;
use crate::{build_runtime_error, dispatcher::download_handle_async, BuiltinResult, RuntimeError};

const NAME: &str = "mtimes";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::mtimes")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mtimes",
    op_kind: GpuOpKind::MatMul,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Binary {
        name: "matmul",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Calls the provider `matmul` hook when available; otherwise gathers inputs and executes the CPU fallback.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::mtimes")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mtimes",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion currently delegates to provider matmul kernels or the CPU fallback.",
};

#[runtime_builtin(
    name = "mtimes",
    category = "math/linalg/ops",
    summary = "Matrix multiplication with MATLAB-compatible semantics.",
    keywords = "mtimes,matrix multiplication,linear algebra,gpu",
    accel = "matmul",
    type_resolver(matmul_type),
    builtin_path = "crate::builtins::math::linalg::ops::mtimes"
)]
async fn mtimes_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    mtimes_eval(&lhs, &rhs).await
}

pub(crate) async fn mtimes_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    if let Some(result) = try_gpu_matmul(lhs, rhs).await? {
        return Ok(result);
    }
    mtimes_cpu(lhs.clone(), rhs.clone()).await
}

async fn try_gpu_matmul(lhs: &Value, rhs: &Value) -> BuiltinResult<Option<Value>> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    if contains_complex(lhs) || contains_complex(rhs) {
        return Ok(None);
    }

    if !matches!(lhs, Value::GpuTensor(_)) && !matches!(rhs, Value::GpuTensor(_)) {
        return Ok(None);
    }

    if let Some(result) = try_gpu_scalar_mul(provider, lhs, rhs).await? {
        return Ok(Some(result));
    }

    let mut lhs_operand = match prepare_gpu_operand(lhs, provider)? {
        Some(op) => op,
        None => return Ok(None),
    };
    let mut rhs_operand = match prepare_gpu_operand(rhs, provider)? {
        Some(op) => op,
        None => {
            release_operand(provider, &mut lhs_operand);
            return Ok(None);
        }
    };

    match provider
        .matmul(lhs_operand.handle(), rhs_operand.handle())
        .await
    {
        Ok(handle) => {
            release_operand(provider, &mut lhs_operand);
            release_operand(provider, &mut rhs_operand);
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(_) => {
            release_operand(provider, &mut lhs_operand);
            release_operand(provider, &mut rhs_operand);
            Ok(None)
        }
    }
}

async fn try_gpu_scalar_mul(
    provider: &'static dyn AccelProvider,
    lhs: &Value,
    rhs: &Value,
) -> BuiltinResult<Option<Value>> {
    if let Some(scalar) = real_scalar_value(provider, lhs).await? {
        if let Some(mut operand) = prepare_gpu_operand(rhs, provider)? {
            let result = provider.scalar_mul(operand.handle(), scalar);
            release_operand(provider, &mut operand);
            return match result {
                Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
                Err(_) => Ok(None),
            };
        }
    }

    if let Some(scalar) = real_scalar_value(provider, rhs).await? {
        if let Some(mut operand) = prepare_gpu_operand(lhs, provider)? {
            let result = provider.scalar_mul(operand.handle(), scalar);
            release_operand(provider, &mut operand);
            return match result {
                Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
                Err(_) => Ok(None),
            };
        }
    }

    Ok(None)
}

async fn real_scalar_value(
    provider: &'static dyn AccelProvider,
    value: &Value,
) -> BuiltinResult<Option<f64>> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(i.to_f64())),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(t.data.first().copied()),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(Some(if logical.data[0] != 0 { 1.0 } else { 0.0 }))
        }
        Value::GpuTensor(handle) if is_scalar_handle(handle) => {
            let host = download_handle_async(provider, handle)
                .await
                .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
            Ok(host.data.first().copied())
        }
        _ => Ok(None),
    }
}

fn is_scalar_handle(handle: &GpuTensorHandle) -> bool {
    crate::builtins::common::shape::is_scalar_shape(&handle.shape)
}

#[async_recursion::async_recursion(?Send)]
async fn mtimes_cpu(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    use Value::*;

    let lhs = crate::dispatcher::gather_if_needed_async(&lhs)
        .await
        .map_err(map_control_flow)?;
    let rhs = crate::dispatcher::gather_if_needed_async(&rhs)
        .await
        .map_err(map_control_flow)?;

    match (lhs, rhs) {
        (LogicalArray(la), other) => {
            let tensor = tensor::logical_to_tensor(&la).map_err(builtin_error)?;
            mtimes_cpu(Value::Tensor(tensor), other).await
        }
        (other, LogicalArray(lb)) => {
            let tensor = tensor::logical_to_tensor(&lb).map_err(builtin_error)?;
            mtimes_cpu(other, Value::Tensor(tensor)).await
        }
        (Bool(b), other) => {
            let scalar = if b { 1.0 } else { 0.0 };
            mtimes_cpu(Value::Num(scalar), other).await
        }
        (other, Bool(b)) => {
            let scalar = if b { 1.0 } else { 0.0 };
            mtimes_cpu(other, Value::Num(scalar)).await
        }
        (Complex(ar, ai), Complex(br, bi)) => Ok(Complex(ar * br - ai * bi, ar * bi + ai * br)),
        (Complex(ar, ai), Num(s)) => Ok(Complex(ar * s, ai * s)),
        (Num(s), Complex(br, bi)) => Ok(Complex(s * br, s * bi)),
        (Tensor(ta), Complex(cr, ci)) => {
            let tensor = linalg::scalar_mul_complex(&ta, cr, ci);
            Ok(complex_tensor_into_value(tensor))
        }
        (Complex(cr, ci), Tensor(tb)) => {
            let tensor = linalg::scalar_mul_complex(&tb, cr, ci);
            Ok(complex_tensor_into_value(tensor))
        }
        (ComplexTensor(ct), Num(s)) => {
            let tensor = linalg::scalar_mul_complex_tensor(&ct, s, 0.0);
            Ok(complex_tensor_into_value(tensor))
        }
        (Num(s), ComplexTensor(ct)) => {
            let tensor = linalg::scalar_mul_complex_tensor(&ct, s, 0.0);
            Ok(complex_tensor_into_value(tensor))
        }
        (ComplexTensor(ct), Complex(cr, ci)) => {
            let tensor = linalg::scalar_mul_complex_tensor(&ct, cr, ci);
            Ok(complex_tensor_into_value(tensor))
        }
        (Complex(cr, ci), ComplexTensor(ct)) => {
            let tensor = linalg::scalar_mul_complex_tensor(&ct, cr, ci);
            Ok(complex_tensor_into_value(tensor))
        }
        (ComplexTensor(ta), ComplexTensor(tb)) => {
            let tensor = linalg::matmul_complex(&ta, &tb).map_err(builtin_error)?;
            Ok(complex_tensor_into_value(tensor))
        }
        (ComplexTensor(ta), Tensor(tb)) => {
            let tensor = linalg::matmul_complex_real(&ta, &tb).map_err(builtin_error)?;
            Ok(complex_tensor_into_value(tensor))
        }
        (Tensor(ta), ComplexTensor(tb)) => {
            let tensor = linalg::matmul_real_complex(&ta, &tb).map_err(builtin_error)?;
            Ok(complex_tensor_into_value(tensor))
        }
        (Tensor(ta), Tensor(tb)) => {
            let tensor = linalg::matmul_real(&ta, &tb).map_err(builtin_error)?;
            Ok(tensor::tensor_into_value(tensor))
        }
        (Tensor(ta), Num(s)) => Ok(tensor::tensor_into_value(linalg::scalar_mul_real(&ta, s))),
        (Num(s), Tensor(tb)) => Ok(tensor::tensor_into_value(linalg::scalar_mul_real(&tb, s))),
        (Tensor(ta), Int(i)) => Ok(tensor::tensor_into_value(linalg::scalar_mul_real(
            &ta,
            i.to_f64(),
        ))),
        (Int(i), Tensor(tb)) => Ok(tensor::tensor_into_value(linalg::scalar_mul_real(
            &tb,
            i.to_f64(),
        ))),
        (Num(x), Num(y)) => Ok(Num(x * y)),
        (Int(x), Num(y)) => Ok(Num(x.to_f64() * y)),
        (Num(x), Int(y)) => Ok(Num(x * y.to_f64())),
        (Int(x), Int(y)) => Ok(Num(x.to_f64() * y.to_f64())),
        _ => Err(builtin_error("mtimes: unsupported operand types")),
    }
}

fn prepare_gpu_operand(
    value: &Value,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Option<PreparedOperand>> {
    match value {
        Value::GpuTensor(handle) => {
            if is_scalar_handle(handle) {
                Ok(None)
            } else {
                Ok(Some(PreparedOperand::borrowed(handle)))
            }
        }
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                Ok(None)
            } else {
                let uploaded = upload_tensor(provider, t)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(None)
            } else {
                let tensor = tensor::logical_to_tensor(logical).map_err(builtin_error)?;
                let uploaded = upload_tensor(provider, &tensor)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        _ => Ok(None),
    }
}

fn upload_tensor(
    provider: &'static dyn AccelProvider,
    tensor: &Tensor,
) -> BuiltinResult<GpuTensorHandle> {
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let handle = provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("mtimes: {e}")))?;
    Ok(handle)
}

fn release_operand(provider: &'static dyn AccelProvider, operand: &mut PreparedOperand) {
    if operand.owned {
        let _ = provider.free(&operand.handle);
        operand.owned = false;
    }
}

fn contains_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

struct PreparedOperand {
    handle: GpuTensorHandle,
    owned: bool,
}

impl PreparedOperand {
    fn borrowed(handle: &GpuTensorHandle) -> Self {
        Self {
            handle: handle.clone(),
            owned: false,
        }
    }

    fn owned(handle: GpuTensorHandle) -> Self {
        Self {
            handle,
            owned: true,
        }
    }

    fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, ResolveContext, Tensor, Type};

    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matrix_product_matches_expected() {
        let a = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0], vec![3, 2]).unwrap();
        let result = mtimes_builtin(Value::Tensor(a), Value::Tensor(b)).expect("mtimes");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![58.0, 139.0, 64.0, 154.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn mtimes_type_infers_output_shape() {
        let out = matmul_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(1)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(1)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_matrix_product() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = mtimes_builtin(Value::Num(0.5), Value::Tensor(a)).expect("mtimes");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![0.5, 1.0, 1.5, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matrix_scalar_product() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = mtimes_builtin(Value::Tensor(a), Value::Num(3.0)).expect("mtimes");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![3.0, 6.0, 9.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot_product_returns_scalar() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let col = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
        let result = mtimes_builtin(Value::Tensor(row), Value::Tensor(col)).expect("mtimes");
        match result {
            Value::Num(value) => assert!((value - 32.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_matrix_product() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let matrix = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let result =
            mtimes_builtin(Value::LogicalArray(logical), Value::Tensor(matrix)).expect("mtimes");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![2.0, 3.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_tensor_product() {
        let ct = runmat_builtins::ComplexTensor::new(
            vec![(1.0, 2.0), (3.0, -4.0), (5.0, 6.0), (7.0, -8.0)],
            vec![2, 2],
        )
        .unwrap();
        let scalar = Value::Complex(1.0, -1.0);
        let result = mtimes_builtin(Value::ComplexTensor(ct.clone()), scalar).expect("mtimes");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, ct.shape);
                assert_eq!(
                    t.data,
                    vec![(3.0, 1.0), (-1.0, -7.0), (11.0, 1.0), (-1.0, -15.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inner_dimension_mismatch_errors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0], vec![3, 1]).unwrap();
        let err = unwrap_error(mtimes_builtin(Value::Tensor(a), Value::Tensor(b)).unwrap_err());
        assert!(
            err.message().contains("Inner matrix dimensions must agree"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mix_int_and_matrix() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            mtimes_builtin(Value::Int(IntValue::I32(2)), Value::Tensor(a)).expect("mtimes");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 4.0, 6.0, 8.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_scalar_matrix_product() {
        test_support::with_test_provider(|provider| {
            let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &matrix.data,
                shape: &matrix.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = mtimes_builtin(Value::Num(2.0), Value::GpuTensor(handle))
                .expect("gpu scalar matmul");
            let gathered = match result {
                Value::GpuTensor(out) => {
                    test_support::gather(Value::GpuTensor(out)).expect("gather")
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            };
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![2.0, 4.0, 6.0, 8.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mtimes_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
            let view_a = runmat_accelerate_api::HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let view_b = runmat_accelerate_api::HostTensorView {
                data: &b.data,
                shape: &b.shape,
            };
            let ha = provider.upload(&view_a).expect("upload A");
            let hb = provider.upload(&view_b).expect("upload B");
            let result =
                mtimes_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb)).expect("mtimes");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![26.0, 38.0, 30.0, 44.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn mtimes_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

        let cpu =
            mtimes_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("cpu mtimes");
        let expected = test_support::gather(cpu).expect("gather cpu");

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_a = runmat_accelerate_api::HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = runmat_accelerate_api::HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload A");
        let hb = provider.upload(&view_b).expect("upload B");

        let gpu = mtimes_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb)).expect("wgpu mtimes");
        let gathered = test_support::gather(gpu).expect("gather gpu");

        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    fn mtimes_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(super::mtimes_builtin(lhs, rhs))
    }
}
