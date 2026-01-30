//! MATLAB-compatible `fix` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding::fix")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fix",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_fix" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_fix to keep fix on device; otherwise the runtime gathers to host and applies CPU truncation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding::fix")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fix",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            let truncated = format!("trunc({input})");
            Ok(format!("select({0}, {1}, {0} == {1})", truncated, zero))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL truncation; providers can substitute custom kernels when unary_fix is available.",
};

const BUILTIN_NAME: &str = "fix";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "fix",
    category = "math/rounding",
    summary = "Round scalars, vectors, matrices, or N-D tensors toward zero.",
    keywords = "fix,truncate,rounding,toward zero,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::rounding::fix"
)]
async fn fix_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => fix_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(fix_scalar(re), fix_scalar(im))),
        Value::ComplexTensor(ct) => fix_complex_tensor(ct),
        Value::CharArray(ca) => fix_char_array(ca),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|err| builtin_error(err))?;
            fix_tensor(tensor).map(tensor::tensor_into_value)
        }
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("fix: expected numeric or logical input"))
        }
        other => fix_numeric(other),
    }
}

async fn fix_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_fix(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    fix_tensor(tensor).map(tensor::tensor_into_value)
}

fn fix_numeric(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(n) => Ok(Value::Num(fix_scalar(n))),
        Value::Int(i) => Ok(Value::Num(fix_scalar(i.to_f64()))),
        Value::Bool(b) => Ok(Value::Num(fix_scalar(if b { 1.0 } else { 0.0 }))),
        Value::Tensor(t) => fix_tensor(t).map(tensor::tensor_into_value),
        other => {
            let tensor =
                tensor::value_into_tensor_for("fix", other).map_err(|err| builtin_error(err))?;
            Ok(fix_tensor(tensor).map(tensor::tensor_into_value)?)
        }
    }
}

fn fix_tensor(mut tensor: Tensor) -> BuiltinResult<Tensor> {
    for value in &mut tensor.data {
        *value = fix_scalar(*value);
    }
    Ok(tensor)
}

fn fix_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let data = ct
        .data
        .iter()
        .map(|&(re, im)| (fix_scalar(re), fix_scalar(im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(data, ct.shape.clone())
        .map_err(|e| builtin_error(format!("fix: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn fix_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| fix_scalar(ch as u32 as f64))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("fix: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn fix_scalar(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let truncated = value.trunc();
    if truncated == 0.0 {
        0.0
    } else {
        truncated
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_builtins::{ComplexTensor, IntValue, LogicalArray, Type};

    fn fix_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::fix_builtin(value))
    }

    fn assert_error_contains(error: RuntimeError, needle: &str) {
        assert!(
            error.message().contains(needle),
            "unexpected error: {}",
            error.message()
        );
    }

    #[test]
    fn fix_type_preserves_tensor_shape() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        }]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn fix_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(&[Type::Tensor {
            shape: Some(vec![Some(1), Some(1)]),
        }]);
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_scalar_positive_and_negative() {
        let input = Value::Tensor(
            Tensor::new(vec![-3.7, -2.4, -0.6, 0.0, 0.6, 2.4, 3.7], vec![7, 1]).unwrap(),
        );
        let result = fix_builtin(input).expect("fix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![-3.0, -2.0, 0.0, 0.0, 0.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_tensor_matrix() {
        let tensor = Tensor::new(vec![1.9, 4.1, -2.8, 0.5], vec![2, 2]).unwrap();
        let result = fix_builtin(Value::Tensor(tensor)).expect("fix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 4.0, -2.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_complex_number() {
        let result = fix_builtin(Value::Complex(1.9, -2.2)).expect("fix");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_char_array_returns_numeric_tensor() {
        let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
        let result = fix_builtin(Value::CharArray(chars)).expect("fix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![65.0, 66.0, 67.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_logical_array() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = fix_builtin(Value::LogicalArray(logical)).expect("fix");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 0.0, 1.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_bool_promotes_to_numeric() {
        let result = fix_builtin(Value::Bool(true)).expect("fix");
        match result {
            Value::Num(v) => assert_eq!(v, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_int_value_promotes() {
        let value = Value::Int(IntValue::I32(-42));
        let result = fix_builtin(value).expect("fix");
        match result {
            Value::Num(v) => assert_eq!(v, -42.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_string_errors() {
        let err = fix_builtin(Value::from("abc")).unwrap_err();
        assert_error_contains(err, "expected numeric");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_preserves_special_values_and_canonicalizes_negative_zero() {
        let tensor = Tensor::new(
            vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0],
            vec![4, 1],
        )
        .unwrap();
        let result = fix_builtin(Value::Tensor(tensor)).expect("fix");
        let Value::Tensor(out) = result else {
            panic!("expected tensor result");
        };
        assert!(out.data[0].is_nan(), "NaN should propagate");
        assert_eq!(out.data[1], f64::INFINITY);
        assert_eq!(out.data[2], f64::NEG_INFINITY);
        assert_eq!(out.data[3], 0.0);
        assert!(
            out.data[3].is_sign_positive(),
            "negative zero should canonicalize to +0"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_complex_tensor_rounds_components() {
        let tensor = ComplexTensor::new(vec![(1.9, -2.6), (-3.4, 0.2)], vec![2, 1]).unwrap();
        let result = fix_builtin(Value::ComplexTensor(tensor)).expect("fix");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.data, vec![(1.0, -2.0), (-3.0, 0.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.9, -0.1, 0.1, 2.6], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fix_builtin(Value::GpuTensor(handle)).expect("fix");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![-1.0, 0.0, 0.0, 2.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fix_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.7, -0.4, 0.4, 3.7], vec![4, 1]).unwrap();
        let cpu = fix_tensor(tensor.clone()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(fix_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }
}
