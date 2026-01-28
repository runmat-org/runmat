//! MATLAB-compatible `isfinite` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isfinite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isfinite",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_isfinite",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dispatches to the provider `logical_isfinite` hook when available; otherwise the runtime gathers to host and computes the mask on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isfinite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isfinite",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0", "1.0"),
                ScalarType::F64 => ("f64(0.0)", "f64(1.0)"),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!("select({zero}, {one}, isFinite({input}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fused kernels emit 0/1 masks; providers can override with native logical-isfinite implementations.",
};

const BUILTIN_NAME: &str = "isfinite";
const IDENTIFIER_INVALID_INPUT: &str = "MATLAB:isfinite:InvalidInput";
const IDENTIFIER_INTERNAL: &str = "RunMat:isfinite:InternalError";

#[runtime_builtin(
    name = "isfinite",
    category = "logical/tests",
    summary = "Return a logical mask indicating which elements of the input are finite.",
    keywords = "isfinite,finite,logical,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::tests::isfinite"
)]
async fn isfinite_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            if let Some(provider) = runmat_accelerate_api::provider() {
                if let Ok(mask) = provider.logical_isfinite(&handle) {
                    return Ok(gpu_helpers::logical_gpu_value(mask));
                }
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| internal_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}")))?;
            isfinite_tensor(BUILTIN_NAME, tensor)
        }
        other => isfinite_host(other),
    }
}

fn isfinite_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(x) => Ok(Value::Bool(x.is_finite())),
        Value::Int(_) | Value::Bool(_) => Ok(Value::Bool(true)),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_finite() && im.is_finite())),
        Value::Tensor(tensor) => isfinite_tensor(BUILTIN_NAME, tensor),
        Value::ComplexTensor(tensor) => isfinite_complex_tensor(BUILTIN_NAME, tensor),
        Value::LogicalArray(array) => logical_full(BUILTIN_NAME, array.shape, true),
        Value::CharArray(array) => logical_full(BUILTIN_NAME, vec![array.rows, array.cols], true),
        Value::String(_) => Ok(Value::Bool(false)),
        Value::StringArray(array) => logical_full(BUILTIN_NAME, array.shape, false),
        _ => Err(build_runtime_error(format!(
            "{BUILTIN_NAME}: expected numeric, logical, char, or string input"
        ))
        .with_identifier(IDENTIFIER_INVALID_INPUT)
        .with_builtin(BUILTIN_NAME)
        .build()),
    }
}

fn isfinite_tensor(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&x| if x.is_finite() { 1u8 } else { 0u8 })
        .collect::<Vec<_>>();
    logical_result(name, data, tensor.shape)
}

fn isfinite_complex_tensor(name: &str, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| {
            if re.is_finite() && im.is_finite() {
                1u8
            } else {
                0u8
            }
        })
        .collect::<Vec<_>>();
    logical_result(name, data, tensor.shape)
}

fn logical_full(name: &str, shape: Vec<usize>, value: bool) -> BuiltinResult<Value> {
    let total = tensor::element_count(&shape);
    if total == 0 {
        return LogicalArray::new(Vec::new(), shape)
            .map(Value::LogicalArray)
            .map_err(|e| logical_array_error(name, e));
    }
    let fill = if value { 1u8 } else { 0u8 };
    let bits = vec![fill; total];
    logical_result(name, bits, shape)
}

fn logical_result(name: &str, bits: Vec<u8>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let total = tensor::element_count(&shape);
    if total != bits.len() {
        return Err(internal_error(
            name,
            format!(
                "{name}: internal error, mask length {} does not match shape {:?}",
                bits.len(),
                shape
            ),
        ));
    }
    if total == 1 {
        Ok(Value::Bool(bits[0] != 0))
    } else {
        LogicalArray::new(bits, shape)
            .map(Value::LogicalArray)
            .map_err(|e| logical_array_error(name, e))
    }
}

fn logical_array_error(name: &str, err: impl std::fmt::Display) -> RuntimeError {
    internal_error(name, format!("{name}: {err}"))
}

fn internal_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(IDENTIFIER_INTERNAL)
        .with_builtin(name)
        .build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntValue, StringArray};

    fn run_isfinite(value: Value) -> BuiltinResult<Value> {
        block_on(super::isfinite_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_scalar_true() {
        let result = run_isfinite(Value::Num(42.0)).expect("isfinite");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_scalar_false_for_nan() {
        let result = run_isfinite(Value::Num(f64::NAN)).expect("isfinite");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_scalar_false_for_inf() {
        let result = run_isfinite(Value::Num(f64::INFINITY)).expect("isfinite");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_int_and_bool_true() {
        let int_val = run_isfinite(Value::Int(IntValue::I32(7))).expect("isfinite");
        let bool_val = run_isfinite(Value::Bool(false)).expect("isfinite");
        assert_eq!(int_val, Value::Bool(true));
        assert_eq!(bool_val, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_complex_requires_both_components_finite() {
        let finite = run_isfinite(Value::Complex(1.0, -2.0)).expect("isfinite");
        assert_eq!(finite, Value::Bool(true));

        let inf_real = run_isfinite(Value::Complex(f64::INFINITY, 0.0)).expect("isfinite");
        assert_eq!(inf_real, Value::Bool(false));

        let nan_imag = run_isfinite(Value::Complex(0.0, f64::NAN)).expect("isfinite");
        assert_eq!(nan_imag, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_tensor_mask() {
        let tensor =
            Tensor::new(vec![1.0, f64::NAN, f64::INFINITY, -5.0], vec![2, 2]).expect("tensor");
        let result = run_isfinite(Value::Tensor(tensor)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![2, 2]);
                assert_eq!(mask.data, vec![1, 0, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_complex_tensor_mask() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (f64::NAN, 0.0), (1.0, f64::INFINITY)],
            vec![3, 1],
        )
        .unwrap();
        let result = run_isfinite(Value::ComplexTensor(tensor)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert_eq!(mask.data, vec![1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_logical_array_returns_ones() {
        let logical = LogicalArray::new(vec![0, 1, 0], vec![3, 1]).unwrap();
        let result = run_isfinite(Value::LogicalArray(logical)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert!(mask.data.iter().all(|&bit| bit == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_char_array_returns_ones() {
        let array = CharArray::new("Run".chars().collect(), 1, 3).unwrap();
        let result = run_isfinite(Value::CharArray(array)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 3]);
                assert_eq!(mask.data, vec![1, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_string_scalar_false() {
        let result = run_isfinite(Value::String("42".to_string())).expect("isfinite");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_string_array_returns_zeros() {
        let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        let result = run_isfinite(Value::StringArray(strings)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 2]);
                assert_eq!(mask.data, vec![0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = run_isfinite(Value::Tensor(tensor)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![0, 3]);
                assert!(mask.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_rejects_unsupported_types() {
        let err = run_isfinite(Value::FunctionHandle("foo".to_string()))
            .expect_err("isfinite should reject function handles");
        assert!(
            err.message()
                .contains("expected numeric, logical, char, or string input"),
            "unexpected error message: {err:?}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, f64::INFINITY, 3.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = run_isfinite(Value::GpuTensor(handle)).expect("isfinite");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isfinite_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new(vec![1.0, f64::NAN, f64::INFINITY, 5.0], vec![4, 1]).expect("tensor");
        let cpu = isfinite_tensor("isfinite", tensor.clone()).expect("cpu path");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = run_isfinite(Value::GpuTensor(handle)).expect("gpu path");
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::LogicalArray(expected), Tensor { data, shape, .. }) => {
                assert_eq!(shape, expected.shape);
                let expected_f64: Vec<f64> = expected
                    .data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect();
                assert_eq!(data, expected_f64);
            }
            (Value::Bool(flag), Tensor { data, .. }) => {
                assert_eq!(data, vec![if flag { 1.0 } else { 0.0 }]);
            }
            other => panic!("unexpected results {other:?}"),
        }
    }
}
