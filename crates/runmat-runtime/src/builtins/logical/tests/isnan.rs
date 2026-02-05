//! MATLAB-compatible `isnan` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::logical::type_resolvers::logical_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isnan")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isnan",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_isnan",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dispatches to the provider `logical_isnan` hook when available; otherwise the runtime gathers to host and builds the logical mask on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isnan")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isnan",
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
            Ok(format!(
                "select({zero}, {one}, isNan({input}))"
            ))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fused kernels emit 0/1 masks; providers can override with native logical-isnan implementations.",
};

const BUILTIN_NAME: &str = "isnan";
const IDENTIFIER_INVALID_INPUT: &str = "MATLAB:isnan:InvalidInput";
const IDENTIFIER_INTERNAL: &str = "RunMat:isnan:InternalError";

#[runtime_builtin(
    name = "isnan",
    category = "logical/tests",
    summary = "Return a logical mask indicating which elements of the input are NaN.",
    keywords = "isnan,nan,logical,gpu",
    accel = "elementwise",
    type_resolver(logical_unary_type),
    builtin_path = "crate::builtins::logical::tests::isnan"
)]
async fn isnan_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| internal_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}")))?;
            isnan_tensor(BUILTIN_NAME, tensor)
        }
        other => isnan_host(other),
    }
}

fn isnan_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(x) => Ok(Value::Bool(x.is_nan())),
        Value::Int(_) | Value::Bool(_) => Ok(Value::Bool(false)),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_nan() || im.is_nan())),
        Value::Tensor(tensor) => isnan_tensor(BUILTIN_NAME, tensor),
        Value::ComplexTensor(tensor) => isnan_complex_tensor(BUILTIN_NAME, tensor),
        Value::LogicalArray(array) => {
            let LogicalArray { shape, .. } = array;
            logical_zeros(BUILTIN_NAME, shape)
        }
        Value::CharArray(array) => {
            let CharArray { rows, cols, .. } = array;
            logical_zeros(BUILTIN_NAME, vec![rows, cols])
        }
        Value::String(_) => Ok(Value::Bool(false)),
        Value::StringArray(array) => {
            let StringArray { shape, .. } = array;
            logical_zeros(BUILTIN_NAME, shape)
        }
        _ => Err(build_runtime_error(format!(
            "{BUILTIN_NAME}: expected numeric, logical, char, or string input"
        ))
        .with_identifier(IDENTIFIER_INVALID_INPUT)
        .with_builtin(BUILTIN_NAME)
        .build()),
    }
}

fn isnan_tensor(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&x| if x.is_nan() { 1u8 } else { 0u8 })
        .collect::<Vec<_>>();
    logical_result(name, data, tensor.shape)
}

fn isnan_complex_tensor(name: &str, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| if re.is_nan() || im.is_nan() { 1u8 } else { 0u8 })
        .collect::<Vec<_>>();
    logical_result(name, data, tensor.shape)
}

fn logical_zeros(name: &str, shape: Vec<usize>) -> BuiltinResult<Value> {
    let total = tensor::element_count(&shape);
    if total == 0 {
        return LogicalArray::new(Vec::new(), shape)
            .map(Value::LogicalArray)
            .map_err(|e| logical_array_error(name, e));
    }
    let data = vec![0u8; total];
    logical_result(name, data, shape)
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
    use runmat_builtins::{ResolveContext, Type};

    #[test]
    fn isnan_type_returns_logical() {
        let out = logical_unary_type(&[Type::Tensor { shape: None }], &ResolveContext::new(Vec::new()));
        assert_eq!(out, Type::logical());
    }

    fn run_isnan(value: Value) -> BuiltinResult<Value> {
        block_on(super::isnan_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_scalar_nan() {
        let result = run_isnan(Value::Num(f64::NAN)).expect("isnan");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_scalar_finite() {
        let result = run_isnan(Value::Num(5.0)).expect("isnan");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_scalar_bool_false() {
        let result = run_isnan(Value::Bool(true)).expect("isnan");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_tensor_mask() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0, f64::NAN], vec![2, 2]).unwrap();
        let result = run_isnan(Value::Tensor(tensor)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![2, 2]);
                assert_eq!(mask.data, vec![0, 1, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_logical_array_returns_zeros() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = run_isnan(Value::LogicalArray(logical)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert!(mask.data.iter().all(|&bit| bit == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_complex_tensor_mask() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (f64::NAN, 0.0), (0.0, f64::NAN)],
            vec![3, 1],
        )
        .unwrap();
        let result = run_isnan(Value::ComplexTensor(tensor)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert_eq!(mask.data, vec![0, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_string_scalar_false() {
        let result = run_isnan(Value::String("NaN".to_string())).expect("isnan");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_string_array_returns_all_false() {
        let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        let result = run_isnan(Value::StringArray(strings)).expect("isnan");
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
    fn isnan_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = run_isnan(Value::Tensor(tensor)).expect("isnan");
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
    fn isnan_rejects_unsupported_types() {
        let err = run_isnan(Value::FunctionHandle("foo".to_string()))
            .expect_err("isnan should reject function handles");
        assert!(
            err.message()
                .contains("expected numeric, logical, char, or string input"),
            "unexpected error message: {err:?}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_char_array_returns_zeros() {
        let array = CharArray::new("NaN".chars().collect(), 1, 3).unwrap();
        let result = run_isnan(Value::CharArray(array)).expect("isnan");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 3]);
                assert_eq!(mask.data, vec![0, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isnan_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, f64::NAN, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = run_isnan(Value::GpuTensor(handle)).expect("isnan");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            assert_eq!(gathered.data, vec![0.0, 1.0, 0.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isnan_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, f64::NAN, 0.0], vec![3, 1]).unwrap();
        let cpu = isnan_tensor("isnan", tensor.clone()).expect("cpu path");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = run_isnan(Value::GpuTensor(handle)).expect("gpu path");
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
