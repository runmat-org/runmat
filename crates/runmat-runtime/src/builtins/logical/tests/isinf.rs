//! MATLAB-compatible `isinf` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isinf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isinf",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_isinf",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dispatches to the provider `logical_isinf` hook when available; otherwise the runtime gathers to host and builds the logical mask on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isinf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isinf",
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
            Ok(format!("select({zero}, {one}, isInf({input}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fused kernels emit 0/1 masks; providers can override with native logical-isinf implementations.",
};

const BUILTIN_NAME: &str = "isinf";
const IDENTIFIER_INVALID_INPUT: &str = "MATLAB:isinf:InvalidInput";
const IDENTIFIER_INTERNAL: &str = "RunMat:isinf:InternalError";

#[runtime_builtin(
    name = "isinf",
    category = "logical/tests",
    summary = "Return a logical mask indicating which elements of the input are ±Inf.",
    keywords = "isinf,infinity,logical,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::tests::isinf"
)]
async fn isinf_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            if let Some(provider) = runmat_accelerate_api::provider() {
                if let Ok(mask) = provider.logical_isinf(&handle) {
                    return Ok(gpu_helpers::logical_gpu_value(mask));
                }
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| internal_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}")))?;
            isinf_tensor(BUILTIN_NAME, tensor)
        }
        other => isinf_host(other),
    }
}

fn isinf_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(x) => Ok(Value::Bool(x.is_infinite())),
        Value::Int(_) | Value::Bool(_) => Ok(Value::Bool(false)),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_infinite() || im.is_infinite())),
        Value::Tensor(tensor) => isinf_tensor(BUILTIN_NAME, tensor),
        Value::ComplexTensor(tensor) => isinf_complex_tensor(BUILTIN_NAME, tensor),
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

fn isinf_tensor(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&x| if x.is_infinite() { 1u8 } else { 0u8 })
        .collect::<Vec<_>>();
    logical_result(name, data, tensor.shape)
}

fn isinf_complex_tensor(name: &str, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| {
            if re.is_infinite() || im.is_infinite() {
                1u8
            } else {
                0u8
            }
        })
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
    use runmat_builtins::IntValue;

    fn run_isinf(value: Value) -> BuiltinResult<Value> {
        block_on(super::isinf_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_positive() {
        let result = run_isinf(Value::Num(f64::INFINITY)).expect("isinf");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_negative() {
        let result = run_isinf(Value::Num(f64::NEG_INFINITY)).expect("isinf");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_finite() {
        let result = run_isinf(Value::Num(42.0)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_nan_false() {
        let result = run_isinf(Value::Num(f64::NAN)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_bool_false() {
        let result = run_isinf(Value::Bool(true)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_int_false() {
        let result = run_isinf(Value::Int(IntValue::I32(7))).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_complex_scalar_detects_infinite_components() {
        let finite = run_isinf(Value::Complex(1.0, 2.0)).expect("isinf");
        assert_eq!(finite, Value::Bool(false));

        let inf_real = run_isinf(Value::Complex(f64::INFINITY, 0.0)).expect("isinf");
        assert_eq!(inf_real, Value::Bool(true));

        let inf_imag = run_isinf(Value::Complex(0.0, f64::NEG_INFINITY)).expect("isinf");
        assert_eq!(inf_imag, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_tensor_mask() {
        let tensor =
            Tensor::new(vec![1.0, f64::INFINITY, -f64::INFINITY, 0.0], vec![2, 2]).unwrap();
        let result = run_isinf(Value::Tensor(tensor)).expect("isinf");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![2, 2]);
                assert_eq!(mask.data, vec![0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_logical_array_returns_zeros() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = run_isinf(Value::LogicalArray(logical)).expect("isinf");
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
    fn isinf_complex_tensor_mask() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (f64::INFINITY, 1.0), (2.0, f64::NEG_INFINITY)],
            vec![3, 1],
        )
        .unwrap();
        let result = run_isinf(Value::ComplexTensor(tensor)).expect("isinf");
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
    fn isinf_string_scalar_false() {
        let result = run_isinf(Value::String("Inf".to_string())).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_string_array_returns_all_false() {
        let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        let result = run_isinf(Value::StringArray(strings)).expect("isinf");
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
    fn isinf_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = run_isinf(Value::Tensor(tensor)).expect("isinf");
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
    fn isinf_singleton_tensor_returns_scalar_bool() {
        let tensor = Tensor::new(vec![f64::INFINITY], vec![1, 1]).unwrap();
        let result = run_isinf(Value::Tensor(tensor)).expect("isinf");
        assert_eq!(result, Value::Bool(true));

        let finite = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = run_isinf(Value::Tensor(finite)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_rejects_unsupported_types() {
        let err = run_isinf(Value::FunctionHandle("foo".to_string()))
            .expect_err("isinf should reject function handles");
        assert!(
            err.message()
                .contains("expected numeric, logical, char, or string input"),
            "unexpected error message: {err:?}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_char_array_returns_zeros() {
        let array = CharArray::new("Inf".chars().collect(), 1, 3).unwrap();
        let result = run_isinf(Value::CharArray(array)).expect("isinf");
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
    fn isinf_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, f64::INFINITY, -f64::INFINITY], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = run_isinf(Value::GpuTensor(handle)).expect("isinf");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            assert_eq!(gathered.data, vec![0.0, 1.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isinf_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new(vec![1.0, f64::INFINITY, -f64::INFINITY, 0.0], vec![2, 2]).unwrap();
        let cpu = isinf_tensor("isinf", tensor.clone()).expect("cpu path");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = run_isinf(Value::GpuTensor(handle)).expect("gpu path");
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
