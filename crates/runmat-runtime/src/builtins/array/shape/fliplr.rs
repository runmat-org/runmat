//! MATLAB-compatible `fliplr` builtin with GPU-aware semantics for RunMat.

use crate::builtins::array::shape::flip::{
    complex_tensor_into_value, flip_char_array_with, flip_complex_tensor_with, flip_gpu_with,
    flip_logical_array_with, flip_string_array_with, flip_tensor_with,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{ComplexTensor, Value};
use runmat_macros::runtime_builtin;

const LR_DIM: [usize; 1] = [2];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::fliplr")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fliplr",
    op_kind: GpuOpKind::Custom("flip"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("flip")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to the generic flip hook with axis=1; falls back to host mirror when the hook is missing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::fliplr")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fliplr",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a data-reordering barrier; fusion planner preserves residency but does not fuse through fliplr.",
};

fn fliplr_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("fliplr").build()
}

#[runtime_builtin(
    name = "fliplr",
    category = "array/shape",
    summary = "Flip an array left-to-right along the second dimension.",
    keywords = "fliplr,flip,horizontal,matrix,gpu",
    accel = "custom",
    builtin_path = "crate::builtins::array::shape::fliplr"
)]
async fn fliplr_builtin(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            Ok(flip_tensor_with("fliplr", tensor, &LR_DIM).map(tensor::tensor_into_value)?)
        }
        Value::LogicalArray(array) => {
            Ok(flip_logical_array_with("fliplr", array, &LR_DIM).map(Value::LogicalArray)?)
        }
        Value::ComplexTensor(ct) => {
            Ok(flip_complex_tensor_with("fliplr", ct, &LR_DIM).map(Value::ComplexTensor)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| fliplr_error(format!("fliplr: {e}")))?;
            Ok(flip_complex_tensor_with("fliplr", tensor, &LR_DIM)
                .map(complex_tensor_into_value)?)
        }
        Value::StringArray(strings) => {
            Ok(flip_string_array_with("fliplr", strings, &LR_DIM).map(Value::StringArray)?)
        }
        Value::CharArray(chars) => {
            Ok(flip_char_array_with("fliplr", chars, &LR_DIM).map(Value::CharArray)?)
        }
        Value::String(scalar) => Ok(Value::String(scalar)),
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("fliplr", Value::Num(n))
                .map_err(|e| fliplr_error(e))?;
            Ok(flip_tensor_with("fliplr", tensor, &LR_DIM).map(tensor::tensor_into_value)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("fliplr", Value::Int(i))
                .map_err(|e| fliplr_error(e))?;
            Ok(flip_tensor_with("fliplr", tensor, &LR_DIM).map(tensor::tensor_into_value)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("fliplr", Value::Bool(flag))
                .map_err(|e| fliplr_error(e))?;
            Ok(flip_tensor_with("fliplr", tensor, &LR_DIM).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => Ok(flip_gpu_with("fliplr", handle, &LR_DIM).await?),
        Value::Cell(_) => Err(fliplr_error("fliplr: cell arrays are not yet supported")),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(fliplr_error("fliplr: unsupported input type")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn fliplr_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::fliplr_builtin(value))
    }
    use crate::builtins::array::shape::flip::{flip_logical_array, flip_tensor};
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, LogicalArray, StringArray, StructValue, Tensor, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_matrix_reverses_columns() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).expect("tensor");
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_row_vector_reverses_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_column_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let expected = tensor.clone();
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_nd_tensor_flips_second_dim_only() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_char_array() {
        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let result = fliplr_builtin(Value::CharArray(chars)).expect("fliplr");
        match result {
            Value::CharArray(out) => {
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "nurtam");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_string_array() {
        let strings =
            StringArray::new(vec!["left".into(), "right".into()], vec![1, 2]).expect("strings");
        let result = fliplr_builtin(Value::StringArray(strings)).expect("fliplr");
        match result {
            Value::StringArray(out) => assert_eq!(out.data, vec!["right", "left"]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_logical_array_preserves_bits() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let expected = flip_logical_array(logical.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::LogicalArray(logical)).expect("fliplr");
        match result {
            Value::LogicalArray(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_scalar_numeric_noop() {
        let result = fliplr_builtin(Value::Num(42.0)).expect("fliplr");
        match result {
            Value::Num(v) => assert_eq!(v, 42.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_string_scalar_noop() {
        let result = fliplr_builtin(Value::String("runmat".into())).expect("fliplr");
        match result {
            Value::String(s) => assert_eq!(s, "runmat"),
            other => panic!("expected string scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=12).map(|v| v as f64).collect(), vec![3, 4]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fliplr_builtin(Value::GpuTensor(handle)).expect("fliplr gpu");
            let gathered = test_support::gather(result).expect("gather");
            let expected = flip_tensor(tensor, &LR_DIM).expect("expected");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.data, expected.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fliplr_rejects_unsupported_type() {
        let value = Value::Struct(StructValue::new());
        let err = fliplr_builtin(value).expect_err("structs are unsupported");
        assert!(
            err.to_string().contains("unsupported input type"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fliplr_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).unwrap();
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");
        let value = fliplr_builtin(Value::GpuTensor(handle)).expect("fliplr gpu");
        let gathered = test_support::gather(value).expect("gather");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }
}
