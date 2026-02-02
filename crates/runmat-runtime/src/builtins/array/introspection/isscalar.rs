//! MATLAB-compatible `isscalar` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::{value_dimensions, value_numel};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::introspection::isscalar"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isscalar",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Inspects tensor metadata; downloads handles only when providers omit shapes.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::isscalar"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isscalar",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that returns a host logical scalar for fusion planning.",
};

#[runtime_builtin(
    name = "isscalar",
    category = "array/introspection",
    summary = "Return true when a value has exactly one element and unit dimensions.",
    keywords = "isscalar,scalar,metadata query,gpu,logical",
    accel = "metadata",
    builtin_path = "crate::builtins::array::introspection::isscalar"
)]
async fn isscalar_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(value_is_scalar(&value).await?))
}

async fn value_is_scalar(value: &Value) -> crate::BuiltinResult<bool> {
    if value_numel(value).await? != 1 {
        return Ok(false);
    }
    Ok(value_dimensions(value)
        .await?
        .into_iter()
        .all(|dim| dim == 1))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn isscalar_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::isscalar_builtin(value))
    }
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_builtins::{CellArray, CharArray, StructValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isscalar_numeric_scalar_returns_true() {
        let result = isscalar_builtin(Value::Num(5.0)).expect("isscalar");
        assert_eq!(result, Value::Bool(true));

        let bool_result = isscalar_builtin(Value::Bool(true)).expect("isscalar bool");
        assert_eq!(bool_result, Value::Bool(true));

        let complex_result = isscalar_builtin(Value::Complex(2.0, -3.0)).expect("isscalar complex");
        assert_eq!(complex_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isscalar_vector_returns_false() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = isscalar_builtin(Value::Tensor(tensor)).expect("isscalar");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isscalar_char_array_obeys_dimensions() {
        let single = CharArray::new(vec!['a'], 1, 1).unwrap();
        let row = CharArray::new_row("RunMat");
        let single_result = isscalar_builtin(Value::CharArray(single)).expect("isscalar single");
        let row_result = isscalar_builtin(Value::CharArray(row)).expect("isscalar row");
        assert_eq!(single_result, Value::Bool(true));
        assert_eq!(row_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isscalar_string_scalar_true_but_empty_array_false() {
        let scalar = runmat_builtins::StringArray::new(vec!["RunMat".into()], vec![1, 1]).unwrap();
        let empty = runmat_builtins::StringArray::new(Vec::new(), vec![0, 1]).unwrap();
        let scalar_result =
            isscalar_builtin(Value::StringArray(scalar)).expect("isscalar string scalar");
        let empty_result =
            isscalar_builtin(Value::StringArray(empty)).expect("isscalar string empty");
        let string_value_result =
            isscalar_builtin(Value::String("scalar".into())).expect("isscalar string value");
        assert_eq!(scalar_result, Value::Bool(true));
        assert_eq!(empty_result, Value::Bool(false));
        assert_eq!(string_value_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isscalar_cell_and_struct_follow_dimensions() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let not_scalar_cell = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let mut struct_scalar = StructValue::new();
        struct_scalar.fields.insert("value".into(), Value::Num(1.0));
        let scalar_cell = isscalar_builtin(Value::Cell(cell)).expect("isscalar cell");
        let nonscalar_cell =
            isscalar_builtin(Value::Cell(not_scalar_cell)).expect("isscalar non-scalar cell");
        let struct_result =
            isscalar_builtin(Value::Struct(struct_scalar)).expect("isscalar struct");
        assert_eq!(scalar_cell, Value::Bool(true));
        assert_eq!(nonscalar_cell, Value::Bool(false));
        assert_eq!(struct_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isscalar_gpu_tensor_checks_dimensions() {
        test_support::with_test_provider(|provider| {
            let scalar_tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let vector_tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let scalar_view = runmat_accelerate_api::HostTensorView {
                data: &scalar_tensor.data,
                shape: &scalar_tensor.shape,
            };
            let vector_view = runmat_accelerate_api::HostTensorView {
                data: &vector_tensor.data,
                shape: &vector_tensor.shape,
            };
            let scalar_handle = provider.upload(&scalar_view).expect("upload scalar");
            let vector_handle = provider.upload(&vector_view).expect("upload vector");
            let scalar_result =
                isscalar_builtin(Value::GpuTensor(scalar_handle)).expect("isscalar gpu scalar");
            let vector_result =
                isscalar_builtin(Value::GpuTensor(vector_handle)).expect("isscalar gpu vector");
            assert_eq!(scalar_result, Value::Bool(true));
            assert_eq!(vector_result, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isscalar_wgpu_provider_respects_metadata() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        assert_eq!(
            handle.shape,
            vec![1, 1],
            "provider should supply tensor shape"
        );
        let result = isscalar_builtin(Value::GpuTensor(handle)).expect("isscalar");
        assert_eq!(result, Value::Bool(true));
    }
}
