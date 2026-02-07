//! MATLAB-compatible `isempty` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_numel;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::isempty")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isempty",
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
    notes: "Queries tensor metadata; gathers only when the provider fails to expose shapes.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::isempty"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isempty",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that returns a host logical scalar for fusion planning.",
};

#[runtime_builtin(
    name = "isempty",
    category = "array/introspection",
    summary = "Return true when an array has zero elements, matching MATLAB semantics.",
    keywords = "isempty,empty array,metadata query,gpu,logical",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    builtin_path = "crate::builtins::array::introspection::isempty"
)]
async fn isempty_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let is_empty = value_is_empty(&value).await?;
    Ok(Value::Bool(is_empty))
}

fn bool_scalar_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

async fn value_is_empty(value: &Value) -> crate::BuiltinResult<bool> {
    Ok(value_numel(value).await? == 0)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn isempty_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::isempty_builtin(value))
    }
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_builtins::{CellArray, CharArray, ResolveContext, Tensor, Type};

    #[test]
    fn isempty_type_returns_bool() {
        assert_eq!(
            super::bool_scalar_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isempty_empty_tensor_returns_true() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = isempty_builtin(Value::Tensor(tensor)).expect("isempty");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isempty_scalar_returns_false() {
        let result = isempty_builtin(Value::Num(5.0)).expect("isempty");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isempty_char_array_behaves_like_matlab() {
        let empty_chars = CharArray::new_row("");
        let non_empty_chars = CharArray::new_row("RunMat");
        let empty = isempty_builtin(Value::CharArray(empty_chars)).expect("isempty");
        let non_empty = isempty_builtin(Value::CharArray(non_empty_chars)).expect("isempty");
        assert_eq!(empty, Value::Bool(true));
        assert_eq!(non_empty, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isempty_cell_array_uses_dimensions() {
        let empty_cell = CellArray::new(Vec::new(), 0, 2).unwrap();
        let populated_cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let empty = isempty_builtin(Value::Cell(empty_cell)).expect("isempty");
        let populated = isempty_builtin(Value::Cell(populated_cell)).expect("isempty");
        assert_eq!(empty, Value::Bool(true));
        assert_eq!(populated, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isempty_string_scalar_is_false_even_if_empty_text() {
        let result = isempty_builtin(Value::String(String::new())).expect("isempty");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isempty_string_array_zero_rows_is_true() {
        let array = runmat_builtins::StringArray::new(Vec::new(), vec![0, 2]).unwrap();
        let result = isempty_builtin(Value::StringArray(array)).expect("isempty");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isempty_gpu_tensor_respects_shape() {
        test_support::with_test_provider(|provider| {
            let empty_tensor = Tensor::new(Vec::new(), vec![0, 4]).unwrap();
            let non_empty_tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();

            let empty_view = runmat_accelerate_api::HostTensorView {
                data: &empty_tensor.data,
                shape: &empty_tensor.shape,
            };
            let non_empty_view = runmat_accelerate_api::HostTensorView {
                data: &non_empty_tensor.data,
                shape: &non_empty_tensor.shape,
            };

            let empty_handle = provider.upload(&empty_view).expect("upload empty");
            let non_empty_handle = provider.upload(&non_empty_view).expect("upload non-empty");

            let empty_result =
                isempty_builtin(Value::GpuTensor(empty_handle)).expect("isempty empty");
            let non_empty_result =
                isempty_builtin(Value::GpuTensor(non_empty_handle)).expect("isempty non-empty");

            assert_eq!(empty_result, Value::Bool(true));
            assert_eq!(non_empty_result, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isempty_wgpu_provider_uses_handle_shape() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let empty_tensor = Tensor::new(Vec::new(), vec![0, 4]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &empty_tensor.data,
            shape: &empty_tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        assert_eq!(
            handle.shape,
            vec![0, 4],
            "provider should surface tensor shape"
        );

        let result = isempty_builtin(Value::GpuTensor(handle)).expect("isempty");
        assert_eq!(result, Value::Bool(true));
    }
}
