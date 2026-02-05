//! MATLAB-compatible `ismatrix` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_dimensions;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::introspection::ismatrix"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ismatrix",
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
    notes: "Consumes tensor shape metadata; falls back to gathering only when providers omit shape information.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::ismatrix"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ismatrix",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that always yields a host logical scalar; fusion treats it as a control predicate.",
};

#[runtime_builtin(
    name = "ismatrix",
    category = "array/introspection",
    summary = "Return true when an array has at most two dimensions (m-by-n, including vectors and scalars).",
    keywords = "ismatrix,matrix detection,metadata query,logical,gpu",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    builtin_path = "crate::builtins::array::introspection::ismatrix"
)]
async fn ismatrix_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(value_is_matrix(&value).await?))
}

fn bool_scalar_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

async fn value_is_matrix(value: &Value) -> crate::BuiltinResult<bool> {
    Ok(value_dimensions(value).await?.len() <= 2)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn ismatrix_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::ismatrix_builtin(value))
    }
    use runmat_builtins::{
        CellArray, CharArray, LogicalArray, ObjectInstance, ResolveContext, StringArray,
        StructValue, Tensor, Type,
    };

    #[test]
    fn ismatrix_type_returns_bool() {
        assert_eq!(
            super::bool_scalar_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_accepts_scalars_vectors_and_matrices() {
        let scalar = ismatrix_builtin(Value::Num(5.0)).expect("ismatrix scalar");
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let col = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let row_result = ismatrix_builtin(Value::Tensor(row)).expect("ismatrix row");
        let col_result = ismatrix_builtin(Value::Tensor(col)).expect("ismatrix col");
        let matrix_result = ismatrix_builtin(Value::Tensor(matrix)).expect("ismatrix matrix");
        assert_eq!(scalar, Value::Bool(true));
        assert_eq!(row_result, Value::Bool(true));
        assert_eq!(col_result, Value::Bool(true));
        assert_eq!(matrix_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_rejects_higher_rank_arrays() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let result = ismatrix_builtin(Value::Tensor(tensor)).expect("ismatrix");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_handles_empty_dimensions_like_matlab() {
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let row_empty = Tensor::new(Vec::new(), vec![1, 0]).unwrap();
        let col_empty = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let empty_3d = Tensor::new(Vec::new(), vec![0, 0, 3]).unwrap();
        let empty_result = ismatrix_builtin(Value::Tensor(empty)).expect("ismatrix []");
        let row_result = ismatrix_builtin(Value::Tensor(row_empty)).expect("ismatrix 1x0");
        let col_result = ismatrix_builtin(Value::Tensor(col_empty)).expect("ismatrix 0x1");
        let empty_3d_result = ismatrix_builtin(Value::Tensor(empty_3d)).expect("ismatrix 0x0x3");
        assert_eq!(empty_result, Value::Bool(true));
        assert_eq!(row_result, Value::Bool(true));
        assert_eq!(col_result, Value::Bool(true));
        assert_eq!(empty_3d_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_handles_scalar_like_runtime_values() {
        let bool_result = ismatrix_builtin(Value::Bool(true)).expect("ismatrix bool");
        let string_result =
            ismatrix_builtin(Value::String("runmat".into())).expect("ismatrix string");
        let func_result =
            ismatrix_builtin(Value::FunctionHandle("sin".into())).expect("ismatrix function");
        let object = Value::Object(ObjectInstance::new("TestClass".into()));
        let object_result = ismatrix_builtin(object).expect("ismatrix object");
        assert_eq!(bool_result, Value::Bool(true));
        assert_eq!(string_result, Value::Bool(true));
        assert_eq!(func_result, Value::Bool(true));
        assert_eq!(object_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_logical_arrays_respect_shape_rank() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).expect("logical array");
        let logical_result =
            ismatrix_builtin(Value::LogicalArray(logical)).expect("ismatrix logical");
        assert_eq!(logical_result, Value::Bool(true));

        let logical3d =
            LogicalArray::new(vec![0, 1, 0, 1], vec![1, 1, 4]).expect("logical 3d array");
        let logical3d_result =
            ismatrix_builtin(Value::LogicalArray(logical3d)).expect("ismatrix logical 3d");
        assert_eq!(logical3d_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_char_string_cell_and_struct_metadata() {
        let char_array = CharArray::new_row("RunMat");
        let string_array = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![1, 3])
            .expect("string array");
        let cell_array =
            CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).expect("cell array");
        let mut struct_value = StructValue::new();
        struct_value.fields.insert("field".into(), Value::Num(1.0));
        let struct_vector = CellArray::new(
            vec![
                Value::Struct(struct_value.clone()),
                Value::Struct(struct_value.clone()),
            ],
            1,
            2,
        )
        .expect("struct array handles");
        let char_result = ismatrix_builtin(Value::CharArray(char_array)).expect("ismatrix char");
        let string_result =
            ismatrix_builtin(Value::StringArray(string_array)).expect("ismatrix string");
        let cell_result = ismatrix_builtin(Value::Cell(cell_array)).expect("ismatrix cell");
        let struct_result =
            ismatrix_builtin(Value::Cell(struct_vector)).expect("ismatrix struct array");
        assert_eq!(char_result, Value::Bool(true));
        assert_eq!(string_result, Value::Bool(true));
        assert_eq!(cell_result, Value::Bool(true));
        assert_eq!(struct_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_rejects_struct_arrays_with_extra_dimensions() {
        let mut struct_value = StructValue::new();
        struct_value.fields.insert("field".into(), Value::Num(1.0));
        let handles = vec![
            Value::Struct(struct_value.clone()),
            Value::Struct(struct_value.clone()),
        ];
        let array = CellArray::new(handles, 1, 2).expect("cell array");
        let nested = Tensor::new(vec![0.0; 2], vec![1, 1, 2]).unwrap();
        let array_result = ismatrix_builtin(Value::Cell(array)).expect("ismatrix cell");
        let nested_result = ismatrix_builtin(Value::Tensor(nested)).expect("ismatrix nested");
        assert_eq!(array_result, Value::Bool(true));
        assert_eq!(nested_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_gpu_tensor_uses_handle_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
            let tensor3d = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
            let tensor_view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let tensor3d_view = runmat_accelerate_api::HostTensorView {
                data: &tensor3d.data,
                shape: &tensor3d.shape,
            };
            let handle = provider.upload(&tensor_view).expect("upload matrix");
            let handle3d = provider.upload(&tensor3d_view).expect("upload 3d");
            let result = ismatrix_builtin(Value::GpuTensor(handle)).expect("ismatrix gpu");
            let result3d = ismatrix_builtin(Value::GpuTensor(handle3d)).expect("ismatrix 3d gpu");
            assert_eq!(result, Value::Bool(true));
            assert_eq!(result3d, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_gpu_tensor_vector_shape_is_matrix() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload vector");
            assert_eq!(handle.shape, vec![3]);
            let result = ismatrix_builtin(Value::GpuTensor(handle)).expect("ismatrix gpu vector");
            assert_eq!(result, Value::Bool(true));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_gpu_handle_without_shape_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let handle = runmat_accelerate_api::GpuTensorHandle {
                shape: Vec::new(),
                device_id: handle.device_id,
                buffer_id: handle.buffer_id,
            };
            let result = ismatrix_builtin(Value::GpuTensor(handle)).expect("ismatrix gpu fallback");
            assert_eq!(result, Value::Bool(true));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_gpu_handle_invalid_buffer_errors() {
        test_support::with_test_provider(|provider| {
            let handle = runmat_accelerate_api::GpuTensorHandle {
                shape: Vec::new(),
                device_id: provider.device_id(),
                buffer_id: u64::MAX,
            };
            let err = ismatrix_builtin(Value::GpuTensor(handle)).unwrap_err();
            assert!(err.message.contains("gather"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn value_is_matrix_matches_dimensions_helper() {
        let tensor = Tensor::new(vec![0.0; 12], vec![3, 4]).unwrap();
        let is_matrix = futures::executor::block_on(value_is_matrix(&Value::Tensor(tensor)))
            .expect("value_is_matrix");
        assert!(is_matrix);
        let higher = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let is_matrix = futures::executor::block_on(value_is_matrix(&Value::Tensor(higher)))
            .expect("value_is_matrix");
        assert!(!is_matrix);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ismatrix_wgpu_provider_populates_shape() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        assert_eq!(handle.shape, vec![2, 2]);
        let result = ismatrix_builtin(Value::GpuTensor(handle)).expect("ismatrix");
        assert_eq!(result, Value::Bool(true));
    }
}
