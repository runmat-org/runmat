//! MATLAB-compatible `isvector` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_dimensions;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::introspection::isvector"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isvector",
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
    notes: "Reads tensor metadata; falls back to gathering when providers omit shape information.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::isvector"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isvector",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that always returns a host logical scalar for fusion planning.",
};

#[runtime_builtin(
    name = "isvector",
    category = "array/introspection",
    summary = "Return true when an array is 1-by-N or N-by-1 (including scalars).",
    keywords = "isvector,vector detection,metadata query,gpu,logical",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    builtin_path = "crate::builtins::array::introspection::isvector"
)]
async fn isvector_builtin(value: Value) -> crate::BuiltinResult<Value> {
    Ok(Value::Bool(value_is_vector(&value).await?))
}

fn bool_scalar_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

async fn value_is_vector(value: &Value) -> crate::BuiltinResult<bool> {
    let dims = value_dimensions(value).await?;
    if dims.len() > 2 {
        return Ok(false);
    }
    let mut non_singleton_dims = 0usize;

    for &dim in dims.iter() {
        if dim != 1 {
            non_singleton_dims += 1;
            if non_singleton_dims > 1 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn isvector_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::isvector_builtin(value))
    }
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_builtins::{CellArray, CharArray, ResolveContext, Tensor, Type};

    #[test]
    fn isvector_type_returns_bool() {
        assert_eq!(
            super::bool_scalar_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isvector_detects_row_and_column_vectors() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let col = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row_result = isvector_builtin(Value::Tensor(row)).expect("isvector row");
        let col_result = isvector_builtin(Value::Tensor(col)).expect("isvector col");
        assert_eq!(row_result, Value::Bool(true));
        assert_eq!(col_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isvector_rejects_matrices_and_higher_dimensions() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let cube = Tensor::new(vec![0.0; 4], vec![1, 1, 4]).unwrap();
        let matrix_result = isvector_builtin(Value::Tensor(matrix)).expect("isvector matrix");
        let cube_result = isvector_builtin(Value::Tensor(cube)).expect("isvector cube");
        assert_eq!(matrix_result, Value::Bool(false));
        assert_eq!(cube_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isvector_counts_scalars_and_empty_one_dimensional_arrays() {
        let scalar_result = isvector_builtin(Value::Num(5.0)).expect("isvector scalar");
        let empty_row = Tensor::new(Vec::new(), vec![1, 0]).unwrap();
        let empty_col = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let empty_wide = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let row_result = isvector_builtin(Value::Tensor(empty_row)).expect("isvector 1x0");
        let col_result = isvector_builtin(Value::Tensor(empty_col)).expect("isvector 0x1");
        let wide_result = isvector_builtin(Value::Tensor(empty_wide)).expect("isvector 0x3");
        assert_eq!(scalar_result, Value::Bool(true));
        assert_eq!(row_result, Value::Bool(true));
        assert_eq!(col_result, Value::Bool(true));
        assert_eq!(wide_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isvector_char_and_cell_arrays_follow_dimensions() {
        let char_row = CharArray::new_row("RunMat");
        let char_matrix = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let cell_vector = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let cell_matrix = CellArray::new(
            vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
            ],
            2,
            2,
        )
        .unwrap();
        let char_row_result = isvector_builtin(Value::CharArray(char_row)).expect("isvector char");
        let char_matrix_result =
            isvector_builtin(Value::CharArray(char_matrix)).expect("isvector char matrix");
        let cell_vector_result = isvector_builtin(Value::Cell(cell_vector)).expect("isvector cell");
        let cell_matrix_result =
            isvector_builtin(Value::Cell(cell_matrix)).expect("isvector cell matrix");
        assert_eq!(char_row_result, Value::Bool(true));
        assert_eq!(char_matrix_result, Value::Bool(false));
        assert_eq!(cell_vector_result, Value::Bool(true));
        assert_eq!(cell_matrix_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isvector_trailing_singleton_dimensions_are_rejected() {
        let scalar_with_extra = Tensor::new(vec![5.0], vec![1, 1, 1]).unwrap();
        let result =
            isvector_builtin(Value::Tensor(scalar_with_extra)).expect("isvector trailing ones");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isvector_gpu_tensor_uses_handle_shape() {
        test_support::with_test_provider(|provider| {
            let vector = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let vector_view = runmat_accelerate_api::HostTensorView {
                data: &vector.data,
                shape: &vector.shape,
            };
            let matrix_view = runmat_accelerate_api::HostTensorView {
                data: &matrix.data,
                shape: &matrix.shape,
            };
            let vector_handle = provider.upload(&vector_view).expect("upload vector");
            let matrix_handle = provider.upload(&matrix_view).expect("upload matrix");
            let vector_result =
                isvector_builtin(Value::GpuTensor(vector_handle)).expect("isvector gpu vector");
            let matrix_result =
                isvector_builtin(Value::GpuTensor(matrix_handle)).expect("isvector gpu matrix");
            assert_eq!(vector_result, Value::Bool(true));
            assert_eq!(matrix_result, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isvector_wgpu_provider_populates_shape() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        assert_eq!(
            handle.shape,
            vec![3, 1],
            "provider should supply tensor shape metadata"
        );
        let result = isvector_builtin(Value::GpuTensor(handle)).expect("isvector");
        assert_eq!(result, Value::Bool(true));
    }
}
