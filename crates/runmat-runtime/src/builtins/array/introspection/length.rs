//! MATLAB-compatible `length` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_dimensions;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::containers::map::map_length;
use crate::runtime_error::RuntimeError;
use runmat_builtins::{Type, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::length")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "length",
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
    notes: "Reads tensor metadata from handles; falls back to gathering only when provider metadata is absent.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::length"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "length",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query; fusion planner treats this as a host scalar lookup.",
};

fn length_type(args: &[Type]) -> Type {
    if args.is_empty() {
        Type::Unknown
    } else {
        Type::Int
    }
}

#[runtime_builtin(
    name = "length",
    category = "array/introspection",
    summary = "Return the length of the largest dimension of scalars, vectors, matrices, and N-D arrays.",
    keywords = "length,largest dimension,vector length,gpu metadata,array size",
    accel = "metadata",
    type_resolver(length_type),
    builtin_path = "crate::builtins::array::introspection::length"
)]
async fn length_builtin(value: Value) -> crate::BuiltinResult<Value> {
    if let Some(count) = map_length(&value) {
        return Ok(Value::Num(count as f64));
    }
    let len = max_dimension(&value).await? as f64;
    Ok(Value::Num(len))
}

async fn max_dimension(value: &Value) -> Result<usize, RuntimeError> {
    let dims = value_dimensions(value).await?;
    Ok(dims.into_iter().max().unwrap_or(0))
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn length_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::length_builtin(value))
    }
    use runmat_builtins::{
        CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Type, Value,
    };

    #[test]
    fn length_type_returns_int() {
        assert_eq!(super::length_type(&[Type::Tensor { shape: None }]), Type::Int);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_scalar_is_one() {
        let result = length_builtin(Value::Num(5.0)).expect("length");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_column_vector_uses_rows() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_matrix_returns_larger_dimension() {
        let tensor = Tensor::new(vec![0.0; 10], vec![2, 5]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_high_rank_tensor_reports_global_max() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_partial_empty_tensor_returns_max_dimension() {
        let tensor = Tensor::new(vec![], vec![0, 0, 5]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_empty_matrix_with_nonzero_dimension() {
        let tensor = Tensor::new(vec![], vec![0, 7]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(7.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_fully_empty_matrix_returns_zero() {
        let tensor = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_character_array_uses_shape() {
        let chars = CharArray::new_row("RunMat");
        let result = length_builtin(Value::CharArray(chars)).expect("length");
        assert_eq!(result, Value::Num(6.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_complex_tensor_uses_shape() {
        let complex = ComplexTensor::new(vec![(0.0, 0.0); 12], vec![3, 4]).unwrap();
        let result = length_builtin(Value::ComplexTensor(complex)).expect("length");
        assert_eq!(result, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_cell_array_respects_dimensions() {
        let cells = CellArray::new(
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
        let result = length_builtin(Value::Cell(cells)).expect("length");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_string_array_defaults_to_shape() {
        let sa = StringArray::new(vec!["a".into(), "bb".into()], vec![2, 1]).unwrap();
        let result = length_builtin(Value::StringArray(sa)).expect("length");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_logical_array_uses_shape() {
        let la = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = length_builtin(Value::LogicalArray(la)).expect("length");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn length_gpu_tensor_reads_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..12).map(|x| x as f64).collect(), vec![3, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = length_builtin(Value::GpuTensor(handle)).expect("length");
            assert_eq!(result, Value::Num(4.0));
        });
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn length_wgpu_tensor_uses_handle_shape() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new((0..24).map(|v| v as f64).collect(), vec![6, 4]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = length_builtin(Value::GpuTensor(handle)).expect("length");
        assert_eq!(result, Value::Num(6.0));
    }
}
