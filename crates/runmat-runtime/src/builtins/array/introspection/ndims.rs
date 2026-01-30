//! MATLAB-compatible `ndims` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_ndims;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{Type, Value};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::ndims")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ndims",
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
    notes: "Metadata-only query; relies on tensor handle shapes and gathers only when provider metadata is unavailable.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::ndims"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ndims",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query; fusion planner bypasses this builtin and emits a host scalar.",
};

fn ndims_type(args: &[Type]) -> Type {
    if args.is_empty() {
        Type::Unknown
    } else {
        Type::Int
    }
}

#[runtime_builtin(
    name = "ndims",
    category = "array/introspection",
    summary = "Return the number of dimensions of scalars, vectors, matrices, and N-D arrays.",
    keywords = "ndims,number of dimensions,array rank,gpu metadata,MATLAB compatibility",
    accel = "metadata",
    type_resolver(ndims_type),
    builtin_path = "crate::builtins::array::introspection::ndims"
)]
async fn ndims_builtin(value: Value) -> crate::BuiltinResult<Value> {
    let rank = value_ndims(&value).await? as f64;
    Ok(Value::Num(rank))
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn ndims_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::ndims_builtin(value))
    }
    use runmat_builtins::{
        CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Type, Value,
    };

    #[test]
    fn ndims_type_returns_int() {
        assert_eq!(super::ndims_type(&[Type::Tensor { shape: None }]), Type::Int);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_scalar_returns_two() {
        let result = ndims_builtin(Value::Num(std::f64::consts::PI)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_row_vector_returns_two() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let result = ndims_builtin(Value::Tensor(tensor)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_three_dimensional_tensor_returns_three() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let result = ndims_builtin(Value::Tensor(tensor)).expect("ndims");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_trailing_singletons_preserved() {
        let tensor = Tensor::new(vec![0.0; 40], vec![5, 1, 1, 8]).unwrap();
        let result = ndims_builtin(Value::Tensor(tensor)).expect("ndims");
        assert_eq!(result, Value::Num(4.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_cell_array_returns_two() {
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
        let result = ndims_builtin(Value::Cell(cells)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_string_array_returns_two() {
        let sa = StringArray::new(vec!["a".into(), "bb".into(), "ccc".into()], vec![3, 1]).unwrap();
        let result = ndims_builtin(Value::StringArray(sa)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_char_array_returns_two() {
        let chars = CharArray::new_row("RunMat");
        let result = ndims_builtin(Value::CharArray(chars)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_complex_tensor_uses_shape() {
        let complex = ComplexTensor::new(vec![(0.0, 0.0); 18], vec![3, 3, 2]).unwrap();
        let result = ndims_builtin(Value::ComplexTensor(complex)).expect("ndims");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_logical_array_returns_two() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let result = ndims_builtin(Value::LogicalArray(logical)).expect("ndims");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_gpu_tensor_reads_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..48).map(|x| x as f64).collect(), vec![4, 3, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = ndims_builtin(Value::GpuTensor(handle)).expect("ndims");
            assert_eq!(result, Value::Num(3.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndims_gpu_tensor_without_metadata_defaults_correctly() {
        // Simulate a provider that does not populate shape metadata.
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let handle = runmat_accelerate_api::GpuTensorHandle {
                shape: vec![],
                device_id: handle.device_id,
                buffer_id: handle.buffer_id,
            };
            let result = ndims_builtin(Value::GpuTensor(handle)).expect("ndims");
            assert_eq!(result, Value::Num(2.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ndims_wgpu_tensor_reads_shape() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new((0..64).map(|x| x as f64).collect(), vec![4, 4, 4]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = ndims_builtin(Value::GpuTensor(handle)).expect("ndims");
        assert_eq!(result, Value::Num(3.0));
    }
}
