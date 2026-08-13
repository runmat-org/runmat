//! MATLAB-compatible `full` conversion for sparse matrix values.

use runmat_builtins::catalog::definitions::{
    FULL_ERROR_INTERNAL, FULL_ERROR_INVALID_INPUT, FULL_INTEGER_SPARSE_EXTENSION,
};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{SparseTensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "full";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::full")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("storage-conversion"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Converts host sparse matrices to dense host tensors; already-full GPU tensors remain resident.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::full")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Storage conversion executes outside numeric fusion.",
};

fn full_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    full_error_with_message(format!("{}: {detail}", error.message), error)
}

fn full_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_input(type_name: &str) -> RuntimeError {
    full_error_with_detail(
        &FULL_ERROR_INVALID_INPUT,
        format!("conversion to full storage is not defined for {type_name}"),
    )
}

#[runtime_builtin(
    name = "full",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::full"
)]
async fn full_builtin(value: Value) -> BuiltinResult<Value> {
    if matches!(&value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some()) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FULL_INTEGER_SPARSE_EXTENSION,
            NAME,
        )?;
    }
    match value {
        Value::SparseTensor(sparse) => full_from_sparse(sparse),
        passthrough @ (Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::Tensor(_)
        | Value::ComplexTensor(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::GpuTensor(_)) => Ok(passthrough),
        Value::String(_) | Value::StringArray(_) => Err(invalid_input("string")),
        Value::Symbolic(_) | Value::SymbolicArray(_) => Err(invalid_input("sym")),
        Value::Cell(_) => Err(invalid_input("cell")),
        Value::Struct(_) => Err(invalid_input("struct")),
        Value::ObjectArray(array) => Err(invalid_input(array.class_name())),
        Value::Object(obj) => Err(invalid_input(&obj.class_name)),
        Value::HandleObject(handle) => Err(invalid_input(&handle.class_name)),
        Value::Listener(_) => Err(invalid_input("event.listener")),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(invalid_input("function_handle")),
        Value::ClassRef(_) => Err(invalid_input("meta.class")),
        Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_) => Err(invalid_input("MException")),
        Value::OutputList(_) => Err(invalid_input("OutputList")),
    }
}

fn full_from_sparse(sparse: SparseTensor) -> BuiltinResult<Value> {
    if sparse.is_logical() {
        return sparse
            .to_dense_logical()
            .map(Value::LogicalArray)
            .map_err(|err| {
                full_error_with_detail(
                    &FULL_ERROR_INTERNAL,
                    format!("failed to densify sparse input: {err}"),
                )
            });
    }
    let tensor = sparse.to_dense().map_err(|err| {
        full_error_with_detail(
            &FULL_ERROR_INTERNAL,
            format!("failed to densify sparse input: {err}"),
        )
    })?;
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelProvider as _, HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView,
        HostTensorView,
    };
    use runmat_builtins::catalog::definitions::{FULL_EXTENSIONS, FULL_INTEGER_CAPABILITIES};
    use runmat_builtins::BuiltinIntegerInputAvailability;
    use runmat_value::{CellArray, IntegerStorage, SparseTensor, Tensor, Value};

    fn run_full(value: Value) -> BuiltinResult<Value> {
        block_on(super::full_builtin(value))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected dense tensor, got {other:?}"),
        }
    }

    fn every_integer_storage() -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(vec![-1, i8::MAX]),
            IntegerStorage::I16(vec![-1, i16::MAX]),
            IntegerStorage::I32(vec![-1, i32::MAX]),
            IntegerStorage::I64(vec![-1, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![0, u64::MAX]),
        ]
    }

    #[test]
    fn full_integer_capabilities_separate_dense_identity_from_sparse_extension() {
        assert_eq!(FULL_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(
            FULL_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert_eq!(
            FULL_INTEGER_CAPABILITIES[1].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
        assert_eq!(FULL_EXTENSIONS[0].id, "full-integer-sparse");
    }

    #[test]
    fn full_preserves_every_integer_class_for_already_full_inputs() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for storage in every_integer_storage() {
            let tensor = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("integer tensor");
            let output = expect_tensor(run_full(Value::Tensor(tensor)).expect("full identity"));
            assert_eq!(output.shape, vec![1, 2]);
            assert_eq!(output.integer_storage(), Some(&storage));
        }
    }

    #[test]
    fn full_preserves_every_integer_class_when_densifying_sparse_extension() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in every_integer_storage() {
            let sparse =
                SparseTensor::new_integer(2, 2, vec![0, 1, 2], vec![1, 0], storage.clone())
                    .expect("integer sparse");
            let output = expect_tensor(run_full(Value::SparseTensor(sparse)).expect("densify"));
            let mut expected = storage.zeros_like(4);
            expected
                .set_value(1, storage.value_at(0).expect("first"))
                .expect("first placement");
            expected
                .set_value(2, storage.value_at(1).expect("second"))
                .expect("second placement");
            assert_eq!(output.shape, vec![2, 2]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn full_integer_sparse_extension_is_independently_gated() {
        let sparse = || {
            SparseTensor::new_integer(
                1,
                1,
                vec![0, 1],
                vec![0],
                IntegerStorage::U64(vec![u64::MAX]),
            )
            .expect("integer sparse")
        };
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = run_full(Value::SparseTensor(sparse())).expect_err("strict rejection");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:FullIntegerSparseExtension")
            );
        }
        {
            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            let output = expect_tensor(
                run_full(Value::SparseTensor(sparse())).expect("extension admission"),
            );
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::U64(vec![u64::MAX]))
            );
        }
    }

    #[test]
    fn full_resident_identity_preserves_owning_provider_without_dispatch() {
        let _guard = test_support::accel_test_lock();
        let owner: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        let owner_dyn: &'static dyn runmat_accelerate_api::AccelProvider = owner;
        let ambient: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
            Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
        );
        unsafe {
            runmat_accelerate_api::register_provider(owner);
            runmat_accelerate_api::register_provider(ambient);
        }
        let _ambient = runmat_accelerate_api::ThreadProviderGuard::set(Some(ambient));
        let classes = [
            HostIntegerDataOwned::I8(vec![-1, i8::MAX]),
            HostIntegerDataOwned::I16(vec![-1, i16::MAX]),
            HostIntegerDataOwned::I32(vec![-1, i32::MAX]),
            HostIntegerDataOwned::I64(vec![-1, i64::MAX]),
            HostIntegerDataOwned::U8(vec![0, u8::MAX]),
            HostIntegerDataOwned::U16(vec![0, u16::MAX]),
            HostIntegerDataOwned::U32(vec![0, u32::MAX]),
            HostIntegerDataOwned::U64(vec![0, u64::MAX]),
        ];
        for class in &classes {
            let data = match class {
                HostIntegerDataOwned::I8(data) => HostIntegerDataView::I8(data),
                HostIntegerDataOwned::I16(data) => HostIntegerDataView::I16(data),
                HostIntegerDataOwned::I32(data) => HostIntegerDataView::I32(data),
                HostIntegerDataOwned::I64(data) => HostIntegerDataView::I64(data),
                HostIntegerDataOwned::U8(data) => HostIntegerDataView::U8(data),
                HostIntegerDataOwned::U16(data) => HostIntegerDataView::U16(data),
                HostIntegerDataOwned::U32(data) => HostIntegerDataView::U32(data),
                HostIntegerDataOwned::U64(data) => HostIntegerDataView::U64(data),
            };
            let input = owner
                .upload_integer(&HostIntegerTensorView {
                    data,
                    shape: &[1, 2],
                })
                .expect("owner upload");
            let output = run_full(Value::GpuTensor(input.clone())).expect("full identity");
            assert_eq!(output, Value::GpuTensor(input.clone()));
            assert!(runmat_accelerate_api::provider_for_handle(&input)
                .is_some_and(|provider| std::ptr::eq(provider, owner_dyn)));
            owner.free(&input).expect("owner cleanup");
        }
    }

    #[test]
    fn full_uses_canonical_catalog_contract_without_legacy_registration() {
        let entry = runmat_builtins::builtin_catalog_entry_by_name("full")
            .expect("canonical full catalog entry");
        assert_eq!(entry.contract.inference_rule.0, "array.full");
        assert!(runmat_builtins::builtin_function_by_name("full").is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sparse_triplet_matrix_densifies_column_major() {
        let sparse =
            SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![10.0, 30.0, 20.0]).unwrap();
        let dense = expect_tensor(run_full(Value::SparseTensor(sparse)).unwrap());
        assert_eq!(dense.shape, vec![3, 2]);
        assert_eq!(
            dense.materialize_f64(),
            vec![10.0, 0.0, 30.0, 0.0, 20.0, 0.0]
        );
    }

    #[test]
    fn full_preserves_exact_uint64_sparse_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 0],
            runmat_value::IntegerStorage::U64(vec![u64::MAX, 7]),
        )
        .expect("uint64 sparse");

        let dense = expect_tensor(run_full(Value::SparseTensor(sparse)).expect("full sparse"));
        assert_eq!(
            dense.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![0, u64::MAX, 7, 0]))
        );
    }

    #[test]
    fn full_preserves_logical_sparse_class_and_shape() {
        let sparse =
            SparseTensor::new_logical(3, 2, vec![0, 2, 3], vec![0, 2, 1]).expect("logical");
        let dense = run_full(Value::SparseTensor(sparse)).expect("full logical sparse");
        let Value::LogicalArray(dense) = dense else {
            panic!("expected logical array");
        };
        assert_eq!(dense.shape, vec![3, 2]);
        assert_eq!(dense.data, vec![1, 0, 1, 0, 1, 0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_sparse_matrix_densifies_to_zero_tensor() {
        let dense = expect_tensor(
            run_full(Value::SparseTensor(SparseTensor::zeros(2, 3))).expect("full sparse"),
        );
        assert_eq!(dense.shape, vec![2, 3]);
        assert_eq!(dense.materialize_f64(), vec![0.0; 6]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn already_full_numeric_values_pass_through() {
        let scalar = run_full(Value::Num(42.0)).unwrap();
        assert_eq!(scalar, Value::Num(42.0));

        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let result = run_full(Value::Tensor(tensor.clone())).unwrap();
        assert_eq!(result, Value::Tensor(tensor));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn already_full_gpu_tensor_remains_resident() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload");
            let result = run_full(Value::GpuTensor(handle.clone())).expect("full gpu");
            assert_eq!(result, Value::GpuTensor(handle.clone()));
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn sparse_dense_size_overflow_returns_internal_error() {
        let sparse = SparseTensor::zeros(usize::MAX, 2);
        let err = run_full(Value::SparseTensor(sparse)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:full:Internal"));
        assert!(err.message().contains("overflow"));
    }

    #[test]
    fn unsupported_values_raise_targeted_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = run_full(Value::Cell(cell)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:full:InvalidInput"));
        assert!(err.message().contains("cell"));
    }
}
