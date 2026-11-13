use super::super::r#false::false_builtin;
use super::super::r#true::true_builtin;
use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
use runmat_value::{IntegerStorage, SparseTensor, Tensor};

#[path = "contract_tests.rs"]
mod contract_tests;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn true_default_scalar() {
    let result = block_on(true_builtin(
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("true");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn false_default_scalar() {
    let result = block_on(false_builtin(
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("false");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn true_with_dims() {
    let args = vec![Value::Num(2.0), Value::Num(1.0)];
    let result = block_on(true_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("true");
    match result {
        Value::LogicalArray(logical) => {
            assert_eq!(logical.shape, vec![2, 1]);
            assert!(logical.data.iter().all(|&x| x == 1));
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn false_from_size_vector() {
    let size_vec = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
    let args = vec![Value::Tensor(size_vec)];
    let result = block_on(false_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("false");
    match result {
        Value::LogicalArray(logical) => {
            assert_eq!(logical.shape, vec![2, 3]);
            assert!(logical.data.iter().all(|&x| x == 0));
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[test]
fn false_reads_every_integer_size_class_exactly_and_clamps_negative() {
    let storages = [
        IntegerStorage::I8(vec![2]),
        IntegerStorage::I16(vec![2]),
        IntegerStorage::I32(vec![2]),
        IntegerStorage::I64(vec![2]),
        IntegerStorage::U8(vec![2]),
        IntegerStorage::U16(vec![2]),
        IntegerStorage::U32(vec![2]),
        IntegerStorage::U64(vec![2]),
    ];
    for storage in storages {
        let size = Tensor::new_integer(storage, vec![1, 1]).expect("size");
        let Value::LogicalArray(output) = block_on(false_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(size)]),
        ))
        .expect("false") else {
            panic!("expected logical array");
        };
        assert_eq!(output.shape, vec![2, 2]);
    }
    let Value::LogicalArray(empty) = block_on(false_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I64(
            -7,
        ))]),
    ))
    .expect("false") else {
        panic!("expected empty logical array");
    };
    assert_eq!(empty.shape, vec![0, 0]);
}

#[test]
fn true_reads_every_integer_size_class_exactly_and_clamps_negative() {
    let storages = [
        IntegerStorage::I8(vec![2]),
        IntegerStorage::I16(vec![2]),
        IntegerStorage::I32(vec![2]),
        IntegerStorage::I64(vec![2]),
        IntegerStorage::U8(vec![2]),
        IntegerStorage::U16(vec![2]),
        IntegerStorage::U32(vec![2]),
        IntegerStorage::U64(vec![2]),
    ];
    for storage in storages {
        let size = Tensor::new_integer(storage, vec![1, 1]).expect("size");
        let Value::LogicalArray(output) = block_on(true_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(size)]),
        ))
        .expect("true") else {
            panic!("expected logical array");
        };
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.data, vec![1; 4]);
    }
    let Value::LogicalArray(empty) = block_on(true_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I64(
            -7,
        ))]),
    ))
    .expect("true") else {
        panic!("expected empty logical array");
    };
    assert_eq!(empty.shape, vec![0, 0]);
}

#[test]
fn false_like_does_not_infer_shape_and_preserves_sparse_logical() {
    let prototype =
        Tensor::new_integer(IntegerStorage::U64(vec![9; 6]), vec![2, 3]).expect("prototype");
    assert_eq!(
        block_on(false_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("like"),
                Value::Tensor(prototype)
            ])
        ))
        .expect("false like"),
        Value::Bool(false)
    );

    let sparse = SparseTensor::zeros(4, 5);
    let output = block_on(false_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::from("like"),
            Value::SparseTensor(sparse),
        ]),
    ))
    .expect("sparse false");
    assert!(
        matches!(output, Value::SparseTensor(ref value) if value.shape() == vec![2, 3] && value.numeric_dtype().is_none())
    );
}

#[test]
fn true_like_does_not_infer_shape_and_preserves_sparse_logical() {
    let prototype =
        Tensor::new_integer(IntegerStorage::U64(vec![9; 6]), vec![2, 3]).expect("prototype");
    assert_eq!(
        block_on(true_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("like"),
                Value::Tensor(prototype)
            ])
        ))
        .expect("true like"),
        Value::Bool(true)
    );

    let sparse = SparseTensor::zeros(4, 5);
    let output = block_on(true_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::from("like"),
            Value::SparseTensor(sparse),
        ]),
    ))
    .expect("sparse true");
    let Value::SparseTensor(output) = output else {
        panic!("expected sparse logical output");
    };
    assert_eq!(output.shape(), vec![2, 3]);
    assert!(output.numeric_dtype().is_none());
    assert_eq!(output.nnz(), 6);
}

#[test]
fn false_strict_mode_gates_legacy_forms_before_evaluation() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let logical = block_on(false_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::from("logical"),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        logical.identifier(),
        Some("RunMat:compatibility:FalseLogicalOptionExtension")
    );
    let matrix = Tensor::new(vec![0.0; 4], vec![2, 2]).expect("matrix");
    let implicit = block_on(false_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(matrix)]),
    ))
    .unwrap_err();
    assert_eq!(
        implicit.identifier(),
        Some("RunMat:compatibility:FalseImplicitPrototypeExtension")
    );
    let single = Tensor::from_f32(vec![2.0], vec![1, 1]).expect("single size");
    let single = block_on(false_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(single)]),
    ))
    .unwrap_err();
    assert_eq!(
        single.identifier(),
        Some("RunMat:compatibility:FalseSingleSizeExtension")
    );
}

#[test]
fn true_strict_mode_gates_legacy_forms_before_evaluation() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let logical = block_on(true_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::from("logical"),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        logical.identifier(),
        Some("RunMat:compatibility:TrueLogicalOptionExtension")
    );
    let matrix = Tensor::new(vec![0.0; 4], vec![2, 2]).expect("matrix");
    let implicit = block_on(true_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(matrix)]),
    ))
    .unwrap_err();
    assert_eq!(
        implicit.identifier(),
        Some("RunMat:compatibility:TrueImplicitPrototypeExtension")
    );
    let single = Tensor::from_f32(vec![2.0], vec![1, 1]).expect("single size");
    let single = block_on(true_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(single)]),
    ))
    .unwrap_err();
    assert_eq!(
        single.identifier(),
        Some("RunMat:compatibility:TrueSingleSizeExtension")
    );
}

#[test]
fn false_resident_integer_size_is_structural_and_returns_host_logical() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let shape = [1usize, 1usize];
        let size = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U16(&[2]),
                shape: &shape,
            })
            .expect("resident size");
        let output = block_on(false_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(size)]),
        ))
        .expect("false");
        let Value::LogicalArray(output) = output else {
            panic!("expected host logical output");
        };
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.data, vec![0; 4]);
    });
}

#[test]
fn false_strict_mode_rejects_resident_size_before_provider_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let invalid = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered allocation")
            .handle,
    );
    let error = block_on(false_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![invalid]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FalseResidentSizeExtension")
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn logical_wgpu_like_integer_prototype_preserves_owner_and_logical_storage() {
    let _guard = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("WGPU provider");
    let provider = runmat_accelerate_api::provider().expect("registered WGPU provider");
    let size = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U16(&[2]),
            shape: &[1, 1],
        })
        .expect("resident integer size");
    let output = block_on(false_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(IntValue::U16(2)),
            Value::from("like"),
            Value::GpuTensor(size.clone()),
        ]),
    ))
    .expect("false");
    let Value::GpuTensor(handle) = output else {
        panic!("expected resident logical output");
    };
    assert_eq!(handle.shape(), vec![2, 2]);
    assert!(runmat_accelerate_api::handle_is_logical(&handle));
    assert!(runmat_accelerate_api::provider_for_handle(&handle)
        .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider)));
    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather false");
    assert_eq!(gathered.shape(), vec![2, 2]);
    assert_eq!(gathered.materialize_f64(), vec![0.0; 4]);

    let output = block_on(true_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(IntValue::U16(2)),
            Value::from("like"),
            Value::GpuTensor(size),
        ]),
    ))
    .expect("true");
    let Value::GpuTensor(handle) = output else {
        panic!("expected resident logical output");
    };
    assert_eq!(handle.shape(), vec![2, 2]);
    assert!(runmat_accelerate_api::handle_is_logical(&handle));
    assert!(runmat_accelerate_api::provider_for_handle(&handle)
        .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider)));
    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather true");
    assert_eq!(gathered.shape(), vec![2, 2]);
    assert_eq!(gathered.materialize_f64(), vec![1.0; 4]);
}
