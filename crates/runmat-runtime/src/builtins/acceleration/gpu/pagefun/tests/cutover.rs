use super::*;

#[test]
fn nonscalar_single_matrix_pages_keep_single_after_host_product() {
    let left = Value::Tensor(Tensor::from_f32(vec![1., 3., 2., 4.], vec![2, 2]).unwrap());
    let right = Value::Tensor(Tensor::from_f32(vec![1., 0., 0., 1.], vec![2, 2]).unwrap());
    let result = block_on(pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        left,
        crate::builtins::common::test_support::builtin_arguments(vec![right]),
    ))
    .unwrap();
    let Value::Tensor(result) = result else {
        panic!("real matrix result");
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F32);
    assert_eq!(result.materialize_f64(), vec![1., 3., 2., 4.]);
}

#[test]
fn empty_and_nonempty_integer_pages_share_class_admission() {
    for pages in [0, 2] {
        let integer = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![9_007_199_254_740_993; pages]),
                vec![1, 1, pages],
            )
            .unwrap(),
        );
        let single = Value::Tensor(Tensor::from_f32(vec![1.; pages], vec![1, 1, pages]).unwrap());
        let other_integer = Value::Tensor(
            Tensor::new_integer(IntegerStorage::I64(vec![1; pages]), vec![1, 1, pages]).unwrap(),
        );
        for other in [single, other_integer, Value::Bool(true)] {
            let error = block_on(pagefun_impl(
                Value::FunctionHandle("mtimes".into()),
                integer.clone(),
                crate::builtins::common::test_support::builtin_arguments(vec![other]),
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), PAGEFUN_ERROR_INVALID_INPUT.identifier);
        }
        let double = Value::Tensor(Tensor::new(vec![1.; pages], vec![1, 1, pages]).unwrap());
        let result = block_on(pagefun_impl(
            Value::FunctionHandle("mtimes".into()),
            integer,
            crate::builtins::common::test_support::builtin_arguments(vec![double]),
        ))
        .unwrap();
        let Value::Tensor(result) = result else {
            panic!("integer output");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993; pages]))
        );
        assert_eq!(result.shape(), &[1, 1, pages]);
    }
}

#[test]
fn catalog_binding_replaces_metadata_and_type_resolver_authorities() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name("pagefun").unwrap();
    assert!(runmat_builtins::builtin_function_by_name("pagefun").is_none());
    let bindings = crate::builtin::runtime_builtin_bindings_by_name("pagefun");
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        entry.binding_identity(&entry.bindings[0])
    );
    assert_eq!(entry.extensions.len(), 2);
}

#[test]
fn illegal_outputs_and_nested_inputs_fail_before_provider_ownership_access() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 2],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let original = allocation.handle.wire_descriptor();
    assert!(gpu_helpers::exact_provider_for_handle(&allocation.handle).is_none());
    let resident = Value::GpuTensor(allocation.handle.clone());
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = block_on(pagefun_builtin(
            Value::FunctionHandle("mtimes".into()),
            resident.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![resident.clone()]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), PAGEFUN_ERROR_OUTPUTS.identifier);
    }
    let nested = Value::Cell(runmat_value::CellArray::new(vec![resident.clone()], 1, 1).unwrap());
    let error = block_on(pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        resident,
        crate::builtins::common::test_support::builtin_arguments(vec![nested]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), PAGEFUN_ERROR_INVALID_INPUT.identifier);
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(
        allocation.probe.retirements(),
        0,
        "caller and managed cell retain their own leases"
    );
}

#[test]
fn empty_page_broadcast_rejects_non_singleton_mismatch() {
    let empty = Value::Tensor(Tensor::new(vec![], vec![2, 2, 0]).unwrap());
    let two = Value::Tensor(Tensor::new(vec![1.; 8], vec![2, 2, 2]).unwrap());
    let error = block_on(pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        empty,
        crate::builtins::common::test_support::builtin_arguments(vec![two]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        PAGEFUN_ERROR_PAGE_DIM_MISMATCH.identifier
    );
}

#[test]
fn complex_integer_pages_reject_before_floating_materialization() {
    let storage = runmat_value::IntegerComplexStorage::new(
        IntegerStorage::U64(vec![9_007_199_254_740_993]),
        IntegerStorage::U64(vec![0]),
    )
    .unwrap();
    let value = Value::ComplexTensor(ComplexTensor::new_integer(storage, vec![1, 1]).unwrap());
    let error = block_on(pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        value,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.)]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), PAGEFUN_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn single_complex_pages_and_empty_pages_retain_class() {
    for shape in [vec![1, 1, 2], vec![1, 1, 0]] {
        let values = if shape[2] == 0 {
            vec![]
        } else {
            vec![(2f32, 1f32), (4f32, -1f32)]
        };
        let value = Value::ComplexTensor(ComplexTensor::from_f32(values, shape.clone()).unwrap());
        let result = block_on(pagefun_impl(
            Value::FunctionHandle("mtimes".into()),
            value,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.)]),
        ))
        .unwrap();
        let Value::ComplexTensor(result) = result else {
            panic!("complex page array");
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(result.shape(), shape);
    }
}
