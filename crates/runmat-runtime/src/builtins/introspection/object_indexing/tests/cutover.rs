use super::*;

#[test]
fn canonical_indexing_bindings_join_without_legacy_authority() {
    for name in [
        "subsref",
        "subsasgn",
        "numArgumentsFromSubscript",
        "superclasses",
        "get.p",
        "set.p",
    ] {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name).expect(name);
        assert!(runmat_builtins::builtin_function_by_name(name).is_none());
        let bindings = crate::builtin::runtime_builtin_bindings_by_name(name);
        assert_eq!(bindings.len(), 1, "{name}");
        assert_eq!(
            bindings[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
    }
}

#[test]
fn requested_cell_outputs_preserve_nested_resident_values_without_provider_access() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 3],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("valid unregistered resident fixture")
            .handle,
    );
    let payload = cell(vec![resident]);
    let source = cell(vec![Value::Int(IntValue::U64(u64::MAX)), payload.clone()]);
    let path = substruct("{}", cell(vec![Value::String(":".into())]));
    for count in [0, 1, 2] {
        let _guard =
            crate::output_count::push_output_count(Some(count)).expect("output-count scope");
        let result = block_on(subsref_builtin(source.clone(), path.clone())).unwrap();
        let Value::OutputList(values) = result.value() else {
            panic!("explicit output carrier");
        };
        assert_eq!(values.len(), count);
        if count > 0 {
            assert_eq!(values[0], Value::Int(IntValue::U64(u64::MAX)));
        }
        if count > 1 {
            assert_eq!(values[1], payload);
        }
    }
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    assert!(block_on(subsref_builtin(source, path)).is_err());
}

#[test]
fn discard_still_validates_paths_and_overload_admission() {
    block_on(crate::register_test_classes_builtin()).unwrap();
    let _guard = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let object = Value::Object(ObjectInstance::new("OverIdx"));
    // Invalid selectors must fail even when the read targets an overloaded class
    // and the caller retains no outputs.
    let path = substruct("{}", Value::Num(1.0));
    assert!(block_on(subsref_builtin(object, path)).is_err());
    assert!(block_on(subsref_builtin(Value::Num(1.0), Value::Num(2.0))).is_err());
}

#[test]
fn fixed_output_rejection_precedes_indexing_dispatch() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(subsasgn_builtin(
        Value::Num(0.0),
        Value::Num(0.0),
        Value::Num(0.0),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:subsasgn:TooManyOutputs"));
    let error = call_err(Value::Num(0.0), Value::Num(0.0), Value::Num(0.0));
    assert_eq!(
        error.identifier(),
        Some("RunMat:numArgumentsFromSubscript:TooManyOutputs")
    );
}

#[test]
fn scalar_integer_tensor_cardinality_never_rounds_through_double() {
    #[cfg(target_pointer_width = "64")]
    {
        let tensor = Tensor::from_numeric_storage(
            runmat_value::NumericStorage::U64(vec![9_007_199_254_740_993]),
            vec![1, 1],
        )
        .unwrap();
        assert_eq!(
            exact_output_count(&Value::Tensor(tensor)).unwrap(),
            9_007_199_254_740_993
        );
        assert!(exact_output_count(&Value::Num(18_446_744_073_709_551_616.0)).is_err());
    }
}
