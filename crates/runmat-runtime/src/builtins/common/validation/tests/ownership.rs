use super::*;

#[test]
fn namedargs_preserves_nested_resident_payloads_and_source_reuse() {
    runmat_gc::gc_test_context(|| {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                vec![1, 2],
            )
            .unwrap();
            let handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let nested =
                Value::Cell(CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).unwrap());
            let mut structure = runmat_value::UnboundedStructValueBuilder::new();
            structure.insert("payload", nested.clone());
            let structure = structure.finish();
            for count in [0, 1] {
                let _outputs = crate::output_count::push_output_count(Some(count))
                    .expect("output-count scope");
                let output = namedargs2cell::evaluate(
                    crate::builtins::common::validation::tests::retained_arguments(vec![
                        Value::Struct(structure.clone()),
                    ]),
                )
                .unwrap();
                let Value::Cell(cell) = output else {
                    panic!("cell output")
                };
                assert_eq!(cell.data()[1], nested);
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&handle),
                    Some(runmat_accelerate_api::IntegerElementType::U64)
                );
                let gathered =
                    futures::executor::block_on(provider.download_integer(&handle)).unwrap();
                assert!(
                    matches!(gathered.data, runmat_accelerate_api::HostIntegerDataOwned::U64(data)
                if data == vec![9_007_199_254_740_993, u64::MAX])
                );
            }
            let weak = handle.downgrade();
            drop(structure);
            drop(nested);
            runmat_gc::gc_collect_major().unwrap();
            assert!(futures::executor::block_on(provider.download_integer(&handle)).is_ok());
            drop(handle);
            assert!(weak.acquire().is_err());
        });
    });
}

#[test]
fn scalar_predicate_and_name_value_conversion_enforce_one_output_before_input_work() {
    for evaluate in [
        isvarname::evaluate as fn(crate::call::BuiltinArguments) -> BuiltinResult<Value>,
        namedargs2cell::evaluate,
    ] {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error =
            evaluate(crate::builtins::common::validation::tests::retained_arguments(vec![]))
                .unwrap_err();
        assert!(error.identifier().unwrap().ends_with(":TooManyOutputs"));
    }
    for count in [0, 1] {
        let _outputs =
            crate::output_count::push_output_count(Some(count)).expect("output-count scope");
        assert_eq!(
            isvarname::evaluate(
                crate::builtins::common::validation::tests::retained_arguments(vec![
                    Value::String("valid_name".into())
                ])
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            isvarname::evaluate(
                crate::builtins::common::validation::tests::retained_arguments(vec![Value::Int(
                    IntValue::U64(u64::MAX)
                )])
            )
            .unwrap(),
            Value::Bool(false)
        );
    }
}
