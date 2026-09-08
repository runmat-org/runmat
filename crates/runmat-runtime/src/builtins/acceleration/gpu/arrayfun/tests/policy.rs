use super::*;

#[test]
fn arrayfun_text_callable_and_host_scalar_expansion_are_mode_gated() {
    let input = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("input"));
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(Value::from("sin"), vec![input.clone()])
            .expect_err("text callable must reject in compatible mode");
        assert_eq!(
            error.identifier(),
            ARRAYFUN_TEXT_CALLABLE_EXTENSION.error_identifier
        );
        let error = call(
            Value::FunctionHandle("atan2".to_string()),
            vec![input.clone(), Value::Num(1.0)],
        )
        .expect_err("host scalar expansion must reject in compatible mode");
        assert_eq!(
            error.identifier(),
            ARRAYFUN_HOST_SCALAR_EXPANSION_EXTENSION.error_identifier
        );
    }
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        assert!(call(Value::from("sin"), vec![input.clone()]).is_ok());
        assert!(call(
            Value::FunctionHandle("atan2".to_string()),
            vec![input, Value::Num(1.0)],
        )
        .is_ok());
    }
}

#[test]
fn arrayfun_error_struct_has_only_documented_fields() {
    let Value::Struct(error) = make_error_struct("RunMat:test:Failure: detail", 4) else {
        panic!("expected error struct");
    };
    let mut fields: Vec<_> = error.fields.keys().map(String::as_str).collect();
    fields.sort_unstable();
    assert_eq!(fields, vec!["identifier", "index", "message"]);
    assert_eq!(error.fields.get("index"), Some(&Value::Num(5.0)));
}

#[test]
fn uniform_classifier_reads_typed_complex_integer_tensor_storage_exactly() {
    let storage = IntegerComplexStorage::new(
        IntegerStorage::I64(vec![9_007_199_254_740_993]),
        IntegerStorage::I64(vec![-9_007_199_254_740_993]),
    )
    .expect("complex integer storage");
    let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex tensor");

    match classify_value(&Value::ComplexTensor(tensor)).expect("classify") {
        ClassifiedValue::IntegerComplex(re, im) => {
            assert_eq!(re, IntValue::I64(9_007_199_254_740_993));
            assert_eq!(im, IntValue::I64(-9_007_199_254_740_993));
        }
        _ => panic!("expected exact integer-complex classification"),
    }
}
