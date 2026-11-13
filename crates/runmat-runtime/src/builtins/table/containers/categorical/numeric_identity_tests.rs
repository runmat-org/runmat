use super::*;

fn categorical(args: &[Value]) -> BuiltinResult<Value> {
    categorical_from_args(crate::call::FunctionArgumentView::from_slice(args))
}

fn named_values() -> Value {
    Value::StringArray(StringArray::new(vec!["left".into(), "right".into()], vec![1, 2]).unwrap())
}

#[test]
fn tiny_float_identities_survive_default_names_and_public_equality() {
    for source in [
        Value::Tensor(Tensor::new(vec![1e-20, 2e-20, 1e-20, f64::NAN], vec![4, 1]).unwrap()),
        Value::Tensor(Tensor::from_f32(vec![1e-20, 2e-20, 1e-20, f32::NAN], vec![4, 1]).unwrap()),
    ] {
        let result = categorical(&[source]).unwrap();
        let Value::Object(object) = &result else {
            panic!("categorical")
        };
        assert_eq!(
            categorical_categories(object).unwrap(),
            vec!["1e-20", "2e-20"]
        );
        let codes = categorical_codes(object).unwrap().materialize_f64();
        assert_eq!(&codes[..3], &[1.0, 2.0, 1.0]);
        assert!(codes[3].is_nan(), "numeric NaN remains missing");
        let comparison =
            categorical_compare(&result, &Value::from("1e-20"), CategoricalComparison::Eq)
                .unwrap()
                .unwrap();
        assert!(matches!(comparison, Value::LogicalArray(array) if array.data == vec![1, 0, 1, 0]));
    }
    assert_eq!(
        categorical_labels(&Value::Num(1e-20)).unwrap(),
        vec!["0.00000000000000000001"]
    );
    assert_eq!(
        categorical_labels(&Value::Num(f64::NAN)).unwrap(),
        vec![StringElement::Missing]
    );
}

#[test]
fn adjacent_values_remain_distinct_but_ambiguous_default_names_still_reject() {
    for (source, valueset) in [
        (
            Value::Tensor(
                Tensor::new(vec![1.0, f64::from_bits(1.0_f64.to_bits() + 1)], vec![1, 2]).unwrap(),
            ),
            Value::Tensor(
                Tensor::new(vec![f64::from_bits(1.0_f64.to_bits() + 1), 1.0], vec![1, 2]).unwrap(),
            ),
        ),
        (
            Value::Tensor(
                Tensor::from_f32(vec![1.0, f32::from_bits(1.0_f32.to_bits() + 1)], vec![1, 2])
                    .unwrap(),
            ),
            Value::Tensor(
                Tensor::from_f32(vec![f32::from_bits(1.0_f32.to_bits() + 1), 1.0], vec![1, 2])
                    .unwrap(),
            ),
        ),
    ] {
        let keys = categorical_labels(&source).unwrap();
        assert_ne!(keys[0], keys[1], "lookup keys must not round together");
        assert!(
            categorical(std::slice::from_ref(&source)).is_err(),
            "five-digit default names remain ambiguous"
        );
        assert!(
            categorical(&[source.clone(), valueset.clone()]).is_err(),
            "explicit valueset without names retains same rejection"
        );
        let Value::Object(object) =
            categorical(&[source.clone(), valueset, named_values()]).unwrap()
        else {
            panic!("categorical")
        };
        assert_eq!(
            categorical_categories(&object).unwrap(),
            vec!["left", "right"]
        );
        assert_eq!(
            categorical_codes(&object).unwrap().materialize_f64(),
            vec![2.0, 1.0]
        );
        assert!(categorical(&[
            source.clone(),
            source,
            named_values(),
            Value::from("Ordinal"),
            Value::Bool(true)
        ])
        .is_ok());
    }
}

#[test]
fn lookup_keys_preserve_float_class_zero_infinity_and_exact_integers() {
    for scalar in [NumericScalar::F64(-0.0), NumericScalar::F32(-0.0)] {
        assert_eq!(categorical_numeric_label(scalar), "-0");
    }
    for scalar in [NumericScalar::F64(0.0), NumericScalar::F32(0.0)] {
        assert_eq!(categorical_numeric_label(scalar), "0");
    }
    assert_eq!(
        categorical_numeric_label(NumericScalar::F64(f64::INFINITY)),
        "inf"
    );
    assert_eq!(
        categorical_numeric_label(NumericScalar::F32(f32::NEG_INFINITY)),
        "-inf"
    );
    assert_eq!(
        categorical_numeric_label(NumericScalar::I64(i64::MIN)),
        "-9223372036854775808"
    );
    assert_eq!(
        categorical_numeric_label(NumericScalar::U64(u64::MAX)),
        "18446744073709551615"
    );
    let single = f32::from_bits(0x3f9e0652);
    let key = categorical_numeric_label(NumericScalar::F32(single));
    assert_eq!(
        key.as_str().unwrap().parse::<f32>().unwrap().to_bits(),
        single.to_bits()
    );
    assert_eq!(
        default_category_names(
            &[key.as_str().unwrap().to_owned()],
            Some(runmat_value::NumericDType::F32)
        )
        .unwrap(),
        vec![format_five_significant(f64::from(single))]
    );
    let source = Value::Tensor(Tensor::new(vec![-0.0, 0.0], vec![1, 2]).unwrap());
    assert!(
        categorical(&[source]).is_err(),
        "existing signed-zero default-name collision remains explicit"
    );
    let source = Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap());
    assert!(
        categorical(&[source, Value::Num(1.0)]).is_err(),
        "numeric valueset must retain original storage class"
    );
}

#[test]
fn missing_category_identity_is_distinct_from_literal_and_empty_text() {
    let source = Value::StringArray(
        StringArray::from_elements_unbounded(
            vec![
                StringElement::Missing,
                "<missing>".into(),
                "<Missing>".into(),
                "".into(),
            ],
            vec![1, 4],
        )
        .unwrap(),
    );
    let Value::Object(default) = categorical(std::slice::from_ref(&source)).unwrap() else {
        panic!("categorical")
    };
    assert_eq!(
        categorical_labels(&Value::Object(default)).unwrap(),
        vec![
            StringElement::Missing,
            "<missing>".into(),
            "<Missing>".into(),
            StringElement::Missing
        ]
    );
    let valueset = Value::StringArray(
        StringArray::from_elements_unbounded(vec![StringElement::Missing, "".into()], vec![1, 2])
            .unwrap(),
    );
    let names = Value::StringArray(
        StringArray::new(vec!["absent".into(), "empty".into()], vec![1, 2]).unwrap(),
    );
    let Value::Object(explicit) = categorical(&[source, valueset, names]).unwrap() else {
        panic!("categorical")
    };
    assert_eq!(
        categorical_labels(&Value::Object(explicit)).unwrap(),
        vec![
            "absent".into(),
            StringElement::Missing,
            StringElement::Missing,
            "empty".into()
        ]
    );
}
