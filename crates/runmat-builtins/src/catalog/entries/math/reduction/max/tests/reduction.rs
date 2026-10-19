use super::*;

#[test]
fn reduction_proves_exact_value_and_double_index_shape() {
    let shape = ShapeFact::from(vec![Some(3), Some(4)]);
    let result = call(numeric(NumericClass::UInt64, shape), vec![], 2);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
    assert_eq!(result.outputs[1].shape, result.outputs[0].shape);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(
        result.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn logical_reduction_uses_double_and_scalar_integer_stays_exact() {
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let result = call(logical, vec![], 1);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let scalar = call(numeric(NumericClass::UInt64, ShapeFact::Scalar), vec![], 2);
    assert_eq!(
        scalar.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(
        scalar.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn scalar_complex_reduction_retains_complex_domain_even_with_zero_imaginary_part() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    let result = call(input, vec![], 2);
    assert_eq!(
        result.outputs[0].numeric().unwrap().domain,
        NumericDomain::Complex
    );
    assert!(result.outputs[0]
        .shape
        .is_proven_equivalent(&ShapeFact::Scalar));
    assert_eq!(
        result.outputs[1].numeric().unwrap().domain,
        NumericDomain::Real
    );
}

#[test]
fn ordinal_categorical_object_stays_dynamic_without_a_false_numeric_error() {
    let input = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::CATEGORICAL.into()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Opaque,
    );
    let result = call(input, vec![], 2);
    assert!(result
        .outputs
        .iter()
        .all(|output| matches!(output.certainty, CertaintyFact::Dynamic(_))));
    assert!(!result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM.MAX.INVALID_INPUT"));
}

#[test]
fn empty_reduction_preserves_known_numeric_classes_and_successful_shapes() {
    let shape = ShapeFact::from(vec![Some(0), Some(2)]);
    let floating = call(numeric(NumericClass::Single, shape.clone()), vec![], 2);
    assert_eq!(floating.outputs[0].shape, shape);
    assert_eq!(floating.outputs[1].shape, floating.outputs[0].shape);
    assert_eq!(
        floating.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        floating.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );

    let integer_all = call(
        numeric(NumericClass::UInt64, shape.clone()),
        vec![LiteralValue::Empty, LiteralValue::String("all".into())],
        2,
    );
    assert_eq!(integer_all.outputs[0].shape, shape);
    assert_eq!(
        integer_all.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(
        integer_all.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );

    let floating_all = call(
        numeric(NumericClass::Single, shape),
        vec![LiteralValue::Empty, LiteralValue::String("all".into())],
        2,
    );
    assert_eq!(
        floating_all.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        floating_all.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(floating_all.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn empty_all_preserves_integer_class_and_resident_results_are_conservative() {
    let empty = call(
        numeric(
            NumericClass::UInt64,
            ShapeFact::from(vec![Some(0), Some(2)]),
        ),
        vec![LiteralValue::Empty, LiteralValue::String("all".into())],
        2,
    );
    assert_eq!(
        empty.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(2)])
    );
    assert_eq!(
        empty.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );

    let mut resident = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(3), Some(1)]),
    );
    resident.residency = ResidencyFact::Device { provider: None };
    let result = call(resident, vec![], 2);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[1].residency, ResidencyFact::Unknown);
}
