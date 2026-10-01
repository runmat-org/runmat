use super::*;
use runmat_types::*;

#[test]
fn shape_class_dynamic_inputs_and_output_arity() {
    let request =
        super::super::test_support::request(super::super::policy::DistributionKind::GenericInv);
    let result = infer(&request, &ICDF_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);

    let mut unknown = request.clone();
    let numeric_start = 1;
    unknown.arguments[numeric_start] = ValueFact::unknown(DynamicReason::RuntimeValue);
    let dynamic = infer(&unknown, &ICDF_CATALOG_ENTRY);
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);

    let mut too_many_outputs = request;
    too_many_outputs.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!infer(&too_many_outputs, &ICDF_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn named_family_alias_and_parameter_count_are_validated() {
    let mut request =
        super::super::test_support::request(super::super::policy::DistributionKind::GenericInv);
    request.literals.literal_args[0] = LiteralValue::String("gaussian".into());
    assert!(infer(&request, &ICDF_CATALOG_ENTRY).diagnostics.is_empty());
    request.literals.literal_args[0] = LiteralValue::String("unknown".into());
    assert!(infer(&request, &ICDF_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FAMILY"));
    request.literals.literal_args[0] = LiteralValue::String("Poisson".into());
    assert!(infer(&request, &ICDF_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "ARITY"));
}

#[test]
fn fitted_object_output_class_follows_probability_not_opaque_parameters() {
    let mut request =
        super::super::test_support::request(super::super::policy::DistributionKind::GenericInv);
    request.arguments[0] = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(runmat_types::standard::PROBABILITY_DISTRIBUTION.owned()),
        properties: std::collections::BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }));
    request.arguments.truncate(2);
    request.literals.literal_args.truncate(2);
    let single = infer(&request, &ICDF_CATALOG_ENTRY);
    assert!(single.diagnostics.is_empty(), "{:?}", single.diagnostics);
    assert_eq!(
        single.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );

    request.arguments[1] = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let double = infer(&request, &ICDF_CATALOG_ENTRY);
    assert!(double.diagnostics.is_empty(), "{:?}", double.diagnostics);
    assert_eq!(
        double.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );

    request.arguments[0] = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(runmat_types::standard::TABLE.owned()),
        properties: std::collections::BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }));
    assert!(infer(&request, &ICDF_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FAMILY"));

    if let ValueKindFact::Object(object) = &mut request.arguments[0].kind {
        object.runtime_class = None;
    }
    let unknown_class = infer(&request, &ICDF_CATALOG_ENTRY);
    assert!(!unknown_class
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FAMILY"));
    assert!(matches!(
        unknown_class.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    ));
}
