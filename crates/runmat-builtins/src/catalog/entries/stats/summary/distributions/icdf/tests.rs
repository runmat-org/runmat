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
