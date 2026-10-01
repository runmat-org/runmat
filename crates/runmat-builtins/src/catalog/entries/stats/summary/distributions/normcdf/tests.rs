use super::*;
use runmat_types::*;

#[test]
fn shape_class_dynamic_inputs_and_output_arity() {
    let request =
        super::super::test_support::request(super::super::policy::DistributionKind::NormalCdf);
    let result = infer(&request, &NORMCDF_CATALOG_ENTRY);
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
    let numeric_start = 0;
    unknown.arguments[numeric_start] = ValueFact::unknown(DynamicReason::RuntimeValue);
    let dynamic = infer(&unknown, &NORMCDF_CATALOG_ENTRY);
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);

    let mut too_many_outputs = request;
    too_many_outputs.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!infer(&too_many_outputs, &NORMCDF_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn upper_tail_is_a_typed_option_not_a_numeric_operand() {
    let mut request =
        super::super::test_support::request(super::super::policy::DistributionKind::NormalCdf);
    request
        .arguments
        .push(ValueFact::scalar(ValueKindFact::String));
    request
        .literals
        .literal_args
        .push(LiteralValue::String("upper".into()));
    let upper = infer(&request, &NORMCDF_CATALOG_ENTRY);
    assert!(upper.diagnostics.is_empty(), "{:?}", upper.diagnostics);
    assert_eq!(
        upper.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    request.literals.literal_args[3] = LiteralValue::String("lower".into());
    assert!(infer(&request, &NORMCDF_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "TAIL"));
}
