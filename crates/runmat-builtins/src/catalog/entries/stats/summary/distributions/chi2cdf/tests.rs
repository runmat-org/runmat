use super::*;
use runmat_types::*;

#[test]
fn shape_class_dynamic_inputs_and_output_arity() {
    let request =
        super::super::test_support::request(super::super::policy::DistributionKind::ChiSquareCdf);
    let result = infer(&request, &CHI2CDF_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);

    let mut unknown = request.clone();
    let numeric_start = 0;
    unknown.arguments[numeric_start] = ValueFact::unknown(DynamicReason::RuntimeValue);
    let dynamic = infer(&unknown, &CHI2CDF_CATALOG_ENTRY);
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);

    let mut too_many_outputs = request;
    too_many_outputs.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(!infer(&too_many_outputs, &CHI2CDF_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn same_size_or_scalar_is_not_general_implicit_expansion() {
    let mut request =
        super::super::test_support::request(super::super::policy::DistributionKind::ChiSquareCdf);
    request.arguments[1].shape = ShapeFact::from(vec![Some(1), Some(3)]);
    let result = infer(&request, &CHI2CDF_CATALOG_ENTRY);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "BROADCAST"));
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}
