use super::*;
use runmat_types::*;
#[test]
fn scalar_value_identity_does_not_claim_mutable_property_types() {
    let result = inference::infer(
        &CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &INTEGER_CONSTRAINT_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("object fact")
    };
    assert_eq!(
        object.runtime_class.as_ref(),
        Some(&standard::INTEGER_CONSTRAINT.owned())
    );
    assert_eq!(object.handle_semantics, Some(false));
    assert!(object.properties.is_empty());
    assert!(!object.properties_complete);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(
        IntegerConstraintField::parse("semicontinuous"),
        Some(IntegerConstraintField::SemiContinuous)
    );
    assert_eq!(IntegerConstraintField::parse("Semi"), None);
}
