use super::*;
use runmat_types::*;

#[test]
fn inference_accepts_slot_vectors_and_rejects_proven_invalid_control_shapes() {
    let scalar = infer_literal(&LiteralValue::Number(2.)).fact;
    for (shape, allowed) in [
        (ShapeFact::Scalar, true),
        (ShapeFact::from(vec![Some(1), Some(4)]), true),
        (ShapeFact::from(vec![Some(4), Some(1)]), true),
        (ShapeFact::from(vec![Some(1), None]), true),
        (ShapeFact::from(vec![Some(2), None]), true),
        (ShapeFact::from(vec![Some(2), Some(2)]), false),
        (ShapeFact::from(vec![Some(0), None]), false),
        (ShapeFact::from(vec![Some(1), Some(1), Some(3)]), false),
    ] {
        let mut position = scalar.clone();
        position.shape = shape;
        let request = CallRequest {
            arguments: vec![scalar.clone(), scalar.clone(), position],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = inference::infer(&request, &SUBPLOT_CATALOG_ENTRY);
        assert_eq!(
            result.diagnostics.is_empty(),
            allowed,
            "{:?}",
            request.arguments[2].shape
        );
    }
    let mut rows = scalar.clone();
    rows.shape = ShapeFact::from(vec![Some(2), None]);
    let request = CallRequest {
        arguments: vec![rows, scalar.clone(), scalar],
        literals: LiteralContext::default(),
        outputs: OutputSelection::default(),
    };
    assert!(!inference::infer(&request, &SUBPLOT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn contract_has_owned_identity_and_bounded_outputs() {
    assert_eq!(SUBPLOT_CATALOG_ENTRY.identity.name, "subplot");
    assert!(matches!(
        SUBPLOT_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(SUBPLOT_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == SUBPLOT_ERROR_TOO_MANY_OUTPUTS.identifier));
}
