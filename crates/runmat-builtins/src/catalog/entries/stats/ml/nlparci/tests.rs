use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[test]
fn descriptor_and_inference_preserve_interval_shape_and_precision() {
    assert_eq!(NLPARCI_DESCRIPTOR.signatures.len(), 3);
    let real = |class, shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    };
    let request = CallRequest {
        arguments: vec![
            real(NumericClass::Single, vec![Some(2), Some(1)]),
            real(NumericClass::Single, vec![Some(4), Some(1)]),
            ValueFact::unknown(runmat_types::DynamicReason::UnsupportedRepresentation),
            real(NumericClass::Double, vec![Some(4), Some(2)]),
        ],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 4]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let inferred = inference::infer(&request, &NLPARCI_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert!(matches!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            ..
        })
    ));
}
