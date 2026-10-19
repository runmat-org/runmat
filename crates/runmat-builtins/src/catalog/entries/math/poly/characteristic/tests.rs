use super::*;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[test]
fn vector_and_square_matrix_infer_exact_row_class() {
    for (shape, length) in [
        (ShapeFact::from(vec![Some(1), Some(3)]), 4),
        (ShapeFact::from(vec![Some(3), Some(3)]), 4),
    ] {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            }),
            shape,
            StorageFact::Dense,
        );
        let result = crate::infer_catalog_call(
            &POLY_CATALOG_ENTRY,
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
            },
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(length)])
        );
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
        assert_eq!(
            result.outputs[0].numeric().unwrap().domain,
            NumericDomain::Complex
        );
    }
}

#[test]
fn automatic_residency_can_gather_but_remote_input_is_diagnosed() {
    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    input.residency = ResidencyFact::Device { provider: None };
    let result = crate::infer_catalog_call(
        &POLY_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );

    let mut remote = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    remote.residency = ResidencyFact::Remote { pool: None };
    let result = crate::infer_catalog_call(
        &POLY_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![remote],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert_eq!(result.diagnostics.len(), 1);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}
