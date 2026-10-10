use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

#[test]
fn literal_factor_preserves_single_and_row_orientation() {
    let request = CallRequest {
        arguments: vec![
            numeric(NumericClass::Single, &[1, 3]),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ],
        literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(2.0)]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = inference::infer(&request, &UPSAMPLE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(6)])
    );
}

#[test]
fn nonscalar_factor_is_rejected() {
    let request = CallRequest {
        arguments: vec![
            numeric(NumericClass::Double, &[1, 1]),
            numeric(NumericClass::Double, &[1, 2]),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    assert!(!inference::infer(&request, &UPSAMPLE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
