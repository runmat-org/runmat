use super::*;
use runmat_types::*;

fn observations(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

#[test]
fn condensed_shape_and_double_output_are_exact() {
    let result = inference::infer(
        &CallRequest {
            arguments: vec![observations(4, 2)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &PDIST_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(6)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn rank_and_class_errors_are_diagnosed() {
    for value in [
        ValueFact::proven(
            ValueKindFact::Logical,
            ShapeFact::from(vec![Some(3), Some(2)]),
            StorageFact::Dense,
        ),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(3), Some(2), Some(2)]),
            StorageFact::Dense,
        ),
    ] {
        let result = inference::infer(
            &CallRequest {
                arguments: vec![value],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
            &PDIST_CATALOG_ENTRY,
        );
        assert!(!result.diagnostics.is_empty());
    }
}
