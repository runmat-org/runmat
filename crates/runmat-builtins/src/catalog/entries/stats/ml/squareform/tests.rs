use super::super::distance_geometry::{condensed_length, matrix_size_for_condensed_length};
use super::*;
use runmat_types::*;

fn infer_for(
    kind: ValueKindFact,
    shape: ShapeFact,
    direction: Option<LiteralValue>,
) -> CallInference {
    let mut arguments = vec![ValueFact::proven(kind, shape, StorageFact::Dense)];
    let mut literals = vec![LiteralValue::Unknown];
    if let Some(literal) = direction {
        arguments.push(ValueFact::scalar(ValueKindFact::String));
        literals.push(literal);
    }
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &SQUAREFORM_CATALOG_ENTRY,
    )
}

#[test]
fn exact_checked_condensed_geometry() {
    assert_eq!(matrix_size_for_condensed_length(0), Some(1));
    assert_eq!(matrix_size_for_condensed_length(3), Some(3));
    assert_eq!(matrix_size_for_condensed_length(4), None);
    assert_eq!(condensed_length(3), Some(3));
    assert_eq!(matrix_size_for_condensed_length(usize::MAX), None);
}

#[test]
fn class_and_shape_are_preserved_for_both_directions() {
    for kind in [
        ValueKindFact::Logical,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
    ] {
        let matrix = infer_for(kind.clone(), ShapeFact::from(vec![Some(1), Some(3)]), None);
        assert!(matrix.diagnostics.is_empty(), "{:?}", matrix.diagnostics);
        assert_eq!(matrix.outputs[0].kind, kind);
        assert_eq!(
            matrix.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(3)])
        );
        let vector = infer_for(kind.clone(), ShapeFact::from(vec![Some(3), Some(3)]), None);
        assert!(vector.diagnostics.is_empty(), "{:?}", vector.diagnostics);
        assert_eq!(vector.outputs[0].kind, kind);
        assert_eq!(
            vector.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(3)])
        );
    }
}

#[test]
fn unknown_force_cannot_fix_an_output_shape() {
    let output = infer_for(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(1), Some(3)]),
        Some(LiteralValue::Unknown),
    );
    assert_eq!(output.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn malformed_geometry_and_direction_are_diagnosed() {
    let kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    assert!(
        !infer_for(kind.clone(), ShapeFact::from(vec![Some(1), Some(4)]), None)
            .diagnostics
            .is_empty()
    );
    assert!(!infer_for(
        kind.clone(),
        ShapeFact::from(vec![Some(2), Some(3)]),
        Some(LiteralValue::String("tovector".to_owned())),
    )
    .diagnostics
    .is_empty());
    assert!(!infer_for(
        kind,
        ShapeFact::from(vec![Some(1), Some(3)]),
        Some(LiteralValue::String("other".to_owned())),
    )
    .diagnostics
    .is_empty());
}
