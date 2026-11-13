use super::*;
use runmat_types::{CellFact, NumericClass, NumericDomain, NumericFact};

#[test]
fn nd_character_names_reject_but_nd_cell_shapes_are_preserved() {
    for name in ["getenv", "isenv"] {
        let rejected = infer(
            name,
            vec![shaped(ValueKindFact::Character, &[1, 2, 3])],
            RequestedOutputCount::One,
        );
        assert!(!rejected.diagnostics.is_empty());
        let cell = CellFact {
            element: Box::new(character_row()),
            elements: vec![],
            elements_complete: false,
        };
        let accepted = infer(
            name,
            vec![shaped(ValueKindFact::Cell(cell), &[2, 2, 2])],
            RequestedOutputCount::One,
        );
        assert!(accepted.diagnostics.is_empty());
        assert_eq!(
            accepted.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(2), Some(2)])
        );
    }
}

#[test]
fn predicates_preserve_container_shape_and_reject_known_extensions() {
    let strings = shaped(ValueKindFact::String, &[2, 3]);
    let result = infer("isenv", vec![strings], RequestedOutputCount::One);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );

    let matrix = shaped(ValueKindFact::Character, &[2, 8]);
    for name in ["isenv", "unsetenv"] {
        let rejected = infer(name, vec![matrix.clone()], RequestedOutputCount::One);
        assert!(rejected
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.argument == Some(0)));
    }
}

#[test]
fn setenv_models_two_outputs_and_validates_typed_inputs() {
    let result = infer(
        "setenv",
        vec![character_row(), ValueFact::scalar(ValueKindFact::String)],
        RequestedOutputCount::Exactly(2),
    );
    assert_eq!(result.outputs.len(), 2);
    assert_eq!(
        result.outputs[0].numeric().map(|numeric| numeric.class),
        Some(NumericClass::Double)
    );
    assert_eq!(result.outputs[1].kind, ValueKindFact::Character);
    assert!(result.diagnostics.is_empty());

    let nonscalar_numeric = shaped(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        &[1, 2],
    );
    let rejected = infer(
        "setenv",
        vec![character_row(), nonscalar_numeric],
        RequestedOutputCount::Zero,
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SETENV-VALUE"));
}
