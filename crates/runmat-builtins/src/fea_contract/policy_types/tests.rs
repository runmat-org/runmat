use super::classes::{SIGNAL_KNOT, SPECTRUM_AXIS};
use super::conformance::{object, run};
use super::inference::{PolicyArg, PolicyCall, SHAPE, TYPED};
use runmat_types::{
    CellFact, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

mod fixture {
    use super::super::classes::SPECTRUM_AXIS;
    use super::super::entry::policy_entry;
    use super::super::inference::PolicyArg;

    policy_entry!(
        FIXTURE_CATALOG_ENTRY,
        name: "fea.SpectrumAxis.fixture",
        result: SPECTRUM_AXIS,
        signature: "axis = fea.SpectrumAxis.fixture(values, knots)",
        inputs: [
            ("values", NumericArray, "Vector.", PolicyArg::RealVector3),
            ("knots", Any, "Knots.", PolicyArg::TypedCell { classes: &[super::SIGNAL_KNOT], minimum: 2 }),
        ],
        output: ("axis", "Fixture axis."),
        integer_capabilities: &[],
        integer_audit: None,
        documentation: crate::BuiltinDocumentation::EMPTY,
    );
}

fn double(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn knots(count: usize, element: ValueFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element.clone()),
            elements: vec![element; count],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(count), Some(1)]),
        StorageFact::Dense,
    )
}

#[test]
fn vector_and_cell_shapes_are_checked_only_when_proven() {
    let entry = &fixture::FIXTURE_CATALOG_ENTRY;
    let valid = run(
        entry,
        vec![
            double(ShapeFact::from(vec![Some(3), Some(1)])),
            knots(3, object(SIGNAL_KNOT)),
        ],
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);

    let matrix = run(
        entry,
        vec![
            double(ShapeFact::from(vec![Some(3), Some(3)])),
            knots(2, object(SIGNAL_KNOT)),
        ],
    );
    assert!(matrix
        .diagnostics
        .iter()
        .any(|d| d.code == SHAPE && d.argument == Some(0)));

    let short = run(
        entry,
        vec![
            double(ShapeFact::from(vec![None, Some(1)])),
            knots(1, object(SIGNAL_KNOT)),
        ],
    );
    assert_eq!(short.diagnostics.len(), 1, "{:?}", short.diagnostics);
    assert_eq!(short.diagnostics[0].code, SHAPE);

    let foreign = run(
        entry,
        vec![double(ShapeFact::Scalar), knots(2, object(SPECTRUM_AXIS))],
    );
    assert!(foreign
        .diagnostics
        .iter()
        .any(|d| d.code == TYPED && d.argument == Some(1)));
    assert!(
        foreign.diagnostics.iter().any(|d| d.argument == Some(0)),
        "scalar is not a 3-vector"
    );
}

#[test]
fn rules_are_plain_static_data() {
    let call = PolicyCall {
        result: SPECTRUM_AXIS,
        arguments: &[PolicyArg::Count],
    };
    assert_eq!(call.arguments.len(), 1);
    assert_eq!(fixture::POLICY_CALL.arguments.len(), 2);
    assert_eq!(fixture::DESCRIPTOR.signatures[0].inputs.len(), 2);
}
