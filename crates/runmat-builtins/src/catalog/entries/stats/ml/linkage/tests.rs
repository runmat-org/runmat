use super::*;
use runmat_types::*;

fn data(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &LINKAGE_CATALOG_ENTRY,
    )
}

fn text(_value: &str) -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

#[test]
fn observation_and_condensed_shapes_produce_host_double_trees() {
    let observations = infer(vec![data(4, 2)], vec![]);
    assert!(
        observations.diagnostics.is_empty(),
        "{:?}",
        observations.diagnostics
    );
    assert_eq!(
        observations.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );
    assert_eq!(
        observations.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(observations.outputs[0].residency, ResidencyFact::Host);

    let condensed = infer(
        vec![data(1, 6), text("complete")],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("complete".into()),
        ],
    );
    assert!(
        condensed.diagnostics.is_empty(),
        "{:?}",
        condensed.diagnostics
    );
    assert_eq!(
        condensed.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(3)])
    );
}

#[test]
fn proven_bad_geometry_and_method_are_diagnosed() {
    assert!(!infer(vec![data(1, 5)], vec![]).diagnostics.is_empty());
    assert!(!infer(vec![data(701, 2)], vec![]).diagnostics.is_empty());
    assert!(!infer(
        vec![data(4, 2), text("unknown")],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("unknown".into())
        ]
    )
    .diagnostics
    .is_empty());
    assert!(!infer(
        vec![data(1, 6), text("single"), text("cityblock")],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("single".into()),
            LiteralValue::String("cityblock".into())
        ]
    )
    .diagnostics
    .is_empty());
}

#[test]
fn typed_method_aliases_and_save_memory_controls_are_checked() {
    assert_eq!(
        LinkageMethod::from_keyword("UPGMA"),
        Some(LinkageMethod::Average)
    );
    let valid = infer(
        vec![data(4, 2), text("ward"), text("savememory"), text("on")],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("ward".into()),
            LiteralValue::String("savememory".into()),
            LiteralValue::String("on".into()),
        ],
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    let invalid = infer(
        vec![data(4, 2), text("single"), text("savememory"), text("on")],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("single".into()),
            LiteralValue::String("savememory".into()),
            LiteralValue::String("on".into()),
        ],
    );
    assert!(!invalid.diagnostics.is_empty());
}
