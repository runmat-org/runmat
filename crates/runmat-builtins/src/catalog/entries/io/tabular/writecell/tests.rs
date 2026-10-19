use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn statement_and_extension_output_are_distinct() {
    let args = vec![
        ValueFact::unknown(DynamicReason::RuntimeValue),
        ValueFact::scalar(ValueKindFact::String),
    ];
    let statement = inference::infer(&request(args.clone(), 0), &WRITECELL_CATALOG_ENTRY);
    assert!(statement.outputs.is_empty());
    assert!(statement.effects.0.contains(&EffectKind::FilesystemWrite));
    let output = inference::infer(&request(args, 1), &WRITECELL_CATALOG_ENTRY);
    assert_eq!(output.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn known_noncell_is_invalid_but_unknown_remains_possible() {
    let path = ValueFact::scalar(ValueKindFact::String);
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(inference::infer(
        &request(vec![numeric, path.clone()], 0),
        &WRITECELL_CATALOG_ENTRY
    )
    .diagnostics
    .iter()
    .any(|diagnostic| diagnostic.code == "RM-CATALOG-WRITECELL-DATA"));
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        inference::infer(&request(vec![unknown, path], 0), &WRITECELL_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}
