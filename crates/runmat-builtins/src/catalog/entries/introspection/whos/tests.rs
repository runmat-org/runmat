use super::*;
use runmat_types::*;

fn infer(
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: RequestedOutputCount,
) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(outputs),
        },
        &WHOS_CATALOG_ENTRY,
    )
}

#[test]
fn workspace_metadata_has_known_fields_and_dynamic_column_length() {
    let result = infer(vec![], vec![], RequestedOutputCount::One);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    let ValueKindFact::Struct(record) = &result.outputs[0].kind else {
        panic!("struct")
    };
    assert!(record.fields_complete);
    assert_eq!(record.fields.len(), 9);
    assert_eq!(record.fields["name"].kind, ValueKindFact::String);
    assert_eq!(record.fields["global"].kind, ValueKindFact::Logical);
    assert!(!result.effects.0.contains(&EffectKind::MaySuspend));
    assert!(!result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn file_mode_and_invalid_numeric_selector_are_distinct() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let file = infer(
        vec![text.clone(), text],
        vec![
            LiteralValue::String("-file".into()),
            LiteralValue::String("snapshot.mat".into()),
        ],
        RequestedOutputCount::One,
    );
    assert!(file.diagnostics.is_empty(), "{:?}", file.diagnostics);
    assert!(file
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(!file.effects.0.contains(&EffectKind::WorkspaceRead));
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let invalid = infer(vec![numeric], vec![], RequestedOutputCount::One);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-WORKSPACE-QUERY-SELECTOR"));
}

#[test]
fn zero_and_excess_outputs_follow_fixed_result_contract() {
    assert!(infer(vec![], vec![], RequestedOutputCount::Zero)
        .outputs
        .is_empty());
    assert!(infer(vec![], vec![], RequestedOutputCount::Exactly(2))
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-TYPE-CALL-ARITY"));
}
