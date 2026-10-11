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
        &WHO_CATALOG_ENTRY,
    )
}

#[test]
fn workspace_form_is_cell_of_host_strings_without_filesystem_capability() {
    let result = infer(vec![], vec![], RequestedOutputCount::One);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("cell")
    };
    assert_eq!(cell.element.kind, ValueKindFact::String);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert!(result.effects.0.contains(&EffectKind::WorkspaceRead));
    assert!(!result.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(!result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}

#[test]
fn file_and_dynamic_option_forms_preserve_effect_unions() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let file = infer(
        vec![text.clone(), text.clone()],
        vec![
            LiteralValue::String("-file".into()),
            LiteralValue::String("test.mat".into()),
        ],
        RequestedOutputCount::One,
    );
    assert!(file.diagnostics.is_empty(), "{:?}", file.diagnostics);
    assert!(file.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(!file.effects.0.contains(&EffectKind::WorkspaceRead));
    assert!(file
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    let uppercase = infer(
        vec![text.clone(), text.clone()],
        vec![
            LiteralValue::String("-FILE".into()),
            LiteralValue::String("test.mat".into()),
        ],
        RequestedOutputCount::One,
    );
    assert!(uppercase.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(!uppercase.effects.0.contains(&EffectKind::WorkspaceRead));
    let dynamic = infer(
        vec![text],
        vec![LiteralValue::Unknown],
        RequestedOutputCount::One,
    );
    assert!(dynamic.effects.0.contains(&EffectKind::WorkspaceRead));
    assert!(dynamic.effects.0.contains(&EffectKind::FilesystemRead));
}

#[test]
fn invalid_selector_and_output_count_are_diagnosed() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    assert!(!infer(vec![numeric], vec![], RequestedOutputCount::One)
        .diagnostics
        .is_empty());
    assert!(infer(vec![], vec![], RequestedOutputCount::Exactly(2))
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-TYPE-CALL-ARITY"));
    assert!(infer(vec![], vec![], RequestedOutputCount::Zero)
        .outputs
        .is_empty());
}
