use super::*;
use runmat_types::*;
#[test]
fn unknown_clear_target_remains_admissible_without_invented_name() {
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_clear(&request, &CLEAR_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
}
#[test]
fn clear_infers_empty_host_result_and_workspace_effect() {
    let request = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_clear(&request, &CLEAR_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    assert!(result.effects.0.contains(&EffectKind::WorkspaceWrite));
}

#[test]
fn discarded_clear_result_retains_workspace_effect() {
    let request = CallRequest {
        arguments: vec![],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Zero),
    };
    let result = inference::infer_clear(&request, &CLEAR_CATALOG_ENTRY);
    assert!(result.outputs.is_empty());
    assert!(result.effects.0.contains(&EffectKind::WorkspaceWrite));
}
