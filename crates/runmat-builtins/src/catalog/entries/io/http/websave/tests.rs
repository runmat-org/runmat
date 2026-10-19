use super::*;
use runmat_types::*;

#[test]
fn websave_filename_is_host_character_row_and_requires_network_and_filesystem() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer(&request, &WEBSAVE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert!(result.effects.0.contains(&EffectKind::FilesystemWrite));
    assert!(result.effects.0.contains(&EffectKind::Network));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
    let invalid = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::Logical),
            request.arguments[1].clone(),
        ],
        ..request
    };
    assert!(!inference::infer(&invalid, &WEBSAVE_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
