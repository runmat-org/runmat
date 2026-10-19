use super::*;
use runmat_types::*;
fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
#[test]
fn unknown_inputs_preserve_output_representation_and_effects() {
    let request = call(vec![ValueFact::unknown(DynamicReason::RuntimeValue); 1], 1);
    let result = infer_catalog_call(&GUNZIP_CATALOG_ENTRY, &request);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(result.outputs[0].storage, StorageFact::Dense);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(
        GUNZIP_CATALOG_ENTRY.contract.capabilities,
        &[
            CapabilityRequirement::Filesystem,
            CapabilityRequirement::Network,
        ]
    );
    for capability in [
        CapabilityRequirement::Filesystem,
        CapabilityRequirement::Network,
    ] {
        assert!(result.capabilities.0.contains(&capability));
    }
    assert!(result.effects.0.contains(&EffectKind::FilesystemWrite));
    assert!(result.effects.0.contains(&EffectKind::MaySuspend));
}
#[test]
fn missing_inputs_numeric_text_and_excess_outputs_are_diagnosed() {
    assert!(!infer_catalog_call(&GUNZIP_CATALOG_ENTRY, &call(vec![], 0))
        .diagnostics
        .is_empty());
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    assert!(
        !infer_catalog_call(&GUNZIP_CATALOG_ENTRY, &call(vec![numeric; 1], 0))
            .diagnostics
            .is_empty()
    );
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        !infer_catalog_call(&GUNZIP_CATALOG_ENTRY, &call(vec![unknown.clone(); 1], 2))
            .diagnostics
            .is_empty()
    );
    assert!(
        infer_catalog_call(&GUNZIP_CATALOG_ENTRY, &call(vec![unknown; 1], 0))
            .diagnostics
            .is_empty()
    );
}
