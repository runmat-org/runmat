use super::*;
use runmat_types::{
    CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueKindFact,
};

#[test]
fn catalog_declares_supported_forms_and_conservative_scalar_schema() {
    assert_eq!(
        ODESET_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Complete
    );
    for effect in [
        EffectKind::FilesystemWrite,
        EffectKind::UserInterface,
        EffectKind::HostCallback,
        EffectKind::MayThrow,
    ] {
        assert!(ODESET_CATALOG_ENTRY.contract.effects.contains(&effect));
    }
    assert!(ODESET_CATALOG_ENTRY.contract.capabilities.is_empty());
    assert_eq!(ODESET_DESCRIPTOR.signatures.len(), 4);
    assert!(ODESET_DESCRIPTOR
        .signatures
        .iter()
        .any(|signature| signature.label == "options = odeset(oldopts, newopts)"));
    let result = inference::infer(
        &CallRequest {
            arguments: vec![],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &ODESET_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Struct(fields) = &result.outputs[0].kind else {
        panic!("scalar structure expected")
    };
    assert!(fields.fields_complete);
    assert_eq!(fields.fields.len(), super::super::OdeOptionField::ALL.len());
    for field in super::super::OdeOptionField::ALL {
        assert!(fields.fields.contains_key(field.public_name()));
    }
    assert_eq!(ODESET_CATALOG_ENTRY.documentation.examples.len(), 2);
    for (argument_count, requested, prints_possible) in [
        (0, RequestedOutputCount::Zero, true),
        (0, RequestedOutputCount::One, false),
        (0, RequestedOutputCount::CurrentFunctionNargout, true),
        (2, RequestedOutputCount::Zero, false),
        (2, RequestedOutputCount::CurrentFunctionNargout, false),
    ] {
        let result = inference::infer(
            &CallRequest {
                arguments: vec![
                    runmat_types::ValueFact::unknown(
                        runmat_types::DynamicReason::RuntimeValue
                    );
                    argument_count
                ],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(requested),
            },
            &ODESET_CATALOG_ENTRY,
        );
        for effect in [
            EffectKind::FilesystemWrite,
            EffectKind::UserInterface,
            EffectKind::HostCallback,
        ] {
            assert_eq!(result.effects.0.contains(&effect), prints_possible);
        }
        assert!(result.effects.0.contains(&EffectKind::MayThrow));
    }
}
