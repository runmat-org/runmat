use super::*;
use runmat_types::*;

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}
fn number() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn contract_and_native_effects_are_inferred() {
    let positive = inference::infer(
        &request(
            vec![text(), text()],
            vec![
                LiteralValue::String("example.h5".into()),
                LiteralValue::Unknown,
            ],
            1,
        ),
        &HDF5READ_CATALOG_ENTRY,
    );
    assert!(
        positive.diagnostics.is_empty(),
        "{:?}",
        positive.diagnostics
    );
    assert_eq!(positive.outputs[0].residency, ResidencyFact::Host);
    assert!(matches!(
        positive.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::ExternalData)
    ));
    assert!(positive.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(positive
        .capabilities
        .0
        .contains(&CapabilityRequirement::NativeCode));
    assert_eq!(
        HDF5READ_CATALOG_ENTRY.placement.portability,
        BuiltinPortability::NativeOnly
    );
    assert_eq!(HDF5READ_CATALOG_ENTRY.bindings, &REQUIRED_DEFAULT_BINDING);
}

#[test]
fn malformed_arity_and_text_are_diagnosed_without_guessing_output() {
    let missing = inference::infer(&request(Vec::new(), Vec::new(), 1), &HDF5READ_CATALOG_ENTRY);
    assert!(!missing.diagnostics.is_empty());
    let mut arguments = vec![text(), text()];
    arguments[0] = number();
    let wrong_filename = inference::infer(
        &request(arguments, vec![LiteralValue::Unknown; 2], 1),
        &HDF5READ_CATALOG_ENTRY,
    );
    assert!(!wrong_filename.diagnostics.is_empty());
}
