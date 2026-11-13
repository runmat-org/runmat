use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn scalar_collection_and_unknown_forms_keep_double_status_and_optional_text() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    for arguments in [
        vec![scalar],
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![],
    ] {
        let inferred = inference::infer(&request(arguments, 2), &FCLOSE_CATALOG_ENTRY);
        assert!(inferred.diagnostics.is_empty());
        assert!(matches!(
            inferred.outputs[0].kind,
            ValueKindFact::Numeric(_)
        ));
        assert!(matches!(inferred.outputs[1].kind, ValueKindFact::Character));
        assert!(inferred.effects.0.contains(&EffectKind::FilesystemWrite));
    }
}

#[test]
fn invalid_class_and_excess_output_are_diagnosed_without_erasing_effects() {
    let invalid = inference::infer(
        &request(
            vec![ValueFact::scalar(ValueKindFact::Struct(
                StructFact::scalar(Default::default(), false),
            ))],
            1,
        ),
        &FCLOSE_CATALOG_ENTRY,
    );
    assert!(!invalid.diagnostics.is_empty());
    assert!(invalid.effects.0.contains(&EffectKind::EnvironmentWrite));
    let too_many = inference::infer(
        &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 3),
        &FCLOSE_CATALOG_ENTRY,
    );
    assert!(!too_many.diagnostics.is_empty());
}
