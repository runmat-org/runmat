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
            vec![text()],
            vec![LiteralValue::String("example.h5".into())],
            1,
        ),
        &H5DISP_CATALOG_ENTRY,
    );
    assert!(
        positive.diagnostics.is_empty(),
        "{:?}",
        positive.diagnostics
    );
    assert_eq!(positive.outputs[0].kind, ValueKindFact::String);
    assert!(positive.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(!positive.effects.0.contains(&EffectKind::UserInterface));
    assert!(!positive.effects.0.contains(&EffectKind::HostCallback));
    assert!(positive
        .capabilities
        .0
        .contains(&CapabilityRequirement::NativeCode));
    assert_eq!(
        H5DISP_CATALOG_ENTRY.placement.portability,
        BuiltinPortability::NativeOnly
    );
    assert_eq!(H5DISP_CATALOG_ENTRY.bindings, &REQUIRED_DEFAULT_BINDING);
    assert_eq!(H5DISP_CATALOG_ENTRY.extensions, &[H5DISP_CAPTURE_EXTENSION]);
}

#[test]
fn statement_and_dynamic_output_contexts_keep_console_effects() {
    for requested in [
        RequestedOutputCount::Zero,
        RequestedOutputCount::Exactly(0),
        RequestedOutputCount::CurrentFunctionNargout,
        RequestedOutputCount::DestinationSequenceCardinality,
    ] {
        let mut call = request(
            vec![text()],
            vec![LiteralValue::String("example.h5".into())],
            0,
        );
        call.outputs = OutputSelection::new(requested);
        let inferred = inference::infer(&call, &H5DISP_CATALOG_ENTRY);
        assert!(inferred.effects.0.contains(&EffectKind::UserInterface));
        assert!(inferred.effects.0.contains(&EffectKind::HostCallback));
        assert!(inferred.effects.0.contains(&EffectKind::FilesystemRead));
    }
    assert_eq!(H5DISP_CATALOG_ENTRY.contract.purity, BuiltinPurity::Impure);
}

#[test]
fn malformed_arity_and_text_are_diagnosed_without_guessing_output() {
    let missing = inference::infer(&request(Vec::new(), Vec::new(), 1), &H5DISP_CATALOG_ENTRY);
    assert!(!missing.diagnostics.is_empty());
    let mut arguments = vec![text()];
    arguments[0] = number();
    let wrong_filename = inference::infer(
        &request(arguments, vec![LiteralValue::Unknown; 1], 1),
        &H5DISP_CATALOG_ENTRY,
    );
    assert!(!wrong_filename.diagnostics.is_empty());
}

#[test]
fn documented_mode_and_utf8_forms_keep_known_controls_useful() {
    for literals in [
        vec![
            LiteralValue::String("sample.h5".into()),
            LiteralValue::String("/data".into()),
            LiteralValue::String("min".into()),
        ],
        vec![
            LiteralValue::String("sample.h5".into()),
            LiteralValue::String("TextEncoding".into()),
            LiteralValue::String("UTF-8".into()),
        ],
        vec![
            LiteralValue::String("sample.h5".into()),
            LiteralValue::String("/data".into()),
            LiteralValue::String("simple".into()),
            LiteralValue::String("TextEncoding".into()),
            LiteralValue::String("UTF-8".into()),
        ],
    ] {
        let inferred = inference::infer(
            &request(vec![text(); literals.len()], literals, 1),
            &H5DISP_CATALOG_ENTRY,
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::String);
    }
    for literals in [
        vec![
            LiteralValue::String("sample.h5".into()),
            LiteralValue::String("/data".into()),
            LiteralValue::String("full".into()),
        ],
        vec![
            LiteralValue::String("sample.h5".into()),
            LiteralValue::String("TextEncoding".into()),
            LiteralValue::String("Latin-1".into()),
        ],
    ] {
        let inferred = inference::infer(
            &request(vec![text(); literals.len()], literals, 1),
            &H5DISP_CATALOG_ENTRY,
        );
        assert!(inferred
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-HDF5-DISPLAY"));
    }
    let unknown = inference::infer(
        &request(vec![text(); 3], vec![LiteralValue::Unknown; 3], 1),
        &H5DISP_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
}
