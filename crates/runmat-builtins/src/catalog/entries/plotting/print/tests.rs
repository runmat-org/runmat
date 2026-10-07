use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn print_controls_share_runtime_decoder() {
    assert_eq!(
        PrintControl::decode("-DPNG").unwrap(),
        PrintControl::Device(PrintDevice::Png)
    );
    assert_eq!(
        PrintControl::decode("-r0").unwrap(),
        PrintControl::Resolution(PRINT_DEFAULT_DPI)
    );
    assert_eq!(
        PrintControl::decode("-r300").unwrap(),
        PrintControl::Resolution(300)
    );
    assert_eq!(print_dimensions(300).unwrap(), (1600, 1200));
    assert!(matches!(
        PrintControl::decode("-dsvg"),
        Err(PrintControlError::UnsupportedDevice)
    ));
    assert!(matches!(
        PrintControl::decode("-r99999"),
        Err(PrintControlError::UnsupportedResolution(_))
    ));
    assert!(matches!(
        PrintControl::decode("-rwat"),
        Err(PrintControlError::ResolutionParse(_))
    ));
}

#[test]
fn print_inference_preserves_statement_and_status_forms() {
    let text = ValueFact::scalar(ValueKindFact::Character);
    for count in [0, 1, 2] {
        let result = inference::infer(
            &request(
                vec![text.clone()],
                vec![LiteralValue::String("plot.png".into())],
                count,
            ),
            &PRINT_CATALOG_ENTRY,
        );
        assert_eq!(!result.diagnostics.is_empty(), count > 1);
        assert_eq!(result.outputs.len(), count);
        if count > 0 {
            assert_eq!(result.outputs[0], ValueFact::scalar(ValueKindFact::Logical));
        }
        if count > 1 {
            assert_eq!(
                result.outputs[1],
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            );
        }
        assert!(result.effects.0.contains(&EffectKind::FilesystemWrite));
        assert!(result.effects.0.contains(&EffectKind::UserInterface));
    }
    assert_eq!(
        PRINT_STATUS_OUTPUT_EXTENSION.mode,
        BuiltinExtensionMode::RunMatOnly
    );
}

#[test]
fn print_inference_rejects_known_invalid_and_preserves_unknown() {
    let text = ValueFact::scalar(ValueKindFact::Character);
    let bad = inference::infer(
        &request(
            vec![text.clone(), text.clone()],
            vec![
                LiteralValue::String("plot.png".into()),
                LiteralValue::String("-dsvg".into()),
            ],
            0,
        ),
        &PRINT_CATALOG_ENTRY,
    );
    assert!(!bad.diagnostics.is_empty());
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(
        inference::infer(&request(vec![unknown], vec![], 0), &PRINT_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    let duplicate = inference::infer(
        &request(
            vec![text.clone(), text],
            vec![
                LiteralValue::String("a.png".into()),
                LiteralValue::String("b.png".into()),
            ],
            0,
        ),
        &PRINT_CATALOG_ENTRY,
    );
    assert!(!duplicate.diagnostics.is_empty());
    let paged_char = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(4), Some(2)]),
        StorageFact::Dense,
    );
    assert!(
        inference::infer(&request(vec![paged_char], vec![], 0), &PRINT_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}

#[test]
fn repeated_resolution_is_not_a_static_error() {
    let text = ValueFact::scalar(ValueKindFact::Character);
    let result = inference::infer(
        &request(
            vec![text.clone(), text.clone(), text],
            vec![
                LiteralValue::String("plot.png".into()),
                LiteralValue::String("-r75".into()),
                LiteralValue::String("-r300".into()),
            ],
            0,
        ),
        &PRINT_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
}

#[test]
fn print_documentation_conserves_historical_examples_and_links() {
    assert_eq!(PRINT_CATALOG_ENTRY.documentation.examples.len(), 5);
    assert_eq!(PRINT_CATALOG_ENTRY.documentation.faqs.len(), 3);
    assert_eq!(PRINT_CATALOG_ENTRY.documentation.links.len(), 5);
    assert_eq!(PRINT_CATALOG_ENTRY.bindings[0].variant, "default");
}
