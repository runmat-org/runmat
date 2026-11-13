use super::*;

#[test]
fn legacy_examples_retained_printed_output_and_plot_oracle() {
    let examples = NYQUIST_CATALOG_ENTRY.documentation.examples;
    for (id, printed, output) in [
        ("explicit", "fprintf(", "1.0 0.5 -0.5 2.0"),
        ("discrete", "fprintf(", "2.0 0.0"),
        ("plot", "fprintf(\"ok\\n\")", "ok"),
    ] {
        let example = examples
            .iter()
            .find(|example| example.id == id)
            .expect("legacy scenario");
        assert!(example.program.contains(printed), "{id}");
        assert_eq!(example.display_output, Some(output), "{id}");
    }
    assert!(NYQUIST_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == Some("RunMat:nyquist:PlotFailed")));
}
use runmat_types::*;
use std::collections::BTreeMap;

fn model() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
#[test]
fn output_facts_follow_requested_count_and_grid_length() {
    let w = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let result = crate::infer_catalog_call(&NYQUIST_CATALOG_ENTRY, &request(vec![model(), w], 3));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 3);
    for output in &result.outputs {
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );
        assert_eq!(output.shape, ShapeFact::from(vec![Some(3), Some(1)]));
        assert_eq!(output.residency, ResidencyFact::Host);
    }
    assert!(!result.effects.0.contains(&EffectKind::UserInterface));
    let plot = crate::infer_catalog_call(&NYQUIST_CATALOG_ENTRY, &request(vec![model()], 0));
    assert!(plot.outputs.is_empty());
    assert!(plot.effects.0.contains(&EffectKind::UserInterface));
}
#[test]
fn known_bad_model_frequency_and_output_count_are_diagnosed() {
    let bad_model = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    assert!(
        crate::infer_catalog_call(&NYQUIST_CATALOG_ENTRY, &request(vec![bad_model], 1))
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-NYQUIST-MODEL")
    );
    let bad_w = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    assert!(
        crate::infer_catalog_call(&NYQUIST_CATALOG_ENTRY, &request(vec![model(), bad_w], 1))
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-NYQUIST-FREQUENCY")
    );
    assert!(
        crate::infer_catalog_call(&NYQUIST_CATALOG_ENTRY, &request(vec![model()], 4))
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-NYQUIST-OUTPUTS")
    );
    assert!(crate::infer_catalog_call(
        &NYQUIST_CATALOG_ENTRY,
        &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1)
    )
    .diagnostics
    .is_empty());
}
