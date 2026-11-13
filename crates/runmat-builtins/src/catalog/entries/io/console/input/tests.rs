use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

fn request(args: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments: args,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn literal_string_mode_has_character_output_and_invalid_flag_diagnoses() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let raw = inference::infer(
        &request(
            vec![text.clone(), text.clone()],
            vec![
                LiteralValue::String("Prompt: ".into()),
                LiteralValue::String("s".into()),
            ],
        ),
        &INPUT_CATALOG_ENTRY,
    );
    assert_eq!(raw.outputs[0].kind, ValueKindFact::Character);
    assert!(raw.diagnostics.is_empty());
    let bad = inference::infer(
        &request(
            vec![text.clone(), text],
            vec![
                LiteralValue::String("Prompt: ".into()),
                LiteralValue::String("x".into()),
            ],
        ),
        &INPUT_CATALOG_ENTRY,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-INPUT-FLAG"));
}

#[test]
fn paged_prompt_rejects_while_trailing_singleton_remains_possible() {
    let char_fact = |dims| {
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(dims),
            StorageFact::Dense,
        )
    };
    let bad = inference::infer(
        &request(vec![char_fact(vec![Some(1), Some(1), Some(2)])], vec![]),
        &INPUT_CATALOG_ENTRY,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-INPUT-PROMPT"));
    let good = inference::infer(
        &request(vec![char_fact(vec![Some(1), Some(2), Some(1)])], vec![]),
        &INPUT_CATALOG_ENTRY,
    );
    assert!(good.diagnostics.is_empty());
}
