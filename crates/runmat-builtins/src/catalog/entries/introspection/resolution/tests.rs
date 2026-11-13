use super::*;
use runmat_types::*;

#[test]
fn typed_option_tokens_match_runtime_flag_spelling() {
    assert_eq!(
        which_option_token(" \t-ALL \n"),
        Some(WhichOptionToken::All)
    );
    assert_eq!(
        which_option_token("-built-in"),
        Some(WhichOptionToken::Builtin)
    );
    assert_eq!(
        which_option_token("-variable"),
        Some(WhichOptionToken::Variable)
    );
    assert_eq!(which_option_token("-file"), Some(WhichOptionToken::File));
    assert_eq!(
        which_option_token(" -unknown"),
        Some(WhichOptionToken::Unrecognized)
    );
    assert_eq!(which_option_token("sin"), None);
}

#[test]
fn invalid_argument_role_uses_only_known_literal_prefix() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let invalid = ValueFact::scalar(ValueKindFact::Logical);
    let cases = [
        (
            vec![text.clone(), invalid.clone()],
            vec![LiteralValue::String("sin".into()), LiteralValue::Bool(true)],
            "RM.WHICH.OPTION_ARG_TYPE",
            1,
        ),
        (
            vec![text.clone(), invalid.clone()],
            vec![
                LiteralValue::String("-all".into()),
                LiteralValue::Bool(true),
            ],
            "RM.WHICH.NAME_ARG_TYPE",
            1,
        ),
        (
            vec![text.clone(), invalid.clone()],
            vec![LiteralValue::Unknown, LiteralValue::Bool(true)],
            "RM.WHICH.NAME_ARG_TYPE",
            1,
        ),
        (
            vec![text.clone(), text, invalid],
            vec![
                LiteralValue::String("-all".into()),
                LiteralValue::String("sin".into()),
                LiteralValue::Bool(true),
            ],
            "RM.WHICH.OPTION_ARG_TYPE",
            2,
        ),
    ];
    for (arguments, literals, code, index) in cases {
        let request = CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result = inference::infer_which(&request, &WHICH_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == code && diagnostic.argument == Some(index) }));
    }
}
#[test]
fn unknown_name_content_does_not_invent_a_resolved_path() {
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_which(&request, &WHICH_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
}
#[test]
fn which_preserves_dynamic_resolution_without_inventing_file() {
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_which(&request, &WHICH_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn which_all_remains_dynamic_when_option_is_source_known() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("-all".into()),
            LiteralValue::String("sin".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_which(&request, &WHICH_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn padded_case_variant_all_literal_remains_dynamic() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::Character(" \t-ALL \n".into()),
            LiteralValue::String("sin".into()),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_which(&request, &WHICH_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn aligned_unknown_option_does_not_prove_character_output() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        literals: LiteralContext::new(vec![
            LiteralValue::String("sin".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let result = inference::infer_which(&request, &WHICH_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}
