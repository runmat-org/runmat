use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount, ValueFact,
    ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(outputs),
    }
}

#[test]
fn assertion_contract_has_no_public_output_and_one_runtime_binding() {
    assert_eq!(ASSERT_CATALOG_ENTRY.bindings.len(), 1);
    assert!(ASSERT_DESCRIPTOR
        .signatures
        .iter()
        .all(|form| form.outputs.is_empty()));
    assert_eq!(ASSERT_EXTENSIONS.len(), 2);
    assert_eq!(ASSERT_INTEGER_CAPABILITIES.len(), 3);
}

#[test]
fn assertion_inference_tracks_arity_condition_and_output_count() {
    let known = request(
        vec![ValueFact::scalar(ValueKindFact::Logical)],
        RequestedOutputCount::Zero,
    );
    let result = inference::infer(&known, &ASSERT_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert!(result.outputs.is_empty());

    let unknown = request(
        vec![ValueFact::unknown(runmat_types::DynamicReason::Unspecified)],
        RequestedOutputCount::Zero,
    );
    assert!(inference::infer(&unknown, &ASSERT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    let invalid = request(
        vec![ValueFact::scalar(ValueKindFact::Exception(
            runmat_types::ExceptionFact { identifier: None },
        ))],
        RequestedOutputCount::Zero,
    );
    assert!(!inference::infer(&invalid, &ASSERT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    let string_condition = request(
        vec![ValueFact::scalar(ValueKindFact::String)],
        RequestedOutputCount::Zero,
    );
    assert!(inference::infer(&string_condition, &ASSERT_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ASSERT-CONDITION"));

    let missing = request(vec![], RequestedOutputCount::Zero);
    assert!(!inference::infer(&missing, &ASSERT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    let requested = request(
        vec![ValueFact::scalar(ValueKindFact::Logical)],
        RequestedOutputCount::One,
    );
    assert!(!inference::infer(&requested, &ASSERT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn assertion_failure_payload_inference_uses_only_known_false_condition() {
    let mut request = request(
        vec![
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            })),
        ],
        RequestedOutputCount::Zero,
    );
    request.arguments[3].shape = runmat_types::ShapeFact::from(vec![Some(1), Some(2)]);
    request.literals.literal_args = vec![
        LiteralValue::Bool(false),
        LiteralValue::String("bad.segment:9bad".into()),
        LiteralValue::String("failure %d".into()),
        LiteralValue::Unknown,
    ];
    let failed = inference::infer(&request, &ASSERT_CATALOG_ENTRY);
    assert!(failed
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ASSERT-IDENTIFIER"));
    assert!(failed
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ASSERT-REPLACEMENT"));

    request.literals.literal_args[0] = LiteralValue::Bool(true);
    assert!(inference::infer(&request, &ASSERT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    request.literals.literal_args[0] = LiteralValue::Unknown;
    assert!(inference::infer(&request, &ASSERT_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    request.literals.literal_args[0] = LiteralValue::Bool(false);
    request.arguments[2] = ValueFact::scalar(ValueKindFact::Logical);
    assert!(inference::infer(&request, &ASSERT_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ASSERT-MESSAGE"));

    request.arguments[2] = ValueFact::scalar(ValueKindFact::String);
    request.literals.literal_args[1] = LiteralValue::String(" good_2 ".into());
    assert!(!inference::infer(&request, &ASSERT_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ASSERT-REPLACEMENT"));
}
