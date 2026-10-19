use super::*;
use runmat_types::{
    CallRequest, ExceptionFact, LiteralContext, LiteralValue, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> CallRequest {
    CallRequest {
        literals: LiteralContext::new(vec![LiteralValue::Unknown; arguments.len()]),
        arguments,
        outputs: OutputSelection::new(outputs),
    }
}

#[test]
fn error_and_rethrow_bindings_and_no_output_forms() {
    assert_eq!(ERROR_CATALOG_ENTRY.bindings.len(), 1);
    assert_eq!(RETHROW_CATALOG_ENTRY.bindings.len(), 1);
    assert!(ERROR_DESCRIPTOR
        .signatures
        .iter()
        .all(|form| form.outputs.is_empty()));
    assert!(RETHROW_DESCRIPTOR
        .signatures
        .iter()
        .all(|form| form.outputs.is_empty()));
    assert_eq!(ERROR_EXTENSIONS.len(), 3);
    assert_eq!(ERROR_INTEGER_CAPABILITIES.len(), 2);
}

#[test]
fn error_inference_checks_arity_and_requested_outputs() {
    let missing = request(vec![], RequestedOutputCount::Zero);
    assert!(!inference::error(&missing, &ERROR_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    let text = request(
        vec![ValueFact::scalar(ValueKindFact::String)],
        RequestedOutputCount::Zero,
    );
    assert!(inference::error(&text, &ERROR_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    let requested = request(
        vec![ValueFact::scalar(ValueKindFact::String)],
        RequestedOutputCount::One,
    );
    assert!(!inference::error(&requested, &ERROR_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    let mut exception_extra = request(
        vec![
            ValueFact::scalar(ValueKindFact::Exception(ExceptionFact { identifier: None })),
            ValueFact::scalar(ValueKindFact::String),
        ],
        RequestedOutputCount::Zero,
    );
    assert!(inference::error(&exception_extra, &ERROR_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ERROR-PAYLOAD-ARITY"));

    exception_extra.arguments[0] = ValueFact::scalar(ValueKindFact::String);
    exception_extra.arguments[1].shape = runmat_types::ShapeFact::from(vec![Some(1), Some(2)]);
    exception_extra.literals.literal_args[0] = LiteralValue::String("Acme:badInput".into());
    assert!(inference::error(&exception_extra, &ERROR_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-ERROR-MESSAGE"));
}

#[test]
fn rethrow_inference_accepts_exception_and_rejects_numeric() {
    let exception = request(
        vec![ValueFact::scalar(ValueKindFact::Exception(ExceptionFact {
            identifier: None,
        }))],
        RequestedOutputCount::Zero,
    );
    assert!(inference::rethrow(&exception, &RETHROW_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let numeric = request(
        vec![ValueFact::scalar(ValueKindFact::Numeric(
            runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            },
        ))],
        RequestedOutputCount::Zero,
    );
    assert!(!inference::rethrow(&numeric, &RETHROW_CATALOG_ENTRY)
        .diagnostics
        .is_empty());

    let mut string_array = request(
        vec![ValueFact::scalar(ValueKindFact::String)],
        RequestedOutputCount::Zero,
    );
    string_array.arguments[0].shape = runmat_types::ShapeFact::from(vec![Some(1), Some(2)]);
    assert!(inference::rethrow(&string_array, &RETHROW_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-RETHROW-PAYLOAD"));
}
