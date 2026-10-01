use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn numeric(rows: usize, cols: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}

#[test]
fn known_matrix_fit_has_a_typed_host_model_result() {
    let call = request(
        vec![numeric(4, 2), numeric(4, 1)],
        vec![LiteralValue::Unknown, LiteralValue::Unknown],
        1,
    );
    let result = inference::infer(&call, &FITLM_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 1);
    let ValueKindFact::Object(model) = &result.outputs[0].kind else {
        panic!("fitlm must infer a fitted model object");
    };
    assert_eq!(
        model.runtime_class,
        Some(runmat_types::standard::LINEAR_MODEL.owned())
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].storage, StorageFact::Opaque);
}

#[test]
fn unknown_data_still_has_a_known_successful_result_class() {
    let call = request(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        vec![LiteralValue::Unknown],
        1,
    );
    let result = inference::infer(&call, &FITLM_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Object(_)));
}

#[test]
fn invalid_known_data_and_unsupported_option_are_diagnosed() {
    let call = request(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::String("invalid".into())],
        1,
    );
    assert!(!inference::infer(&call, &FITLM_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    let call = request(
        vec![
            numeric(4, 1),
            numeric(4, 1),
            ValueFact::scalar(ValueKindFact::String),
            numeric(1, 1),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("RobustOpts".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(!inference::infer(&call, &FITLM_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}

#[test]
fn discarded_result_and_too_many_outputs_follow_fixed_contract() {
    let args = vec![numeric(4, 1), numeric(4, 1)];
    let none = request(args.clone(), vec![LiteralValue::Unknown; 2], 0);
    assert!(inference::infer(&none, &FITLM_CATALOG_ENTRY)
        .outputs
        .is_empty());
    let two = request(args, vec![LiteralValue::Unknown; 2], 2);
    assert!(!inference::infer(&two, &FITLM_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
