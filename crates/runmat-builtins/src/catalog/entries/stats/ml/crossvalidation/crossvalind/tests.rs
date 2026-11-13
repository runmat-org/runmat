use super::*;
use runmat_types::{
    CallRequest, EffectKind, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn numeric() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    )
}

fn request(method: &str, value: Option<LiteralValue>, outputs: usize) -> CallRequest {
    let mut arguments = vec![ValueFact::scalar(ValueKindFact::String), numeric()];
    let mut literals = vec![
        LiteralValue::String(method.into()),
        LiteralValue::Integer {
            text: "8".into(),
            class: NumericClass::UInt64,
        },
    ];
    if let Some(value) = value {
        arguments.push(numeric());
        literals.push(value);
    }
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn known_methods_select_numeric_or_logical_output_and_rng_effect() {
    let folds = inference::infer(
        &request("KFold", Some(LiteralValue::Number(2.0)), 1),
        &CROSSVALIND_CATALOG_ENTRY,
    );
    assert!(folds.diagnostics.is_empty(), "{:?}", folds.diagnostics);
    assert_eq!(
        folds.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        folds.outputs[0].shape,
        ShapeFact::from(vec![Some(8), Some(1)])
    );
    assert!(folds.effects.0.contains(&EffectKind::Randomness));

    let holdout = inference::infer(
        &request("HoldOut", Some(LiteralValue::Number(0.5)), 2),
        &CROSSVALIND_CATALOG_ENTRY,
    );
    assert!(holdout.diagnostics.is_empty(), "{:?}", holdout.diagnostics);
    assert!(matches!(holdout.outputs[0].kind, ValueKindFact::Logical));
    assert!(matches!(holdout.outputs[1].kind, ValueKindFact::Logical));

    let resub = inference::infer(
        &request("Resubstitution", None, 2),
        &CROSSVALIND_CATALOG_ENTRY,
    );
    assert!(resub.diagnostics.is_empty(), "{:?}", resub.diagnostics);
    assert!(!resub.effects.0.contains(&EffectKind::Randomness));
}

#[test]
fn logical_scalar_count_matches_runtime_admission() {
    let mut call = request("Resubstitution", None, 1);
    call.arguments[1] = ValueFact::scalar(ValueKindFact::Logical);
    call.literals.literal_args[1] = LiteralValue::Bool(true);
    let result = inference::infer(&call, &CROSSVALIND_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
}

#[test]
fn invalid_known_controls_and_output_count_do_not_prove_facts() {
    let invalid_fold = inference::infer(
        &request("KFold", Some(LiteralValue::Number(9.0)), 1),
        &CROSSVALIND_CATALOG_ENTRY,
    );
    assert!(!invalid_fold.diagnostics.is_empty());
    assert!(matches!(
        invalid_fold.outputs[0].kind,
        ValueKindFact::Unknown
    ));
    let invalid_outputs = inference::infer(
        &request("KFold", Some(LiteralValue::Number(2.0)), 2),
        &CROSSVALIND_CATALOG_ENTRY,
    );
    assert!(!invalid_outputs.diagnostics.is_empty());
    let invalid_method = inference::infer(
        &request("LeaveOut", Some(LiteralValue::Number(2.0)), 1),
        &CROSSVALIND_CATALOG_ENTRY,
    );
    assert!(!invalid_method.diagnostics.is_empty());
}

#[test]
fn class_length_and_one_observation_holdout_are_rejected_statically() {
    let mut class_call = request("KFold", Some(LiteralValue::Number(2.0)), 1);
    class_call.arguments.extend([
        ValueFact::scalar(ValueKindFact::Character),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(3), Some(1)]),
            StorageFact::Dense,
        ),
    ]);
    class_call.literals.literal_args.extend([
        LiteralValue::Character("Classes".into()),
        LiteralValue::Unknown,
    ]);
    let result = inference::infer(&class_call, &CROSSVALIND_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));

    let mut one_observation = request("HoldOut", Some(LiteralValue::Number(0.5)), 2);
    one_observation.literals.literal_args[1] = LiteralValue::Number(1.0);
    let result = inference::infer(&one_observation, &CROSSVALIND_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Unknown));
}
