use super::*;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn call(arguments: Vec<ValueFact>, outputs: usize) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            literals: LiteralContext::new(Vec::new()),
            arguments,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &LSLINE_CATALOG_ENTRY,
    )
}

#[test]
fn no_output_and_handle_forms_keep_ui_effect_and_unknown_handle_count() {
    let no_output = call(vec![], 0);
    assert!(no_output.diagnostics.is_empty());
    assert!(no_output
        .effects
        .0
        .contains(&runmat_types::EffectKind::UserInterface));
    assert!(no_output.outputs.is_empty());
    let result = call(vec![], 1);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn invalid_axes_and_output_count_never_prove_extra_results() {
    let possible_axes = call(
        vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))],
        1,
    );
    assert!(possible_axes.diagnostics.is_empty());
    assert_eq!(
        possible_axes.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(possible_axes.outputs[0].shape, ShapeFact::Unknown);
    let vector = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let invalid = call(vec![vector], 1);
    assert!(!invalid.diagnostics.is_empty());
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));
    let extra = call(vec![], 2);
    assert!(!extra.diagnostics.is_empty());
    assert!(matches!(extra.outputs[1].kind, ValueKindFact::Unknown));
}
