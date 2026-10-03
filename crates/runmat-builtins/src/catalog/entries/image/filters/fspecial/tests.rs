use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, OutputSelection,
    RequestedOutputCount, ShapeFact, ValueFact, ValueKindFact,
};

#[test]
fn optional_resident_route_declares_its_transitive_suspension_effect() {
    assert_eq!(
        FSPECIAL_CATALOG_ENTRY.contract.async_behavior,
        BuiltinAsyncBehavior::MaySuspend
    );
    assert!(FSPECIAL_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::EnvironmentRead));
    assert!(FSPECIAL_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::MaySuspend));
    assert!(FSPECIAL_CATALOG_ENTRY
        .contract
        .effects
        .contains(&EffectKind::MayThrow));
}

fn call(
    arguments: Vec<LiteralValue>,
    output_count: RequestedOutputCount,
) -> runmat_types::CallInference {
    let facts = arguments
        .iter()
        .enumerate()
        .map(|(index, _)| {
            ValueFact::scalar(if index == 0 {
                ValueKindFact::String
            } else {
                ValueKindFact::Unknown
            })
        })
        .collect();
    inference::infer(
        &CallRequest {
            arguments: facts,
            literals: LiteralContext::new(arguments),
            outputs: OutputSelection::new(output_count),
        },
        &FSPECIAL_CATALOG_ENTRY,
    )
}

#[test]
fn fixed_default_and_literal_kernel_shapes_preserve_dynamic_presentation() {
    for (arguments, shape) in [
        (
            vec![LiteralValue::String("average".into())],
            vec![Some(3), Some(3)],
        ),
        (
            vec![LiteralValue::String("LOG".into())],
            vec![Some(5), Some(5)],
        ),
        (
            vec![
                LiteralValue::String("motion".into()),
                LiteralValue::Integer {
                    text: "14".into(),
                    class: runmat_types::NumericClass::UInt64,
                },
            ],
            vec![Some(15), Some(15)],
        ),
    ] {
        let inferred = call(arguments, RequestedOutputCount::One);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs[0].shape, ShapeFact::from(shape));
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(
            inferred.outputs[0].certainty,
            runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
        );
    }
}

#[test]
fn known_invalid_form_and_output_count_have_stable_diagnostics() {
    let invalid_form = call(
        vec![
            LiteralValue::String("sobel".into()),
            LiteralValue::Number(3.0),
        ],
        RequestedOutputCount::One,
    );
    assert!(invalid_form
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-FSPECIAL-FORM"));
    let invalid_outputs = call(
        vec![LiteralValue::String("average".into())],
        RequestedOutputCount::Exactly(2),
    );
    assert!(invalid_outputs
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-FSPECIAL-OUTPUTS"));
}
