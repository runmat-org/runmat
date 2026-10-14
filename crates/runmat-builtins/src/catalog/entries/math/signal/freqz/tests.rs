use super::*;
use runmat_types::*;

fn coefficient(class: NumericClass) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    )
}

fn infer(n: LiteralValue, outputs: RequestedOutputCount) -> CallInference {
    crate::infer_catalog_call(
        &FREQZ_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![
                coefficient(NumericClass::Single),
                coefficient(NumericClass::Double),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Unknown, n]),
            outputs: OutputSelection::new(outputs),
        },
    )
}

#[test]
fn response_precision_shape_and_frequency_class_follow_adapter() {
    let result = infer(LiteralValue::Number(8.0), RequestedOutputCount::Exactly(2));
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs.len(), 2);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(8), Some(1)])
    );
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex
        })
    );
    assert_eq!(
        result.outputs[1].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
    assert!(!result.effects.0.contains(&EffectKind::UserInterface));
    assert!(!result
        .capabilities
        .0
        .contains(&CapabilityRequirement::UserInterface));
}

#[test]
fn zero_output_plot_requires_ui_but_positive_outputs_do_not() {
    let plotted = infer(LiteralValue::Number(8.0), RequestedOutputCount::Exactly(0));
    assert!(plotted.effects.0.contains(&EffectKind::UserInterface));
    assert!(plotted
        .capabilities
        .0
        .contains(&CapabilityRequirement::UserInterface));
    assert!(plotted.outputs.is_empty());
    let unknown = infer(LiteralValue::Unknown, RequestedOutputCount::Exactly(2));
    assert_eq!(
        unknown.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
}

#[test]
fn invalid_point_count_and_empty_default_are_distinct() {
    let invalid = infer(LiteralValue::Number(1.0), RequestedOutputCount::One);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FREQZ-N"));
    let default = infer(LiteralValue::Empty, RequestedOutputCount::One);
    assert!(default.diagnostics.is_empty());
    assert_eq!(
        default.outputs[0].shape,
        ShapeFact::from(vec![Some(512), Some(1)])
    );
}

#[test]
fn ranked_coefficient_fact_does_not_allocate_rank_sized_geometry() {
    let mut b = coefficient(NumericClass::Double);
    b.shape = ShapeFact::Ranked { rank: usize::MAX };
    let result = crate::infer_catalog_call(
        &FREQZ_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![b, coefficient(NumericClass::Double)],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(result.diagnostics.is_empty());
}
