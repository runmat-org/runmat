use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, count: usize) -> CallInference {
    crate::infer_catalog_call(
        &IMPULSE_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
    )
}

fn model() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn response_contract_is_host_double_and_plot_is_effectful() {
    let data = call(vec![model()], 2);
    assert!(data.diagnostics.is_empty(), "{:?}", data.diagnostics);
    for output in &data.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(100), Some(1)]));
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );
        assert_eq!(output.residency, ResidencyFact::Host);
    }
    assert!(!data.effects.0.contains(&EffectKind::UserInterface));
    assert!(call(vec![model()], 0)
        .effects
        .0
        .contains(&EffectKind::UserInterface));
}

#[test]
fn known_other_class_and_excess_inputs_are_diagnosed() {
    let mut other = model();
    let ValueKindFact::Object(object) = &mut other.kind else {
        unreachable!()
    };
    object.runtime_class = None;
    object.class = Some(ClassId(9));
    assert!(!call(vec![other], 1).diagnostics.is_empty());
    assert!(!call(
        vec![
            model(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::unknown(DynamicReason::RuntimeValue)
        ],
        1
    )
    .diagnostics
    .is_empty());
}
