use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn tf_object() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TRANSFER_FUNCTION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
fn infer(args: Vec<ValueFact>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &DCGAIN_CATALOG_ENTRY,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn tf_gain_is_host_scalar_but_result_class_remains_dynamic() {
    let result = infer(vec![tf_object()], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].storage, StorageFact::Scalar);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert!(result.effects.0.contains(&EffectKind::MayThrow));
    assert!(result.effects.0.contains(&EffectKind::MaySuspend));
}

#[test]
fn invalid_models_arity_and_outputs_are_diagnosed() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let mut other = tf_object();
    let ValueKindFact::Object(ref mut object) = other.kind else {
        unreachable!()
    };
    object.runtime_class = Some(standard::STATE_SPACE.owned());
    for input in [
        numeric,
        other,
        ValueFact::proven(
            tf_object().kind,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Opaque,
        ),
    ] {
        assert!(infer(vec![input], 1)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-DCGAIN-MODEL"));
    }
    assert!(infer(vec![], 1)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DCGAIN-ARITY"));
    assert!(infer(vec![tf_object(), tf_object()], 1)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-DCGAIN-ARITY"));
    assert!(infer(vec![tf_object()], 0).outputs.is_empty());
    assert!(!infer(vec![tf_object()], 2).diagnostics.is_empty());
}
