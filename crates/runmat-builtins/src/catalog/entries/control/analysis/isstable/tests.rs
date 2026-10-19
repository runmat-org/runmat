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

fn infer(arguments: Vec<ValueFact>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &ISSTABLE_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn known_tf_and_unknown_model_have_host_logical_scalar_result() {
    for model in [tf_object(), ValueFact::unknown(DynamicReason::RuntimeValue)] {
        let result = infer(vec![model], 1);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        assert!(result.effects.0.contains(&EffectKind::MayThrow));
        assert!(result.effects.0.contains(&EffectKind::MaySuspend));
    }
}

#[test]
fn wrong_class_numeric_and_nonscalar_models_are_diagnosed() {
    let mut other_class = tf_object();
    let ValueKindFact::Object(ref mut object) = other_class.kind else {
        unreachable!();
    };
    object.runtime_class = Some(standard::STATE_SPACE.owned());
    for input in [
        other_class,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })),
        ValueFact::proven(
            tf_object().kind,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Opaque,
        ),
    ] {
        let result = infer(vec![input], 1);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-ISSTABLE-MODEL"));
    }
}

#[test]
fn arity_and_requested_outputs_are_checked() {
    for arguments in [vec![], vec![tf_object(), tf_object()]] {
        assert!(infer(arguments, 1)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-ISSTABLE-ARITY"));
    }
    let discarded = infer(vec![tf_object()], 0);
    assert!(discarded.outputs.is_empty());
    let excessive = infer(vec![tf_object()], 2);
    assert!(!excessive.diagnostics.is_empty());
}
