use super::*;
use runmat_types::*;

fn infer(entry: &BuiltinCatalogEntry, arguments: Vec<ValueFact>, outputs: usize) -> CallInference {
    let request = CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    };
    match entry.contract.inference_rule {
        BuiltinInferenceRule::Identity(rule) => rule.infer(&request, entry),
        other => panic!("expected identity inference, got {other:?}"),
    }
}

#[test]
fn constructor_produces_host_handle_identity_for_known_and_unknown_inputs() {
    for args in [
        vec![],
        vec![ValueFact::unknown(DynamicReason::RuntimeValue); 2],
    ] {
        let result = infer(&constructor::ENTRY, args, 1);
        assert!(result.diagnostics.is_empty());
        let output = &result.outputs[0];
        assert_eq!(output.residency, ResidencyFact::Host);
        assert_eq!(output.storage, StorageFact::Opaque);
        assert_eq!(output.alias, AliasFact::Identity);
        assert_eq!(output.mutation, MutationFact::HandleSemantics);
        let ValueKindFact::Object(object) = &output.kind else {
            panic!("expected object");
        };
        assert_eq!(
            object.runtime_class,
            Some(StaticClassIdentity::new("containers.Map").owned())
        );
        assert_eq!(object.handle_semantics, Some(true));
    }
    assert!(!infer(
        &constructor::ENTRY,
        vec![ValueFact::scalar(ValueKindFact::String)],
        1
    )
    .diagnostics
    .is_empty());
}

#[test]
fn methods_preserve_output_count_and_handle_effects() {
    let receiver = inference::map_fact();
    let keys = infer(&keys::ENTRY, vec![receiver.clone()], 1);
    assert_eq!(keys.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
    assert!(matches!(keys.outputs[0].kind, ValueKindFact::Cell(_)));

    let character_key = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let membership = infer(&is_key::ENTRY, vec![receiver.clone(), character_key], 1);
    assert_eq!(membership.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(membership.outputs[0].kind, ValueKindFact::Logical);

    let multirow_key = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let rejected = infer(&is_key::ENTRY, vec![receiver.clone(), multirow_key], 1);
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CONTAINERS-MAP-SCALAR-KEY"));
    assert_eq!(rejected.outputs[0].shape, ShapeFact::Unknown);
    assert!(matches!(
        rejected.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));

    let unknown_rows = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![None, Some(3)]),
        StorageFact::Dense,
    );
    let uncertain = infer(&is_key::ENTRY, vec![receiver.clone(), unknown_rows], 1);
    assert_eq!(uncertain.outputs[0].shape, ShapeFact::Unknown);
    assert!(matches!(
        uncertain.outputs[0].certainty,
        CertaintyFact::Dynamic(_)
    ));

    let selected = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::Character)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let values = infer(&values::ENTRY, vec![receiver.clone(), selected], 1);
    assert_eq!(
        values.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );

    let removed = infer(
        &remove::ENTRY,
        vec![receiver.clone(), ValueFact::scalar(ValueKindFact::String)],
        0,
    );
    assert!(removed.outputs.is_empty());
    assert!(removed.effects.0.contains(&EffectKind::MaySuspend));
    assert!(removed.effects.0.contains(&EffectKind::MayThrow));
    assert!(!infer(
        &remove::ENTRY,
        vec![receiver.clone(), ValueFact::scalar(ValueKindFact::String)],
        1,
    )
    .diagnostics
    .is_empty());

    let assigned = infer(
        &subsasgn::ENTRY,
        vec![
            receiver.clone(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        1,
    );
    assert_eq!(assigned.outputs[0].alias, AliasFact::Identity);
    assert_eq!(assigned.outputs[0].mutation, MutationFact::HandleSemantics);

    let bad_receiver = infer(
        &keys::ENTRY,
        vec![ValueFact::scalar(ValueKindFact::Logical)],
        1,
    );
    assert!(!bad_receiver.diagnostics.is_empty());
}
