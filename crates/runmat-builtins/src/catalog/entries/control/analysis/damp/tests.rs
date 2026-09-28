use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn model(class: StaticClassIdentity, properties: BTreeMap<String, ValueFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn call(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> CallInference {
    crate::infer_catalog_call(
        &DAMP_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(outputs),
        },
    )
}

fn double_real() -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    })
}

#[test]
fn state_space_matrix_size_sets_all_column_lengths_without_guessing_pole_domain() {
    let mut properties = BTreeMap::new();
    properties.insert(
        "A".into(),
        ValueFact::proven(
            double_real(),
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        ),
    );
    let inferred = call(
        vec![model(standard::STATE_SPACE, properties)],
        RequestedOutputCount::Exactly(3),
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs.len(), 3);
    for output in &inferred.outputs {
        assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(1)]));
        assert_eq!(output.residency, ResidencyFact::Host);
        assert_eq!(output.storage, StorageFact::Dense);
    }
    assert_eq!(inferred.outputs[0].kind, double_real());
    assert_eq!(inferred.outputs[1].kind, double_real());
    assert_eq!(inferred.outputs[2].kind, ValueKindFact::Unknown);
    assert!(matches!(
        inferred.outputs[2].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    ));
    assert!(inferred.effects.0.contains(&EffectKind::MayThrow));
    assert!(inferred.effects.0.contains(&EffectKind::MaySuspend));
}

#[test]
fn static_gain_proves_empty_real_columns_but_unknown_model_does_not() {
    let mut properties = BTreeMap::new();
    properties.insert("Denominator".into(), ValueFact::scalar(double_real()));
    let known = call(
        vec![model(standard::TRANSFER_FUNCTION, properties)],
        RequestedOutputCount::Exactly(3),
    );
    assert!(known.diagnostics.is_empty(), "{:?}", known.diagnostics);
    assert_eq!(
        known.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(1)])
    );
    assert_eq!(known.outputs[2].kind, double_real());

    let unknown = call(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
        RequestedOutputCount::Exactly(3),
    );
    assert!(unknown.diagnostics.is_empty(), "{:?}", unknown.diagnostics);
    assert_eq!(
        unknown.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert_eq!(unknown.outputs[2].kind, ValueKindFact::Unknown);
}

#[test]
fn requested_output_count_and_discard_are_precise() {
    let sys = model(standard::TRANSFER_FUNCTION, BTreeMap::new());
    for (count, expected) in [(0, 0), (1, 1), (2, 2), (3, 3)] {
        let result = call(vec![sys.clone()], RequestedOutputCount::Exactly(count));
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), expected);
    }
    let excessive = call(vec![sys.clone()], RequestedOutputCount::Exactly(4));
    assert!(excessive
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-DAMP-OUTPUTS"));
    let dynamic = call(vec![sys], RequestedOutputCount::CurrentFunctionNargout);
    assert!(dynamic.dynamic_outputs);
    assert_eq!(dynamic.outputs.len(), 3);
}

#[test]
fn numeric_wrong_class_and_arity_are_diagnosed() {
    for value in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })),
        model(standard::DATETIME, BTreeMap::new()),
    ] {
        let result = call(vec![value], RequestedOutputCount::One);
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-DAMP-MODEL"));
    }
    for arguments in [
        vec![],
        vec![model(standard::STATE_SPACE, BTreeMap::new()); 2],
    ] {
        assert!(call(arguments, RequestedOutputCount::One)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-DAMP-ARITY"));
    }
}
