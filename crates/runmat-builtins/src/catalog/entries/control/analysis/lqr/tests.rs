use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn matrix(rows: usize, columns: usize, class: NumericClass) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}
fn call(args: Vec<ValueFact>, outputs: RequestedOutputCount) -> CallInference {
    crate::infer_catalog_call(
        &LQR_CATALOG_ENTRY,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(outputs),
        },
    )
}
fn matrices() -> Vec<ValueFact> {
    vec![
        matrix(2, 2, NumericClass::Double),
        matrix(2, 1, NumericClass::Double),
        matrix(2, 2, NumericClass::Double),
        matrix(1, 1, NumericClass::Double),
    ]
}
fn ss() -> ValueFact {
    let mut properties = BTreeMap::new();
    properties.insert("A".into(), matrix(2, 2, NumericClass::Double));
    properties.insert("B".into(), matrix(2, 1, NumericClass::Double));
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::STATE_SPACE.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn matrix_and_ss_forms_infer_requested_host_output_shapes() {
    for args in [
        matrices(),
        vec![
            ss(),
            matrix(2, 2, NumericClass::Double),
            matrix(1, 1, NumericClass::Double),
        ],
    ] {
        let result = call(args, RequestedOutputCount::Exactly(3));
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 3);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(2)])
        );
        assert_eq!(
            result.outputs[1].shape,
            ShapeFact::from(vec![Some(2), Some(2)])
        );
        assert_eq!(
            result.outputs[2].shape,
            ShapeFact::from(vec![Some(2), Some(1)])
        );
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
        assert!(matches!(
            result.outputs[2].certainty,
            CertaintyFact::Dynamic(_)
        ));
        assert!(result.effects.0.contains(&EffectKind::MaySuspend));
    }

    // CARE and DLQR use different public output orders over the same
    // standard matrix geometry. Check the canonical entry dispatch as well
    // as the shared inference helper.
    for (entry, shapes) in [
        (&crate::CARE_CATALOG_ENTRY, [[2, 2], [2, 1], [1, 2]]),
        (&crate::DLQR_CATALOG_ENTRY, [[1, 2], [2, 2], [2, 1]]),
    ] {
        let result = crate::infer_catalog_call(
            entry,
            &CallRequest {
                arguments: matrices(),
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
            },
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 3);
        for (output, [rows, columns]) in result.outputs.iter().zip(shapes) {
            assert_eq!(
                output.shape,
                ShapeFact::from(vec![Some(rows), Some(columns)])
            );
            assert_eq!(output.residency, ResidencyFact::Host);
        }
    }
}

#[test]
fn one_state_pole_is_real_and_unknown_shape_does_not_invent_dimensions() {
    let result = call(
        vec![
            matrix(1, 1, NumericClass::Double),
            matrix(1, 1, NumericClass::Double),
            matrix(1, 1, NumericClass::Double),
            matrix(1, 1, NumericClass::Double),
        ],
        RequestedOutputCount::Exactly(3),
    );
    assert_eq!(
        result.outputs[2].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
    let unknown = call(
        vec![ValueFact::unknown(DynamicReason::RuntimeValue); 4],
        RequestedOutputCount::Exactly(2),
    );
    assert!(unknown.outputs[0]
        .shape
        .known_dims()
        .is_some_and(|dims| dims.iter().all(Option::is_none)));

    // A numeric scalar is an admitted 1-by-1 matrix in the shared runtime
    // request, including the CARE form with its default identity R.
    let scalar = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    for (entry, count) in [
        (&crate::CARE_CATALOG_ENTRY, 3),
        (&crate::DLQR_CATALOG_ENTRY, 4),
    ] {
        let result = crate::infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![scalar.clone(); count],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
            },
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 3);
        for output in &result.outputs {
            assert_eq!(output.shape, ShapeFact::from(vec![Some(1), Some(1)]));
            assert_eq!(output.residency, ResidencyFact::Host);
        }
        assert_eq!(
            result.outputs[1].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        );
    }
}

#[test]
fn invalid_dimensions_class_and_output_count_are_diagnosed() {
    let mut bad_q = matrices();
    bad_q[2] = matrix(3, 3, NumericClass::Double);
    assert!(call(bad_q, RequestedOutputCount::Exactly(1))
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-LQR-DIMENSIONS"));
    let mut bad_complex = matrices();
    bad_complex[1].kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    assert!(call(bad_complex, RequestedOutputCount::Exactly(1))
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-LQR-INPUT"));
    let mut unsupported = ss();
    let ValueKindFact::Object(ref mut object) = unsupported.kind else {
        unreachable!()
    };
    object.runtime_class = Some(standard::TRANSFER_FUNCTION.owned());
    let mut known_other_without_runtime_class = ss();
    let ValueKindFact::Object(ref mut other) = known_other_without_runtime_class.kind else {
        unreachable!()
    };
    other.runtime_class = None;
    other.class = Some(ClassId(7));
    assert!(call(
        vec![
            known_other_without_runtime_class,
            matrix(2, 2, NumericClass::Double),
            matrix(1, 1, NumericClass::Double)
        ],
        RequestedOutputCount::Exactly(1)
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-LQR-MODEL"));
    assert!(call(
        vec![
            unsupported,
            matrix(2, 2, NumericClass::Double),
            matrix(1, 1, NumericClass::Double)
        ],
        RequestedOutputCount::Exactly(1)
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-LQR-MODEL"));
    assert!(!call(matrices(), RequestedOutputCount::Exactly(4))
        .diagnostics
        .is_empty());
}

#[test]
fn zero_requested_outputs_still_validate_and_may_throw() {
    let result = call(
        vec![ValueFact::scalar(ValueKindFact::String)],
        RequestedOutputCount::Exactly(0),
    );
    assert!(result.outputs.is_empty());
    assert!(!result.diagnostics.is_empty());
    assert!(!result.effects.0.is_empty());
}
