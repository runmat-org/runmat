use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn model() -> ValueFact {
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

fn infer_tolerance(tolerance: ValueFact, literal: Option<LiteralValue>) -> CallInference {
    let literals = literal.map_or_else(LiteralContext::default, |literal| {
        LiteralContext::new(vec![LiteralValue::Unknown, literal])
    });
    crate::infer_catalog_call(
        &MINREAL_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![model(), tolerance],
            literals,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

#[test]
fn known_tolerance_requires_host_real_float_scalar_and_positive_literal() {
    let single = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    }));
    assert!(infer_tolerance(
        single.clone(),
        Some(LiteralValue::Real {
            text: "1e-4".into(),
            class: NumericClass::Single,
        }),
    )
    .diagnostics
    .is_empty());
    for (fact, literal) in [
        (single, Some(LiteralValue::Number(0.0))),
        (
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })),
            None,
        ),
        (
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(1), Some(2)]),
                StorageFact::Dense,
            ),
            None,
        ),
    ] {
        assert!(infer_tolerance(fact, literal)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-MINREAL-TOLERANCE"));
    }

    let mut stale_model = model();
    if let ValueKindFact::Object(object) = &mut stale_model.kind {
        object.properties.insert(
            "Numerator".into(),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(1), Some(3)]),
                StorageFact::Dense,
            ),
        );
        object.properties_complete = true;
    }
    let inferred = crate::infer_catalog_call(
        &MINREAL_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![stale_model],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert!(inferred.diagnostics.is_empty());
    let ValueKindFact::Object(result) = &inferred.outputs[0].kind else {
        panic!("minreal should infer a tf result");
    };
    assert_eq!(
        result.runtime_class,
        Some(standard::TRANSFER_FUNCTION.owned())
    );
    assert!(result.properties.is_empty());
    assert!(!result.properties_complete);
}
