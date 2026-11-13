use super::*;
use runmat_types::{
    DimensionFact, NumericClass, NumericDomain, NumericFact, ShapeFact, ValueKindFact,
};

#[test]
fn pure_numeric_fact_route_matches_public_inference() {
    let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    for (entry, arguments) in [
        (
            &crate::PLUS_CATALOG_ENTRY,
            vec![input.clone(), input.clone()],
        ),
        (&crate::SIN_CATALOG_ENTRY, vec![input.clone()]),
        (&crate::MEAN_CATALOG_ENTRY, vec![input.clone()]),
        (&crate::SINGLE_CATALOG_ENTRY, vec![input.clone()]),
        (&crate::SQRT_CATALOG_ENTRY, vec![input.clone()]),
        (
            &crate::MAX_CATALOG_ENTRY,
            vec![input.clone(), input.clone()],
        ),
    ] {
        assert!(finite_fusion_fact_route(entry, arguments.len()));
        let request = CallRequest {
            arguments,
            literals: Default::default(),
            outputs: Default::default(),
        };
        let public = infer(entry, &request);
        assert!(public.diagnostics.is_empty());
        assert_eq!(public.outputs.len(), 1);
        assert_eq!(
            infer_fusion_numeric_fact(entry, &request).unwrap().unwrap(),
            public.outputs[0]
        );
    }
}

#[test]
fn finite_numeric_fact_rejects_incompatible_broadcast_without_changing_public_error() {
    let shaped = |columns| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(2), DimensionFact::Known(columns)],
            },
            runmat_types::StorageFact::Dense,
        )
    };
    let request = CallRequest {
        arguments: vec![shaped(3), shaped(4)],
        literals: Default::default(),
        outputs: Default::default(),
    };
    let public = infer(&crate::PLUS_CATALOG_ENTRY, &request);
    assert_eq!(public.diagnostics.len(), 1);
    assert_eq!(public.diagnostics[0].code, "RM-TYPE-BROADCAST");
    assert!(
        infer_fusion_numeric_fact(&crate::PLUS_CATALOG_ENTRY, &request)
            .unwrap()
            .is_err()
    );
}
