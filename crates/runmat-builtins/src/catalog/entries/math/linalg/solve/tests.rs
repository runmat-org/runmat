use super::*;
use crate::{infer_catalog_call, BuiltinDocumentationAuthority, BuiltinExampleVerification};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

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

fn request(arguments: Vec<ValueFact>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn all_eleven_solve_identities_own_typed_contracts_and_original_programs() {
    for (entry, expected_examples) in [
        (&COND_CATALOG_ENTRY, 6),
        (&DET_CATALOG_ENTRY, 6),
        (&INV_CATALOG_ENTRY, 6),
        (&LINSOLVE_CATALOG_ENTRY, 5),
        (&NORM_CATALOG_ENTRY, 6),
        (&NULL_CATALOG_ENTRY, 3),
        (&PINV_CATALOG_ENTRY, 5),
        (&RANK_CATALOG_ENTRY, 7),
        (&RCOND_CATALOG_ENTRY, 6),
        (&RREF_CATALOG_ENTRY, 3),
        (&VECNORM_CATALOG_ENTRY, 6),
    ] {
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(entry.documentation.examples.len(), expected_examples);
        assert!(entry.documentation.examples.iter().all(|example| {
            matches!(
                example.verification,
                BuiltinExampleVerification::Assertions { .. }
                    | BuiltinExampleVerification::ExpectedError { .. }
            )
        }));
        assert_eq!(entry.placement.fusion, crate::BuiltinFusionPolicy::Boundary);
    }
}

#[test]
fn solve_shape_and_host_precision_facts_follow_actual_output_geometry() {
    let input = matrix(3, 2, NumericClass::Single);
    for class in [NumericClass::Single, NumericClass::Double] {
        let pinv = infer_catalog_call(&PINV_CATALOG_ENTRY, &request(vec![matrix(3, 2, class)], 1));
        assert_eq!(
            pinv.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert!(matches!(pinv.outputs[0].kind, ValueKindFact::Numeric(n) if n.class == class));
    }

    let null = infer_catalog_call(&NULL_CATALOG_ENTRY, &request(vec![input.clone()], 1));
    assert_eq!(null.outputs[0].shape, ShapeFact::from(vec![Some(2), None]));

    let det = infer_catalog_call(
        &DET_CATALOG_ENTRY,
        &request(vec![matrix(2, 2, NumericClass::Single)], 1),
    );
    assert!(matches!(
        det.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    ));
    let rcond = infer_catalog_call(
        &RCOND_CATALOG_ENTRY,
        &request(vec![matrix(2, 2, NumericClass::Single)], 1),
    );
    assert!(matches!(
        rcond.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    ));

    let rhs = matrix(3, 4, NumericClass::Double);
    let solve = infer_catalog_call(
        &LINSOLVE_CATALOG_ENTRY,
        &request(vec![matrix(3, 3, NumericClass::Single), rhs], 2),
    );
    assert_eq!(
        solve.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
    assert!(
        matches!(solve.outputs[0].kind, ValueKindFact::Numeric(n) if n.class == NumericClass::Double)
    );
    assert_eq!(solve.outputs[1].shape, ShapeFact::Scalar);

    let mut resident_rhs = matrix(3, 1, NumericClass::Single);
    resident_rhs.residency = ResidencyFact::Device { provider: None };
    let mixed = infer_catalog_call(
        &LINSOLVE_CATALOG_ENTRY,
        &request(vec![matrix(3, 3, NumericClass::Single), resident_rhs], 1),
    );
    assert_eq!(mixed.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(mixed.outputs[0].residency, ResidencyFact::Unknown);

    let mut resident = matrix(2, 2, NumericClass::Single);
    resident.residency = ResidencyFact::Device { provider: None };
    for entry in [&COND_CATALOG_ENTRY, &DET_CATALOG_ENTRY, &NORM_CATALOG_ENTRY] {
        let inferred = infer_catalog_call(entry, &request(vec![resident.clone()], 1));
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);
    }
}

#[test]
fn vecnorm_reduces_only_a_proven_axis_and_preserves_nd_rank() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        StorageFact::Dense,
    );
    let default = infer_catalog_call(&VECNORM_CATALOG_ENTRY, &request(vec![input.clone()], 1));
    assert_eq!(
        default.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3), Some(4)])
    );
    let mut explicit = request(
        vec![
            input.clone(),
            ValueFact::scalar(ValueKindFact::Unknown),
            ValueFact::scalar(ValueKindFact::Unknown),
        ],
        1,
    );
    explicit.literals.literal_args = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Number(3.0),
    ];
    let third = infer_catalog_call(&VECNORM_CATALOG_ENTRY, &explicit);
    assert_eq!(
        third.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3), Some(1)])
    );
    explicit.literals.literal_args[2] = LiteralValue::Unknown;
    let dynamic = infer_catalog_call(&VECNORM_CATALOG_ENTRY, &explicit);
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Ranked { rank: 3 });
}
