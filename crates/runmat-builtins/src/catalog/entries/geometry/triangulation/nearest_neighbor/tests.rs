use super::*;
use runmat_types::{
    CallRequest, DimensionFact, DynamicReason, LiteralContext, NumericClass, NumericDomain,
    NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn receiver() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(super::super::delaunay_tri::DELAUNAY_TRI_CLASS.owned()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: Some(false),
    }))
}

fn query(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

#[test]
fn both_aliases_prove_query_count_without_guessing_unknown_rows() {
    for entry in [
        &NEAREST_NEIGHBOR_CATALOG_ENTRY,
        &DELAUNAY_TRI_NEAREST_NEIGHBOR_CATALOG_ENTRY,
    ] {
        for (arguments, rows) in [
            (
                vec![receiver(), query(ShapeFact::from(vec![Some(4), Some(2)]))],
                Some(4),
            ),
            (vec![receiver(), query(ShapeFact::Unknown)], None),
            (
                vec![
                    receiver(),
                    query(ShapeFact::Scalar),
                    query(ShapeFact::Scalar),
                ],
                Some(1),
            ),
        ] {
            let result = inference::infer(&request(arguments), entry);
            assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![rows, Some(1)])
            );
            assert_eq!(
                result.outputs[0].numeric().unwrap().class,
                NumericClass::Double
            );
        }
    }
}

#[test]
fn definite_bad_receiver_and_query_geometry_are_diagnosed() {
    let wrong = inference::infer(
        &request(vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            query(ShapeFact::from(vec![Some(3), Some(2)])),
        ]),
        &NEAREST_NEIGHBOR_CATALOG_ENTRY,
    );
    assert!(wrong.diagnostics.is_empty());
    let mut wrong_receiver = receiver();
    let ValueKindFact::Object(object) = &mut wrong_receiver.kind else {
        unreachable!()
    };
    object.runtime_class = Some(runmat_types::StaticClassIdentity::new("OtherMesh").owned());
    let invalid = inference::infer(
        &request(vec![
            wrong_receiver,
            query(ShapeFact::from(vec![Some(2), Some(3)])),
        ]),
        &NEAREST_NEIGHBOR_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-NEARESTNEIGHBOR-RECEIVER"));
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-NEARESTNEIGHBOR-SHAPE"));
    let mismatched = inference::infer(
        &request(vec![
            receiver(),
            query(ShapeFact::from(vec![Some(2), Some(1)])),
            query(ShapeFact::from(vec![Some(3), Some(1)])),
        ]),
        &NEAREST_NEIGHBOR_CATALOG_ENTRY,
    );
    assert!(mismatched
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-NEARESTNEIGHBOR-SHAPE"));
    assert_eq!(
        mismatched.outputs[0].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    assert_eq!(
        DimensionFact::Known(1),
        match &mismatched.outputs[0].shape {
            ShapeFact::Shaped { dims } => dims[1].clone(),
            _ => unreachable!(),
        }
    );
}
