use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>, requested: RequestedOutputCount) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(requested),
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
fn function_and_method_prove_one_or_two_outputs_with_shared_query_rows() {
    for entry in [
        &POINT_LOCATION_CATALOG_ENTRY,
        &DELAUNAY_TRI_POINT_LOCATION_CATALOG_ENTRY,
    ] {
        for (requested, expected) in [
            (RequestedOutputCount::One, 1),
            (RequestedOutputCount::Exactly(2), 2),
        ] {
            let result = inference::infer(
                &request(
                    vec![receiver(), query(ShapeFact::from(vec![Some(5), Some(2)]))],
                    requested,
                ),
                entry,
            );
            assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
            assert_eq!(result.outputs.len(), expected);
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![Some(5), Some(1)])
            );
            assert_eq!(
                result.outputs[0].numeric().unwrap().class,
                NumericClass::Double
            );
            if expected == 2 {
                assert_eq!(
                    result.outputs[1].shape,
                    ShapeFact::from(vec![Some(5), Some(3)])
                );
            }
        }
    }
}

#[test]
fn unknown_query_rows_remain_unknown_and_bad_geometry_is_diagnosed() {
    let unknown = inference::infer(
        &request(
            vec![receiver(), ValueFact::unknown(DynamicReason::RuntimeValue)],
            RequestedOutputCount::Exactly(2),
        ),
        &POINT_LOCATION_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(
        unknown.outputs[1].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
    let invalid = inference::infer(
        &request(
            vec![receiver(), query(ShapeFact::from(vec![Some(2), Some(3)]))],
            RequestedOutputCount::One,
        ),
        &POINT_LOCATION_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-POINTLOCATION-SHAPE"));
    let outputs = inference::infer(
        &request(
            vec![receiver(), query(ShapeFact::from(vec![Some(1), Some(2)]))],
            RequestedOutputCount::Exactly(3),
        ),
        &POINT_LOCATION_CATALOG_ENTRY,
    );
    assert!(!outputs.diagnostics.is_empty());
}
