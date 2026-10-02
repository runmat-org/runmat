use super::*;
use runmat_types::{
    DimensionFact, DynamicReason, NumericClass, NumericDomain, NumericFact, ObjectFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>) -> runmat_types::CallRequest {
    runmat_types::CallRequest {
        arguments,
        literals: runmat_types::LiteralContext::default(),
        outputs: runmat_types::OutputSelection::new(runmat_types::RequestedOutputCount::One),
    }
}

fn receiver(triangles: Option<usize>) -> ValueFact {
    let mut properties = BTreeMap::new();
    properties.insert(
        "ConnectivityList".into(),
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![triangles, Some(3)]),
            StorageFact::Dense,
        ),
    );
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(runmat_types::StaticClassIdentity::new("DelaunayTri").owned()),
        properties,
        properties_complete: true,
        handle_semantics: Some(false),
    }))
}

#[test]
fn both_bindings_prove_double_edge_columns_without_guessing_topology() {
    for entry in [
        &FREE_BOUNDARY_CATALOG_ENTRY,
        &DELAUNAY_TRI_FREE_BOUNDARY_CATALOG_ENTRY,
    ] {
        for (triangles, rows) in [(Some(0), Some(0)), (Some(2), None), (None, None)] {
            let result = inference::infer(&request(vec![receiver(triangles)]), entry);
            assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
            assert_eq!(
                result.outputs[0].numeric().unwrap().class,
                NumericClass::Double
            );
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![rows, Some(2)])
            );
        }
    }
}

#[test]
fn unknown_receiver_stays_conservative_and_wrong_class_is_diagnosed() {
    let unknown = inference::infer(
        &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)]),
        &FREE_BOUNDARY_CATALOG_ENTRY,
    );
    assert!(unknown.diagnostics.is_empty());
    assert_eq!(
        unknown.outputs[0].shape,
        ShapeFact::from(vec![None, Some(2)])
    );

    let mut wrong = receiver(Some(1));
    let ValueKindFact::Object(object) = &mut wrong.kind else {
        unreachable!()
    };
    object.runtime_class = Some(runmat_types::StaticClassIdentity::new("OtherMesh").owned());
    let invalid = inference::infer(&request(vec![wrong]), &FREE_BOUNDARY_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FREEBOUNDARY-RECEIVER"));
    assert_eq!(
        ShapeFact::from(vec![None, Some(2)]),
        invalid.outputs[0].shape
    );
    assert_eq!(
        DimensionFact::Known(2),
        match &invalid.outputs[0].shape {
            ShapeFact::Shaped { dims } => dims[1].clone(),
            _ => unreachable!(),
        }
    );
}
