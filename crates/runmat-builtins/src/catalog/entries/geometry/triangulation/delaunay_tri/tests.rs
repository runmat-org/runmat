use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn points(rows: Option<usize>, cols: Option<usize>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![rows, cols]),
        StorageFact::Dense,
    )
}

#[test]
fn constructor_proves_class_aliases_and_empty_topology_without_fabricated_triangles() {
    for (arguments, rows, triangles) in [
        (vec![], Some(0), Some(0)),
        (vec![points(Some(2), Some(2))], Some(2), Some(0)),
        (vec![points(Some(4), Some(2))], Some(4), None),
        (
            vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
            None,
            None,
        ),
    ] {
        let result = inference::infer(&request(arguments), &DELAUNAY_TRI_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        let ValueKindFact::Object(ObjectFact {
            runtime_class: Some(class),
            properties,
            ..
        }) = &result.outputs[0].kind
        else {
            panic!("constructor must infer DelaunayTri object");
        };
        assert_eq!(class, &DELAUNAY_TRI_CLASS.owned());
        assert_eq!(properties["X"], properties["Points"]);
        assert_eq!(properties["Triangulation"], properties["ConnectivityList"]);
        assert_eq!(properties["X"].shape, ShapeFact::from(vec![rows, Some(2)]));
        assert_eq!(
            properties["Triangulation"].shape,
            ShapeFact::from(vec![triangles, Some(3)])
        );
        assert_eq!(
            properties["Constraints"].shape,
            ShapeFact::from(vec![Some(0), Some(2)])
        );
    }
}

#[test]
fn definite_bad_point_shape_and_complex_domain_are_diagnosed() {
    let shape = inference::infer(
        &request(vec![points(Some(3), Some(3))]),
        &DELAUNAY_TRI_CATALOG_ENTRY,
    );
    assert!(shape
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-DELAUNAYTRI-SHAPE"));
    let mut complex = points(Some(3), Some(2));
    complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    let domain = inference::infer(&request(vec![complex]), &DELAUNAY_TRI_CATALOG_ENTRY);
    assert!(domain
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-DELAUNAYTRI-CLASS"));
}
