use super::*;
use runmat_types::*;
use std::collections::BTreeMap;

fn call(entry: &BuiltinCatalogEntry, args: Vec<ValueFact>) -> CallInference {
    crate::infer_catalog_call(
        entry,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    )
}

fn layer() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("nnet.cnn.layer.ReLULayer").owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn graph_identity_and_layer_count_are_statically_available() {
    let empty = call(&LAYER_GRAPH_CATALOG_ENTRY, vec![]);
    assert!(empty.diagnostics.is_empty());
    let ValueKindFact::Object(graph) = &empty.outputs[0].kind else {
        panic!("graph object fact expected")
    };
    assert_eq!(
        graph.runtime_class,
        Some(StaticClassIdentity::new("nnet.cnn.LayerGraph").owned())
    );
    assert_eq!(
        graph.properties["Layers"].shape,
        ShapeFact::from(vec![Some(0), Some(1)])
    );
    assert_eq!(empty.outputs[0].residency, ResidencyFact::Host);

    let one = call(&LAYER_GRAPH_CATALOG_ENTRY, vec![layer()]);
    let ValueKindFact::Object(graph) = &one.outputs[0].kind else {
        panic!("graph object fact expected")
    };
    assert_eq!(
        graph.properties["LayerNames"].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
}

#[test]
fn analysis_uses_known_graph_layer_shape_without_guessing_unknown_contents() {
    let graph = call(&LAYER_GRAPH_CATALOG_ENTRY, vec![layer()])
        .outputs
        .remove(0);
    let result = call(&ANALYZE_NETWORK_CATALOG_ENTRY, vec![graph]);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(analysis) = &result.outputs[0].kind else {
        panic!("analysis object fact expected")
    };
    assert_eq!(
        analysis.runtime_class,
        Some(StaticClassIdentity::new("nnet.cnn.NetworkAnalysis").owned())
    );
    assert_eq!(
        analysis.properties["LayerNames"].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
    assert_eq!(analysis.properties["NumLayers"].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);

    let unknown = call(
        &ANALYZE_NETWORK_CATALOG_ENTRY,
        vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
    );
    let ValueKindFact::Object(analysis) = &unknown.outputs[0].kind else {
        panic!("analysis object fact expected")
    };
    assert_eq!(
        analysis.properties["LayerNames"].shape,
        ShapeFact::from(vec![None, Some(1)])
    );

    let open_graph = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("nnet.cnn.LayerGraph").owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let result = call(&ANALYZE_NETWORK_CATALOG_ENTRY, vec![open_graph.clone()]);
    let ValueKindFact::Object(analysis) = &result.outputs[0].kind else {
        panic!("analysis object fact expected")
    };
    assert_eq!(
        analysis.properties["LayerNames"].shape,
        ShapeFact::from(vec![None, Some(1)])
    );

    let mut bad_graph = open_graph;
    let ValueKindFact::Object(object) = &mut bad_graph.kind else {
        unreachable!()
    };
    object.properties.insert(
        "Layers".into(),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    );
    assert!(!call(&ANALYZE_NETWORK_CATALOG_ENTRY, vec![bad_graph])
        .diagnostics
        .is_empty());
}

#[test]
fn invalid_numeric_input_and_arity_are_diagnosed() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    for (entry, args) in [
        (&LAYER_GRAPH_CATALOG_ENTRY, vec![numeric.clone()]),
        (&LAYER_GRAPH_CATALOG_ENTRY, vec![layer(), layer()]),
        (&ANALYZE_NETWORK_CATALOG_ENTRY, vec![numeric]),
        (&ANALYZE_NETWORK_CATALOG_ENTRY, vec![]),
    ] {
        assert!(
            !call(entry, args).diagnostics.is_empty(),
            "{}",
            entry.identity.name
        );
    }
    let numeric_cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    );
    assert!(
        !call(&LAYER_GRAPH_CATALOG_ENTRY, vec![numeric_cell.clone()])
            .diagnostics
            .is_empty()
    );
    assert!(!call(&ANALYZE_NETWORK_CATALOG_ENTRY, vec![numeric_cell])
        .diagnostics
        .is_empty());
}
