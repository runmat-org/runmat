use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ShapeFact, StaticClassIdentity, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

const LAYER_GRAPH: StaticClassIdentity = StaticClassIdentity::new("nnet.cnn.LayerGraph");
const NETWORK_ANALYSIS: StaticClassIdentity = StaticClassIdentity::new("nnet.cnn.NetworkAnalysis");

fn layer_count(value: &ValueFact) -> Result<Option<usize>, ()> {
    match &value.kind {
        ValueKindFact::Unknown => Ok(None),
        // layerGraph(layers) treats any object as one layer; analyzeNetwork
        // handles its layer-graph class separately before this helper.
        ValueKindFact::Object(_) => Ok(Some(1)),
        ValueKindFact::Cell(cell) if cell_layers_possible(cell) => Ok(value.shape.element_count()),
        ValueKindFact::Cell(_) => Err(()),
        _ => Err(()),
    }
}

fn analysis_count(value: &ValueFact) -> Result<Option<usize>, ()> {
    if let ValueKindFact::Object(object) = &value.kind {
        if object.runtime_class.as_ref() == Some(&LAYER_GRAPH.owned()) {
            return match object.properties.get("Layers") {
                Some(layers) => layer_count(layers),
                None if object.properties_complete => Ok(Some(0)),
                None => Ok(None),
            };
        }
    }
    layer_count(value)
}

fn cell_layers_possible(cell: &CellFact) -> bool {
    fn possible(kind: &ValueKindFact) -> bool {
        matches!(kind, ValueKindFact::Unknown | ValueKindFact::Object(_))
    }
    possible(&cell.element.kind) && cell.elements.iter().all(|element| possible(&element.kind))
}

fn column(count: Option<usize>) -> ShapeFact {
    ShapeFact::from(vec![count, Some(1)])
}

fn text_column(count: Option<usize>) -> ValueFact {
    ValueFact::proven(ValueKindFact::String, column(count), StorageFact::Dense)
}

fn layer_cell(count: Option<usize>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        column(count),
        StorageFact::Dense,
    )
}

fn object(class: StaticClassIdentity, properties: BTreeMap<String, ValueFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn layer_graph(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-DL-LAYERGRAPH-ARITY",
            "layerGraph accepts zero inputs or one layer collection",
            1,
        ));
    }
    let count = request.arguments.first().map_or(Some(0), |value| {
        layer_count(value).unwrap_or_else(|_| {
            diagnostics.push(argument_error(
                "RM-DL-LAYERGRAPH-LAYERS",
                "layers must be a layer object or cell collection",
                0,
            ));
            None
        })
    });
    let properties = BTreeMap::from([
        ("Layers".into(), layer_cell(count)),
        ("LayerNames".into(), text_column(count)),
        (
            "Connections".into(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                BTreeMap::new(),
                true,
            ))),
        ),
    ]);
    finish_fixed(entry, request, object(LAYER_GRAPH, properties), diagnostics)
}

pub(super) fn analyze_network(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-DL-ANALYZENETWORK-ARITY",
            "analyzeNetwork expects exactly one graph or layer collection",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let count = request.arguments.first().and_then(|value| {
        analysis_count(value).unwrap_or_else(|_| {
            diagnostics.push(argument_error(
                "RM-DL-ANALYZENETWORK-LAYERS",
                "input must be a layer graph, layer object, or cell collection",
                0,
            ));
            None
        })
    });
    let properties = BTreeMap::from([
        (
            "NumLayers".into(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ),
        ("LayerNames".into(), text_column(count)),
        (
            "Diagnostics".into(),
            ValueFact::scalar(ValueKindFact::String),
        ),
    ]);
    finish_fixed(
        entry,
        request,
        object(NETWORK_ANALYSIS, properties),
        diagnostics,
    )
}
