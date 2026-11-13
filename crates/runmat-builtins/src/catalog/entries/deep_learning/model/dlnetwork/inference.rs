use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry, DeepLearningLayerKind, DeepLearningNetworkOption,
};
use runmat_types::{
    standard, CallInference, CallRequest, CellFact, DynamicReason, NumericDomain, ObjectFact,
    ShapeFact, StaticClassIdentity, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

const LAYER_GRAPH: StaticClassIdentity = StaticClassIdentity::new("nnet.cnn.LayerGraph");
fn cell_layers_possible(cell: &CellFact) -> bool {
    let possible = |fact: &ValueFact| match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .is_none_or(|identity| DeepLearningLayerKind::from_identity(identity).is_some()),
        _ => false,
    };
    possible(&cell.element) && cell.elements.iter().all(possible)
}

fn layers_count(value: &ValueFact) -> Result<Option<usize>, ()> {
    match &value.kind {
        ValueKindFact::Unknown => Ok(None),
        ValueKindFact::Cell(cell) if cell_layers_possible(cell) => {
            if cell.elements_complete {
                if value
                    .shape
                    .element_count()
                    .is_some_and(|count| count != cell.elements.len())
                {
                    return Err(());
                }
                Ok(Some(cell.elements.len()))
            } else {
                Ok(value.shape.element_count())
            }
        }
        ValueKindFact::Cell(_) => Err(()),
        ValueKindFact::Object(object) if object.runtime_class.is_none() => Ok(None),
        ValueKindFact::Object(object)
            if [
                standard::DL_NETWORK,
                standard::SERIES_NETWORK,
                standard::DAG_NETWORK,
            ]
            .iter()
            .any(|class| {
                object
                    .runtime_class
                    .as_ref()
                    .is_some_and(|identity| identity.is(*class))
            }) || object
                .runtime_class
                .as_ref()
                .is_some_and(|identity| identity.is(LAYER_GRAPH)) =>
        {
            match object.properties.get("Layers") {
                Some(layers) => layers_count(layers),
                None if object.properties_complete => Err(()),
                None => Ok(None),
            }
        }
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_some_and(|identity| {
                DeepLearningLayerKind::from_identity(identity).is_some()
            }) =>
        {
            Ok(Some(1))
        }
        _ => Err(()),
    }
}

fn column(count: Option<usize>) -> ShapeFact {
    ShapeFact::from(vec![count, Some(1)])
}

fn text(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(ValueKindFact::String, shape, StorageFact::Dense)
}

fn cell(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn stored_layers(source: Option<&ValueFact>, count: Option<usize>) -> ValueFact {
    let source = source.and_then(|source| match &source.kind {
        ValueKindFact::Cell(_) => Some(source.clone()),
        ValueKindFact::Object(object) if object.runtime_class.is_none() => None,
        ValueKindFact::Object(object)
            if [
                standard::DL_NETWORK,
                standard::SERIES_NETWORK,
                standard::DAG_NETWORK,
                LAYER_GRAPH,
            ]
            .iter()
            .any(|class| {
                object
                    .runtime_class
                    .as_ref()
                    .is_some_and(|identity| identity.is(*class))
            }) =>
        {
            object.properties.get("Layers").cloned()
        }
        ValueKindFact::Object(_) => Some(ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(source.clone()),
                elements: vec![source.clone()],
                elements_complete: true,
            }),
            column(Some(1)),
            StorageFact::Dense,
        )),
        _ => None,
    });
    match source {
        Some(mut layers) if matches!(layers.kind, ValueKindFact::Cell(_)) => {
            layers.shape = column(count);
            layers
        }
        _ => cell(column(count)),
    }
}

fn empty_struct() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        true,
    )))
}

pub(super) fn network_rows_table_fact() -> ValueFact {
    state_rows_table_fact(None)
}

pub(in crate::catalog::entries::deep_learning::model) fn state_rows_table_fact(
    rows: Option<usize>,
) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TABLE.owned()),
            properties: BTreeMap::from([(
                "__table_variables".into(),
                ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                    BTreeMap::from([
                        ("Layer".into(), text(column(rows))),
                        ("Parameter".into(), text(column(rows))),
                        ("Value".into(), cell(column(rows))),
                    ]),
                    true,
                ))),
            )]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len().is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-DL-DLNETWORK-ARITY",
            "dlnetwork requires a layer collection followed by complete name-value pairs",
            request.arguments.len().saturating_sub(1),
        ));
    }
    let count = request.arguments.first().and_then(|source| {
        layers_count(source).unwrap_or_else(|_| {
            diagnostics.push(argument_error(
                "RM-DL-DLNETWORK-LAYERS",
                "layers must be a supported layer object, graph, or cell collection",
                0,
            ));
            None
        })
    });
    if count == Some(0) {
        diagnostics.push(argument_error(
            "RM-DL-DLNETWORK-EMPTY",
            "dlnetwork requires at least one supported layer",
            0,
        ));
    }
    for index in (1..request.arguments.len()).step_by(2) {
        let name = &request.arguments[index];
        if !matches!(
            &name.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-DL-DLNETWORK-OPTIONS",
                "name-value keys must be text",
                index,
            ));
        }
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|name| DeepLearningNetworkOption::from_name(&name).is_none())
        {
            diagnostics.push(argument_error(
                "RM-DL-DLNETWORK-OPTIONS",
                "only Initialize and OutputNames are supported constructor options",
                index,
            ));
        }
        if let (Some(option), Some(value)) = (
            request.literals.literal_string_at(index),
            request.arguments.get(index + 1),
        ) {
            let admissible = match DeepLearningNetworkOption::from_name(&option) {
                Some(DeepLearningNetworkOption::Initialize) => {
                    let class_possible = match &value.kind {
                        ValueKindFact::Logical | ValueKindFact::Unknown => true,
                        ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Real,
                        _ => false,
                    };
                    let literal_possible = request
                        .literals
                        .numeric_at(index + 1)
                        .is_none_or(|number| number == 0.0 || number == 1.0);
                    class_possible
                        && literal_possible
                        && value.shape.element_count().is_none_or(|count| count == 1)
                }
                Some(DeepLearningNetworkOption::OutputNames) => {
                    matches!(
                        &value.kind,
                        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                    ) && value.shape.element_count().is_none_or(|count| count == 1)
                }
                None => true,
            };
            if !admissible {
                diagnostics.push(argument_error(
                    "RM-DL-DLNETWORK-OPTION-VALUE",
                    "Initialize must be a scalar logical-like control; OutputNames must be one text name",
                    index + 1,
                ));
            }
        }
    }
    let mut properties = BTreeMap::from([
        (
            "Layers".into(),
            stored_layers(request.arguments.first(), count),
        ),
        ("LayerNames".into(), text(column(count))),
        (
            "InputNames".into(),
            text(ShapeFact::from(vec![Some(1), Some(1)])),
        ),
        (
            "OutputNames".into(),
            text(ShapeFact::from(vec![Some(1), Some(1)])),
        ),
        ("Connections".into(), empty_struct()),
        ("Learnables".into(), network_rows_table_fact()),
        ("State".into(), network_rows_table_fact()),
        (
            "Initialized".into(),
            ValueFact::scalar(ValueKindFact::Logical),
        ),
        ("SupportedExecution".into(), text(ShapeFact::Scalar)),
    ]);
    if (1..request.arguments.len()).step_by(2).any(|index| {
        request
            .literals
            .literal_string_at(index)
            .is_some_and(|name| {
                DeepLearningNetworkOption::from_name(&name)
                    == Some(DeepLearningNetworkOption::Initialize)
            })
    }) {
        properties.insert(
            "Initialize".into(),
            ValueFact::scalar(ValueKindFact::Logical),
        );
    }
    let result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::DL_NETWORK.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, result, diagnostics)
}
