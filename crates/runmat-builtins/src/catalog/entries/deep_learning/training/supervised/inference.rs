use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, ClassIdentity, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ShapeFact, StaticClassIdentity, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

const LAYER_GRAPH: StaticClassIdentity = StaticClassIdentity::new("nnet.cnn.LayerGraph");
const DLNETWORK: StaticClassIdentity = StaticClassIdentity::new("dlnetwork");
const SERIES_NETWORK: StaticClassIdentity = StaticClassIdentity::new("SeriesNetwork");
const DAG_NETWORK: StaticClassIdentity = StaticClassIdentity::new("DAGNetwork");

fn known_object_class(object: &ObjectFact) -> Option<&ClassIdentity> {
    object.runtime_class.as_ref()
}

fn possible_real_matrix(value: &ValueFact) -> bool {
    let kind = match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::Logical => true,
        ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
        _ => false,
    };
    kind && value.shape.known_dims().is_none_or(|dims| dims.len() <= 2)
}

fn possible_targets(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown
        | ValueKindFact::String
        | ValueKindFact::Character
        | ValueKindFact::Cell(_) => true,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical => possible_real_matrix(value),
        ValueKindFact::Object(object) => known_object_class(object)
            .is_none_or(|class| class.is(runmat_types::standard::CATEGORICAL)),
        _ => false,
    }
}

fn possible_layers(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Cell(cell) => {
            matches!(
                cell.element.kind,
                ValueKindFact::Unknown | ValueKindFact::Object(_)
            ) && cell.elements.iter().all(|element| {
                matches!(
                    element.kind,
                    ValueKindFact::Unknown | ValueKindFact::Object(_)
                )
            })
        }
        ValueKindFact::Object(object) => known_object_class(object).is_none_or(|class| {
            class.is(LAYER_GRAPH)
                || class.is(DLNETWORK)
                || class.is(SERIES_NETWORK)
                || class.is(DAG_NETWORK)
                || [
                    crate::FEATURE_INPUT_LAYER_CLASS,
                    crate::FULLY_CONNECTED_LAYER_CLASS,
                    crate::RELU_LAYER_CLASS,
                    crate::ELU_LAYER_CLASS,
                    crate::SOFTMAX_LAYER_CLASS,
                    crate::CLASSIFICATION_OUTPUT_LAYER_CLASS,
                    crate::REGRESSION_OUTPUT_LAYER_CLASS,
                ]
                .into_iter()
                .any(|supported| class.is(supported))
        }),
        _ => false,
    }
}

fn possible_options(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => known_object_class(object)
            .is_none_or(|class| crate::TrainingSolver::from_identity(class).is_some()),
        _ => false,
    }
}

fn possible_host_loss(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Character => value
            .shape
            .known_dims()
            .is_none_or(|dims| dims.first().copied().flatten().is_none_or(|rows| rows == 1)),
        _ => false,
    }
}

fn trained_network(class: Option<StaticClassIdentity>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: class.map(StaticClassIdentity::owned),
            properties: BTreeMap::from([
                ("TrainingInfo".into(), training_info()),
                (
                    "TrainedWith".into(),
                    ValueFact::scalar(ValueKindFact::String),
                ),
                (
                    "TrainingLoss".into(),
                    ValueFact::scalar(ValueKindFact::String),
                ),
            ]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn training_info() -> ValueFact {
    let number = || {
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))
    };
    let loss_history = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Dense,
    );
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::from([
            (
                "SolverName".into(),
                ValueFact::scalar(ValueKindFact::String),
            ),
            ("MaxEpochs".into(), number()),
            ("MiniBatchSize".into(), number()),
            ("InitialLearnRate".into(), number()),
            ("Iteration".into(), number()),
            ("Epoch".into(), number()),
            ("FinalLoss".into(), number()),
            ("TrainingLoss".into(), loss_history),
            (
                "StopReason".into(),
                ValueFact::scalar(ValueKindFact::String),
            ),
        ]),
        true,
    )))
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, is_trainnet: bool) -> CallInference {
    let mut diagnostics = Vec::new();
    let expected = if is_trainnet { 5 } else { 4 };
    if request.arguments.len() != expected {
        diagnostics.push(argument_error(
            "RM-DL-SUPERVISED-ARITY",
            format!("{} requires exactly {expected} inputs", entry.identity.name),
            request.arguments.len().saturating_sub(1),
        ));
    }
    for (index, message, possible) in [
        (
            0,
            "X must be a real numeric or logical matrix",
            possible_real_matrix as fn(&ValueFact) -> bool,
        ),
        (
            1,
            "Y must be a supported regression matrix or classification target",
            possible_targets,
        ),
        (
            2,
            "network must be a supported sequential layer collection",
            possible_layers,
        ),
    ] {
        if request
            .arguments
            .get(index)
            .is_some_and(|value| !possible(value))
        {
            diagnostics.push(argument_error("RM-DL-SUPERVISED-INPUT", message, index));
        }
    }
    let options_index = if is_trainnet { 4 } else { 3 };
    if request
        .arguments
        .get(options_index)
        .is_some_and(|value| !possible_options(value))
    {
        diagnostics.push(argument_error(
            "RM-DL-SUPERVISED-OPTIONS",
            "options must be a trainingOptions object",
            options_index,
        ));
    }
    if is_trainnet {
        if request
            .arguments
            .get(3)
            .is_some_and(|value| !possible_host_loss(value))
        {
            diagnostics.push(argument_error(
                "RM-DL-TRAINNET-LOSS",
                "lossFcn must name a supported built-in loss; function handles are unsupported",
                3,
            ));
        }
        if let Some(name) = request.literals.literal_string_at(3) {
            if !matches!(
                name.to_ascii_lowercase().as_str(),
                "mse"
                    | "mean-squared-error"
                    | "mean_squared_error"
                    | "crossentropy"
                    | "cross-entropy"
            ) {
                diagnostics.push(argument_error(
                    "RM-DL-TRAINNET-LOSS",
                    "unsupported trainnet lossFcn",
                    3,
                ));
            }
        }
    }
    let class = if is_trainnet {
        Some(DLNETWORK)
    } else {
        match request.arguments.get(2).map(|value| &value.kind) {
            Some(ValueKindFact::Object(object))
                if known_object_class(object).is_some_and(|class| class.is(LAYER_GRAPH)) =>
            {
                Some(DAG_NETWORK)
            }
            Some(ValueKindFact::Unknown) => None,
            Some(ValueKindFact::Object(object)) if known_object_class(object).is_none() => None,
            _ => Some(SERIES_NETWORK),
        }
    };
    finish_fixed_outputs(
        entry,
        request,
        vec![trained_network(class), training_info()],
        diagnostics,
    )
}

pub(super) fn train_network(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, false)
}

pub(super) fn trainnet(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, true)
}
