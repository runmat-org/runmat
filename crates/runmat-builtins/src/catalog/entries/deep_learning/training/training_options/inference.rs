use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, ObjectFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn defaults() -> BTreeMap<String, ValueFact> {
    BTreeMap::from([
        (
            "SolverName".into(),
            ValueFact::scalar(ValueKindFact::String),
        ),
        (
            "MaxEpochs".into(),
            ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            })),
        ),
        (
            "MiniBatchSize".into(),
            ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            })),
        ),
        (
            "InitialLearnRate".into(),
            ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::Double,
                domain: runmat_types::NumericDomain::Real,
            })),
        ),
        ("Shuffle".into(), ValueFact::scalar(ValueKindFact::String)),
        ("Verbose".into(), ValueFact::scalar(ValueKindFact::Logical)),
        ("Plots".into(), ValueFact::scalar(ValueKindFact::String)),
        (
            "ExecutionEnvironment".into(),
            ValueFact::scalar(ValueKindFact::String),
        ),
    ])
}

fn host_scalar_text(value: &ValueFact) -> bool {
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

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() % 2 == 0 {
        diagnostics.push(argument_error(
            "RM-DL-TRAININGOPTIONS-ARITY",
            "trainingOptions requires a solver followed by complete name-value pairs",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(solver) = request.arguments.first() {
        if !host_scalar_text(solver) {
            diagnostics.push(argument_error(
                "RM-DL-TRAININGOPTIONS-SOLVER",
                "solver must be host text naming sgdm, adam, or rmsprop",
                0,
            ));
        }
        if let Some(name) = request.literals.literal_string_at(0) {
            if !matches!(
                name.to_ascii_lowercase().as_str(),
                "sgdm" | "adam" | "rmsprop"
            ) {
                diagnostics.push(argument_error(
                    "RM-DL-TRAININGOPTIONS-SOLVER",
                    "unsupported training solver",
                    0,
                ));
            }
        }
    }
    let mut properties = defaults();
    let mut complete = true;
    for index in (1..request.arguments.len()).step_by(2) {
        if !host_scalar_text(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-DL-TRAININGOPTIONS-OPTION",
                "option name must be host text",
                index,
            ));
        }
        let Some(name) = request.literals.literal_string_at(index) else {
            complete = false;
            continue;
        };
        if let Some(value) = request.arguments.get(index + 1) {
            properties.insert(super::canonical_training_option_name(name), value.clone());
        }
    }
    if !complete {
        // A dynamic key may overwrite any default property, so none of those
        // facts can be considered proven until the name is known.
        properties.clear();
    }
    let result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::TRAINING_OPTIONS_CLASS.owned()),
            properties,
            properties_complete: complete,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, result, diagnostics)
}
