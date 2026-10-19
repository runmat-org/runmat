//! Table target projection; column names are structural metadata, not roles.
use super::*;
use crate::DeepLearningParameterTableColumn as Column;
use runmat_types::{standard, ClassIdentity, ObjectFact, ShapeFact, StructFact};
use std::collections::BTreeMap;

// The table domain owns this physical property. Keep the spelling confined to
// the adapter; callers reason about the projected Variables fact instead.
const VARIABLES: &str = "__table_variables";

pub(super) fn is_network(class: &ClassIdentity) -> bool {
    class.is(standard::DL_NETWORK)
        || class.is(standard::SERIES_NETWORK)
        || class.is(standard::DAG_NETWORK)
}

fn parameter_columns(fields: &BTreeMap<String, ValueFact>, complete: bool) -> bool {
    complete && Column::has_exact_columns(fields.keys().map(String::as_str))
}

fn columns(object: &ObjectFact) -> Option<&StructFact> {
    match &object.properties.get(VARIABLES)?.kind {
        ValueKindFact::Struct(columns) => Some(columns),
        _ => None,
    }
}

fn gradient_columns(columns: &StructFact) -> StructFact {
    let parameter = parameter_columns(&columns.fields, columns.fields_complete);
    let fields = columns
        .fields
        .iter()
        .map(|(name, value)| {
            let result = if parameter
                && matches!(
                    Column::from_name(name),
                    Some(Column::Layer | Column::Parameter)
                ) {
                super::retained_metadata(value)
            } else {
                super::gradient(value)
            };
            (name.clone(), result)
        })
        .collect();
    // Table Variables is a scalar StructValue, not a StructArray.
    StructFact::scalar(fields, columns.fields_complete)
}

pub(super) fn gradient(target: &ValueFact, object: &ObjectFact) -> ValueFact {
    let mut projected = ObjectFact {
        properties: object
            .properties
            .iter()
            .map(|(name, value)| (name.clone(), super::retained_metadata(value)))
            .collect(),
        ..object.clone()
    };
    if let Some(columns) = columns(object) {
        let mut variables = ValueFact::scalar(ValueKindFact::Struct(gradient_columns(columns)));
        variables.alias = AliasFact::Unknown;
        projected.properties.insert(VARIABLES.into(), variables);
    } else if object.properties.contains_key(VARIABLES) {
        projected.properties.insert(
            VARIABLES.into(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        );
    } else {
        // Replacement installs Variables even if the input fact did not
        // expose that physical property.
        projected.properties_complete = false;
    }
    let mut result = ValueFact::proven(
        ValueKindFact::Object(projected),
        target.shape.clone(),
        target.storage,
    );
    result.alias = AliasFact::Unknown;
    result
}

pub(super) fn network_gradient(network: &ObjectFact) -> ValueFact {
    if let Some(learnables) = network.properties.get("Learnables") {
        if let ValueKindFact::Object(table) = &learnables.kind {
            if table
                .runtime_class
                .as_ref()
                .is_some_and(|class| class.is(standard::TABLE))
            {
                return gradient(learnables, table);
            }
        }
    }
    // Runtime projects canonical Layers into a fresh public table. A network
    // fact without a Learnables projection proves that class, not row data.
    let mut result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TABLE.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    result.alias = AliasFact::Unknown;
    result
}

pub(super) fn check(object: &ObjectFact, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    let Some(columns) = columns(object) else {
        return;
    };
    if parameter_columns(&columns.fields, columns.fields_complete) {
        let known_bad = [Column::Layer, Column::Parameter]
            .iter()
            .filter_map(|column| columns.fields.get(column.name()))
            .any(|column| !matches!(column.kind, ValueKindFact::String | ValueKindFact::Unknown))
            || columns
                .fields
                .get(Column::Value.name())
                .is_some_and(|column| {
                    !matches!(column.kind, ValueKindFact::Cell(_) | ValueKindFact::Unknown)
                });
        if known_bad {
            diagnostics.push(argument_error(
                "RM-DL-DLGRADIENT-TABLE",
                "parameter table requires string Layer/Parameter columns and a cell Value column",
                index,
            ));
        }
        if let Some(value) = columns.fields.get(Column::Value.name()) {
            super::check(value, index, diagnostics);
        }
    } else if columns.fields_complete {
        for value in columns.fields.values() {
            super::check(value, index, diagnostics);
        }
    }
}

pub(super) fn check_network(
    object: &ObjectFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let Some(ValueFact {
        kind: ValueKindFact::Object(table),
        ..
    }) = object.properties.get("Learnables")
    else {
        return;
    };
    if table
        .runtime_class
        .as_ref()
        .is_some_and(|class| class.is(standard::TABLE))
    {
        check(table, index, diagnostics);
    }
}
