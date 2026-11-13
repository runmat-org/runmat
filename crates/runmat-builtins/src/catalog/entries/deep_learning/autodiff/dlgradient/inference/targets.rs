//! Position-preserving facts for recursively differentiated targets.
use super::*;
use runmat_types::{CellFact, InferenceDiagnostic};

mod structs;
mod tables;

/// Metadata is copied into a new aggregate. Its nested payloads may be
/// deep-copied or CoW-shared depending on their runtime representation.
fn retained_metadata(source: &ValueFact) -> ValueFact {
    let mut result = source.clone();
    result.alias = AliasFact::Unknown;
    result.kind = match &source.kind {
        ValueKindFact::Cell(cell) => ValueKindFact::Cell(CellFact {
            element: Box::new(retained_metadata(&cell.element)),
            elements: cell.elements.iter().map(retained_metadata).collect(),
            elements_complete: cell.elements_complete,
        }),
        ValueKindFact::Struct(value) => ValueKindFact::Struct(runmat_types::StructFact {
            fields: value
                .fields
                .iter()
                .map(|(name, value)| (name.clone(), retained_metadata(value)))
                .collect(),
            fields_complete: value.fields_complete,
            elements: value
                .elements
                .iter()
                .map(|element| {
                    element
                        .iter()
                        .map(|(name, value)| (name.clone(), retained_metadata(value)))
                        .collect()
                })
                .collect(),
            elements_complete: value.elements_complete,
        }),
        ValueKindFact::Object(object) => ValueKindFact::Object(ObjectFact {
            properties: object
                .properties
                .iter()
                .map(|(name, value)| (name.clone(), retained_metadata(value)))
                .collect(),
            ..object.clone()
        }),
        _ => source.kind.clone(),
    };
    result
}

pub(super) fn gradient(target: &ValueFact) -> ValueFact {
    match &target.kind {
        ValueKindFact::Cell(cell) => {
            let mut result = ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(gradient(&cell.element)),
                    elements: cell.elements.iter().map(gradient).collect(),
                    elements_complete: cell.elements_complete,
                }),
                target.shape.clone(),
                target.storage,
            );
            // Distinct containers can retain the same cached leaf adjoints.
            result.alias = AliasFact::Unknown;
            result
        }
        ValueKindFact::Struct(value) => structs::gradient(target, value),
        ValueKindFact::Object(object)
            if object
                .runtime_class
                .as_ref()
                .is_some_and(runmat_types::standard::is_tabular) =>
        {
            tables::gradient(target, object)
        }
        ValueKindFact::Object(object)
            if object
                .runtime_class
                .as_ref()
                .is_some_and(tables::is_network) =>
        {
            tables::network_gradient(object)
        }
        _ => gradient_for_target(target),
    }
}

pub(super) fn check(target: &ValueFact, index: usize, diagnostics: &mut Vec<InferenceDiagnostic>) {
    match &target.kind {
        ValueKindFact::Cell(cell) => {
            // The common fact proves a leaf exists only when the cell is
            // known nonempty. Empty cells never visit their common fact.
            if !cell.elements_complete
                && target.shape.element_count().is_some_and(|count| count > 0)
            {
                check(&cell.element, index, diagnostics);
            }
            for element in &cell.elements {
                check(element, index, diagnostics);
            }
        }
        ValueKindFact::Object(object)
            if object
                .runtime_class
                .as_ref()
                .is_some_and(runmat_types::standard::is_tabular) =>
        {
            tables::check(object, index, diagnostics);
        }
        ValueKindFact::Object(object)
            if object
                .runtime_class
                .as_ref()
                .is_some_and(tables::is_network) =>
        {
            tables::check_network(object, index, diagnostics);
        }
        ValueKindFact::Object(_) => check_dlarray_data(target, index, diagnostics),
        ValueKindFact::Struct(value) => structs::check(target, value, index, diagnostics),
        ValueKindFact::Unknown | ValueKindFact::Never => {}
        _ => diagnostics.push(argument_error(
            "RM-DL-DLGRADIENT-TARGET",
            "targets must be traced dlarray values or supported learnables trees",
            index,
        )),
    }
}
