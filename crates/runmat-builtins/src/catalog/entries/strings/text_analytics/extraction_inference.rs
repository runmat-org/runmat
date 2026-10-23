//! Declaration-selected HTML/file facts. Content-dependent matches stay dynamic.
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(super) enum ExtractionOperation {
    Text,
    Subtrees,
    Attribute,
    File,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: ExtractionOperation,
) -> CallInference {
    use ExtractionOperation::*;
    let count = request.arguments.len();
    let mut diagnostics = Vec::new();
    let valid = match operation {
        Text => count == 1 || count == 3,
        Subtrees | Attribute => count == 2,
        File => count > 0 && count % 2 == 1,
    };
    if !valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXTRACTION-ARITY",
            "inputs do not match the declared extraction form",
            count,
        ));
    }
    if let Some(first) = request.arguments.first() {
        let valid = match operation {
            Text => matches!(
                first.kind,
                ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Cell(_)
                    | ValueKindFact::Object(_)
                    | ValueKindFact::Unknown
            ),
            Subtrees | Attribute => matches!(
                first.kind,
                ValueKindFact::Cell(_) | ValueKindFact::Object(_) | ValueKindFact::Unknown
            ),
            File => matches!(
                first.kind,
                ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Cell(_)
                    | ValueKindFact::Unknown
            ),
        };
        if !valid {
            diagnostics.push(argument_error(
                "RM-CATALOG-EXTRACTION-INPUT",
                "input requires host text or the declared HTML tree container",
                0,
            ));
        }
        if let ValueKindFact::Object(object) = &first.kind {
            if object
                .runtime_class
                .as_ref()
                .is_some_and(|class| *class != StaticClassIdentity::new("htmlTree").owned())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-EXTRACTION-CLASS",
                    "object input requires htmlTree",
                    0,
                ));
            }
        }
    }
    let shape = match operation {
        File => ShapeFact::Scalar,
        Subtrees => ShapeFact::from(vec![None, Some(1)]),
        Attribute => match request.arguments.first() {
            Some(value) if matches!(value.kind, ValueKindFact::Object(_)) => ShapeFact::Scalar,
            Some(value) if matches!(value.kind, ValueKindFact::Cell(_)) => {
                string_output_shape(&value.shape)
            }
            _ => ShapeFact::Unknown,
        },
        Text => match request.arguments.first() {
            Some(value) if matches!(value.kind, ValueKindFact::String | ValueKindFact::Cell(_)) => {
                string_output_shape(&value.shape)
            }
            Some(value) if matches!(value.kind, ValueKindFact::Object(_)) => ShapeFact::Scalar,
            // Character matrices are extension-gated and projected row-wise.
            Some(value) if matches!(value.kind, ValueKindFact::Character) => ShapeFact::from(vec![
                value
                    .shape
                    .known_dims()
                    .and_then(|dims| dims.first().copied().flatten()),
                Some(1),
            ]),
            _ => ShapeFact::Unknown,
        },
    };
    let output = if diagnostics.is_empty() {
        let kind = if matches!(operation, Subtrees) {
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::proven(
                    ValueKindFact::Object(ObjectFact {
                        class: None,
                        runtime_class: Some(StaticClassIdentity::new("htmlTree").owned()),
                        properties: BTreeMap::new(),
                        properties_complete: false,
                        handle_semantics: Some(false),
                    }),
                    ShapeFact::Scalar,
                    StorageFact::Opaque,
                )),
                elements: Vec::new(),
                elements_complete: false,
            })
        } else {
            ValueKindFact::String
        };
        ValueFact::proven(
            kind,
            shape,
            if matches!(operation, Subtrees) {
                StorageFact::Opaque
            } else {
                StorageFact::Dense
            },
        )
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed(entry, request, output, diagnostics)
}

// The existing HTML string owner scalarizes a single result, but preserves
// explicit empty/non-singleton geometry. Do not invent object-array geometry.
fn string_output_shape(shape: &ShapeFact) -> ShapeFact {
    match shape.element_count() {
        Some(1) => ShapeFact::Scalar,
        Some(_) => shape.clone(),
        None if shape.known_dims().is_some_and(|dims| dims.len() == 2) => shape.clone(),
        None => ShapeFact::Unknown,
    }
}
