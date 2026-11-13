use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_insert(request, entry)
}

pub(in crate::catalog::entries::strings::transform) fn infer_insert(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    use super::super::extractbefore::{
        boundary_element_shape, invalid_boundary_cell_element, invalid_boundary_kind,
        invalid_position_literal, invalid_text_element, text_element_shape,
    };
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-INSERT-ARITY",
            format!("{name} requires exactly three arguments"),
            request.arguments.len().min(3),
        ));
    }
    let text = request.arguments.first();
    if let Some(text) = text {
        if !matches!(
            text.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-INSERT-TEXT",
                format!("{name} requires text input"),
                0,
            ));
        }
        if let ValueKindFact::Cell(cell) = &text.kind {
            if text.shape.element_count() != Some(0)
                && cell
                    .elements
                    .iter()
                    .chain(std::iter::once(cell.element.as_ref()))
                    .any(invalid_text_element)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INSERT-CELL",
                    "source cells must contain text scalars",
                    0,
                ));
            }
        }
    }
    let boundary = request.arguments.get(1);
    if let Some(boundary) = boundary {
        if invalid_boundary_kind(boundary) {
            diagnostics.push(argument_error(
                "RM-CATALOG-INSERT-BOUNDARY",
                format!("{name} requires a text, pattern, or real numeric boundary"),
                1,
            ));
        }
        if let ValueKindFact::Cell(cell) = &boundary.kind {
            if boundary.shape.element_count() != Some(0)
                && cell
                    .elements
                    .iter()
                    .chain(std::iter::once(cell.element.as_ref()))
                    .any(invalid_boundary_cell_element)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INSERT-BOUNDARY-CELL",
                    "boundary cells must contain text scalars",
                    1,
                ));
            }
        }
        if request
            .literals
            .literal_args
            .get(1)
            .is_some_and(invalid_position_literal)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-INSERT-POSITION",
                "numeric positions must be positive integers",
                1,
            ));
        }
    }
    let replacement = request.arguments.get(2);
    if let Some(replacement) = replacement {
        if !matches!(
            replacement.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-INSERT-REPLACEMENT",
                "newText must be text",
                2,
            ));
        }
        if let ValueKindFact::Cell(cell) = &replacement.kind {
            if replacement.shape.element_count() != Some(0)
                && cell
                    .elements
                    .iter()
                    .chain(std::iter::once(cell.element.as_ref()))
                    .any(invalid_boundary_cell_element)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INSERT-REPLACEMENT-CELL",
                    "newText cells must contain text scalars",
                    2,
                ));
            }
        }
    }
    let text_shape = text.map(text_element_shape).unwrap_or(ShapeFact::Unknown);
    for (index, control) in [(1, boundary), (2, replacement)] {
        if let Some(control) = control {
            let control_shape = boundary_element_shape(control);
            if control_shape
                .element_count()
                .is_some_and(|count| count != 1)
                && text_shape.element_count().is_some()
                && !text_shape.is_proven_equivalent(&control_shape)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-INSERT-SHAPE",
                    "boundary and newText must be scalar or the same size as str",
                    index,
                ));
            }
        }
    }
    let output = match text {
        Some(ValueFact {
            kind: ValueKindFact::String,
            shape,
            storage,
            ..
        }) => ValueFact::proven(ValueKindFact::String, shape.clone(), *storage),
        Some(ValueFact {
            kind: ValueKindFact::Character,
            shape,
            ..
        }) => ValueFact::proven(
            ValueKindFact::Character,
            shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                ShapeFact::from(vec![dims.first().copied().flatten(), None])
            }),
            StorageFact::Dense,
        ),
        Some(ValueFact {
            kind: ValueKindFact::Cell(_),
            shape,
            storage,
            ..
        }) => ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: Vec::new(),
                elements_complete: false,
            }),
            shape.clone(),
            *storage,
        ),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection};

    #[test]
    fn preserves_text_shape_and_rejects_invalid_position() {
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        );
        let pos = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Real,
        }));
        let inserted = ValueFact::scalar(ValueKindFact::String);
        let request = CallRequest {
            arguments: vec![text.clone(), pos, inserted],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            infer(&request, &super::super::INSERT_BEFORE_CATALOG_ENTRY).outputs[0].shape,
            text.shape
        );
    }
}
