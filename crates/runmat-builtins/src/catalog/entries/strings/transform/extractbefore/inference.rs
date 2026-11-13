use crate::{
    catalog::inference::{argument_error, finish_fixed, possible_character_row},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, LiteralValue, NumericDomain, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_boundary(request, entry)
}

pub(in crate::catalog::entries::strings::transform) fn infer_boundary(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-EXTRACT-ARITY",
            format!("{name} requires text and one boundary"),
            request.arguments.len().min(2),
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
                "RM-CATALOG-EXTRACT-TEXT",
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
                    "RM-CATALOG-EXTRACT-CELL",
                    "text cells must contain character vectors or string scalars",
                    0,
                ));
            }
        }
    }
    let boundary = request.arguments.get(1);
    if let Some(boundary) = boundary {
        if invalid_boundary_kind(boundary) {
            diagnostics.push(argument_error(
                "RM-CATALOG-EXTRACT-BOUNDARY",
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
                    "RM-CATALOG-EXTRACT-BOUNDARY-CELL",
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
                "RM-CATALOG-EXTRACT-POSITION",
                "numeric positions must be positive integers",
                1,
            ));
        }
        let text_shape = text.map(text_element_shape).unwrap_or(ShapeFact::Unknown);
        let boundary_shape = boundary_element_shape(boundary);
        if boundary_shape
            .element_count()
            .is_some_and(|count| count != 1)
            && text_shape.element_count().is_some()
            && !text_shape.is_proven_equivalent(&boundary_shape)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-EXTRACT-SHAPE",
                "boundary must be scalar or the same size as text",
                1,
            ));
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

pub(in crate::catalog::entries::strings::transform) fn text_element_shape(
    value: &ValueFact,
) -> ShapeFact {
    match &value.kind {
        ValueKindFact::Character => value.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
            ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
        }),
        ValueKindFact::String | ValueKindFact::Cell(_) => value.shape.clone(),
        _ => ShapeFact::Unknown,
    }
}

pub(in crate::catalog::entries::strings::transform) fn boundary_element_shape(
    value: &ValueFact,
) -> ShapeFact {
    match &value.kind {
        ValueKindFact::Character => value.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
            ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
        }),
        ValueKindFact::String | ValueKindFact::Cell(_) | ValueKindFact::Numeric(_) => {
            value.shape.clone()
        }
        ValueKindFact::Object(object)
            if object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class == &runmat_types::standard::PATTERN.owned()) =>
        {
            ShapeFact::Scalar
        }
        _ => ShapeFact::Unknown,
    }
}

pub(in crate::catalog::entries::strings::transform) fn invalid_boundary_kind(
    value: &ValueFact,
) -> bool {
    match &value.kind {
        ValueKindFact::Unknown
        | ValueKindFact::String
        | ValueKindFact::Character
        | ValueKindFact::Cell(_) => false,
        ValueKindFact::Numeric(number) => number.domain != NumericDomain::Real,
        ValueKindFact::Object(object) => object
            .runtime_class
            .as_ref()
            .is_some_and(|class| class != &runmat_types::standard::PATTERN.owned()),
        _ => true,
    }
}

pub(in crate::catalog::entries::strings::transform) fn invalid_text_element(
    value: &ValueFact,
) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => value.shape.element_count().is_some_and(|count| count != 1),
        ValueKindFact::Character => !possible_character_row(&value.shape),
        ValueKindFact::Cell(cell) => cell
            .elements
            .iter()
            .chain(std::iter::once(cell.element.as_ref()))
            .any(invalid_text_element),
        _ => true,
    }
}

pub(in crate::catalog::entries::strings::transform) fn invalid_boundary_cell_element(
    value: &ValueFact,
) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => value.shape.element_count().is_some_and(|count| count != 1),
        ValueKindFact::Character => !possible_character_row(&value.shape),
        _ => true,
    }
}

pub(in crate::catalog::entries::strings::transform) fn invalid_position_literal(
    literal: &LiteralValue,
) -> bool {
    match literal {
        LiteralValue::Integer { text, .. } => text.parse::<i128>().is_ok_and(|value| value <= 0),
        LiteralValue::Number(value) => !value.is_finite() || *value < 1.0 || value.fract() != 0.0,
        LiteralValue::Real { text, .. } => text
            .parse::<f64>()
            .is_ok_and(|value| !value.is_finite() || value < 1.0 || value.fract() != 0.0),
        LiteralValue::Vector(values) => values.iter().any(invalid_position_literal),
        LiteralValue::Matrix(rows) => rows.iter().flatten().any(invalid_position_literal),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, NumericClass, NumericFact, OutputSelection};

    fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
        infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::new(literals),
                outputs: OutputSelection::default(),
            },
            &super::super::EXTRACT_BEFORE_CATALOG_ENTRY,
        )
    }

    #[test]
    fn preserves_array_container_and_reports_bad_controls() {
        let text = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        );
        let boundary = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        assert_eq!(
            call(vec![text.clone(), boundary.clone()], vec![]).outputs[0].shape,
            text.shape
        );
        let bad = call(
            vec![text, boundary],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Integer {
                    text: "0".into(),
                    class: NumericClass::UInt64,
                },
            ],
        );
        assert!(bad
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-EXTRACT-POSITION"));
    }
}
