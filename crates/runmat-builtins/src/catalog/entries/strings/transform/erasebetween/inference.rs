use crate::{
    catalog::inference::{argument_error, finish_fixed, possible_character_row},
    BuiltinCatalogEntry,
};
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, CellFact, CertaintyFact, DynamicReason,
    LiteralValue, NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum BoundaryDomain {
    Text,
    Position,
    Unknown,
    Invalid,
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let count = request.arguments.len();
    if count < 3 || !(count - 3).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ERASE-BETWEEN-ARITY",
            "eraseBetween requires text, two boundaries, and complete name-value pairs",
            count.saturating_sub(1),
        ));
    }
    let input = request.arguments.first();
    if let Some(text) = input {
        if !matches!(
            text.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERASE-BETWEEN-TEXT",
                "eraseBetween requires text input",
                0,
            ));
        }
        if let ValueKindFact::Cell(cell) = &text.kind {
            if text.shape.element_count() != Some(0)
                && cell
                    .elements
                    .iter()
                    .chain(std::iter::once(cell.element.as_ref()))
                    .any(invalid_text_cell_element)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ERASE-BETWEEN-CELL",
                    "eraseBetween text cells must contain string scalars or character vectors",
                    0,
                ));
            }
        }
    }
    let domains = [request.arguments.get(1), request.arguments.get(2)].map(|input| {
        input
            .map(boundary_domain)
            .unwrap_or(BoundaryDomain::Unknown)
    });
    for (index, domain) in domains.iter().enumerate() {
        if *domain == BoundaryDomain::Invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERASE-BETWEEN-BOUNDARY",
                "eraseBetween boundaries must be text, patterns, or real numeric positions",
                index + 1,
            ));
        }
    }
    if matches!(
        domains,
        [BoundaryDomain::Text, BoundaryDomain::Position]
            | [BoundaryDomain::Position, BoundaryDomain::Text]
    ) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ERASE-BETWEEN-MIXED",
            "eraseBetween boundaries must use the same representation",
            1,
        ));
    }
    if domains[0] == BoundaryDomain::Position && domains[1] == BoundaryDomain::Position {
        for index in 1..=2 {
            if request
                .literals
                .literal_args
                .get(index)
                .is_some_and(invalid_position_literal)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ERASE-BETWEEN-POSITION",
                    "position boundaries must be positive integers",
                    index,
                ));
            }
        }
    }
    for index in (3..count).step_by(2) {
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|name| name != "boundaries")
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERASE-BETWEEN-OPTION",
                "the option name must be Boundaries",
                index,
            ));
        }
        if request.arguments.get(index).is_some_and(|arg| {
            !matches!(
                arg.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Cell(_)
            )
        }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERASE-BETWEEN-OPTION",
                "the option name must be text",
                index,
            ));
        }
        if request
            .literals
            .literal_string_at(index + 1)
            .is_some_and(|value| !matches!(value.as_str(), "inclusive" | "exclusive"))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ERASE-BETWEEN-OPTION",
                "Boundaries must be inclusive or exclusive",
                index + 1,
            ));
        }
    }
    let text_shape = input.map(text_element_shape).unwrap_or(ShapeFact::Unknown);
    let mut output_shape = text_shape.clone();
    for index in 1..=2 {
        let boundary_shape = request
            .arguments
            .get(index)
            .map(boundary_element_shape)
            .unwrap_or(ShapeFact::Unknown);
        match broadcast_shape(&output_shape, &boundary_shape) {
            Ok(shape) => output_shape = shape,
            Err(error) => {
                diagnostics.push(error);
                output_shape = ShapeFact::Unknown;
            }
        }
    }
    let output = match input.map(|value| &value.kind) {
        Some(ValueKindFact::String) => {
            let storage = match output_shape.element_count() {
                Some(1) => StorageFact::Scalar,
                Some(_) => StorageFact::Dense,
                None => StorageFact::Unknown,
            };
            let mut fact = ValueFact::proven(ValueKindFact::String, output_shape, storage);
            if fact.shape == ShapeFact::Unknown {
                fact.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
            }
            fact
        }
        Some(ValueKindFact::Character) => {
            diagnose_nonexpanding_text(&text_shape, &output_shape, &mut diagnostics);
            ValueFact::proven(
                ValueKindFact::Character,
                text_shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                    ShapeFact::from(vec![dims.first().copied().flatten(), None])
                }),
                StorageFact::Dense,
            )
        }
        Some(ValueKindFact::Cell(_)) => {
            diagnose_nonexpanding_text(&text_shape, &output_shape, &mut diagnostics);
            ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                    elements: Vec::new(),
                    elements_complete: false,
                }),
                text_shape,
                input.map_or(StorageFact::Unknown, |value| value.storage),
            )
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn diagnose_nonexpanding_text(
    text: &ShapeFact,
    output: &ShapeFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    if text.element_count().is_some()
        && output.element_count().is_some()
        && !text.is_proven_equivalent(output)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-ERASE-BETWEEN-SHAPE",
            "character and cell text cannot expand to the boundary shape",
            0,
        ));
    }
}

fn text_element_shape(value: &ValueFact) -> ShapeFact {
    match value.kind {
        ValueKindFact::Character => value.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
            ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
        }),
        ValueKindFact::String | ValueKindFact::Cell(_) => value.shape.clone(),
        _ => ShapeFact::Unknown,
    }
}
fn boundary_element_shape(value: &ValueFact) -> ShapeFact {
    match &value.kind {
        ValueKindFact::Object(object)
            if object.runtime_class.as_ref() == Some(&runmat_types::standard::PATTERN.owned()) =>
        {
            ShapeFact::Scalar
        }
        ValueKindFact::Character => value.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
            ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
        }),
        ValueKindFact::String | ValueKindFact::Numeric(_) | ValueKindFact::Cell(_) => {
            value.shape.clone()
        }
        _ => ShapeFact::Unknown,
    }
}
fn boundary_domain(value: &ValueFact) -> BoundaryDomain {
    match &value.kind {
        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Cell(_) => {
            BoundaryDomain::Text
        }
        ValueKindFact::Object(object)
            if object
                .runtime_class
                .as_ref()
                .is_none_or(|class| class == &runmat_types::standard::PATTERN.owned()) =>
        {
            BoundaryDomain::Text
        }
        ValueKindFact::Numeric(number) if number.domain == NumericDomain::Real => {
            BoundaryDomain::Position
        }
        ValueKindFact::Unknown => BoundaryDomain::Unknown,
        _ => BoundaryDomain::Invalid,
    }
}
fn invalid_text_cell_element(element: &ValueFact) -> bool {
    match &element.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::String => element
            .shape
            .element_count()
            .is_some_and(|count| count != 1),
        ValueKindFact::Character => !possible_character_row(&element.shape),
        _ => true,
    }
}
fn invalid_position_literal(literal: &LiteralValue) -> bool {
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
    fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        }
    }
    #[test]
    fn string_scalar_can_expand_from_position_vector() {
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let positions = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int32,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(3)]),
            StorageFact::Dense,
        );
        let result = infer(
            &request(vec![scalar, positions.clone(), positions], vec![]),
            &super::super::ERASE_BETWEEN_CATALOG_ENTRY,
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(3)])
        );
    }
    #[test]
    fn mixed_domains_and_invalid_positions_are_diagnosed() {
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let position = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Real,
        }));
        let mixed = infer(
            &request(vec![scalar.clone(), scalar, position.clone()], vec![]),
            &super::super::ERASE_BETWEEN_CATALOG_ENTRY,
        );
        assert!(mixed
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-ERASE-BETWEEN-MIXED"));
        let bad = infer(
            &request(
                vec![
                    ValueFact::scalar(ValueKindFact::String),
                    position.clone(),
                    position,
                ],
                vec![
                    LiteralValue::Unknown,
                    LiteralValue::Integer {
                        text: "0".into(),
                        class: NumericClass::Int32,
                    },
                    LiteralValue::Unknown,
                ],
            ),
            &super::super::ERASE_BETWEEN_CATALOG_ENTRY,
        );
        assert!(bad
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-ERASE-BETWEEN-POSITION"));
    }
}
