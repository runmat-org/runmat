use runmat_types::{
    broadcast_shape, DimensionFact, DynamicReason, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

/// A scalar string or one character row is required at name-value boundaries.
/// Unknown geometry stays conservative rather than inventing a scalar proof.
pub(crate) fn known_invalid_text_scalar(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::String => match &value.shape {
            ShapeFact::Shaped { dims } => dims
                .iter()
                .any(|dim| matches!(dim, DimensionFact::Known(extent) if *extent != 1)),
            _ => false,
        },
        ValueKindFact::Character => match &value.shape {
            ShapeFact::Shaped { dims } => {
                dims.first()
                    .is_some_and(|dim| matches!(dim, DimensionFact::Known(rows) if *rows != 1))
                    || dims
                        .iter()
                        .skip(2)
                        .any(|dim| matches!(dim, DimensionFact::Known(extent) if *extent != 1))
            }
            _ => false,
        },
        _ => false,
    }
}

/// A row of characters, allowing trailing singleton dimensions. Unknown
/// dimensions cannot prove rejection; zero-row empty text retains its policy.
pub(crate) fn possible_character_row(shape: &ShapeFact) -> bool {
    shape.known_dims().is_none_or(|dims| {
        dims.first().copied().flatten().is_none_or(|rows| rows <= 1)
            && !dims
                .iter()
                .skip(2)
                .any(|dim| dim.is_some_and(|size| size != 1))
    })
}

/// Shape of the text elements compared by strcmp/strcmpi, not the shape of
/// the character storage. Character rows each contribute one text element.
pub(crate) fn text_comparison_shape(value: &ValueFact) -> Option<ShapeFact> {
    match &value.kind {
        ValueKindFact::String => Some(value.shape.clone()),
        ValueKindFact::Character => value
            .shape
            .known_dims()
            .map(|dims| ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])),
        ValueKindFact::Cell(_) => Some(value.shape.clone()),
        _ => None,
    }
}

/// Known nontext elements of a cell container are an input error. Direct
/// numeric operands are handled separately by strcmp's documented false
/// result and therefore are not rejected here.
pub(crate) fn invalid_text_comparison_container(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character => false,
        ValueKindFact::Cell(cell) => {
            fn invalid_element(value: &ValueFact) -> bool {
                match &value.kind {
                    ValueKindFact::Unknown => false,
                    ValueKindFact::String => {
                        value.shape.element_count().is_some_and(|count| count != 1)
                    }
                    ValueKindFact::Character => !possible_character_row(&value.shape),
                    _ => true,
                }
            }
            invalid_element(&cell.element) || cell.elements.iter().any(invalid_element)
        }
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Symbolic => false,
        _ => true,
    }
}

pub(crate) fn text_comparison_result(
    left: &ValueFact,
    right: &ValueFact,
) -> (ValueFact, Option<runmat_types::InferenceDiagnostic>) {
    // The runtime returns scalar false before provider access for direct
    // numeric/resident operands. Unknown inputs can still produce a logical
    // result, but their shape cannot be established from an argument shape.
    if matches!(
        &left.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Symbolic
    ) || matches!(
        &right.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Symbolic
    ) {
        return (ValueFact::scalar(ValueKindFact::Logical), None);
    }
    let (Some(left_shape), Some(right_shape)) =
        (text_comparison_shape(left), text_comparison_shape(right))
    else {
        let mut result = ValueFact::unknown(DynamicReason::RuntimeValue);
        result.kind = ValueKindFact::Logical;
        return (result, None);
    };
    match broadcast_shape(&left_shape, &right_shape) {
        Ok(shape) => {
            let storage = match shape.element_count() {
                Some(1) => StorageFact::Scalar,
                Some(_) => StorageFact::Dense,
                None => StorageFact::Unknown,
            };
            let mut output = ValueFact::proven(ValueKindFact::Logical, shape, storage);
            if output.shape == ShapeFact::Unknown {
                output.certainty =
                    runmat_types::CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
            }
            (output, None)
        }
        Err(diagnostic) => (
            ValueFact::unknown(DynamicReason::RuntimeValue),
            Some(diagnostic),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{CellFact, NumericClass, NumericDomain, NumericFact, ValueKindFact};

    #[test]
    fn character_rows_preserve_unknowns_and_semantic_rank() {
        for shape in [
            vec![Some(1), Some(4)],
            vec![Some(1), Some(4), Some(1)],
            vec![None, Some(4), None],
            vec![Some(0), Some(0)],
        ] {
            assert!(possible_character_row(&ShapeFact::from(shape)));
        }
        for shape in [
            vec![Some(2), Some(2)],
            vec![Some(1), Some(2), Some(2)],
            vec![None, None, Some(2)],
        ] {
            assert!(!possible_character_row(&ShapeFact::from(shape)));
        }
    }

    #[test]
    fn text_comparison_uses_element_shapes_not_character_widths() {
        let character_rows = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(3), Some(8)]),
            StorageFact::Dense,
        );
        let string_row = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let (output, diagnostic) = text_comparison_result(&character_rows, &string_row);
        assert!(diagnostic.is_none());
        assert_eq!(output.kind, ValueKindFact::Logical);
        assert_eq!(output.shape, ShapeFact::from(vec![Some(3), Some(2)]));
        assert_eq!(output.storage, StorageFact::Dense);
    }

    #[test]
    fn text_comparison_does_not_invent_shape_from_unknown_inputs() {
        let mut unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        unknown.shape = ShapeFact::from(vec![Some(1), Some(2)]);
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let (output, diagnostic) = text_comparison_result(&unknown, &scalar);
        assert!(diagnostic.is_none());
        assert_eq!(output.kind, ValueKindFact::Logical);
        assert_eq!(output.shape, ShapeFact::Unknown);
    }

    #[test]
    fn numeric_input_returns_scalar_false_and_invalid_cell_is_diagnosed() {
        let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }));
        let scalar = ValueFact::scalar(ValueKindFact::String);
        let (output, diagnostic) = text_comparison_result(&numeric, &scalar);
        assert!(diagnostic.is_none());
        assert_eq!(output, ValueFact::scalar(ValueKindFact::Logical));

        let bad_cell = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(numeric),
                elements: vec![],
                elements_complete: false,
            }),
            ShapeFact::Scalar,
            StorageFact::Dense,
        );
        assert!(invalid_text_comparison_container(&bad_cell));
    }

    #[test]
    fn cell_string_elements_must_be_scalar_when_shape_is_known() {
        let array_element = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let invalid = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                elements: vec![array_element],
                elements_complete: true,
            }),
            ShapeFact::Scalar,
            StorageFact::Dense,
        );
        assert!(invalid_text_comparison_container(&invalid));

        let scalar = ValueFact::scalar(ValueKindFact::String);
        let valid = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(scalar),
                elements: vec![],
                elements_complete: true,
            }),
            ShapeFact::Scalar,
            StorageFact::Dense,
        );
        assert!(!invalid_text_comparison_container(&valid));
    }

    #[test]
    fn incompatible_text_shapes_produce_a_diagnostic() {
        let left = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        );
        let right = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(3), Some(2)]),
            StorageFact::Dense,
        );
        let (_, diagnostic) = text_comparison_result(&left, &right);
        assert!(diagnostic.is_some());
    }
}
