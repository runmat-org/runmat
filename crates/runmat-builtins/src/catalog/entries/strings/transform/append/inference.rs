use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    broadcast_shape, CallInference, CallRequest, CellFact, CertaintyFact, DynamicReason, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    if request.arguments.is_empty() {
        return finish_fixed(
            entry,
            request,
            ValueFact::scalar(ValueKindFact::String),
            Vec::new(),
        );
    }
    let mut diagnostics = Vec::new();
    let mut output_kind = ValueKindFact::Character;
    let mut shape = ShapeFact::Scalar;
    let mut unknown_kind = false;
    for (index, input) in request.arguments.iter().enumerate() {
        let input_shape = match &input.kind {
            ValueKindFact::String => {
                output_kind = ValueKindFact::String;
                input.shape.clone()
            }
            ValueKindFact::Cell(cell) => {
                if input.shape.element_count() != Some(0)
                    && !cell
                        .elements
                        .iter()
                        .chain(std::iter::once(cell.element.as_ref()))
                        .all(|value| {
                            matches!(
                                value.kind,
                                ValueKindFact::Unknown | ValueKindFact::Character
                            )
                        })
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-APPEND-CELLSTR",
                        "append cell elements must be character vectors",
                        index,
                    ));
                }
                if output_kind != ValueKindFact::String {
                    output_kind = ValueKindFact::Cell(CellFact {
                        element: Box::new(ValueFact::proven(
                            ValueKindFact::Character,
                            ShapeFact::Unknown,
                            StorageFact::Unknown,
                        )),
                        elements: Vec::new(),
                        elements_complete: false,
                    });
                }
                input.shape.clone()
            }
            ValueKindFact::Character => {
                if input.shape.known_dims().is_some_and(|dims| {
                    dims.iter().skip(2).any(|dim| dim.is_some_and(|n| n != 1))
                        || dims.first().copied().flatten().is_some_and(|n| n > 1)
                            && dims.get(1).copied().flatten().is_some_and(|n| n > 1)
                }) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-APPEND-CHAR",
                        "append character inputs must be vectors",
                        index,
                    ));
                }
                ShapeFact::Scalar
            }
            ValueKindFact::Unknown => {
                unknown_kind = true;
                ShapeFact::Unknown
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-APPEND-TEXT",
                    "append requires text values",
                    index,
                ));
                unknown_kind = true;
                ShapeFact::Unknown
            }
        };
        match broadcast_shape(&shape, &input_shape) {
            Ok(next) => shape = next,
            Err(error) => {
                diagnostics.push(error);
                shape = ShapeFact::Unknown;
            }
        }
    }
    let output = if unknown_kind {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else if output_kind == ValueKindFact::Character {
        // Character output is one row; its width depends on runtime text contents.
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        )
    } else {
        let storage = match shape.element_count() {
            Some(1) => StorageFact::Scalar,
            Some(_) => StorageFact::Dense,
            None => StorageFact::Unknown,
        };
        let mut result = ValueFact::proven(output_kind, shape, storage);
        if result.shape == ShapeFact::Unknown {
            result.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
        }
        result
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{CellFact, LiteralContext, OutputSelection};

    fn call(arguments: Vec<ValueFact>) -> CallInference {
        infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::APPEND_CATALOG_ENTRY,
        )
    }

    #[test]
    fn string_precedence_and_broadcast_shape() {
        let column = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        );
        let row = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(3)]),
            StorageFact::Dense,
        );
        let result = call(vec![column, row]);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::String);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }

    #[test]
    fn unknown_or_nontext_never_proves_a_text_output() {
        assert_eq!(
            call(vec![ValueFact::unknown(DynamicReason::RuntimeValue)]).outputs[0].kind,
            ValueKindFact::Unknown
        );
        assert!(call(vec![ValueFact::scalar(ValueKindFact::Logical)])
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-APPEND-TEXT"));
    }

    #[test]
    fn empty_cell_does_not_invent_an_invalid_element() {
        let empty = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::never()),
                elements: Vec::new(),
                elements_complete: true,
            }),
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        );
        assert!(call(vec![empty]).diagnostics.is_empty());
    }
}
