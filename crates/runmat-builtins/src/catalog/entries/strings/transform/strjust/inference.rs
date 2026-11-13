use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, DynamicReason, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRJUST-ARITY",
            "strjust requires text input",
            0,
        ));
    }
    if let Some(side) = request.arguments.get(1) {
        if !matches!(
            &side.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) || matches!(&side.kind, ValueKindFact::String)
            && side.shape.element_count().is_some_and(|count| count != 1)
            || matches!(&side.kind, ValueKindFact::Character)
                && side.shape.known_dims().is_some_and(|dims| {
                    dims.first().copied().flatten().is_some_and(|rows| rows > 1)
                })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRJUST-SIDE",
                "strjust alignment must be scalar text",
                1,
            ));
        }
    }
    let output = request.arguments.first().map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        |input| match &input.kind {
            ValueKindFact::String => {
                ValueFact::proven(ValueKindFact::String, input.shape.clone(), input.storage)
            }
            ValueKindFact::Character => ValueFact::proven(
                ValueKindFact::Character,
                match input.shape.known_dims() {
                    Some(dimensions) if dimensions.len() == 2 => input.shape.clone(),
                    _ => ShapeFact::Unknown,
                },
                StorageFact::Dense,
            ),
            ValueKindFact::Cell(_) => ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
                    elements: Vec::new(),
                    elements_complete: false,
                }),
                input.shape.clone(),
                input.storage,
            ),
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRJUST-TEXT",
                    "strjust requires host text",
                    0,
                ));
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
        },
    );
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection, ShapeFact};

    fn call(arguments: Vec<ValueFact>) -> CallInference {
        infer(
            &CallRequest {
                arguments,
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::STRJUST_CATALOG_ENTRY,
        )
    }

    #[test]
    fn character_width_and_container_shape_are_preserved() {
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(7)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(vec![chars]).outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(7)])
        );
        let pages = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(7), Some(3)]),
            StorageFact::Dense,
        );
        assert_eq!(call(vec![pages]).outputs[0].shape, ShapeFact::Unknown);
        let strings = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(vec![strings]).outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
    }

    #[test]
    fn known_invalid_input_and_side_are_diagnosed() {
        assert!(call(vec![ValueFact::scalar(ValueKindFact::Logical)])
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-STRJUST-TEXT"));
        assert!(call(vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::Logical)
        ])
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-STRJUST-SIDE"));
    }
}
