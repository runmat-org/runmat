use crate::{
    catalog::inference::{argument_error, finish_fixed, invalid_text_comparison_container},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CertaintyFact, DynamicReason, NumericClass, NumericDomain,
    NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STR2DOUBLE-ARITY",
            "str2double requires exactly one input",
            0,
        ));
    }
    let shape = match request.arguments.first() {
        Some(input) => match &input.kind {
            ValueKindFact::String => input.shape.clone(),
            ValueKindFact::Character => {
                input.shape.known_dims().map_or(ShapeFact::Unknown, |dims| {
                    ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
                })
            }
            ValueKindFact::Cell(_) if invalid_text_comparison_container(input) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STR2DOUBLE-CELL-ELEMENT",
                    "str2double cell elements must be scalar strings or character rows",
                    0,
                ));
                ShapeFact::Unknown
            }
            ValueKindFact::Cell(_) => input.shape.clone(),
            ValueKindFact::Unknown => ShapeFact::Unknown,
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STR2DOUBLE-TEXT",
                    "str2double requires text input",
                    0,
                ));
                ShapeFact::Unknown
            }
        },
        None => ShapeFact::Unknown,
    };
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    if request
        .arguments
        .first()
        .is_none_or(|input| matches!(&input.kind, ValueKindFact::Unknown))
    {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{CellFact, LiteralContext, OutputSelection};

    fn call(input: ValueFact) -> CallInference {
        infer(
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::default(),
                outputs: OutputSelection::default(),
            },
            &super::super::STR2DOUBLE_CATALOG_ENTRY,
        )
    }

    #[test]
    fn parses_text_container_shapes_to_host_double() {
        let strings = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let result = call(strings);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert_eq!(result.outputs[0].storage, StorageFact::Dense);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );

        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(4), Some(8)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(chars).outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(1)])
        );
        let cells = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::scalar(ValueKindFact::String)),
                elements: vec![],
                elements_complete: false,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        assert_eq!(
            call(cells).outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(2)])
        );
    }

    #[test]
    fn unknown_and_numeric_inputs_do_not_invent_shapes() {
        let unknown = call(ValueFact::unknown(DynamicReason::RuntimeValue));
        assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
        assert!(matches!(
            unknown.outputs[0].certainty,
            CertaintyFact::Dynamic(_)
        ));
        let invalid = call(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })));
        assert_eq!(invalid.outputs[0].shape, ShapeFact::Unknown);
        assert!(invalid
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-STR2DOUBLE-TEXT"));
        let bad_cell = ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }))),
                elements: vec![],
                elements_complete: false,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let invalid_cell = call(bad_cell);
        assert_eq!(invalid_cell.outputs[0].shape, ShapeFact::Unknown);
        assert!(invalid_cell
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-STR2DOUBLE-CELL-ELEMENT"));
    }
}
