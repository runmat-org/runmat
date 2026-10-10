//! Static facts for host text conversion. Geometry never stands in for text contents.
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TextConversion {
    Dec2Bin,
    Dec2Hex,
    Bin2Dec,
    Hex2Dec,
}

fn known_invalid_cell_text(fact: &ValueFact) -> bool {
    match fact.kind {
        ValueKindFact::String => fact.shape.element_count().is_some_and(|count| count != 1),
        ValueKindFact::Character => fact
            .shape
            .known_dims()
            .and_then(|dims| dims.first().copied().flatten())
            .is_some_and(|rows| rows != 1),
        ValueKindFact::Unknown => false,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{CellFact, LiteralContext, OutputSelection, RequestedOutputCount};

    fn request(arguments: Vec<ValueFact>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        }
    }

    #[test]
    fn decimal_text_inference_knows_rows_but_not_unobserved_digits() {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let width = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        for (conversion, entry) in [
            (
                TextConversion::Dec2Bin,
                &super::super::DEC2BIN_CATALOG_ENTRY,
            ),
            (
                TextConversion::Dec2Hex,
                &super::super::DEC2HEX_CATALOG_ENTRY,
            ),
        ] {
            let result = infer(
                &request(vec![input.clone(), width.clone()]),
                entry,
                conversion,
            );
            assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
            assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
            assert_eq!(
                result.outputs[0].shape.known_dims(),
                Some(vec![Some(6), None])
            );
            assert_eq!(
                result.outputs[0].residency,
                runmat_types::ResidencyFact::Host
            );
            assert_eq!(entry.bindings, &crate::REQUIRED_DEFAULT_BINDING);
        }
    }

    #[test]
    fn text_to_decimal_preserves_string_shape_and_char_row_count() {
        let strings = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let chars = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(3), Some(8)]),
            StorageFact::Dense,
        );
        for (conversion, entry) in [
            (
                TextConversion::Bin2Dec,
                &super::super::BIN2DEC_CATALOG_ENTRY,
            ),
            (
                TextConversion::Hex2Dec,
                &super::super::HEX2DEC_CATALOG_ENTRY,
            ),
        ] {
            let result = infer(&request(vec![strings.clone()]), entry, conversion);
            assert_eq!(result.outputs[0].shape, strings.shape);
            assert_eq!(
                result.outputs[0].numeric().unwrap().class,
                NumericClass::Double
            );
            let result = infer(&request(vec![chars.clone()]), entry, conversion);
            assert_eq!(
                result.outputs[0].shape.known_dims(),
                Some(vec![Some(3), Some(1)])
            );
        }
    }

    #[test]
    fn invalid_class_and_width_are_diagnosed_without_claiming_width_from_shape() {
        let invalid = ValueFact::scalar(ValueKindFact::String);
        let width = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let result = infer(
            &request(vec![invalid, width]),
            &super::super::DEC2BIN_CATALOG_ENTRY,
            TextConversion::Dec2Bin,
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM.TEXTCONVERSION.INPUT"));
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM.TEXTCONVERSION.WIDTH"));
        let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        let result = infer(
            &request(vec![unknown]),
            &super::super::DEC2HEX_CATALOG_ENTRY,
            TextConversion::Dec2Hex,
        );
        assert_eq!(result.outputs[0].shape.known_dims(), Some(vec![None, None]));
    }

    #[test]
    fn requested_and_discarded_outputs_keep_fixed_single_slot() {
        let mut call = request(vec![ValueFact::scalar(ValueKindFact::String)]);
        call.outputs.discarded.insert(0);
        let result = infer(
            &call,
            &super::super::BIN2DEC_CATALOG_ENTRY,
            TextConversion::Bin2Dec,
        );
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.discarded, vec![0]);
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Double
        );
        call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
        let invalid = infer(
            &call,
            &super::super::BIN2DEC_CATALOG_ENTRY,
            TextConversion::Bin2Dec,
        );
        assert!(!invalid.diagnostics.is_empty());
    }

    #[test]
    fn logical_width_is_admitted_but_nonscalar_logical_width_is_not() {
        let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }));
        let logical = ValueFact::scalar(ValueKindFact::Logical);
        for (conversion, entry) in [
            (
                TextConversion::Dec2Bin,
                &super::super::DEC2BIN_CATALOG_ENTRY,
            ),
            (
                TextConversion::Dec2Hex,
                &super::super::DEC2HEX_CATALOG_ENTRY,
            ),
        ] {
            let result = infer(
                &request(vec![input.clone(), logical.clone()]),
                entry,
                conversion,
            );
            assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
            let nonscalar = ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::from(vec![Some(1), Some(2)]),
                StorageFact::Dense,
            );
            let invalid = infer(&request(vec![input.clone(), nonscalar]), entry, conversion);
            assert!(invalid
                .diagnostics
                .iter()
                .any(|d| d.code == "RM.TEXTCONVERSION.WIDTH"));
        }
    }

    #[test]
    fn singleton_text_arrays_scalarize_but_empty_string_shape_survives() {
        for (conversion, entry) in [
            (
                TextConversion::Bin2Dec,
                &super::super::BIN2DEC_CATALOG_ENTRY,
            ),
            (
                TextConversion::Hex2Dec,
                &super::super::HEX2DEC_CATALOG_ENTRY,
            ),
        ] {
            for dims in [vec![Some(1)], vec![Some(1), Some(1), Some(1)]] {
                let input = ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(dims),
                    StorageFact::Dense,
                );
                let result = infer(&request(vec![input]), entry, conversion);
                assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
                assert_eq!(result.outputs[0].storage, StorageFact::Scalar);
            }
            let empty_shape = ShapeFact::from(vec![Some(0), Some(2)]);
            let empty = ValueFact::proven(
                ValueKindFact::String,
                empty_shape.clone(),
                StorageFact::Dense,
            );
            let result = infer(&request(vec![empty]), entry, conversion);
            assert_eq!(result.outputs[0].shape, empty_shape);
            assert_eq!(result.outputs[0].storage, StorageFact::Dense);
        }
    }

    #[test]
    fn known_invalid_nested_text_geometry_is_diagnosed() {
        let cell = |element: ValueFact| {
            ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(element.clone()),
                    elements: vec![element],
                    elements_complete: true,
                }),
                ShapeFact::Scalar,
                StorageFact::Dense,
            )
        };
        let multi_string = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(1), Some(2)]),
            StorageFact::Dense,
        );
        let multi_char = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        for nested in [multi_string, multi_char] {
            let result = infer(
                &request(vec![cell(nested)]),
                &super::super::HEX2DEC_CATALOG_ENTRY,
                TextConversion::Hex2Dec,
            );
            assert!(result
                .diagnostics
                .iter()
                .any(|d| d.code == "RM.TEXTCONVERSION.CELL"));
        }
        let uncertain = ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![None, Some(1)]),
            StorageFact::Dense,
        );
        let result = infer(
            &request(vec![cell(uncertain)]),
            &super::super::BIN2DEC_CATALOG_ENTRY,
            TextConversion::Bin2Dec,
        );
        assert!(!result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM.TEXTCONVERSION.CELL"));
    }
}

impl TextConversion {
    fn name(self) -> &'static str {
        match self {
            Self::Dec2Bin => "dec2bin",
            Self::Dec2Hex => "dec2hex",
            Self::Bin2Dec => "bin2dec",
            Self::Hex2Dec => "hex2dec",
        }
    }

    fn numeric_to_text(self) -> bool {
        matches!(self, Self::Dec2Bin | Self::Dec2Hex)
    }
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    conversion: TextConversion,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let input = request.arguments.first();
    let numeric_to_text = conversion.numeric_to_text();
    let expected = if numeric_to_text { "one or two" } else { "one" };
    if request.arguments.len() != 1 && !(numeric_to_text && request.arguments.len() == 2) {
        diagnostics.push(argument_error(
            "RM.TEXTCONVERSION.ARITY",
            format!("{} expects {expected} input(s)", conversion.name()),
            0,
        ));
    }
    if let Some(input) = input {
        let admitted = if numeric_to_text {
            matches!(
                input.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Character
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            )
        } else {
            matches!(
                input.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Character
                    | ValueKindFact::String
                    | ValueKindFact::Cell(_)
            )
        };
        if !admitted {
            diagnostics.push(argument_error(
                "RM.TEXTCONVERSION.INPUT",
                format!("{} input has an unsupported class", conversion.name()),
                0,
            ));
        }
        if !numeric_to_text {
            if let ValueKindFact::Cell(cell) = &input.kind {
                if known_invalid_cell_text(&cell.element)
                    || cell.elements.iter().any(known_invalid_cell_text)
                {
                    diagnostics.push(argument_error(
                        "RM.TEXTCONVERSION.CELL",
                        "cell elements must be text",
                        0,
                    ));
                }
            }
        }
    }
    if numeric_to_text {
        if let Some(width) = request.literals.literal_args.get(1) {
            if runmat_types::LiteralContext::numeric_from_literal(width)
                .is_some_and(|value| !value.is_finite() || value < 0.0 || value.fract() != 0.0)
            {
                diagnostics.push(argument_error(
                    "RM.TEXTCONVERSION.WIDTH",
                    "minDigits must be a nonnegative integer scalar",
                    1,
                ));
            }
        }
        if let Some(control) = request.arguments.get(1) {
            if !matches!(
                control.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            ) || control.shape.is_definitely_nonscalar()
            {
                diagnostics.push(argument_error(
                    "RM.TEXTCONVERSION.WIDTH",
                    "minDigits must be a numeric or logical scalar",
                    1,
                ));
            }
        }
    }
    let output = if numeric_to_text {
        let rows = input.and_then(|fact| fact.shape.element_count());
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![rows, None]),
            StorageFact::Dense,
        )
    } else {
        let shape = match input.map(|fact| &fact.kind) {
            Some(ValueKindFact::String) => {
                input.map_or(ShapeFact::Unknown, |fact| fact.shape.clone())
            }
            Some(ValueKindFact::Character) => {
                let rows = input
                    .and_then(|fact| fact.shape.known_dims())
                    .and_then(|dims| dims.first().copied().flatten());
                ShapeFact::from(vec![rows, Some(1)])
            }
            Some(ValueKindFact::Cell(_)) => ShapeFact::from(vec![
                input.and_then(|fact| fact.shape.element_count()),
                Some(1),
            ]),
            _ => ShapeFact::Unknown,
        };
        let shape = if shape.element_count() == Some(1) {
            ShapeFact::Scalar
        } else {
            shape
        };
        let storage = if shape.element_count() == Some(1) {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        };
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            storage,
        )
    };
    let mut result = finish_fixed(entry, request, output, diagnostics);
    if !result.diagnostics.is_empty() && input.is_none() {
        for output in &mut result.outputs {
            *output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    }
    result
}
