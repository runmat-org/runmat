use super::CharacterPropertyCategory;
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, CertaintyFact, DynamicReason, LiteralValue, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 2 | 4) {
        diagnostics.push(argument_error(
            "RM-CATALOG-ISSTRPROP-ARITY",
            "isstrprop requires text, category, and optionally a ForceCellOutput pair",
            request.arguments.len().min(2),
        ));
    }
    if let Some(category) = request.arguments.get(1) {
        if !matches!(
            &category.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-ISSTRPROP-CATEGORY",
                "category must be scalar text",
                1,
            ));
        }
        if matches!(&category.kind, ValueKindFact::String)
            && category
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ISSTRPROP-CATEGORY",
                "category must be scalar text",
                1,
            ));
        }
        if matches!(&category.kind, ValueKindFact::Character)
            && category
                .shape
                .known_dims()
                .is_some_and(|dims| dims.first().copied().flatten().is_some_and(|rows| rows > 1))
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-ISSTRPROP-CATEGORY",
                "category must be a character row",
                1,
            ));
        }
        if let Some(category) = request.literals.literal_string_at(1) {
            if CharacterPropertyCategory::parse(&category).is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ISSTRPROP-CATEGORY",
                    "unknown character category",
                    1,
                ));
            }
        }
    }
    let force_cell = if request.arguments.len() == 4 {
        if let Some(option) = request.arguments.get(2) {
            if !matches!(
                &option.kind,
                ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ISSTRPROP-OPTION",
                    "ForceCellOutput option name must be text",
                    2,
                ));
            }
        }
        if let Some(option) = request.literals.literal_string_at(2) {
            if option != "forcecelloutput" {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ISSTRPROP-OPTION",
                    "only ForceCellOutput is supported",
                    2,
                ));
            }
        }
        if let Some(flag) = request.arguments.get(3) {
            if !matches!(
                &flag.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(runmat_types::NumericFact {
                        class: runmat_types::NumericClass::Double,
                        ..
                    })
            ) || flag.shape.element_count().is_some_and(|count| count != 1)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ISSTRPROP-FLAG",
                    "ForceCellOutput must be a logical scalar or numeric zero/one",
                    3,
                ));
            }
        }
        match request.literals.literal_args.get(3) {
            Some(LiteralValue::Bool(value)) => Some(*value),
            Some(LiteralValue::Number(value)) if *value == 0.0 || *value == 1.0 => {
                Some(*value == 1.0)
            }
            Some(LiteralValue::Number(_)) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-ISSTRPROP-FLAG",
                    "ForceCellOutput numeric flag must be zero or one",
                    3,
                ));
                None
            }
            _ => None,
        }
    } else {
        Some(false)
    };
    let output = match request.arguments.first() {
        Some(input) => {
            let base = match &input.kind {
                ValueKindFact::Numeric(_) | ValueKindFact::Character => {
                    logical(input.shape.clone())
                }
                ValueKindFact::String if input.storage == StorageFact::Scalar => {
                    let shape = match request.literals.literal_args.first() {
                        Some(LiteralValue::String(text)) => {
                            ShapeFact::from(vec![Some(1), Some(text.chars().count())])
                        }
                        _ => ShapeFact::Unknown,
                    };
                    logical(shape)
                }
                ValueKindFact::String if input.storage == StorageFact::Dense => {
                    cell(input.shape.clone())
                }
                ValueKindFact::String => ValueFact::unknown(DynamicReason::RuntimeValue),
                ValueKindFact::Cell(facts) => {
                    if invalid_cell_text(&facts.element)
                        || facts.elements.iter().any(invalid_cell_text)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-ISSTRPROP-TEXT",
                            "cell elements must be text containers",
                            0,
                        ));
                    }
                    cell(input.shape.clone())
                }
                ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
                _ => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-ISSTRPROP-TEXT",
                        "text must be character, string, cell, or numeric character codes",
                        0,
                    ));
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
            };
            match force_cell {
                Some(true) if matches!(&base.kind, ValueKindFact::Unknown) => {
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
                Some(true) if !matches!(&base.kind, ValueKindFact::Cell(_)) => {
                    cell(ShapeFact::Scalar)
                }
                Some(_) => base,
                None => ValueFact::unknown(DynamicReason::RuntimeValue),
            }
        }
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn invalid_cell_text(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character => false,
        ValueKindFact::Cell(facts) => {
            invalid_cell_text(&facts.element) || facts.elements.iter().any(invalid_cell_text)
        }
        _ => true,
    }
}

fn logical(shape: ShapeFact) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut output = ValueFact::proven(ValueKindFact::Logical, shape, storage);
    if output.shape == ShapeFact::Unknown {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}

fn cell(shape: ShapeFact) -> ValueFact {
    let mut output = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(logical(ShapeFact::Unknown)),
            elements: vec![],
            elements_complete: false,
        }),
        shape,
        StorageFact::Dense,
    );
    if output.shape == ShapeFact::Unknown {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    #[test]
    fn numeric_shape_is_preserved_and_force_cell_wraps() {
        let numeric = ValueFact::proven(
            ValueKindFact::Numeric(runmat_types::NumericFact {
                class: runmat_types::NumericClass::UInt16,
                domain: runmat_types::NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let mut request = CallRequest {
            arguments: vec![numeric, ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("alpha".into()),
            ]),
            outputs: OutputSelection::default(),
        };
        let entry = &super::super::ISSTRPROP_CATALOG_ENTRY;
        assert_eq!(
            infer(&request, entry).outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)])
        );
        request.arguments.extend([
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ]);
        request.literals.literal_args.extend([
            LiteralValue::String("ForceCellOutput".into()),
            LiteralValue::Bool(true),
        ]);
        let result = infer(&request, entry);
        assert!(matches!(&result.outputs[0].kind, ValueKindFact::Cell(_)));
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    }

    #[test]
    fn known_invalid_category_and_flag_are_diagnosed() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
                    class: runmat_types::NumericClass::Double,
                    domain: runmat_types::NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::String("a".into()),
                LiteralValue::String("unknown".into()),
                LiteralValue::String("ForceCellOutput".into()),
                LiteralValue::Number(2.0),
            ]),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::ISSTRPROP_CATALOG_ENTRY);
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-ISSTRPROP-CATEGORY"));
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-ISSTRPROP-FLAG"));
    }
}
