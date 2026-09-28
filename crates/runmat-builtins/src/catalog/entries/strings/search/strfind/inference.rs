use crate::{
    catalog::inference::{
        argument_error, finish_fixed, invalid_text_comparison_container, numeric_kind,
        text_comparison_result,
    },
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, CellFact, CertaintyFact, DynamicReason, NumericClass,
    NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRFIND-ARITY",
            "strfind requires subject and pattern",
            request.arguments.len(),
        ));
    }
    if !matches!(request.arguments.len(), 2 | 4) {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRFIND-ARITY",
            "strfind accepts two inputs or one ForceCellOutput name-value pair",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request.arguments.len() == 4
        && request
            .literals
            .literal_string_at(2)
            .is_some_and(|name| name != "forcecelloutput")
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-STRFIND-OPTION",
            "strfind expects the ForceCellOutput option name",
            2,
        ));
    }
    for (index, argument) in request.arguments.iter().take(2).enumerate() {
        let invalid = invalid_text_comparison_container(argument)
            || matches!(
                &argument.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Symbolic
            );
        if invalid {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRFIND-TEXT",
                "strfind requires host text arguments",
                index,
            ));
        }
    }
    let output = match (request.arguments.first(), request.arguments.get(1)) {
        (Some(subject), Some(pattern)) => {
            let (comparison, shape_error) = text_comparison_result(subject, pattern);
            if let Some(error) = shape_error {
                diagnostics.push(error);
            }
            indices_output(subject, pattern, comparison.shape, request)
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn indices_output(
    subject: &ValueFact,
    pattern: &ValueFact,
    shape: ShapeFact,
    request: &CallRequest,
) -> ValueFact {
    if shape == ShapeFact::Unknown {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let option_is_dynamic =
        request.arguments.len() > 2 && request.literals.literal_bool_at(3).is_none();
    if option_is_dynamic {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let forced = request.literals.literal_bool_at(3).unwrap_or(false);
    let is_cell = forced
        || matches!(&subject.kind, ValueKindFact::Cell(_))
        || matches!(&pattern.kind, ValueKindFact::Cell(_));
    let return_cell = match shape.element_count() {
        Some(1) => is_cell,
        Some(_) => true,
        None if is_cell => true,
        None => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut indices = ValueFact::proven(
        numeric_kind(NumericClass::Double, NumericDomain::Real),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    indices.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    if !return_cell {
        return indices;
    }
    let mut output = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(indices),
            elements: vec![],
            elements_complete: false,
        }),
        shape,
        StorageFact::Dense,
    );
    if output.shape.element_count().is_none() {
        output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    #[test]
    fn known_wrong_option_is_diagnosed() {
        let entry = &super::super::STRFIND_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            literals: LiteralContext::new(vec![
                runmat_types::LiteralValue::Unknown,
                runmat_types::LiteralValue::Unknown,
                runmat_types::LiteralValue::String("ignorecase".into()),
                runmat_types::LiteralValue::Bool(true),
            ]),
            outputs: OutputSelection::default(),
        };
        assert!(infer(&request, entry)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STRFIND-OPTION"));
    }

    #[test]
    fn known_text_and_unknown_shapes_are_distinct() {
        let entry = &super::super::STRFIND_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(2), Some(1)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let known = infer(&request, entry);
        assert!(known.diagnostics.is_empty());
        assert_eq!(
            known.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(1)])
        );
        let mut unknown = request;
        unknown.arguments[0] = ValueFact::unknown(DynamicReason::RuntimeValue);
        assert_eq!(infer(&unknown, entry).outputs[0].shape, ShapeFact::Unknown);
    }
    #[test]
    fn invalid_text_is_diagnosed() {
        let entry = &super::super::STRFIND_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert!(infer(&request, entry)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-STRFIND-TEXT"));
    }

    #[test]
    fn unknown_element_count_does_not_choose_numeric_or_cell_output() {
        let entry = &super::super::STRFIND_CATALOG_ENTRY;
        let request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(1), None]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        assert_eq!(
            infer(&request, entry).outputs[0].kind,
            ValueKindFact::Unknown
        );
    }
}
