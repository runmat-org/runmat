use super::super::file_dialog::{double_scalar, filter, text, text_or_cancel};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, CellFact, FactJoin, NumericDomain, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut end = request.arguments.len();
    let mut positional_known = true;
    while end >= 2 {
        match request.literals.literal_string_at(end - 2) {
            Some(name) if name == "multiselect" => {
                if !possible_control(&request.arguments[end - 1]) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-UIGETFILE-CONTROL",
                        "MultiSelect expects an admitted scalar control",
                        end - 1,
                    ));
                }
                end -= 2;
            }
            Some(_) => break,
            None if text(&request.arguments[end - 2]) => {
                positional_known = false;
                break;
            }
            None => break,
        }
    }
    if positional_known {
        if end > 3 {
            diagnostics.push(argument_error(
                "RM-CATALOG-UIGETFILE-ARITY",
                "uigetfile accepts filter, title, default name, and MultiSelect pairs",
                3,
            ));
        }
        if end > 0 && !filter(&request.arguments[0]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-UIGETFILE-FILTER",
                "uigetfile expects scalar text or a one/two-column text filter cell",
                0,
            ));
        }
        for index in 1..end.min(3) {
            if !text(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-UIGETFILE-TEXT",
                    "uigetfile expects scalar text",
                    index,
                ));
            }
        }
    }
    let multiple = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::from(vec![Some(1), None]),
                StorageFact::Dense,
            )),
            elements: vec![],
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    finish_fixed_outputs(
        entry,
        request,
        vec![
            text_or_cancel().join(&multiple),
            text_or_cancel(),
            double_scalar(),
        ],
        diagnostics,
    )
}

fn possible_control(value: &ValueFact) -> bool {
    if matches!(value.kind, ValueKindFact::Unknown) {
        return true;
    }
    match value.kind {
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            !matches!(value.residency, ResidencyFact::Device { .. })
                && value.shape.element_count().is_none_or(|count| count == 1)
        }
        ValueKindFact::Logical => value.shape.element_count().is_none_or(|count| count == 1),
        _ => text(value),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::*;
    #[test]
    fn dynamic_option_name_does_not_become_a_positional_type_error() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        };
        let result = infer(&request, &super::super::UIGETFILE_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[2], double_scalar());
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
    #[test]
    fn known_option_allows_exact_integer_scalar_without_widening() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt64,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::String("MultiSelect".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        assert!(infer(&request, &super::super::UIGETFILE_CATALOG_ENTRY)
            .diagnostics
            .is_empty());
    }
}
