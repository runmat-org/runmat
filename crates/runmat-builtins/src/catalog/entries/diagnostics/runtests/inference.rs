use super::RuntestsOption;
use crate::catalog::inference::{argument_error, finish_fixed_outputs, known_invalid_text_scalar};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn text_collection(value: &ValueFact) -> bool {
    if matches!(
        value.residency,
        ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
    ) {
        return false;
    }
    match &value.kind {
        ValueKindFact::Unknown | ValueKindFact::String => true,
        ValueKindFact::Character => !known_invalid_text_scalar(value),
        ValueKindFact::Cell(cell) => {
            scalar_text(&cell.element) && cell.elements.iter().all(scalar_text)
        }
        _ => false,
    }
}

fn scalar_text(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    ) && !known_invalid_text_scalar(value)
        && !matches!(
            value.residency,
            ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
        )
}

fn literal_flag(value: &LiteralValue) -> Option<Result<bool, ()>> {
    match value {
        LiteralValue::Bool(value) => Some(Ok(*value)),
        LiteralValue::Number(value) => Some(if *value == 0.0 {
            Ok(false)
        } else if *value == 1.0 {
            Ok(true)
        } else {
            Err(())
        }),
        LiteralValue::Integer { .. } => value.integer_value_i128().map(|value| match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(()),
        }),
        _ => None,
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let first_option = request
        .literals
        .literal_string_at(0)
        .and_then(|name| RuntestsOption::parse(&name));
    let start = usize::from(!request.arguments.is_empty() && first_option.is_none());
    if start == 1 && !text_collection(&request.arguments[0]) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RUNTESTS-TARGET",
            "test targets must be host text or a cell of scalar text",
            0,
        ));
    }
    // A dynamic first text argument could be either a target or an option.
    let dynamic_first =
        !request.arguments.is_empty() && request.literals.literal_string_at(0).is_none();
    if !dynamic_first && !(request.arguments.len() - start).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-RUNTESTS-PAIRS",
            "test options require name/value pairs",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if !dynamic_first {
        for index in (start..request.arguments.len()).step_by(2) {
            if !scalar_text(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RUNTESTS-OPTION",
                    "test option names must be scalar text",
                    index,
                ));
            }
            let Some(name) = request.literals.literal_string_at(index) else {
                continue;
            };
            let Some(option) = RuntestsOption::parse(&name) else {
                diagnostics.push(argument_error(
                    "RM-CATALOG-RUNTESTS-OPTION",
                    "unknown runtests option",
                    index,
                ));
                continue;
            };
            let Some(value) = request.arguments.get(index + 1) else {
                continue;
            };
            match option {
                RuntestsOption::UseParallel
                | RuntestsOption::IncludeSubfolders
                | RuntestsOption::Coverage => {
                    if matches!(
                        value.residency,
                        ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
                    ) || !matches!(
                        value.kind,
                        ValueKindFact::Unknown | ValueKindFact::Numeric(_) | ValueKindFact::Logical
                    ) || matches!(value.shape.element_count(), Some(count) if count != 1)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-RUNTESTS-FLAG",
                            "test flags require a host scalar zero or one",
                            index + 1,
                        ));
                    }
                    let literal = request
                        .literals
                        .literal_args
                        .get(index + 1)
                        .and_then(literal_flag);
                    if literal == Some(Err(())) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-RUNTESTS-FLAG",
                            "test flags must equal zero or one",
                            index + 1,
                        ));
                    }
                    if option == RuntestsOption::UseParallel && literal == Some(Ok(true)) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-RUNTESTS-PARALLEL",
                            "in-program UseParallel=true is unsupported",
                            index + 1,
                        ));
                    }
                }
                RuntestsOption::OutputDetail | RuntestsOption::LoggingLevel => {
                    if !scalar_text(value) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-RUNTESTS-TEXT",
                            "presentation controls require host scalar text",
                            index + 1,
                        ));
                    }
                }
                _ => {
                    if !text_collection(value) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-RUNTESTS-TEXT",
                            "test selection controls require host text collections",
                            index + 1,
                        ));
                    }
                }
            }
        }
    }
    let results = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(StaticClassIdentity::new("matlab.unittest.TestResult").owned()),
            properties: std::collections::BTreeMap::new(),
            properties_complete: false,
            handle_semantics: None,
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Opaque,
    );
    finish_fixed_outputs(entry, request, vec![results], diagnostics)
}
