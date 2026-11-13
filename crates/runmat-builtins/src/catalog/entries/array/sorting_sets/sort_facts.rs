use super::facts::*;
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SortForm {
    Values,
    Indices,
    Predicate,
}

pub(super) fn options(
    request: &CallRequest,
    predicate: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let mut dimension_seen = false;
    let mut index = 1;
    while index < request.arguments.len() {
        if let Some(option) = request.literals.literal_string_at(index) {
            let option = option.trim();
            let accepted_flag = matches!(option, "ascend" | "ascending" | "descend" | "descending")
                || (predicate
                    && matches!(
                        option,
                        "rows" | "monotonic" | "strictascend" | "strictdescend" | "strictmonotonic"
                    ));
            if accepted_flag {
                index += 1;
                continue;
            }
            let choices: Option<&[&str]> = match option {
                "comparisonmethod" => Some(&["auto", "real", "abs", "magnitude"]),
                "missingplacement" => Some(&["auto", "first", "last"]),
                _ => None,
            };
            if let Some(choices) = choices {
                index += 1;
                if index == request.arguments.len() {
                    diagnostics.push(argument_error(
                        "RM-SORTING-OPTION-VALUE",
                        "name-value option requires a value",
                        index - 1,
                    ));
                    break;
                }
                if text_option(request, index, diagnostics)
                    .is_some_and(|value| !choices.contains(&value.as_str()))
                {
                    diagnostics.push(argument_error(
                        "RM-SORTING-OPTION-VALUE",
                        "unsupported option value",
                        index,
                    ));
                }
            } else {
                diagnostics.push(argument_error(
                    "RM-SORTING-OPTION",
                    "unsupported sorting option",
                    index,
                ));
            }
        } else {
            let value = &request.arguments[index];
            if value.shape.element_count() == Some(0) && !dimension_seen {
                index += 1;
                continue;
            }
            match value.kind {
                ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String => {}
                ValueKindFact::Numeric(numeric) => {
                    let invalid = numeric.domain != NumericDomain::Real
                        || dimension_seen
                        || value.shape.is_definitely_nonscalar()
                        || request.literals.numeric_at(index).is_some_and(|number| {
                            !number.is_finite() || number <= 0.0 || number.fract() != 0.0
                        });
                    if invalid {
                        diagnostics.push(argument_error(
                            "RM-SORTING-DIMENSION",
                            "dimension must be a positive integer scalar",
                            index,
                        ));
                    }
                    dimension_seen = true;
                }
                _ => diagnostics.push(argument_error(
                    "RM-SORTING-OPTION",
                    "unsupported dimension or option",
                    index,
                )),
            }
        }
        index += 1;
    }
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    form: SortForm,
) -> CallInference {
    let mut diagnostics = Vec::new();
    count(request, 1, &mut diagnostics);
    let input = argument(request, 0);
    accepted_data(
        &input,
        form == SortForm::Predicate,
        false,
        &mut diagnostics,
        0,
    );
    options(request, form == SortForm::Predicate, &mut diagnostics);
    let outputs = match form {
        SortForm::Predicate => vec![logical(ShapeFact::Scalar, ResidencyFact::Host)],
        SortForm::Indices => vec![index(input.shape.clone(), input.residency.clone())],
        SortForm::Values => vec![
            input.clone(),
            index(input.shape.clone(), input.residency.clone()),
        ],
    };
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
