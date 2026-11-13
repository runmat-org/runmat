use super::facts::*;
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    predicate: bool,
) -> CallInference {
    let mut diagnostics = Vec::new();
    count(request, 1, &mut diagnostics);
    let input = argument(request, 0);
    accepted_data(&input, true, true, &mut diagnostics, 0);
    if matches!(input.kind, ValueKindFact::String) {
        diagnostics.push(argument_error(
            "RM-ROW-SORT-INPUT",
            "string arrays are not supported by the row sorter",
            0,
        ));
    }
    if !matches!(input.kind, ValueKindFact::Object(_)) {
        matrix(&input, 0, &mut diagnostics);
    }
    let mut index = 1;
    while index < request.arguments.len() {
        if let Some(text) = request.literals.literal_string_at(index) {
            let text = text.trim();
            if matches!(text, "comparisonmethod" | "missingplacement") {
                let choices: &[&str] = if text == "comparisonmethod" {
                    &["auto", "real", "abs", "magnitude"]
                } else {
                    &["auto", "first", "last"]
                };
                index += 1;
                if index == request.arguments.len() {
                    diagnostics.push(argument_error(
                        "RM-ROW-SORT-OPTION",
                        "name-value option requires a value",
                        index - 1,
                    ));
                    break;
                }
                if text_option(request, index, &mut diagnostics)
                    .is_some_and(|value| !choices.contains(&value.as_str()))
                {
                    diagnostics.push(argument_error(
                        "RM-ROW-SORT-OPTION",
                        "unsupported row sorting option value",
                        index,
                    ));
                }
            } else {
                let direction = matches!(text, "ascend" | "descend" | "ascending" | "descending");
                let predicate_mode = predicate
                    && matches!(
                        text,
                        "monotonic" | "strictascend" | "strictdescend" | "strictmonotonic"
                    );
                let table_selector = matches!(input.kind, ValueKindFact::Object(_));
                if !(direction || predicate_mode || table_selector) {
                    diagnostics.push(argument_error(
                        "RM-ROW-SORT-OPTION",
                        "unsupported row sorting option",
                        index,
                    ));
                }
            }
        } else if let ValueKindFact::Numeric(numeric) = request.arguments[index].kind {
            if numeric.domain != NumericDomain::Real
                || request.literals.numeric_at(index).is_some_and(|number| {
                    !number.is_finite()
                        || number == 0.0
                        || number.fract() != 0.0
                        || dimension(&input.shape, 1)
                            .is_some_and(|columns| number.abs() > columns as f64)
                })
            {
                diagnostics.push(argument_error(
                    "RM-ROW-SORT-COLUMN",
                    "columns must be nonzero integer indices within the input",
                    index,
                ));
            }
        }
        index += 1;
    }
    let outputs = if predicate {
        vec![logical(ShapeFact::Scalar, ResidencyFact::Host)]
    } else {
        vec![
            input.clone(),
            super::facts::index(column(dimension(&input.shape, 0)), input.residency.clone()),
        ]
    };
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
