//! Literal checks for the actual semi-infinite scalar control family.
use super::*;
pub(super) fn inspect(
    request: &CallRequest,
    field: Field,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> bool {
    match field {
        Field::Diagnostics | Field::FunValCheck => {
            if request
                .literals
                .literal_string_at(index)
                .is_some_and(|s| !matches!(s.to_ascii_lowercase().as_str(), "on" | "off"))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-SWITCH",
                    "option must be on or off",
                    index,
                ));
            }
        }
        Field::DiffMaxChange
        | Field::DiffMinChange
        | Field::RelLineSrchBnd
        | Field::TolConSQP
        | Field::MaxSQPIter
        | Field::RelLineSrchBndDuration => {
            if request.literals.numeric_at(index).is_some_and(|n| {
                let infinity = field == Field::DiffMaxChange && n == f64::INFINITY;
                !infinity
                    && (!n.is_finite()
                        || n < 0.0
                        || (n == 0.0
                            && matches!(
                                field,
                                Field::DiffMaxChange
                                    | Field::TolConSQP
                                    | Field::MaxSQPIter
                                    | Field::RelLineSrchBndDuration
                            ))
                        || (matches!(field, Field::MaxSQPIter | Field::RelLineSrchBndDuration)
                            && n.fract() != 0.0))
            }) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-SEMI-RANGE",
                    "invalid semi-infinite scalar control range",
                    index,
                ));
            }
        }
        _ => return false,
    }
    true
}
