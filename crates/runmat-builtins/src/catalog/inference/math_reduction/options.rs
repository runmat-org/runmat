//! Abstract facts about reduction options, not a runtime value parser.
use super::argument_error;
use runmat_types::{CallRequest, InferenceDiagnostic, LiteralContext, LiteralValue};

pub(super) enum Selector {
    Default,
    Dimensions(Vec<usize>),
    All,
    Unknown,
}

pub(super) fn analyze(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Selector {
    let mut selector = None;
    let mut unknown_control = false;
    let mut nan_flag = false;
    for index in 1..request.arguments.len() {
        let literal = request
            .literals
            .literal_args
            .get(index)
            .unwrap_or(&LiteralValue::Unknown);
        let next = match literal {
            LiteralValue::Empty => continue,
            LiteralValue::Vector(values) if values.is_empty() => continue,
            LiteralValue::Unknown => {
                unknown_control = true;
                continue;
            }
            LiteralValue::Character(text)
            | LiteralValue::String(text)
            | LiteralValue::Keyword(text) => match text.to_ascii_lowercase().as_str() {
                "all" => Selector::All,
                "omitnan" | "includenan" => {
                    if nan_flag {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-LOGICAL-REDUCTION-ARGUMENTS",
                            "only one NaN handling option may be specified",
                            index,
                        ));
                    }
                    nan_flag = true;
                    continue;
                }
                _ => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-LOGICAL-REDUCTION-ARGUMENTS",
                        "unknown reduction option",
                        index,
                    ));
                    unknown_control = true;
                    continue;
                }
            },
            LiteralValue::Vector(values) => dimensions(values, index, diagnostics),
            LiteralValue::Number(_) | LiteralValue::Real { .. } | LiteralValue::Integer { .. } => {
                dimensions(std::slice::from_ref(literal), index, diagnostics)
            }
            _ => {
                unknown_control = true;
                continue;
            }
        };
        if selector.is_some() {
            diagnostics.push(argument_error(
                "RM-CATALOG-LOGICAL-REDUCTION-ARGUMENTS",
                "only one nonempty dimension or all selector may be specified",
                index,
            ));
        } else {
            selector = Some(next);
        }
    }
    // Once a selector is known, a successful call can only use empty controls
    // or a NaN flag elsewhere. Unknown controls must not overwrite that fact.
    selector.unwrap_or(if unknown_control {
        Selector::Unknown
    } else {
        Selector::Default
    })
}

fn dimensions(
    values: &[LiteralValue],
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Selector {
    let mut dimensions = Vec::new();
    let mut unknown = false;
    for value in values {
        let dimension = match value {
            LiteralValue::Integer { text, .. } => {
                text.parse::<usize>().ok().filter(|value| *value > 0)
            }
            LiteralValue::Number(_) | LiteralValue::Real { .. } => {
                LiteralContext::numeric_from_literal(value)
                    .filter(|value| {
                        value.is_finite() && (value.round() - value).abs() <= f64::EPSILON
                    })
                    .map(f64::round)
                    .filter(|value| {
                        *value >= 1.0
                            && (*value < usize::MAX as f64
                                || (usize::BITS < 64 && *value == usize::MAX as f64))
                    })
                    .map(|value| value as usize)
            }
            _ => {
                unknown = true;
                continue;
            }
        };
        match dimension {
            Some(dimension) => dimensions.push(dimension),
            None => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-LOGICAL-REDUCTION-ARGUMENTS",
                    "dimensions must be finite positive integers",
                    index,
                ));
                unknown = true;
            }
        }
    }
    if unknown {
        Selector::Unknown
    } else {
        Selector::Dimensions(dimensions)
    }
}
