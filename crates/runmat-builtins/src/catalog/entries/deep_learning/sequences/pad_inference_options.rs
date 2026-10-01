//! Source-known padsequences controls; public spellings use the runtime's types.

use super::{PadBooleanKeyword, PadDirection, PadLengthKeyword, PadOptionKey};
use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, InferenceDiagnostic, LiteralContext, LiteralValue, NumericDomain, ValueKindFact,
};
use std::collections::BTreeMap;

fn positive_dimension_literal(value: &LiteralValue) -> Option<bool> {
    match value {
        LiteralValue::Integer { text, .. } => {
            Some(text.parse::<usize>().is_ok_and(|value| value >= 1))
        }
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            let value = LiteralContext::numeric_from_literal(value)?;
            Some(
                value.is_finite()
                    && value.fract().abs() <= f64::EPSILON
                    && value >= 1.0
                    && (value < usize::MAX as f64
                        || (usize::BITS < 64 && value == usize::MAX as f64)),
            )
        }
        LiteralValue::Unknown => None,
        _ => Some(false),
    }
}

fn boolean_literal(value: &LiteralValue) -> Result<Option<bool>, ()> {
    match value {
        LiteralValue::Bool(value) => Ok(Some(*value)),
        LiteralValue::String(value)
        | LiteralValue::Character(value)
        | LiteralValue::Keyword(value) => PadBooleanKeyword::parse(value)
            .map(|keyword| Some(keyword.value()))
            .ok_or(()),
        LiteralValue::Integer { text, .. } => match text.parse::<u8>() {
            Ok(0) => Ok(Some(false)),
            Ok(1) => Ok(Some(true)),
            _ => Err(()),
        },
        LiteralValue::Number(_) | LiteralValue::Real { .. } => {
            match LiteralContext::numeric_from_literal(value) {
                Some(0.0) => Ok(Some(false)),
                Some(1.0) => Ok(Some(true)),
                Some(_) => Err(()),
                None => Ok(None),
            }
        }
        LiteralValue::Unknown => Ok(None),
        _ => Err(()),
    }
}

pub(super) fn validate_dimension(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request
        .literals
        .literal_args
        .get(1)
        .and_then(positive_dimension_literal)
        == Some(false)
    {
        diagnostics.push(argument_error(
            "RM-DL-PADSEQUENCES-DIMENSION",
            "paddingDim must be a positive integer scalar",
            1,
        ));
    }
}

pub(super) fn validate_options(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<bool> {
    let mut uniform = Some(true);
    let mut unknown_name = false;
    // The runtime keeps the final value for a repeated spelling, then admits
    // options in key order. Inference must not diagnose an overwritten value.
    let mut options = BTreeMap::new();
    let mut unsupported = Vec::new();
    for index in (2..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            match request.literals.literal_args.get(index) {
                Some(LiteralValue::Unknown) | None => unknown_name = true,
                Some(_) => diagnostics.push(argument_error(
                    "RM-DL-PADSEQUENCES-OPTION",
                    "option name must be text",
                    index,
                )),
            }
            continue;
        };
        let Some(option) = PadOptionKey::parse(&name) else {
            unsupported.push(index);
            continue;
        };
        options.insert(option, index);
    }
    // A later dynamic spelling may replace an earlier known one in the
    // runtime's last-wins map. Do not turn either into a proven diagnosis or
    // output class before that spelling is known.
    if unknown_name {
        return None;
    }
    for index in unsupported {
        diagnostics.push(argument_error(
            "RM-DL-PADSEQUENCES-OPTION",
            "unsupported padsequences option",
            index,
        ));
    }
    for (option, index) in &options {
        let value = request.literals.literal_args.get(*index + 1);
        let fact = request.arguments.get(*index + 1);
        let invalid = match option {
            PadOptionKey::PaddingValue => {
                fact.is_some_and(|value| {
                    !matches!(
                        value.kind,
                        ValueKindFact::Unknown
                            | ValueKindFact::Numeric(runmat_types::NumericFact {
                                domain: NumericDomain::Real,
                                ..
                            })
                    ) || value.shape.element_count().is_some_and(|count| count != 1)
                }) || value.is_some_and(|literal| match literal {
                    LiteralValue::Unknown => false,
                    LiteralValue::Number(_)
                    | LiteralValue::Real { .. }
                    | LiteralValue::Integer { .. } => LiteralContext::numeric_from_literal(literal)
                        .is_some_and(|number| !number.is_finite()),
                    _ => true,
                })
            }
            PadOptionKey::PaddingDirection | PadOptionKey::DirectionAlias => {
                value.is_some_and(|literal| match literal {
                    LiteralValue::String(text)
                    | LiteralValue::Character(text)
                    | LiteralValue::Keyword(text) => PadDirection::parse(text).is_none(),
                    LiteralValue::Unknown => false,
                    _ => true,
                })
            }
            PadOptionKey::Length => {
                fact.is_some_and(|value| {
                    matches!(value.kind, ValueKindFact::Numeric(_))
                        && value.shape.element_count().is_some_and(|count| count != 1)
                }) || value.is_some_and(|literal| match literal {
                    LiteralValue::String(text)
                    | LiteralValue::Character(text)
                    | LiteralValue::Keyword(text) => PadLengthKeyword::parse(text).is_none(),
                    _ => positive_dimension_literal(literal) == Some(false),
                })
            }
            PadOptionKey::UniformOutput => {
                let decoded = value.map_or(Ok(None), boolean_literal);
                uniform = decoded.as_ref().ok().copied().flatten();
                decoded.is_err()
                    || fact.is_some_and(|value| {
                        value.shape.element_count().is_some_and(|count| count != 1)
                            && !matches!(
                                request.literals.literal_args.get(*index + 1),
                                Some(
                                    LiteralValue::String(_)
                                        | LiteralValue::Character(_)
                                        | LiteralValue::Keyword(_)
                                )
                            )
                    })
            }
        };
        if invalid {
            diagnostics.push(argument_error(
                "RM-DL-PADSEQUENCES-OPTION",
                "invalid padsequences option value",
                *index + 1,
            ));
        }
    }
    if options.contains_key(&PadOptionKey::PaddingDirection)
        && options.contains_key(&PadOptionKey::DirectionAlias)
    {
        diagnostics.push(argument_error(
            "RM-DL-PADSEQUENCES-OPTION",
            "PaddingDirection and Direction cannot both be specified",
            2,
        ));
    }
    uniform
}
