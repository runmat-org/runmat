//! Validator-specific class-name and range control parsing.

use std::borrow::Cow;

use super::{invalid_argument_error, value_texts, RangeInclusivity, RuntimeError, Value};

pub(super) fn type_names_arg(
    args: crate::call::FunctionArgumentView<'_>,
    index: usize,
) -> Result<Vec<String>, RuntimeError> {
    match args.get(index) {
        Some(value) => {
            value_texts(value).map(|texts| texts.into_iter().map(Cow::into_owned).collect())
        }
        None => Err(invalid_argument_error(
            "argumentValidation",
            "missing type name argument",
        )),
    }
}

pub(super) fn range_inclusivity_arg(
    builtin: &str,
    args: crate::call::FunctionArgumentView<'_>,
) -> Result<RangeInclusivity, RuntimeError> {
    match (args.len(), args.get(0), args.get(1)) {
        (0, _, _) => Ok(RangeInclusivity::CLOSED),
        (1, Some(flag), _) => {
            range_inclusivity_single_flag(builtin, text_scalar_arg(builtin, flag)?.as_ref())
        }
        (2, Some(lower), Some(upper)) => {
            let lower =
                range_bound_inclusive_flag(builtin, text_scalar_arg(builtin, lower)?.as_ref())?;
            let upper =
                range_bound_inclusive_flag(builtin, text_scalar_arg(builtin, upper)?.as_ref())?;
            Ok(RangeInclusivity { lower, upper })
        }
        _ => Err(invalid_argument_error(
            builtin,
            "invalid range inclusivity flags",
        )),
    }
}

fn range_inclusivity_single_flag(
    builtin: &str,
    flag: &str,
) -> Result<RangeInclusivity, RuntimeError> {
    match flag.trim().to_ascii_lowercase().as_str() {
        "inclusive" => Ok(RangeInclusivity::CLOSED),
        "exclusive" => Ok(RangeInclusivity::OPEN),
        "exclude-lower" | "openleft" | "open-left" => Ok(RangeInclusivity::OPEN_LEFT),
        "exclude-upper" | "openright" | "open-right" => Ok(RangeInclusivity::OPEN_RIGHT),
        _ => Err(invalid_argument_error(
            builtin,
            "range flag must be 'inclusive', 'exclusive', 'exclude-lower', or 'exclude-upper'",
        )),
    }
}

fn range_bound_inclusive_flag(builtin: &str, flag: &str) -> Result<bool, RuntimeError> {
    match flag.trim().to_ascii_lowercase().as_str() {
        "inclusive" => Ok(true),
        "exclusive" => Ok(false),
        _ => Err(invalid_argument_error(
            builtin,
            "range bound flag must be 'inclusive' or 'exclusive'",
        )),
    }
}

pub(super) fn text_scalar_arg<'a>(
    builtin: &str,
    value: &'a Value,
) -> Result<Cow<'a, str>, RuntimeError> {
    let mut texts = value_texts(value)?;
    if texts.len() == 1 {
        Ok(texts.pop().expect("one validated text"))
    } else {
        Err(invalid_argument_error(builtin, "expected text scalar"))
    }
}
