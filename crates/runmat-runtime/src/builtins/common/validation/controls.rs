//! Validator-specific class-name and range control parsing.

use super::{invalid_argument_error, value_texts, RangeInclusivity, RuntimeError, Value};

pub(super) fn type_names_arg(args: &[Value], index: usize) -> Result<Vec<String>, RuntimeError> {
    match args.get(index) {
        Some(value) => value_texts(value),
        None => Err(invalid_argument_error(
            "argumentValidation",
            "missing type name argument",
        )),
    }
}

pub(super) fn range_inclusivity_arg(
    builtin: &str,
    args: &[Value],
) -> Result<RangeInclusivity, RuntimeError> {
    match args {
        [] => Ok(RangeInclusivity::CLOSED),
        [flag] => range_inclusivity_single_flag(builtin, text_scalar_arg(builtin, flag)?.as_str()),
        [lower, upper] => {
            let lower =
                range_bound_inclusive_flag(builtin, text_scalar_arg(builtin, lower)?.as_str())?;
            let upper =
                range_bound_inclusive_flag(builtin, text_scalar_arg(builtin, upper)?.as_str())?;
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

pub(super) fn text_scalar_arg(builtin: &str, value: &Value) -> Result<String, RuntimeError> {
    let texts = value_texts(value)?;
    match texts.as_slice() {
        [text] => Ok(text.clone()),
        _ => Err(invalid_argument_error(builtin, "expected text scalar")),
    }
}
