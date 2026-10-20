//! Admitted text-line controls; parsed completely before filesystem access.
use super::{line_error, scalar_text};
use crate::BuiltinResult;

pub(super) fn parse_empty_line_rule(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<bool> {
    let mut omit_empty = false;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(line_error(
                "readlines",
                "readlines: name-value options must be paired",
            ));
        }
        let name = scalar_text(&args[idx], "readlines", "option")?;
        if name.eq_ignore_ascii_case("EmptyLineRule") {
            let value = scalar_text(&args[idx + 1], "readlines", "EmptyLineRule")?;
            omit_empty = match value.to_ascii_lowercase().as_str() {
                "skip" | "omit" | "omitempty" => true,
                "read" => false,
                _ => {
                    return Err(line_error(
                        "readlines",
                        "readlines: unsupported EmptyLineRule",
                    ))
                }
            };
        } else {
            return Err(line_error(
                "readlines",
                format!("readlines: unsupported option {name}"),
            ));
        }
        idx += 2;
    }
    Ok(omit_empty)
}

pub(super) fn parse_newline(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<String>> {
    let mut newline = None;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(line_error(
                "writelines",
                "writelines: name-value options must be paired",
            ));
        }
        let name = scalar_text(&args[idx], "writelines", "option")?;
        if name.eq_ignore_ascii_case("LineEnding") {
            let value = scalar_text(&args[idx + 1], "writelines", "LineEnding")?;
            newline = Some(match value.to_ascii_lowercase().as_str() {
                "windows" | "pc" | "crlf" => "\r\n".to_string(),
                "mac" | "cr" => "\r".to_string(),
                "unix" | "lf" => "\n".to_string(),
                _ => {
                    return Err(line_error(
                        "writelines",
                        "writelines: unsupported LineEnding",
                    ))
                }
            });
        } else {
            return Err(line_error(
                "writelines",
                format!("writelines: unsupported option {name}"),
            ));
        }
        idx += 2;
    }
    Ok(newline)
}
