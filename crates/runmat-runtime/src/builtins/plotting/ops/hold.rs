//! MATLAB-compatible `hold` builtin.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::plotting_error;
use super::state::{set_hold, HoldMode};
use crate::builtins::plotting::type_resolvers::string_type;

use crate::BuiltinResult;

fn parse_mode(value: &Value) -> BuiltinResult<HoldMode> {
    match value {
        Value::CharArray(chars) => {
            let text: String = chars.data.iter().collect();
            parse_mode_str(text.trim())
        }
        Value::String(s) => parse_mode_str(s.trim()),
        Value::Num(v) => Ok(if *v == 0.0 {
            HoldMode::Off
        } else {
            HoldMode::On
        }),
        Value::Bool(b) => Ok(if *b { HoldMode::On } else { HoldMode::Off }),
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(plotting_error("hold", "hold: logical scalar expected"));
            }
            Ok(if tensor.data[0] == 0.0 {
                HoldMode::Off
            } else {
                HoldMode::On
            })
        }
        _ => Err(plotting_error("hold", "hold: unsupported argument type")),
    }
}

fn parse_mode_str(text: &str) -> BuiltinResult<HoldMode> {
    match text.to_ascii_lowercase().as_str() {
        "on" | "all" => Ok(HoldMode::On),
        "off" => Ok(HoldMode::Off),
        "" => Ok(HoldMode::Toggle),
        _ => Err(plotting_error("hold", "hold: expected 'on' or 'off'")),
    }
}

#[runtime_builtin(
    name = "hold",
    category = "plotting",
    summary = "Toggle whether plots replace or append to the current axes.",
    keywords = "hold,plotting",
    suppress_auto_output = true,
    type_resolver(string_type),
    builtin_path = "crate::builtins::plotting::hold"
)]
pub fn hold_builtin(rest: Vec<Value>) -> crate::BuiltinResult<String> {
    let mode = if rest.is_empty() {
        HoldMode::Toggle
    } else {
        parse_mode(&rest[0])?
    };
    let enabled = set_hold(mode);
    Ok(if enabled {
        "hold is on".to_string()
    } else {
        "hold is off".to_string()
    })
}
