//! MATLAB-compatible `hold` builtin.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::state::{set_hold, HoldMode};

fn parse_mode(value: &Value) -> Result<HoldMode, String> {
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
                return Err("hold: logical scalar expected".to_string());
            }
            Ok(if tensor.data[0] == 0.0 {
                HoldMode::Off
            } else {
                HoldMode::On
            })
        }
        _ => Err("hold: unsupported argument type".to_string()),
    }
}

fn parse_mode_str(text: &str) -> Result<HoldMode, String> {
    match text.to_ascii_lowercase().as_str() {
        "on" | "all" => Ok(HoldMode::On),
        "off" => Ok(HoldMode::Off),
        "" => Ok(HoldMode::Toggle),
        _ => Err("hold: expected 'on' or 'off'".to_string()),
    }
}

#[runtime_builtin(
    name = "hold",
    category = "plotting",
    summary = "Toggle whether plots replace or append to the current axes.",
    keywords = "hold,plotting"
)]
pub fn hold_builtin(rest: Vec<Value>) -> Result<String, String> {
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
