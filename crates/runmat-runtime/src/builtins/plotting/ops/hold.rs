//! MATLAB-compatible `hold` builtin.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use super::op_common::cmd_parsing::parse_hold_mode;
use super::state::{set_hold, HoldMode};
use crate::builtins::plotting::type_resolvers::bool_type;

#[runtime_builtin(
    name = "hold",
    category = "plotting",
    summary = "Toggle whether plots replace or append to the current axes.",
    keywords = "hold,plotting",
    suppress_auto_output = true,
    type_resolver(bool_type),
    builtin_path = "crate::builtins::plotting::hold"
)]
pub fn hold_builtin(rest: Vec<Value>) -> crate::BuiltinResult<bool> {
    let mode = if rest.is_empty() {
        HoldMode::Toggle
    } else {
        parse_hold_mode(&rest[0])?
    };
    let enabled = set_hold(mode);
    Ok(enabled)
}
