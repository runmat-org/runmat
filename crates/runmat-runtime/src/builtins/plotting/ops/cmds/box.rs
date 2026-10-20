use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::parse_on_off;
use crate::builtins::plotting::state::{set_box_enabled, toggle_box};
use runmat_builtins::{
    BOX_DESCRIPTOR, BOX_ERROR_INVALID_ARGUMENT, BOX_ERROR_OUTPUTS, BOX_STATE_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
#[path = "box/tests.rs"]
mod tests;
#[runtime_builtin(
    name = "box",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::cmds::r#box"
)]
pub fn box_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<bool> {
    crate::output_count::validate_fixed_outputs(&BOX_DESCRIPTOR, &BOX_ERROR_OUTPUTS, "box")?;
    if crate::output_count::current_output_count() == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(&BOX_STATE_OUTPUT_EXTENSION, "box")?;
    }

    if args.len() > 1 {
        return Err(cmd_error_with_message(
            "box",
            "box: expected at most one mode argument",
            &BOX_ERROR_INVALID_ARGUMENT,
        ));
    }
    match parse_on_off("box", args.first()).map_err(|err| {
        cmd_error_with_message(
            "box",
            format!("{}: {}", BOX_ERROR_INVALID_ARGUMENT.message, err.message()),
            &BOX_ERROR_INVALID_ARGUMENT,
        )
    })? {
        Some(enabled) => {
            set_box_enabled(enabled);
            Ok(enabled)
        }
        None => {
            let enabled = toggle_box();
            Ok(enabled)
        }
    }
}
