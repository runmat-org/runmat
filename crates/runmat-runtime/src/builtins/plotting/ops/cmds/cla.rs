use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::as_lower_str;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{clear_axes_target, current_axes_state};
use runmat_builtins::{
    CLA_DESCRIPTOR, CLA_ERROR_INVALID_ARGUMENT, CLA_ERROR_OUTPUTS, CLA_STATE_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
#[cfg(test)]
#[path = "cla/tests.rs"]
mod tests;
#[runtime_builtin(
    name = "cla",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::cmds"
)]
pub fn cla_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    crate::output_count::validate_fixed_outputs(&CLA_DESCRIPTOR, &CLA_ERROR_OUTPUTS, "cla")?;
    if crate::output_count::current_output_count() == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(&CLA_STATE_OUTPUT_EXTENSION, "cla")?;
    }
    let invalid = || {
        cmd_error_with_message(
            "cla",
            "cla: expected an optional axes handle followed by reset",
            &CLA_ERROR_INVALID_ARGUMENT,
        )
    };
    if args.len() > 2 {
        return Err(invalid());
    }
    let (target, rest) = if let Some(first) = args.first() {
        if let Ok(PlotHandle::Axes(handle, axes)) = resolve_plot_handle(first, "cla") {
            (Some((handle, axes)), &args[1..])
        } else {
            (None, args.as_slice())
        }
    } else {
        (None, args.as_slice())
    };
    let reset = match rest {
        [] => false,
        [value] if as_lower_str(value).is_some_and(|text| text.trim() == "reset") => true,
        _ => return Err(invalid()),
    };
    let (handle, axes) = target.unwrap_or_else(|| {
        let state = current_axes_state();
        (state.handle, state.active_index)
    });
    clear_axes_target(handle, axes, reset).map_err(|err| {
        cmd_error_with_message("cla", format!("cla: {err}"), &CLA_ERROR_INVALID_ARGUMENT)
    })?;
    Ok(true)
}
