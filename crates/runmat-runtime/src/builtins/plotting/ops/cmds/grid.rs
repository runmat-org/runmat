use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::as_lower_str;
use crate::builtins::plotting::state::{
    set_grid_and_minor_grid_enabled, toggle_grid, toggle_minor_grid,
};
use runmat_builtins::{
    GRID_DESCRIPTOR, GRID_ERROR_INVALID_ARGUMENT, GRID_ERROR_OUTPUTS, GRID_STATE_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
enum GridMode {
    ToggleMajor,
    Major(bool),
    ToggleMinor,
}
#[runtime_builtin(
    name = "grid",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::cmds::grid"
)]
pub fn grid_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<bool> {
    crate::output_count::validate_fixed_outputs(&GRID_DESCRIPTOR, &GRID_ERROR_OUTPUTS, "grid")?;
    if crate::output_count::current_output_count() == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GRID_STATE_OUTPUT_EXTENSION,
            "grid",
        )?;
    }

    match parse_grid_mode(args.view())? {
        GridMode::ToggleMajor => {
            let enabled = toggle_grid();
            Ok(enabled)
        }
        GridMode::Major(enabled) => {
            set_grid_and_minor_grid_enabled(enabled, if enabled { None } else { Some(false) });
            Ok(enabled)
        }
        GridMode::ToggleMinor => {
            let enabled = toggle_minor_grid();
            Ok(enabled)
        }
    }
}

fn parse_grid_mode(args: crate::call::FunctionArgumentView<'_>) -> crate::BuiltinResult<GridMode> {
    if args.len() > 1 {
        return Err(cmd_error_with_message(
            "grid",
            format!(
                "{}: expected at most one mode argument",
                GRID_ERROR_INVALID_ARGUMENT.message
            ),
            &GRID_ERROR_INVALID_ARGUMENT,
        ));
    }

    let Some(arg) = args.first() else {
        return Ok(GridMode::ToggleMajor);
    };
    let Some(mode) = as_lower_str(arg).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source("cmds", source.to_string(), source)
    })?
    else {
        return Err(cmd_error_with_message(
            "grid",
            format!(
                "{}: expected string argument",
                GRID_ERROR_INVALID_ARGUMENT.message
            ),
            &GRID_ERROR_INVALID_ARGUMENT,
        ));
    };
    match mode.trim() {
        "on" => Ok(GridMode::Major(true)),
        "off" => Ok(GridMode::Major(false)),
        "minor" => Ok(GridMode::ToggleMinor),
        other => Err(cmd_error_with_message(
            "grid",
            format!(
                "{}: expected 'on', 'off', or 'minor' (got '{other}')",
                GRID_ERROR_INVALID_ARGUMENT.message
            ),
            &GRID_ERROR_INVALID_ARGUMENT,
        )),
    }
}
#[cfg(test)]
#[path = "grid/tests.rs"]
mod tests;
