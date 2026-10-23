use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::as_lower_str;
use crate::builtins::plotting::state::set_surface_shading;
use runmat_builtins::{
    SHADING_DESCRIPTOR, SHADING_ERROR_INVALID_ARGUMENT, SHADING_ERROR_OUTPUTS,
    SHADING_STATE_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
#[path = "shading/tests.rs"]
mod tests;
#[runtime_builtin(
    name = "shading",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::cmds::shading"
)]
pub fn shading_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<bool> {
    crate::output_count::validate_fixed_outputs(
        &SHADING_DESCRIPTOR,
        &SHADING_ERROR_OUTPUTS,
        "shading",
    )?;
    if crate::output_count::current_output_count() == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SHADING_STATE_OUTPUT_EXTENSION,
            "shading",
        )?;
    }

    if args.len() != 1 {
        return Err(cmd_error_with_message(
            "shading",
            "shading: expected one mode argument",
            &SHADING_ERROR_INVALID_ARGUMENT,
        ));
    }
    let Some(arg) = args.first() else {
        return Err(cmd_error_with_message(
            "shading",
            SHADING_ERROR_INVALID_ARGUMENT.message,
            &SHADING_ERROR_INVALID_ARGUMENT,
        ));
    };
    let Some(mode) = as_lower_str(arg).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source("cmds", source.to_string(), source)
    })?
    else {
        return Err(cmd_error_with_message(
            "shading",
            SHADING_ERROR_INVALID_ARGUMENT.message,
            &SHADING_ERROR_INVALID_ARGUMENT,
        ));
    };
    let shading = match mode.trim() {
        "flat" => runmat_plot::plots::surface::ShadingMode::Flat,
        "interp" => runmat_plot::plots::surface::ShadingMode::Smooth,
        "faceted" => runmat_plot::plots::surface::ShadingMode::Faceted,
        other => {
            return Err(cmd_error_with_message(
                "shading",
                format!(
                    "{}: unknown mode '{other}'",
                    SHADING_ERROR_INVALID_ARGUMENT.message
                ),
                &SHADING_ERROR_INVALID_ARGUMENT,
            ))
        }
    };
    set_surface_shading(shading);
    Ok(true)
}
