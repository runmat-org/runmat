use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::parse_on_off;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    axes_metadata_snapshot, current_axes_state, ensure_colorbar_for_axes,
    set_colorbar_enabled_for_axes,
};
use runmat_builtins::{
    COLORBAR_DESCRIPTOR, COLORBAR_ERROR_INVALID_ARGUMENT, COLORBAR_ERROR_OUTPUTS,
};
use runmat_value::Value;
#[cfg(test)]
#[path = "colorbar/tests.rs"]
mod tests;

#[runmat_macros::runtime_builtin(
    name = "colorbar",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::cmds::colorbar"
)]
pub fn colorbar_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &COLORBAR_DESCRIPTOR,
        &COLORBAR_ERROR_OUTPUTS,
        "colorbar",
    )?;
    if args.len() > 2 {
        return Err(cmd_error_with_message(
            "colorbar",
            "colorbar: expected an optional axes handle and mode",
            &COLORBAR_ERROR_INVALID_ARGUMENT,
        ));
    }
    let (target, rest) = if let Some(first) = args.first() {
        if let Ok(PlotHandle::Axes(handle, axes)) = resolve_plot_handle(first, "colorbar") {
            (Some((handle, axes)), args.view().range(1..)?)
        } else {
            (None, args.view())
        }
    } else {
        (None, args.view())
    };
    if rest.len() > 1 {
        return Err(cmd_error_with_message(
            "colorbar",
            "colorbar: expected at most one mode",
            &COLORBAR_ERROR_INVALID_ARGUMENT,
        ));
    }
    let mode = parse_on_off("colorbar", rest.first()).map_err(|error| {
        cmd_error_with_message(
            "colorbar",
            error.message(),
            &COLORBAR_ERROR_INVALID_ARGUMENT,
        )
    })?;
    let runmat = crate::compatibility::runmat_extensions_enabled();
    if !runmat && mode == Some(false) && crate::output_count::current_output_count() == Some(1) {
        return Err(cmd_error_with_message(
            "colorbar",
            "colorbar: off has no output in MATLAB compatibility mode",
            &COLORBAR_ERROR_OUTPUTS,
        ));
    }
    let (handle, axes) = target.unwrap_or_else(|| {
        let state = current_axes_state();
        (state.handle, state.active_index)
    });
    let enabled = match mode {
        Some(enabled) => enabled,
        None if runmat => {
            !axes_metadata_snapshot(handle, axes)
                .map_err(|error| {
                    cmd_error_with_message(
                        "colorbar",
                        format!("colorbar: {error}"),
                        &COLORBAR_ERROR_INVALID_ARGUMENT,
                    )
                })?
                .colorbar_enabled
        }
        None => true,
    };
    if enabled {
        let bar = ensure_colorbar_for_axes(handle, axes).map_err(|error| {
            cmd_error_with_message(
                "colorbar",
                format!("colorbar: {error}"),
                &COLORBAR_ERROR_INVALID_ARGUMENT,
            )
        })?;
        Ok(if runmat {
            Value::Bool(true)
        } else {
            Value::Num(bar)
        })
    } else {
        set_colorbar_enabled_for_axes(handle, axes, false).map_err(|error| {
            cmd_error_with_message(
                "colorbar",
                format!("colorbar: {error}"),
                &COLORBAR_ERROR_INVALID_ARGUMENT,
            )
        })?;
        Ok(if runmat {
            Value::Bool(false)
        } else {
            Value::output_list(Vec::new())
        })
    }
}
#[cfg(test)]
pub(crate) fn colorbar_builtin_source(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    colorbar_builtin(crate::call::BuiltinArguments::from_borrowed(&args, None)?)
}
