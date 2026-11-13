use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::parse_on_off;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    axes_metadata_snapshot, current_axes_state, current_figure_handle,
    set_hidden_line_removal_for_axes,
};
use runmat_builtins::{
    HIDDEN_DESCRIPTOR, HIDDEN_ERROR_INVALID_ARGUMENT, HIDDEN_ERROR_OUTPUTS,
    HIDDEN_STATE_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;
type HiddenAxesTarget = Option<(crate::builtins::plotting::state::FigureHandle, usize)>;
#[runtime_builtin(
    name = "hidden",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::cmds::hidden"
)]
pub fn hidden_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<bool> {
    crate::output_count::validate_fixed_outputs(
        &HIDDEN_DESCRIPTOR,
        &HIDDEN_ERROR_OUTPUTS,
        "hidden",
    )?;
    if crate::output_count::current_output_count() == Some(1) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HIDDEN_STATE_OUTPUT_EXTENSION,
            "hidden",
        )?;
    }

    let (target, rest) = parse_hidden_target(args)?;
    let requested = parse_on_off("hidden", rest.first()).map_err(|err| {
        cmd_error_with_message(
            "hidden",
            format!(
                "{}: {}",
                HIDDEN_ERROR_INVALID_ARGUMENT.message,
                err.message()
            ),
            &HIDDEN_ERROR_INVALID_ARGUMENT,
        )
    })?;
    if rest.len() > 1 {
        return Err(cmd_error_with_message(
            "hidden",
            "hidden: expected at most one mode argument after optional axes handle",
            &HIDDEN_ERROR_INVALID_ARGUMENT,
        ));
    }
    let (handle, axes_index) = target.unwrap_or_else(|| {
        let axes = current_axes_state();
        (current_figure_handle(), axes.active_index)
    });
    let current = axes_metadata_snapshot(handle, axes_index)
        .map_err(|err| {
            cmd_error_with_message(
                "hidden",
                format!("{}: {err}", HIDDEN_ERROR_INVALID_ARGUMENT.message),
                &HIDDEN_ERROR_INVALID_ARGUMENT,
            )
        })?
        .hidden_line_removal;
    let enabled = requested.unwrap_or(!current);
    set_hidden_line_removal_for_axes(handle, axes_index, enabled).map_err(|err| {
        cmd_error_with_message(
            "hidden",
            format!("{}: {err}", HIDDEN_ERROR_INVALID_ARGUMENT.message),
            &HIDDEN_ERROR_INVALID_ARGUMENT,
        )
    })?;
    Ok(enabled)
}

fn parse_hidden_target(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<(HiddenAxesTarget, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(&first, "hidden") {
        return Ok((Some((handle, axes_index)), iter.collect()));
    }
    let mut rest = Vec::with_capacity(iter.size_hint().0 + 1);
    rest.push(first);
    rest.extend(iter);
    Ok((None, rest))
}
#[cfg(test)]
#[path = "hidden/tests.rs"]
mod tests;
