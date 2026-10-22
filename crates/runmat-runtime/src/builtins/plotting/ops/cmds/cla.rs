use super::error::cmd_error_with_message;
use crate::builtins::plotting::op_common::cmd_parsing::as_lower_str;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    clear_axes_target_for_property, current_axes_state_for_property, PropertyMutationTarget,
};
use runmat_builtins::{
    CLA_DESCRIPTOR, CLA_ERROR_INVALID_ARGUMENT, CLA_ERROR_OUTPUTS, CLA_STATE_OUTPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Value;
#[cfg(test)]
#[path = "cla/tests.rs"]
mod tests;
#[runtime_builtin(
    name = "cla",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::cmds::cla"
)]
pub fn cla_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<bool> {
    cla_builtin_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn cla_builtin(args: Vec<Value>) -> crate::BuiltinResult<bool> {
    cla_builtin_with_runtime(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
}

fn cla_builtin_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<bool> {
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
    let original = args
        .first()
        .map(|value| crate::builtins::plotting::properties::select_graphics_property(value, "cla"))
        .transpose()?
        .flatten();
    let (target, rest) = if let Some(runmat_value::GraphicsReceiverKind::Axes { figure, index }) =
        original
            .as_ref()
            .map(|selection| selection.receiver().kind())
    {
        (
            Some((
                crate::builtins::plotting::state::FigureHandle::from(figure),
                index,
            )),
            args.view().range(1..)?,
        )
    } else if let Some(first) = args.first() {
        if let Ok(PlotHandle::Axes(handle, axes)) = resolve_plot_handle(first, "cla") {
            (Some((handle, axes)), args.view().range(1..)?)
        } else {
            (None, args.view())
        }
    } else {
        (None, args.view())
    };
    let reset = match (rest.len(), rest.first()) {
        (0, _) => false,
        (1, Some(value))
            if as_lower_str(value)
                .map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "cmds",
                        source.to_string(),
                        source,
                    )
                })?
                .is_some_and(|text| text.trim() == "reset") =>
        {
            true
        }
        _ => return Err(invalid()),
    };
    let selected = target.and(original);
    let property = selected
        .map_or_else(
            PropertyMutationTarget::alias,
            PropertyMutationTarget::selected,
        )
        .with_runtime(runtime);
    let (handle, axes) = match target {
        Some(target) => target,
        None => {
            let state = current_axes_state_for_property(&property).map_err(|err| {
                crate::builtins::plotting::properties::map_figure_error("cla", err)
            })?;
            (state.handle, state.active_index)
        }
    };
    clear_axes_target_for_property(&property, handle, axes, reset).map_err(|err| {
        cmd_error_with_message("cla", format!("cla: {err}"), &CLA_ERROR_INVALID_ARGUMENT)
    })?;
    Ok(true)
}
