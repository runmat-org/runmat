use super::error::cmd_error_with_message;
use crate::builtins::plotting::colormap_arrays::{colormap_tensor, parse_rgb_colormap_tensor};
use crate::builtins::plotting::op_common::cmd_parsing::as_lower_str;
use crate::builtins::plotting::properties::{resolve_plot_handle, PlotHandle};
use crate::builtins::plotting::state::{
    axes_metadata_snapshot, colormap_length_for_axes, current_axes_state, current_colormap_length,
    current_figure_handle, figure_colormap_snapshot, set_colormap_for_axes_with_length,
    set_colormap_for_figure_with_length,
};
use runmat_builtins::{
    COLORMAP_DESCRIPTOR, COLORMAP_ERROR_INVALID_ARGUMENT, COLORMAP_ERROR_OUTPUTS,
    COLORMAP_NON_UINT8_INTEGER_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

#[runtime_builtin(
    name = "colormap",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::cmds::colormap"
)]
pub fn colormap_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &COLORMAP_DESCRIPTOR,
        &COLORMAP_ERROR_OUTPUTS,
        "colormap",
    )?;
    if args.len() > 2 {
        return Err(invalid(
            "expected colormap(), colormap(map), colormap(target), or colormap(target, map)",
        ));
    }
    if args.is_empty() {
        let state = current_axes_state();
        return if crate::compatibility::runmat_extensions_enabled() {
            query(PlotHandle::Axes(state.handle, state.active_index))
        } else {
            query(PlotHandle::Figure(state.handle))
        };
    }
    if args.len() == 1 && !typed_integer(&args[0]) {
        if let Ok(target) = resolve_plot_handle(&args[0], "colormap") {
            return query(target);
        }
    }
    let target = if args.len() == 2 {
        if typed_integer(&args[0]) {
            return Err(invalid(
                "target must be a figure or axes graphics handle, not a typed integer",
            ));
        }
        let target = resolve_plot_handle(&args[0], "colormap")?;
        require_palette_target(&target)?;
        Some(target)
    } else {
        None
    };
    // Named palettes retain the addressed palette's length, not the selected axes length.
    // Parsing precedes current-figure creation and any mutation.
    let length = match target.as_ref() {
        Some(PlotHandle::Axes(handle, axes)) => colormap_length_for_axes(*handle, *axes),
        Some(PlotHandle::Figure(handle)) => {
            figure_colormap_snapshot(*handle).map_err(state_error)?.1
        }
        None if crate::compatibility::runmat_extensions_enabled() => current_colormap_length(),
        None => figure_colormap_snapshot(current_figure_handle())
            .map(|(_, length)| length)
            .unwrap_or(256),
        _ => unreachable!("palette target admitted"),
    };
    let (map, length) = colormap_parse_map(args.last().expect("nonempty arguments"), length)?;
    let output = Value::Tensor(colormap_tensor(map.clone(), length));
    let target = target.unwrap_or_else(|| {
        let state = current_axes_state();
        if crate::compatibility::runmat_extensions_enabled() {
            PlotHandle::Axes(state.handle, state.active_index)
        } else {
            PlotHandle::Figure(state.handle)
        }
    });
    match target {
        PlotHandle::Axes(handle, axes) => {
            set_colormap_for_axes_with_length(handle, axes, map, length)
        }
        PlotHandle::Figure(handle) => set_colormap_for_figure_with_length(handle, map, length),
        _ => unreachable!("palette target admitted"),
    }
    .map_err(state_error)?;
    Ok(output)
}

fn typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn require_palette_target(target: &PlotHandle) -> crate::BuiltinResult<()> {
    if matches!(target, PlotHandle::Axes(..) | PlotHandle::Figure(..)) {
        Ok(())
    } else {
        Err(invalid("target must be a figure or axes handle"))
    }
}

fn query(target: PlotHandle) -> crate::BuiltinResult<Value> {
    let (map, length) = match target {
        PlotHandle::Axes(handle, axes) => {
            let metadata = axes_metadata_snapshot(handle, axes).map_err(state_error)?;
            (metadata.colormap, metadata.colormap_length)
        }
        PlotHandle::Figure(handle) => figure_colormap_snapshot(handle).map_err(state_error)?,
        _ => return Err(invalid("target must be a figure or axes handle")),
    };
    Ok(Value::Tensor(colormap_tensor(map, length)))
}

fn invalid(message: impl Into<String>) -> crate::RuntimeError {
    cmd_error_with_message("colormap", message, &COLORMAP_ERROR_INVALID_ARGUMENT)
}

fn state_error(error: crate::builtins::plotting::state::FigureError) -> crate::RuntimeError {
    invalid(format!("colormap: {error}"))
}

fn colormap_parse_map(
    arg: &Value,
    length: usize,
) -> crate::BuiltinResult<(runmat_plot::plots::surface::ColorMap, usize)> {
    if let Some(name) = as_lower_str(arg).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source("cmds", source.to_string(), source)
    })? {
        if name.trim() == "default" {
            return Ok((runmat_plot::plots::surface::ColorMap::Parula, 256));
        }
        let Some(cmap) = runmat_plot::plots::surface::ColorMap::from_name(&name) else {
            let other = name.trim();
            return Err(cmd_error_with_message(
                "colormap",
                format!(
                    "{}: unknown colormap '{other}'",
                    COLORMAP_ERROR_INVALID_ARGUMENT.message
                ),
                &COLORMAP_ERROR_INVALID_ARGUMENT,
            ));
        };
        return Ok((cmap, length));
    }

    if let Value::Tensor(tensor) = arg {
        if tensor.integer_storage().is_some() && tensor.numeric_dtype() != NumericDType::U8 {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COLORMAP_NON_UINT8_INTEGER_EXTENSION,
                "colormap",
            )?;
        }
        return parse_rgb_colormap_tensor(tensor, "colormap");
    };

    Err(cmd_error_with_message(
        "colormap",
        COLORMAP_ERROR_INVALID_ARGUMENT.message,
        &COLORMAP_ERROR_INVALID_ARGUMENT,
    ))
}
#[cfg(test)]
#[path = "colormap/tests.rs"]
mod tests;
