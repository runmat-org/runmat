use runmat_value::Value;

use super::axis_ticks::TickAxis;
use super::properties::{resolve_plot_handle, PlotHandle};
use super::state::{
    axis_tick_formats_snapshot_for_axes_for_property, decode_axes_handle,
    set_axis_tick_formats_for_axes_for_property,
};
use super::{plotting_error, plotting_error_with_source};
use crate::builtins::common::tensor;
use crate::builtins::plotting::style::value_as_string;
use crate::BuiltinResult;

const DEFAULT_TICK_FORMAT: &str = "%g";

enum TickFormatTarget {
    Current,
    Axes(Vec<(super::state::FigureHandle, usize)>),
    Native(super::op_common::axes_target::PreparedAxesTarget),
}

pub fn axis_tick_format_builtin(
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
    axis: TickAxis,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let (target, args) = prepare_call(runtime, builtin, args)?;
    match args.len() {
        0 => {
            let mut targets = prepared_targets(runtime, target).into_iter();
            let first = targets.next().expect("current or nonempty admitted axes");
            if targets.next().is_some() {
                return Err(plotting_error(
                    builtin,
                    format!("{builtin}: query form requires a scalar axes handle"),
                ));
            }
            query_format(builtin, axis, &first.resolve(builtin)?)
        }
        1 => {
            let value = args.first().expect("one format argument");
            let format = format_from_value(value, builtin)?;
            super::op_common::axes_output::validate_setter_output(builtin)?;
            for target in prepared_targets(runtime, target) {
                set_format(builtin, axis, &target.resolve(builtin)?, format.clone())?;
            }
            Ok(Value::string(format))
        }
        _ => Err(plotting_error(
            builtin,
            format!("{builtin}: expected zero or one tick-format argument"),
        )),
    }
}

fn prepare_call(
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<(TickFormatTarget, crate::call::BuiltinArguments)> {
    if let Some(first) = args
        .first()
        .filter(|value| matches!(value, Value::Object(_) | Value::HandleObject(_)))
    {
        if let Some(prepared) =
            super::op_common::axes_target::prepare_graphics_target(runtime, builtin, first)?
        {
            return Ok((
                TickFormatTarget::Native(prepared.into_axes(builtin)?),
                args.into_range(1..).expect("nonempty axes arguments"),
            ));
        }
    }
    split_optional_axes_target(builtin, args)
}

fn prepared_targets(
    runtime: Option<&crate::context::RuntimeContext>,
    target: TickFormatTarget,
) -> Vec<super::op_common::axes_target::PreparedAxesTarget> {
    use super::op_common::axes_target::PreparedAxesTarget;
    match target {
        TickFormatTarget::Current => vec![PreparedAxesTarget::current(runtime)],
        TickFormatTarget::Axes(targets) => targets
            .into_iter()
            .map(|(handle, index)| PreparedAxesTarget::alias(runtime, handle, index))
            .collect(),
        TickFormatTarget::Native(target) => vec![target],
    }
}

fn split_optional_axes_target(
    builtin: &'static str,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<(TickFormatTarget, crate::call::BuiltinArguments)> {
    let Some(first) = args.first() else {
        return Ok((TickFormatTarget::Current, args));
    };
    if let Ok(PlotHandle::Axes(handle, axes_index)) = resolve_plot_handle(first, builtin) {
        return Ok((
            TickFormatTarget::Axes(vec![(handle, axes_index)]),
            args.into_range(1..).expect("nonempty axes arguments"),
        ));
    }
    if let Some(targets) = axes_array_targets(first) {
        return Ok((
            TickFormatTarget::Axes(targets),
            args.into_range(1..).expect("nonempty axes arguments"),
        ));
    }
    Ok((TickFormatTarget::Current, args))
}

fn axes_array_targets(value: &Value) -> Option<Vec<(super::state::FigureHandle, usize)>> {
    let Value::Tensor(tensor) = value else {
        return None;
    };
    let data = tensor::tensor_values_f64(tensor);
    if data.is_empty() {
        return None;
    }
    let mut targets = Vec::with_capacity(data.len());
    for scalar in &data {
        let Ok((handle, axes_index)) = decode_axes_handle(*scalar) else {
            return None;
        };
        if !super::state::axes_handle_exists(handle, axes_index) {
            return None;
        }
        targets.push((handle, axes_index));
    }
    Some(targets)
}

fn query_format(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
) -> BuiltinResult<Value> {
    let (handle, index) = target.coordinates();
    let (x, y) =
        axis_tick_formats_snapshot_for_axes_for_property(target.authority(), handle, index)
            .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
    Ok(Value::string(
        match axis {
            TickAxis::X => x,
            TickAxis::Y => y,
        }
        .unwrap_or_else(|| DEFAULT_TICK_FORMAT.into()),
    ))
}

fn set_format(
    builtin: &'static str,
    axis: TickAxis,
    target: &super::state::ResolvedAxesTarget,
    format: String,
) -> BuiltinResult<()> {
    let (handle, index) = target.coordinates();
    let property = target.authority();
    let (x, y) = axis_tick_formats_snapshot_for_axes_for_property(property, handle, index)
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))?;
    let (x, y) = match axis {
        TickAxis::X => (Some(format), y),
        TickAxis::Y => (x, Some(format)),
    };
    set_axis_tick_formats_for_axes_for_property(property, handle, index, x, y)
        .map_err(|err| plotting_error_with_source(builtin, format!("{builtin}: {err}"), err))
}

fn format_from_value(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    let format = value_as_string(value)
        .map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                builtin,
                source.to_string(),
                source,
            )
        })?
        .ok_or_else(|| {
            plotting_error(
                builtin,
                format!("{builtin}: tick format must be a string scalar"),
            )
        })?;
    Ok(runmat_plot::core::plot_renderer::plot_utils::canonical_tick_label_format(&format))
}
