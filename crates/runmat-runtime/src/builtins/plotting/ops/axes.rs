//! MATLAB-compatible `axes` builtin.

use runmat_builtins::{AXES_DESCRIPTOR, AXES_ERROR_TOO_MANY_OUTPUTS};
use runmat_value::Value;

use super::plotting_error;
use super::properties::{
    map_figure_error, resolve_plot_handle, select_graphics_property, set_properties_with_authority,
    PlotHandle,
};
use super::state::{
    create_axes_for_figure_for_property, encode_axes_handle, select_axes_for_figure_for_property,
    FigureHandle, PropertyMutationTarget,
};

const BUILTIN_NAME: &str = "axes";

#[runmat_macros::runtime_builtin(
    name = "axes",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::axes"
)]
pub fn axes_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    axes_builtin_with_runtime(runtime.as_ref(), args)
}

#[cfg(test)]
pub fn axes_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    axes_builtin_with_runtime(
        None,
        crate::call::BuiltinArguments::from_borrowed(&args, None)?,
    )
}

fn axes_builtin_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &AXES_DESCRIPTOR,
        &AXES_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    axes_state(runtime, args).map_err(|mut error| {
        if error.identifier.is_none() {
            error.identifier = Some("RunMat:axes:InvalidArgument".into());
        }
        error.context = error.context.with_builtin(BUILTIN_NAME);
        error
    })
}

fn axes_state(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    let selected = args
        .first()
        .map(|value| select_graphics_property(value, BUILTIN_NAME))
        .transpose()?
        .flatten();
    let (target, property_args) = match selected
        .as_ref()
        .map(|selection| selection.receiver().kind())
    {
        Some(runmat_value::GraphicsReceiverKind::Axes { figure, index }) => (
            Some(AxesTarget::ExistingAxes(FigureHandle::from(figure), index)),
            args.into_range(1..)?,
        ),
        Some(runmat_value::GraphicsReceiverKind::Figure { figure }) => (
            Some(AxesTarget::ParentFigure(FigureHandle::from(figure))),
            args.into_range(1..)?,
        ),
        _ => split_axes_target(args)?,
    };
    let property = selected
        .map_or_else(
            PropertyMutationTarget::alias,
            PropertyMutationTarget::selected,
        )
        .with_runtime(runtime);
    if !property_args.len().is_multiple_of(2) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "axes: property/value arguments must come in pairs",
        ));
    }
    match target {
        Some(AxesTarget::ExistingAxes(handle, axes_index)) => {
            select_axes_for_figure_for_property(&property, handle, axes_index)
                .map_err(|err| map_figure_error(BUILTIN_NAME, err))?;
            if !property_args.is_empty() {
                set_properties_with_authority(
                    &property,
                    PlotHandle::Axes(handle, axes_index),
                    property_args.view(),
                    BUILTIN_NAME,
                )?;
            }
            Ok(encode_axes_handle(handle, axes_index))
        }
        Some(AxesTarget::ParentFigure(handle)) => {
            create_and_configure_axes(&property, Some(handle), property_args)
        }
        None => {
            let (parent, selected, filtered) = extract_parent_property(property_args, runtime)?;
            let property = selected
                .map_or(property, PropertyMutationTarget::selected)
                .with_runtime(runtime);
            create_and_configure_axes(&property, parent, filtered)
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AxesTarget {
    ExistingAxes(FigureHandle, usize),
    ParentFigure(FigureHandle),
}

fn split_axes_target(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<(Option<AxesTarget>, crate::call::BuiltinArguments)> {
    let Some(first) = args.first() else {
        return Ok((None, args));
    };
    if starts_property_pairs(first) {
        return Ok((None, args));
    }
    match resolve_plot_handle(first, BUILTIN_NAME) {
        Ok(PlotHandle::Axes(handle, index)) => Ok((
            Some(AxesTarget::ExistingAxes(handle, index)),
            args.into_range(1..)?,
        )),
        Ok(PlotHandle::Figure(handle)) => Ok((
            Some(AxesTarget::ParentFigure(handle)),
            args.into_range(1..)?,
        )),
        Ok(_) => Err(plotting_error(
            BUILTIN_NAME,
            "axes: target must be an axes or figure handle",
        )),
        Err(error) if args.len() == 1 => Err(error),
        Err(_) => Ok((None, args)),
    }
}

fn create_and_configure_axes(
    property: &PropertyMutationTarget,
    parent: Option<FigureHandle>,
    property_args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    if !property_args.len().is_multiple_of(2) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "axes: property/value arguments must come in pairs",
        ));
    }
    let (handle, axes_index) = create_axes_for_figure_for_property(property, parent)
        .map_err(|err| map_figure_error(BUILTIN_NAME, err))?;
    if !property_args.is_empty() {
        set_properties_with_authority(
            property,
            PlotHandle::Axes(handle, axes_index),
            property_args.view(),
            BUILTIN_NAME,
        )?;
    }
    Ok(encode_axes_handle(handle, axes_index))
}

fn extract_parent_property(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<(
    Option<FigureHandle>,
    Option<super::state::GraphicsPropertySelection>,
    crate::call::BuiltinArguments,
)> {
    if !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "axes: property/value arguments must come in pairs",
        ));
    }
    let mut parent = None;
    let mut selected = None;
    let mut retained_count = 0usize;
    for pair in args.chunks_exact(2) {
        if property_name_is(&pair[0], "parent") {
            selected = select_graphics_property(&pair[1], BUILTIN_NAME)?;
            parent = Some(
                match selected
                    .as_ref()
                    .map(|selection| selection.receiver().kind())
                {
                    Some(runmat_value::GraphicsReceiverKind::Figure { figure }) => {
                        FigureHandle::from(figure)
                    }
                    _ => parent_figure_from_value(&pair[1])?,
                },
            );
        } else {
            retained_count += 2;
        }
    }
    let mut cursor = 0usize;
    let filtered = args.try_map_preserving_roots(retained_count, runtime, |_, source| {
        while cursor < source.len() && property_name_is(&source[cursor - cursor % 2], "parent") {
            cursor = (cursor / 2 + 1) * 2;
        }
        let value = source[cursor].clone();
        cursor += 1;
        Ok(value)
    })?;
    Ok((parent, selected, filtered))
}

fn parent_figure_from_value(value: &Value) -> crate::BuiltinResult<FigureHandle> {
    match resolve_plot_handle(value, BUILTIN_NAME)? {
        PlotHandle::Figure(handle) => Ok(handle),
        _ => Err(plotting_error(
            BUILTIN_NAME,
            "axes: Parent must be a figure handle",
        )),
    }
}

fn starts_property_pairs(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::CharArray(_))
}

fn property_name_is(value: &Value, expected: &str) -> bool {
    crate::builtins::plotting::style::value_text_equals_ascii(value, expected, true)
}

#[cfg(test)]
#[path = "axes/tests.rs"]
mod tests;
