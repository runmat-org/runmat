//! MATLAB-compatible `axes` builtin.

use runmat_builtins::{AXES_DESCRIPTOR, AXES_ERROR_TOO_MANY_OUTPUTS};
use runmat_value::Value;

use super::plotting_error;
use super::properties::{map_figure_error, resolve_plot_handle, set_properties, PlotHandle};
use super::state::{
    create_axes_for_figure, encode_axes_handle, select_axes_for_figure, FigureHandle,
};
use super::style::value_as_string;

const BUILTIN_NAME: &str = "axes";

#[runmat_macros::runtime_builtin(
    name = "axes",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::axes"
)]
pub fn axes_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &AXES_DESCRIPTOR,
        &AXES_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    axes_state(args).map_err(|mut error| {
        if error.identifier.is_none() {
            error.identifier = Some("RunMat:axes:InvalidArgument".into());
        }
        error.context = error.context.with_builtin(BUILTIN_NAME);
        error
    })
}

fn axes_state(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target, property_args) = split_axes_target(args)?;
    if !property_args.len().is_multiple_of(2) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "axes: property/value arguments must come in pairs",
        ));
    }
    match target {
        Some(AxesTarget::ExistingAxes(handle, axes_index)) => {
            select_axes_for_figure(handle, axes_index)
                .map_err(|err| map_figure_error(BUILTIN_NAME, err))?;
            if !property_args.is_empty() {
                set_properties(
                    PlotHandle::Axes(handle, axes_index),
                    &property_args,
                    BUILTIN_NAME,
                )?;
            }
            Ok(encode_axes_handle(handle, axes_index))
        }
        Some(AxesTarget::ParentFigure(handle)) => {
            create_and_configure_axes(Some(handle), property_args)
        }
        None => {
            let (parent, filtered) = extract_parent_property(property_args)?;
            create_and_configure_axes(parent, filtered)
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AxesTarget {
    ExistingAxes(FigureHandle, usize),
    ParentFigure(FigureHandle),
}

fn split_axes_target(args: Vec<Value>) -> crate::BuiltinResult<(Option<AxesTarget>, Vec<Value>)> {
    let mut iter = args.into_iter();
    let Some(first) = iter.next() else {
        return Ok((None, Vec::new()));
    };
    if starts_property_pairs(&first) {
        let mut property_args = vec![first];
        property_args.extend(iter);
        return Ok((None, property_args));
    }
    match resolve_plot_handle(&first, BUILTIN_NAME) {
        Ok(PlotHandle::Axes(handle, axes_index)) => Ok((
            Some(AxesTarget::ExistingAxes(handle, axes_index)),
            iter.collect(),
        )),
        Ok(PlotHandle::Figure(handle)) => {
            Ok((Some(AxesTarget::ParentFigure(handle)), iter.collect()))
        }
        Ok(_) => Err(plotting_error(
            BUILTIN_NAME,
            "axes: target must be an axes or figure handle",
        )),
        Err(err) if iter.len() == 0 => Err(err),
        Err(_) => {
            let mut property_args = vec![first];
            property_args.extend(iter);
            Ok((None, property_args))
        }
    }
}

fn create_and_configure_axes(
    parent: Option<FigureHandle>,
    property_args: Vec<Value>,
) -> crate::BuiltinResult<f64> {
    if !property_args.len().is_multiple_of(2) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "axes: property/value arguments must come in pairs",
        ));
    }
    let (handle, axes_index) =
        create_axes_for_figure(parent).map_err(|err| map_figure_error(BUILTIN_NAME, err))?;
    if !property_args.is_empty() {
        set_properties(
            PlotHandle::Axes(handle, axes_index),
            &property_args,
            BUILTIN_NAME,
        )?;
    }
    Ok(encode_axes_handle(handle, axes_index))
}

fn extract_parent_property(
    args: Vec<Value>,
) -> crate::BuiltinResult<(Option<FigureHandle>, Vec<Value>)> {
    if !args.len().is_multiple_of(2) {
        return Err(plotting_error(
            BUILTIN_NAME,
            "axes: property/value arguments must come in pairs",
        ));
    }
    let mut parent = None;
    let mut filtered = Vec::with_capacity(args.len());
    for pair in args.chunks_exact(2) {
        if property_name_is(&pair[0], "parent") {
            parent = Some(parent_figure_from_value(&pair[1])?);
        } else {
            filtered.push(pair[0].clone());
            filtered.push(pair[1].clone());
        }
    }
    Ok((parent, filtered))
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
    value_as_string(value).is_some()
}

fn property_name_is(value: &Value, expected: &str) -> bool {
    value_as_string(value)
        .map(|name| name.trim().eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

#[cfg(test)]
#[path = "axes/tests.rs"]
mod tests;
