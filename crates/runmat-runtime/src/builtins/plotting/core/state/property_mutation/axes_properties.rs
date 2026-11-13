//! Existing property algorithms at the selected registry transaction boundary.

use super::super::*;

pub(crate) fn set_grid_enabled_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state.figure.set_axes_grid_enabled(axes_index, enabled);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_minor_grid_enabled_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state
                .figure
                .set_axes_minor_grid_enabled(axes_index, enabled);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_box_enabled_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state.figure.set_axes_box_enabled(axes_index, enabled);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axes_style_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    style: TextStyle,
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state.figure.set_axes_style(axes_index, style);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_color_limits_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    limits: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state.figure.set_axes_color_limits(axes_index, limits);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_legend_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
    labels: Option<&[String]>,
    style: Option<LegendStyle>,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state.figure.set_axes_legend_enabled(axes_index, enabled);
            if let Some(labels) = labels {
                state.figure.set_labels_for_axes(axes_index, labels);
            }
            if let Some(style) = style {
                state.figure.set_axes_legend_style(axes_index, style);
            }
            encode_plot_object_handle(handle, axes_index, PlotObjectKind::Legend)
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(object_handle)
}

pub(crate) fn set_axes_position_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    position: [f64; 4],
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state.figure.set_axes_position(axes_index, position);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axes_units_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    units: String,
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state.figure.set_axes_units(axes_index, units);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}
