//! Existing property algorithms at the selected registry transaction boundary.

use super::super::*;

pub(crate) fn set_sg_title_properties_for_figure_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    text: Option<String>,
    style: Option<TextStyle>,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) = with_figure_mut_for_property(property, handle, |state| {
        if let Some(text) = text {
            state.figure.set_sg_title(text);
        }
        if let Some(style) = style {
            state.figure.set_sg_title_style(style);
        }
        encode_plot_object_handle(handle, 0, PlotObjectKind::SuperTitle)
    })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(object_handle)
}

pub(crate) fn set_figure_name_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    name: String,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut_for_property(property, handle, |state| {
        state.figure.set_name(name);
    })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_figure_tag_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    tag: String,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut_for_property(property, handle, |state| {
        state.tag = tag;
    })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_figure_number_title_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut_for_property(property, handle, |state| {
        state.figure.set_number_title(enabled);
    })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_figure_visible_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    visible: bool,
) -> Result<(bool, Figure), FigureError> {
    let ((was_visible, now_visible), figure_clone) =
        with_figure_mut_for_property(property, handle, |state| {
            let was_visible = state.figure.visible;
            state.figure.set_visible(visible);
            (was_visible, state.figure.visible)
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok((!was_visible && now_visible, figure_clone))
}

pub(crate) fn set_figure_position_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    position: [f64; 4],
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut_for_property(property, handle, |state| {
        state.figure.set_position(position);
    })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_figure_background_color_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    color: Vec4,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_figure_mut_for_property(property, handle, |state| {
        state.figure.set_background_color(color);
    })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}
