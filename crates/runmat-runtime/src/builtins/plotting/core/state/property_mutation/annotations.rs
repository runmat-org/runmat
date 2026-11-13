//! Existing property algorithms at the selected registry transaction boundary.

use super::super::*;

pub(crate) fn set_text_properties_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    kind: PlotObjectKind,
    text: Option<String>,
    style: Option<TextStyle>,
) -> Result<f64, FigureError> {
    let (object_handle, figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            if let Some(text) = text {
                match kind {
                    PlotObjectKind::Title => state.figure.set_axes_title(axes_index, text),
                    PlotObjectKind::Subtitle => state.figure.set_axes_subtitle(axes_index, text),
                    PlotObjectKind::XLabel => state.figure.set_axes_xlabel(axes_index, text),
                    PlotObjectKind::YLabel => state.figure.set_axes_ylabel(axes_index, text),
                    PlotObjectKind::ZLabel => state.figure.set_axes_zlabel(axes_index, text),
                    PlotObjectKind::Legend | PlotObjectKind::XAxis | PlotObjectKind::YAxis => {}
                    PlotObjectKind::SuperTitle => state.figure.set_sg_title(text),
                }
            }
            if let Some(style) = style {
                match kind {
                    PlotObjectKind::Title => state.figure.set_axes_title_style(axes_index, style),
                    PlotObjectKind::Subtitle => {
                        state.figure.set_axes_subtitle_style(axes_index, style)
                    }
                    PlotObjectKind::XLabel => state.figure.set_axes_xlabel_style(axes_index, style),
                    PlotObjectKind::YLabel => state.figure.set_axes_ylabel_style(axes_index, style),
                    PlotObjectKind::ZLabel => state.figure.set_axes_zlabel_style(axes_index, style),
                    PlotObjectKind::Legend | PlotObjectKind::XAxis | PlotObjectKind::YAxis => {}
                    PlotObjectKind::SuperTitle => state.figure.set_sg_title_style(style),
                }
            }
            encode_plot_object_handle(handle, axes_index, kind)
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(object_handle)
}

pub(crate) fn set_text_annotation_properties_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    annotation_index: usize,
    text: Option<String>,
    position: Option<glam::Vec3>,
    style: Option<TextStyle>,
) -> Result<(), FigureError> {
    let ((), figure_clone) = with_annotation_target_mut_for_property(
        property,
        handle,
        axes_index,
        annotation_index,
        |state| {
            if let Some(text) = text {
                state
                    .figure
                    .set_axes_text_annotation_text(axes_index, annotation_index, text);
            }
            if let Some(position) = position {
                state.figure.set_axes_text_annotation_position(
                    axes_index,
                    annotation_index,
                    position,
                );
            }
            if let Some(style) = style {
                state
                    .figure
                    .set_axes_text_annotation_style(axes_index, annotation_index, style);
            }
        },
    )?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn update_text_annotation_position_source_for_property(
    property: &PropertyMutationTarget,
    figure: FigureHandle,
    axes_index: usize,
    annotation_index: usize,
    position_source: runmat_plot::plots::NumericPlotData,
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Annotation(
        figure,
        axes_index,
        annotation_index,
    ))?;
    reg.for_each_child_mut_where(
        |state| {
            matches!(state, PlotChildHandleState::Text(text) if text.figure == figure
            && text.axes_index == axes_index && text.annotation_index == annotation_index)
        },
        |state| {
            if let PlotChildHandleState::Text(text) = state {
                text.position_source = Some(position_source.clone());
            }
        },
    );
    Ok(())
}
