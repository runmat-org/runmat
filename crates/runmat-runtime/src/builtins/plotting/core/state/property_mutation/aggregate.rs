//! Existing property algorithms at the selected registry transaction boundary.

use super::super::*;

pub(crate) fn update_textscatter_handle_state_for_property(
    property: &PropertyMutationTarget,

    handle: f64,
    state: TextScatterHandleState,
) -> Result<(), FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let mut reg = property.registry(PropertyLocation::Child(property.child_key(handle)?))?;
    let id = property.child_key(handle)?;
    let figure = state.figure;
    let axes = state.axes_index;
    if reg.plot_children.replace_if(id, |slot| matches!(slot, PlotChildHandleState::TextScatter(slot) if slot.figure == figure && slot.axes_index == axes), PlotChildHandleState::TextScatter(state)) {
        Ok(())
    } else {
        Err(FigureError::InvalidPlotObjectHandle)
    }
}

pub(crate) fn update_textscatter_figure_for_property(
    property: &PropertyMutationTarget,

    figure: FigureHandle,
    axes_index: usize,
    mut apply: impl FnMut(&mut Figure) -> Result<(), FigureError>,
) -> Result<(), FigureError> {
    let (result, figure_clone) =
        with_axes_target_mut_for_property(property, figure, axes_index, |figure_state| {
            apply(&mut figure_state.figure)
        })?;
    result?;
    property.notify(figure, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn update_wordcloud_handle_state_for_property(
    property: &PropertyMutationTarget,

    handle: f64,
    state: WordCloudHandleState,
) -> Result<(), FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let mut reg = property.registry(PropertyLocation::Child(property.child_key(handle)?))?;
    let id = property.child_key(handle)?;
    let figure = state.figure;
    let axes = state.axes_index;
    if reg.plot_children.replace_if(id, |slot| matches!(slot, PlotChildHandleState::WordCloud(slot) if slot.figure == figure && slot.axes_index == axes), PlotChildHandleState::WordCloud(state)) {
        Ok(())
    } else {
        Err(FigureError::InvalidPlotObjectHandle)
    }
}

pub(crate) fn update_wordcloud_figure_for_property(
    property: &PropertyMutationTarget,

    figure: FigureHandle,
    axes_index: usize,
    mut apply: impl FnMut(&mut Figure) -> Result<(), FigureError>,
) -> Result<(), FigureError> {
    let (result, figure_clone) =
        with_axes_target_mut_for_property(property, figure, axes_index, |figure_state| {
            apply(&mut figure_state.figure)
        })?;
    result?;
    property.notify(figure, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn update_stackedplot_handle_state_for_property(
    property: &PropertyMutationTarget,

    handle: f64,
    state: StackedPlotHandleState,
) -> Result<(), FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let mut reg = property.registry(PropertyLocation::Child(property.child_key(handle)?))?;
    let id = property.child_key(handle)?;
    let figure = state.figure;
    if reg.plot_children.replace_if(
        id,
        |slot| matches!(slot, PlotChildHandleState::StackedPlot(slot) if slot.figure == figure),
        PlotChildHandleState::StackedPlot(state),
    ) {
        Ok(())
    } else {
        Err(FigureError::InvalidPlotObjectHandle)
    }
}

pub(crate) fn update_stackedplot_figure_for_property(
    property: &PropertyMutationTarget,

    state: &StackedPlotHandleState,
    mut apply: impl FnMut(&mut Figure) -> Result<(), FigureError>,
) -> Result<(), FigureError> {
    let axes_index = state.axes_indices.first().copied().unwrap_or(0);
    let (result, figure_clone) =
        with_axes_target_mut_for_property(property, state.figure, axes_index, |figure_state| {
            apply(&mut figure_state.figure)
        })?;
    result?;
    property.notify(state.figure, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}
