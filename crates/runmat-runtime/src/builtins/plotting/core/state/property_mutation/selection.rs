//! Existing property algorithms at the selected registry transaction boundary.

use super::super::*;

pub(crate) fn data_aspect_ratio_snapshot_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
) -> Result<([f64; 3], String), FigureError> {
    let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = get_state_for_query(&mut reg, handle);
    validate_axes_index(state, axes_index)?;
    Ok(state
        .figure
        .axes_metadata(axes_index)
        .map(|meta| (meta.data_aspect_ratio, meta.data_aspect_ratio_mode.clone()))
        .unwrap_or(([1.0, 1.0, 1.0], "auto".into())))
}

pub(crate) fn axis_display_bounds_snapshot_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisDisplayBoundsSnapshot, FigureError> {
    let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = reg
        .figures
        .get_mut(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    Ok(display_bounds_for_state_axes(state, axes_index))
}

pub(crate) fn plot_child_handle_snapshot_for_property(
    property: &PropertyMutationTarget,
    handle: f64,
) -> Result<PlotChildHandleState, FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let reg = property.registry(PropertyLocation::Child(property.child_key(handle)?))?;
    reg.plot_children
        .get(&(property.child_key(handle)?))
        .cloned()
        .ok_or(FigureError::InvalidPlotObjectHandle)
}

pub(crate) fn select_current_figure_if_exists_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Figure(handle))?;
    if !reg.figures.contains_key(&handle) {
        return Err(FigureError::InvalidHandle(handle.as_u32()));
    }
    reg.current = handle;
    Ok(())
}

pub(crate) fn axes_metadata_snapshot_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
) -> Result<runmat_plot::plots::AxesMetadata, FigureError> {
    let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = get_state_for_query(&mut reg, handle);
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .ok_or(FigureError::InvalidAxesHandle)
}

pub(crate) fn select_axes_for_figure_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<(), FigureError> {
    select_axes(property, handle, axes_index, None)
}

pub(crate) fn select_axes_receiver_for_property(
    property: &PropertyMutationTarget,
    receiver: &GraphicsReceiverRef,
) -> Result<(), FigureError> {
    let GraphicsReceiverKind::Axes { figure, index } = receiver.kind() else {
        return Err(FigureError::InvalidAxesHandle);
    };
    select_axes(property, FigureHandle::from(figure), index, Some(receiver))
}

fn select_axes(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
    source: Option<&GraphicsReceiverRef>,
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    if source.is_some_and(|source| {
        !reg.figures
            .get(&handle)
            .and_then(|state| state.axes_receivers.get(&axes_index))
            .is_some_and(|receiver| receiver.shares_lease(source))
    }) {
        return Err(FigureError::StalePlotSnapshot);
    }
    let state = get_state_for_query(&mut reg, handle);
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    reg.current = handle;
    let state = get_state_mut(&mut reg, handle);
    state.active_axes = axes_index;
    state.figure.set_active_axes_index(axes_index);
    state.revision = state.revision.wrapping_add(1);
    let figure_clone = state.figure.clone();
    drop(reg);
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}
