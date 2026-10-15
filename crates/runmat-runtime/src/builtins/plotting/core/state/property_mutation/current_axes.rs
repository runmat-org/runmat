//! Existing ordinary axes mutations with explicit invocation authority.

use super::super::*;

pub(crate) fn axes_state_snapshot_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<FigureAxesState, FigureError> {
    let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = get_state_for_query(&mut reg, handle);
    validate_axes_index(state, axes_index)?;
    Ok(FigureAxesState {
        handle,
        rows: state.figure.axes_rows.max(1),
        cols: state.figure.axes_cols.max(1),
        active_index: axes_index,
    })
}

pub(crate) fn current_axes_handle_for_figure_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
) -> Result<f64, FigureError> {
    let mut reg = property.registry(PropertyLocation::Figure(handle))?;
    let state = get_state_for_query(&mut reg, handle);
    Ok(encode_axes_handle(handle, state.active_axes))
}

pub(crate) fn current_axes_state_for_property(
    property: &PropertyMutationTarget,
) -> Result<FigureAxesState, FigureError> {
    let mut reg = property.registry(PropertyLocation::CurrentFigure)?;
    let handle = reg.current;
    let state = get_state_for_query(&mut reg, handle);
    Ok(FigureAxesState {
        handle,
        rows: state.figure.axes_rows.max(1),
        cols: state.figure.axes_cols.max(1),
        active_index: state.active_axes,
    })
}

pub(crate) fn set_axis_ticks_for_property(
    property: &PropertyMutationTarget,
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,
) -> Result<(), FigureError> {
    let (handle, figure_clone) = {
        let mut reg = property.registry(PropertyLocation::CurrentFigure)?;
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_ticks(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axis_tick_labels_for_property(
    property: &PropertyMutationTarget,
    x: Option<Vec<String>>,
    y: Option<Vec<String>>,
) -> Result<(), FigureError> {
    let (handle, figure_clone) = {
        let mut reg = property.registry(PropertyLocation::CurrentFigure)?;
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_tick_labels(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axis_tick_formats_for_property(
    property: &PropertyMutationTarget,
    x: Option<String>,
    y: Option<String>,
) -> Result<(), FigureError> {
    let (handle, figure_clone) = {
        let mut reg = property.registry(PropertyLocation::CurrentFigure)?;
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_tick_formats(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axis_tick_angles_for_property(
    property: &PropertyMutationTarget,
    x: Option<f64>,
    y: Option<f64>,
) -> Result<(), FigureError> {
    let (handle, figure_clone) = {
        let mut reg = property.registry(PropertyLocation::CurrentFigure)?;
        let handle = reg.current;
        let state = get_state_mut(&mut reg, handle);
        let axes = state.active_axes;
        state.figure.set_axes_tick_label_rotations(axes, x, y);
        state.revision = state.revision.wrapping_add(1);
        (handle, state.figure.clone())
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}
