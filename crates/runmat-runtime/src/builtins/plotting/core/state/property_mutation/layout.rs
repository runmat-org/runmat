//! Existing ordinary axes mutations with explicit invocation authority.

use super::super::*;

pub(crate) fn color_order_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<Vec<Vec4>, FigureError> {
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
    if let Some(order) = state
        .figure
        .axes_metadata(axes_index)
        .and_then(|meta| meta.color_order.clone())
    {
        return Ok(order);
    }
    Ok(state
        .figure_color_order
        .clone()
        .unwrap_or_else(default_color_order))
}

pub(crate) fn color_order_for_figure_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
) -> Result<Vec<Vec4>, FigureError> {
    let mut reg = property.registry(PropertyLocation::Figure(handle))?;
    let state = get_state_for_query(&mut reg, handle);
    Ok(state
        .figure_color_order
        .clone()
        .unwrap_or_else(default_color_order))
}

pub(crate) fn set_color_order_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
    colors: &[Vec4],
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        // This setter historically creates a default Figure for a missing
        // handle. Keep that admission behavior, but validate the axes before
        // acquiring a successor for the existing scene.
        let state = get_state_for_query(&mut reg, handle);
        validate_axes_index(state, axes_index)?;
        let state = get_state_mut(&mut reg, handle);
        state.color_cycle_for_axes_mut(axes_index).set_order(colors);
        state
            .figure
            .set_axes_color_order(axes_index, colors.to_vec());
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_color_order_for_figure_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    colors: &[Vec4],
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = property.registry(PropertyLocation::Figure(handle))?;
        let state = get_state_mut(&mut reg, handle);
        state.figure_color_order = Some(colors.to_vec());
        let total_axes = axes_count(state);
        for axes_index in 0..total_axes {
            state.color_cycle_for_axes_mut(axes_index).set_order(colors);
        }
        state.figure.set_all_axes_color_order(colors.to_vec());
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn configure_subplot_for_property(
    property: &PropertyMutationTarget,
    rows: usize,
    cols: usize,
    index: usize,
) -> Result<(), FigureError> {
    configure_subplot_state_for_property(property, rows, cols, index).map(|_| ())
}

pub(crate) fn configure_subplot_state_for_property(
    property: &PropertyMutationTarget,
    rows: usize,
    cols: usize,
    index: usize,
) -> Result<FigureAxesState, FigureError> {
    if rows == 0 || cols == 0 {
        return Err(FigureError::InvalidSubplotGrid { rows, cols });
    }
    let total_axes = rows
        .checked_mul(cols)
        .ok_or(FigureError::InvalidSubplotGrid { rows, cols })?;
    if index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex { rows, cols, index });
    }
    let mut reg = property.registry(PropertyLocation::CurrentFigure)?;
    let handle = reg.current;
    clear_axes::configure_grid(&mut reg, handle, rows, cols);
    let state = get_state_mut(&mut reg, handle);
    state.active_axes = index;
    state.figure.set_active_axes_index(index);
    Ok(FigureAxesState {
        handle,
        rows: state.figure.axes_rows.max(1),
        cols: state.figure.axes_cols.max(1),
        active_index: index,
    })
}
