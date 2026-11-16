//! Existing ordinary axes mutations with explicit invocation authority.

use super::super::*;
use runmat_plot::plots::AxesGridCell;

pub(crate) fn select_subplot_region_for_property(
    property: &PropertyMutationTarget,
    region: AxesGridCell,
) -> Result<FigureAxesState, FigureError> {
    let rows = region.rows as usize;
    let cols = region.columns as usize;
    if !region.is_valid() {
        return Err(FigureError::InvalidSubplotGrid { rows, cols });
    }
    let mut reg = property.registry(PropertyLocation::CurrentFigure)?;
    let handle = reg.current;
    let existed = reg.figures.contains_key(&handle);
    let (matching, targets, index) = {
        let state = get_state_for_query(&mut reg, handle);
        let figure = &state.figure;
        let count = figure.axes_count();
        let matching = (0..count).find(|index| {
            figure.axes_metadata(*index).is_some_and(|metadata| {
                metadata.overlay_parent.is_none()
                    && figure
                        .resolved_axes_grid_cell(*index)
                        .is_some_and(|cell| cell.same_region(region))
            })
        });
        if let Some(index) = matching {
            (true, Vec::new(), index)
        } else {
            let mut targets: Vec<usize> = (0..count)
                .filter(|index| {
                    figure.axes_metadata(*index).is_some_and(|metadata| {
                        metadata.overlay_parent.is_none()
                            && subplot_region_overlaps(figure, *index, region)
                    })
                })
                .collect();
            // Include dependent overlay axes before retiring their parent.
            loop {
                let previous = targets.len();
                for index in 0..count {
                    if figure
                        .axes_metadata(index)
                        .and_then(|metadata| metadata.overlay_parent)
                        .is_some_and(|parent| targets.contains(&parent))
                        && !targets.contains(&index)
                    {
                        targets.push(index);
                    }
                }
                if targets.len() == previous {
                    break;
                }
            }
            let index = targets
                .first()
                .copied()
                .or_else(|| (0..count).find(|index| !figure.axes_is_live(*index)))
                .unwrap_or(count);
            (false, targets, index)
        }
    };
    if !targets.is_empty() {
        clear_axes::retire_registered_axes(&mut reg, handle, &targets);
    }
    let state = get_state_mut(&mut reg, handle);
    if !matching {
        state.figure.ensure_axes(index);
        state
            .figure
            .set_axes_grid_cell(index, region)
            .map_err(|_| FigureError::InvalidSubplotGrid { rows, cols })?;
        state.reset_cycle(index);
    }
    state.active_axes = index;
    state.figure.set_active_axes_index(index);
    state.revision = state.revision.wrapping_add(1);
    let figure = state.figure.clone();
    drop(reg);
    property.notify(
        handle,
        &figure,
        if existed {
            FigureEventKind::Updated
        } else {
            FigureEventKind::Created
        },
    )?;
    Ok(FigureAxesState {
        handle,
        rows,
        cols,
        active_index: index,
    })
}

fn subplot_region_overlaps(figure: &Figure, index: usize, region: AxesGridCell) -> bool {
    if let Some(cell) = figure.resolved_axes_grid_cell(index) {
        return cell.overlaps(region);
    }
    let Some(metadata) = figure.axes_metadata(index) else {
        return false;
    };
    let Some(target) = region.normalized_position() else {
        return false;
    };
    let mut position = metadata.position;
    if !metadata.units.eq_ignore_ascii_case("normalized") {
        let [_, _, width, height] = figure.position;
        if !(width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0) {
            return false;
        }
        position[0] /= width;
        position[1] /= height;
        position[2] /= width;
        position[3] /= height;
    }
    position[0] < target[0] + target[2]
        && target[0] < position[0] + position[2]
        && position[1] < target[1] + target[3]
        && target[1] < position[1] + position[3]
}

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
            if state.figure.axes_is_live(axes_index) {
                state.color_cycle_for_axes_mut(axes_index).set_order(colors);
            }
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
