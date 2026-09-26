//! Axis direction and manual limit locking share the figure/axes state owner.
use super::*;
use runmat_plot::plots::figure::YDirection;
#[cfg(test)]
#[path = "axes_modes/tests.rs"]
mod tests;

/// Z limits exist for represented 3-D content or an explicitly configured Z
/// range. Camera angles do not change the dimensionality of the data.
pub fn axis_z_limits_snapshot_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<Option<(f64, f64)>, FigureError> {
    let mut reg = registry();
    let state = reg
        .figures
        .get_mut(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    validate_axes_index(state, axes_index)?;
    Ok(z_limits_for_state_axes(state, axes_index))
}

fn z_limits_for_state_axes(state: &mut FigureState, axes_index: usize) -> Option<(f64, f64)> {
    if let Some(limits) = state
        .figure
        .axes_metadata(axes_index)
        .and_then(|meta| meta.z_limits)
    {
        return Some(limits);
    }
    if !state.figure.axes_has_3d_content(axes_index) {
        return None;
    }
    let bounds = state.figure.data_bounds_for_axes(axes_index);
    Some(finite_span(bounds.min.z as f64, bounds.max.z as f64))
}

pub fn set_hidden_line_removal_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let figure = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        state
            .figure
            .set_axes_hidden_line_removal(axes_index, enabled);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure, FigureEventKind::Updated);
    Ok(())
}

pub fn set_y_direction_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    direction: YDirection,
) -> Result<(), FigureError> {
    let figure = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        state.figure.set_axes_y_direction(axes_index, direction);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    notify_with_figure(handle, &figure, FigureEventKind::Updated);
    Ok(())
}

/// Freeze currently represented display limits; later appended data cannot
/// expand them. Link groups continue to use the existing shared limit owner.
pub fn lock_axes_limits(handle: FigureHandle, axes_index: usize) -> Result<(), FigureError> {
    let updates = {
        let mut reg = registry();
        let state = reg
            .figures
            .get_mut(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        let (x0, x1, y0, y1) =
            display_bounds_for_state_axes(state, axes_index).unwrap_or((0., 1., 0., 1.));
        let x = finite_span(x0, x1);
        let y = finite_span(y0, y1);
        if let Some(z) = z_limits_for_state_axes(state, axes_index) {
            state.figure.set_axes_z_limits(axes_index, Some(z));
        }
        set_axes_limits_with_links(&mut reg, handle, axes_index, Some(x), Some(y))?
    };
    notify_figure_updates(updates);
    Ok(())
}

fn finite_span(lower: f64, upper: f64) -> (f64, f64) {
    if lower.is_finite() && upper.is_finite() {
        if upper > lower {
            return (lower, upper);
        }
        if upper == lower {
            let padding = (lower.abs() * 0.05).max(0.5);
            return (lower - padding, upper + padding);
        }
    }
    (0., 1.)
}
