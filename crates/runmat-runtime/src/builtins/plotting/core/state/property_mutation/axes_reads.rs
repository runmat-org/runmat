//! Existing query algorithms read through the same retained target guard.
use super::super::axes_modes::{
    color_limits_for_state_axes, displayed_color_limits, z_limits_for_state_axes,
};
use super::super::*;
use runmat_plot::plots::figure::AxisLimitMethod;

pub(crate) fn axis_ticks_snapshot_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisTickSnapshot, FigureError> {
    let reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    Ok((meta.x_ticks, meta.y_ticks))
}

pub(crate) fn axis_tick_labels_snapshot_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisTickLabelSnapshot, FigureError> {
    let reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    Ok((meta.x_tick_labels, meta.y_tick_labels))
}

pub(crate) fn axis_tick_formats_snapshot_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisTickFormatSnapshot, FigureError> {
    let reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    Ok((meta.x_tick_format, meta.y_tick_format))
}

pub(crate) fn axis_tick_angles_snapshot_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<AxisTickAngleSnapshot, FigureError> {
    let reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .cloned()
        .unwrap_or_default();
    Ok((meta.x_tick_label_rotation, meta.y_tick_label_rotation))
}

pub(crate) fn axis_displayed_limits_snapshot_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
    axis: AxisLimitAxis,
) -> Result<Option<(f64, f64)>, FigureError> {
    let axis = match axis {
        AxisLimitAxis::X => LinkAxesAxis::X,
        AxisLimitAxis::Y => LinkAxesAxis::Y,
        AxisLimitAxis::Z => {
            return axis_z_limits_snapshot_for_axes_for_property(property, handle, axes_index)
        }
    };
    let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = reg
        .figures
        .get_mut(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    validate_axes_index(state, axes_index)?;
    let explicit = state
        .figure
        .axes_metadata(axes_index)
        .and_then(|meta| match axis {
            LinkAxesAxis::X => meta.x_limits,
            LinkAxesAxis::Y => meta.y_limits,
        });
    let displayed = displayed_xy_limit(display_bounds_for_state_axes(state, axes_index), axis);
    Ok(displayed.or(explicit))
}

pub(crate) fn axis_limit_method_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
    axis: AxisLimitAxis,
) -> Result<AxisLimitMethod, FigureError> {
    let reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    validate_axes_index(state, axes_index)?;
    let meta = state
        .figure
        .axes_metadata(axes_index)
        .expect("validated axes");
    Ok(match axis {
        AxisLimitAxis::X => meta.x_limit_method,
        AxisLimitAxis::Y => meta.y_limit_method,
        AxisLimitAxis::Z => meta.z_limit_method,
    })
}

pub(crate) fn axis_color_limits_snapshot_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<runmat_plot::plots::surface::ScalarColorRange, FigureError> {
    let reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = reg
        .figures
        .get(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    validate_axes_index(state, axes_index)?;
    Ok(color_limits_for_state_axes(state, axes_index))
}

pub(crate) fn axis_displayed_color_limits_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<Option<(f64, f64)>, FigureError> {
    let range = axis_color_limits_snapshot_for_axes_for_property(property, handle, axes_index)?;
    Ok(displayed_color_limits(range))
}

pub(crate) fn axis_z_limits_snapshot_for_axes_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
) -> Result<Option<(f64, f64)>, FigureError> {
    let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
    let state = reg
        .figures
        .get_mut(&handle)
        .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
    validate_axes_index(state, axes_index)?;
    Ok(z_limits_for_state_axes(state, axes_index))
}
