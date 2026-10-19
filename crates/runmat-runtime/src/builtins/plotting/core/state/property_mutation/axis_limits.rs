//! Existing property algorithms at the selected registry transaction boundary.

use super::super::*;

pub(crate) fn set_axis_equal_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state.figure.set_axes_axis_equal(axes_index, enabled);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_data_aspect_ratio_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    ratio: [f64; 3],
    mode: &str,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = get_state_for_query(&mut reg, handle);
        validate_axes_index(state, axes_index)?;
        let state = get_state_mut(&mut reg, handle);
        state
            .figure
            .set_axes_data_aspect_ratio(axes_index, ratio, mode.to_string());
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axis_manual_limits_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    x: Option<(f64, f64)>,
    y: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    let updates = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = reg
            .figures
            .get(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        let updates = set_axes_limits_with_links(&mut reg, handle, axes_index, x, y)?;
        super::PreparedFigureUpdates::capture(property, &reg, updates)?
    };
    updates.publish(property)?;
    Ok(())
}

pub(crate) fn set_axis_ticks_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    x: Option<Vec<f64>>,
    y: Option<Vec<f64>>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = existing_axes_state_mut(&mut reg, handle, axes_index)?;
        state.figure.set_axes_ticks(axes_index, x, y);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axis_tick_labels_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    x: Option<Vec<String>>,
    y: Option<Vec<String>>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = existing_axes_state_mut(&mut reg, handle, axes_index)?;
        state.figure.set_axes_tick_labels(axes_index, x, y);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axis_tick_formats_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    x: Option<String>,
    y: Option<String>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = existing_axes_state_mut(&mut reg, handle, axes_index)?;
        state.figure.set_axes_tick_formats(axes_index, x, y);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axis_tick_angles_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    x: Option<f64>,
    y: Option<f64>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = existing_axes_state_mut(&mut reg, handle, axes_index)?;
        state.figure.set_axes_tick_label_rotations(axes_index, x, y);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axis_z_limits_preserving_selection_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    limits: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    let figure = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = reg
            .figures
            .get(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        let state = get_state_mut(&mut reg, handle);
        state.figure.set_axes_z_limits(axes_index, limits);
        state.revision = state.revision.wrapping_add(1);
        state.figure.owned_clone()
    };
    property.notify(handle, &figure, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_axis_color_limits_preserving_selection_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    limits: Option<(f64, f64)>,
) -> Result<(), FigureError> {
    let figure = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = reg
            .figures
            .get(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        let state = get_state_mut(&mut reg, handle);
        state.figure.set_axes_color_limits(axes_index, limits);
        state.revision = state.revision.wrapping_add(1);
        state.figure.owned_clone()
    };
    property.notify(handle, &figure, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_log_modes_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    x_log: bool,
    y_log: bool,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = existing_axes_state_mut(&mut reg, handle, axes_index)?;
        state.figure.set_axes_log_modes(axes_index, x_log, y_log);
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_y_axis_location_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    location: String,
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state.figure.set_axes_y_axis_location(axes_index, location);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_view_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    azimuth_deg: f32,
    elevation_deg: f32,
) -> Result<(), FigureError> {
    let ((), figure_clone) =
        with_axes_target_mut_for_property(property, handle, axes_index, |state| {
            state
                .figure
                .set_axes_view(axes_index, azimuth_deg, elevation_deg);
        })?;
    property.notify(handle, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}
