//! Existing property algorithms at the selected registry transaction boundary.

use super::super::*;

pub(in super::super) fn with_axes_target_mut_for_property<R>(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    f: impl FnOnce(&mut FigureState) -> R,
) -> Result<(R, Figure), FigureError> {
    with_axes_target_mut_at(
        property,
        handle,
        axes_index,
        PropertyLocation::Axes(handle, axes_index),
        f,
    )
}

pub(in super::super) fn with_annotation_target_mut_for_property<R>(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
    annotation_index: usize,
    f: impl FnOnce(&mut FigureState) -> R,
) -> Result<(R, Figure), FigureError> {
    with_axes_target_mut_at(
        property,
        handle,
        axes_index,
        PropertyLocation::Annotation(handle, axes_index, annotation_index),
        f,
    )
}

pub(in super::super) fn with_figure_mut_for_property<R>(
    property: &PropertyMutationTarget,

    handle: impl Into<FigureTarget>,
    f: impl FnOnce(&mut FigureState) -> R,
) -> Result<(R, Figure), FigureError> {
    let target = handle.into();
    let handle = target.coordinate();
    let mut reg = property.registry(PropertyLocation::Figure(handle))?;
    target.validate(&reg)?;
    let state = get_state_mut(&mut reg, handle);
    let result = f(state);
    state.revision = state.revision.wrapping_add(1);
    let figure = state.figure.owned_clone();
    drop(reg);
    Ok((result, figure))
}

pub(in super::super) fn with_axes_target_mut_at<R>(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
    location: PropertyLocation,
    f: impl FnOnce(&mut FigureState) -> R,
) -> Result<(R, Figure), FigureError> {
    let mut reg = property.registry(location)?;
    // Invalid axes do not acquire a new scene lease or mutate Figure state.
    let state = reg
        .figures
        .entry(handle)
        .or_insert_with(|| FigureState::new(handle));
    let total_axes = axes_count(state);
    if axes_index >= total_axes {
        return Err(FigureError::InvalidSubplotIndex {
            rows: state.figure.axes_rows.max(1),
            cols: state.figure.axes_cols.max(1),
            index: axes_index,
        });
    }
    let state = get_state_mut(&mut reg, handle);
    state.active_axes = axes_index;
    state.figure.set_active_axes_index(axes_index);
    let result = f(state);
    state.revision = state.revision.wrapping_add(1);
    let figure = state.figure.owned_clone();
    drop(reg);
    Ok((result, figure))
}
