//! Colorbar identity and visibility in the existing graphics child registry.
use super::*;

#[cfg(test)]
#[path = "colorbar/tests.rs"]
mod tests;

#[derive(Clone, Debug)]
pub struct ColorBarHandleState {
    pub figure: FigureHandle,
    pub axes_index: usize,
    pub visible: bool,
}

/// Reconcile a published scene before observers can inspect its object hierarchy.
/// A hidden colorbar remains live; only explicit deletion or removal of its
/// associated axes invalidates its identity.
pub(super) fn reconcile(reg: &mut PlotRegistry, handle: FigureHandle, figure: &Figure) {
    let mut existing_axes = HashSet::new();
    reg.plot_children.retain(|_, child| {
        if let PlotChildHandleState::ColorBar(bar) = child {
            if bar.figure == handle {
                if figure.axes_metadata.get(bar.axes_index).is_none() {
                    return false;
                }
                existing_axes.insert(bar.axes_index);
            }
        }
        true
    });
    reg.plot_children.for_each_mut_where(
        |child| matches!(child, PlotChildHandleState::ColorBar(bar) if bar.figure == handle
            && figure.axes_metadata.get(bar.axes_index).is_some_and(|meta| meta.colorbar_enabled != bar.visible)),
        |child| { if let PlotChildHandleState::ColorBar(bar) = child {
            bar.visible = figure.axes_metadata[bar.axes_index].colorbar_enabled;
        } },
    );
    for (axes_index, meta) in figure.axes_metadata.iter().enumerate() {
        if meta.colorbar_enabled && !existing_axes.contains(&axes_index) {
            let id = reg.next_plot_child_handle;
            reg.next_plot_child_handle += 1;
            reg.plot_children.insert(
                id,
                PlotChildHandleState::ColorBar(ColorBarHandleState {
                    figure: handle,
                    axes_index,
                    visible: true,
                }),
            );
        }
    }
}

pub(super) fn reconcile_publication(handle: FigureHandle, figure: &Figure) {
    let mut reg = registry();
    reconcile(&mut reg, handle, figure);
}

pub(super) fn handles_in_registry(reg: &PlotRegistry, handle: FigureHandle) -> Vec<f64> {
    let mut handles: Vec<_> = reg
        .plot_children
        .iter()
        .filter_map(|(id, child)| {
            matches!(child, PlotChildHandleState::ColorBar(bar) if bar.figure == handle)
                .then_some(*id as f64)
        })
        .collect();
    handles.sort_by(f64::total_cmp);
    handles
}

pub fn colorbar_handle_for_axes(handle: FigureHandle, axes_index: usize) -> Option<f64> {
    registry().plot_children.iter().find_map(|(id, child)| {
        matches!(child, PlotChildHandleState::ColorBar(bar)
            if bar.figure == handle && bar.axes_index == axes_index)
        .then_some(*id as f64)
    })
}

pub fn ensure_colorbar_for_axes(
    handle: FigureHandle,
    axes_index: usize,
) -> Result<f64, FigureError> {
    set_colorbar_enabled_for_axes(handle, axes_index, true)?;
    colorbar_handle_for_axes(handle, axes_index).ok_or(FigureError::InvalidPlotObjectHandle)
}

pub fn set_colorbar_enabled_for_axes(
    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    set_colorbar_enabled_for_axes_for_property(
        &PropertyMutationTarget::alias(),
        handle,
        axes_index,
        enabled,
    )
}

pub(crate) fn set_colorbar_enabled_for_axes_for_property(
    property: &PropertyMutationTarget,

    handle: FigureHandle,
    axes_index: usize,
    enabled: bool,
) -> Result<(), FigureError> {
    let figure = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = reg
            .figures
            .get(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        if axes_index >= axes_count(state) {
            return Err(FigureError::InvalidAxesHandle);
        }
        let state = get_state_mut(&mut reg, handle);
        state.figure.set_axes_colorbar_enabled(axes_index, enabled);
        state.revision = state.revision.wrapping_add(1);
        let figure = state.figure.clone();
        if !enabled {
            reg.plot_children.retain(|_, child| {
                !matches!(child, PlotChildHandleState::ColorBar(bar)
                    if bar.figure == handle && bar.axes_index == axes_index)
            });
        }
        reconcile(&mut reg, handle, &figure);
        figure
    };
    property.notify(handle, &figure, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn set_colorbar_visible_for_property(
    property: &PropertyMutationTarget,
    handle: f64,
    visible: bool,
) -> Result<(), FigureError> {
    let (figure_handle, figure) = {
        let mut reg = property.registry(PropertyLocation::Child(property.child_key(handle)?))?;
        let Some(PlotChildHandleState::ColorBar(bar)) =
            reg.plot_children.get(&property.child_key(handle)?)
        else {
            return Err(FigureError::InvalidPlotObjectHandle);
        };
        let (figure_handle, axes_index) = (bar.figure, bar.axes_index);
        let state = existing_axes_state_mut(&mut reg, figure_handle, axes_index)?;
        state.figure.set_axes_colorbar_enabled(axes_index, visible);
        state.revision = state.revision.wrapping_add(1);
        let figure = state.figure.clone();
        reconcile(&mut reg, figure_handle, &figure);
        (figure_handle, figure)
    };
    property.notify(figure_handle, &figure, FigureEventKind::Updated)?;
    Ok(())
}
