//! Axes clearing owns both scene compaction and registered plot-index rebasing.
use super::*;
#[cfg(test)]
#[path = "clear_axes/tests.rs"]
mod tests;

/// Clear one or more axes under the registry lock, retaining every field of
/// surviving handles except their relocated dense scene indices.
pub(super) fn clear_registered_axes(
    reg: &mut PlotRegistry,
    handle: FigureHandle,
    targets: &[usize],
) {
    let state = get_state_mut(reg, handle);
    state
        .optimization_plots
        .retain(|(axes, _), _| !targets.contains(axes));
    let mut next = 0;
    let mapping: Vec<Option<usize>> = state
        .figure
        .plot_axes_indices()
        .iter()
        .map(|axes| {
            if targets.contains(axes) {
                None
            } else {
                let index = next;
                next += 1;
                Some(index)
            }
        })
        .collect();
    for &axes in targets {
        state.figure.clear_axes(axes);
        state.reset_cycle(axes);
    }
    reg.plot_children.retain(|_, child| {
        let (figure, axes) = child.figure_axes();
        if figure != handle {
            return true;
        }
        // Colorbars are figure children associated with, not owned by, the axes.
        if matches!(child, PlotChildHandleState::ColorBar(_)) {
            return true;
        }
        if targets.contains(&axes)
            || matches!(child,
            PlotChildHandleState::StackedPlot(stacked)
                if stacked.axes_indices.iter().any(|axes| targets.contains(axes)))
        {
            return false;
        }
        true
    });
    // Drop-only admission does not copy a retained payload. Only survivors
    // whose Figure layout is being remapped become writable.
    remap_survivors(reg, handle, &mapping);
}
/// Remove the already-selected chart payload, preserving unrelated charts on
/// the same axes. The caller retires its exact optimization slot first.
pub(super) fn clear_registered_plot(
    reg: &mut PlotRegistry,
    handle: FigureHandle,
    index: usize,
) -> Result<(), String> {
    let count = get_state_mut(reg, handle).figure.plot_axes_indices().len();
    if index >= count {
        return Err("selected optimization chart index was retired".into());
    }
    let mapping: Vec<Option<usize>> = (0..count)
        .map(|old| {
            if old == index {
                None
            } else {
                Some(old - usize::from(old > index))
            }
        })
        .collect();
    get_state_mut(reg, handle).figure.remove_plot(index)?;
    reg.plot_children
        .retain(|_, child| child.figure_axes().0 != handle || child.plot_index() != Some(index));
    remap_survivors(reg, handle, &mapping);
    Ok(())
}
fn remap_survivors(reg: &mut PlotRegistry, handle: FigureHandle, mapping: &[Option<usize>]) {
    reg.plot_children.retain_mut_where(
        |child| {
            child.figure_axes().0 == handle && !matches!(child, PlotChildHandleState::ColorBar(_))
        },
        |_, child| remap_indices(child, mapping),
    );
}

/// Grid expansion can reclaim former overlay axes. Compact their plot data
/// while the original index map still exists, then replace their metadata.
pub(super) fn configure_grid(
    reg: &mut PlotRegistry,
    handle: FigureHandle,
    rows: usize,
    cols: usize,
) {
    let state = get_state_mut(reg, handle);
    let grid_axes = rows.saturating_mul(cols);
    let reclaimed: Vec<_> = state
        .figure
        .axes_metadata
        .iter()
        .enumerate()
        .filter_map(|(index, meta)| {
            (index < grid_axes && meta.overlay_parent.is_some()).then_some(index)
        })
        .collect();
    if !reclaimed.is_empty() {
        clear_registered_axes(reg, handle, &reclaimed);
        let state = get_state_mut(reg, handle);
        for &index in &reclaimed {
            state.axes_receivers.remove(&index);
        }
        reg.plot_children.retain(|_, child| !matches!(child,
            PlotChildHandleState::ColorBar(bar) if bar.figure == handle && reclaimed.contains(&bar.axes_index)));
    }
    get_state_mut(reg, handle)
        .figure
        .set_subplot_grid(rows, cols);
}

fn remap_indices(child: &mut PlotChildHandleState, mapping: &[Option<usize>]) -> bool {
    let remap = |index: &mut usize| {
        let Some(Some(next)) = mapping.get(*index) else {
            return false;
        };
        *index = *next;
        true
    };
    match child {
        PlotChildHandleState::Line(state)
        | PlotChildHandleState::Scatter(state)
        | PlotChildHandleState::Bar(state)
        | PlotChildHandleState::Stairs(state)
        | PlotChildHandleState::Surface(state)
        | PlotChildHandleState::Patch(state)
        | PlotChildHandleState::Line3(state)
        | PlotChildHandleState::Scatter3(state)
        | PlotChildHandleState::Contour(state)
        | PlotChildHandleState::ContourFill(state)
        | PlotChildHandleState::ReferenceLine(state)
        | PlotChildHandleState::Pie(state) => remap(&mut state.plot_index),
        PlotChildHandleState::Histogram(state) => remap(&mut state.plot_index),
        PlotChildHandleState::Histogram2(state) => remap(&mut state.plot_index),
        PlotChildHandleState::AnimatedLine(state) => remap(&mut state.plot_index),
        PlotChildHandleState::Stem(state) => remap(&mut state.plot_index),
        PlotChildHandleState::ErrorBar(state) => remap(&mut state.plot_index),
        PlotChildHandleState::Quiver(state) => remap(&mut state.plot_index),
        PlotChildHandleState::Image(state) => remap(&mut state.plot_index),
        PlotChildHandleState::Heatmap(state) => remap(&mut state.plot_index),
        PlotChildHandleState::Binscatter(state) => remap(&mut state.plot_index),
        PlotChildHandleState::FunctionSurface(state) => remap(&mut state.plot_index),
        PlotChildHandleState::FunctionContour(state) => remap(&mut state.plot_index),
        PlotChildHandleState::Area(state) => remap(&mut state.plot_index),
        PlotChildHandleState::TextScatter(state) => {
            state.marker_plot_index.as_mut().is_none_or(remap)
        }
        PlotChildHandleState::StackedPlot(state) => state.line_plot_indices.iter_mut().all(remap),
        PlotChildHandleState::ColorBar(_)
        | PlotChildHandleState::Text(_)
        | PlotChildHandleState::WordCloud(_) => true,
    }
}

#[cfg(test)]
pub fn clear_axes_target(
    handle: FigureHandle,
    axes_index: usize,
    reset: bool,
) -> Result<(), FigureError> {
    clear_axes_target_for_property(&PropertyMutationTarget::alias(), handle, axes_index, reset)
}

pub(crate) fn clear_axes_target_for_property(
    property: &PropertyMutationTarget,
    handle: FigureHandle,
    axes_index: usize,
    reset: bool,
) -> Result<(), FigureError> {
    let figure = {
        let mut reg = property.registry(PropertyLocation::Axes(handle, axes_index))?;
        let state = reg
            .figures
            .get(&handle)
            .ok_or(FigureError::InvalidHandle(handle.as_u32()))?;
        validate_axes_index(state, axes_index)?;
        clear_registered_axes(&mut reg, handle, &[axes_index]);
        let state = get_state_mut(&mut reg, handle);
        if reset {
            state.figure.reset_axes_properties(axes_index);
            state.hold_per_axes.remove(&axes_index);
            state.line_style_cycles.remove(&axes_index);
            state.line_color_cycles.remove(&axes_index);
            state.zoom_baselines.remove(&axes_index);
            state.zoom_axes_modes.remove(&axes_index);
            state.pan_axes_modes.remove(&axes_index);
        }
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(handle, &figure, FigureEventKind::Updated)?;
    Ok(())
}

#[cfg(test)]
pub fn clear_current_axes() {
    let state = current_axes_state();
    clear_axes_target(state.handle, state.active_index, false).expect("current axes exists");
}
