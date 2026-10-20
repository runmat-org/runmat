//! Existing property algorithms at the selected registry transaction boundary.

use super::super::*;

pub(crate) fn set_heatmap_color_limits_for_property(
    property: &PropertyMutationTarget,
    handle: f64,
    limits: Tensor,
) -> Result<(), FigureError> {
    if !handle.is_finite() || handle <= 0.0 {
        return Err(FigureError::InvalidPlotObjectHandle);
    }
    let mut reg = property.registry(PropertyLocation::Child(property.child_key(handle)?))?;
    match reg.plot_children.get_mut(&(property.child_key(handle)?)) {
        Some(PlotChildHandleState::Heatmap(state)) => {
            state.color_limits = Some(limits);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_binscatter_handle_for_plot_for_property(
    property: &PropertyMutationTarget,

    figure: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut BinscatterHandleState),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure, plot_index))?;
    let state = reg.find_child_mut(|state| matches!(state,
        PlotChildHandleState::Binscatter(chart) if chart.figure == figure && chart.plot_index == plot_index))
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    let PlotChildHandleState::Binscatter(chart) = state else {
        return Err(FigureError::InvalidPlotObjectHandle);
    };
    updater(chart);
    Ok(())
}

pub(crate) fn set_heatmap_display_labels_for_property(
    property: &PropertyMutationTarget,

    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    x_labels: Option<Vec<String>>,
    y_labels: Option<Vec<String>>,
) -> Result<(), FigureError> {
    let figure_clone = {
        let mut reg = property.registry(PropertyLocation::Plot(figure, plot_index))?;
        let (current_x_labels, current_y_labels) = {
            let state = reg.find_child_mut(|state| match state {
                PlotChildHandleState::Heatmap(heatmap) => {
                    heatmap.figure == figure
                        && heatmap.axes_index == axes_index
                        && heatmap.plot_index == plot_index
                }
                _ => false,
            });
            let PlotChildHandleState::Heatmap(heatmap) =
                state.ok_or(FigureError::InvalidPlotObjectHandle)?
            else {
                return Err(FigureError::InvalidPlotObjectHandle);
            };
            if let Some(labels) = x_labels {
                heatmap.x_labels = labels;
            }
            if let Some(labels) = y_labels {
                heatmap.y_labels = labels;
            }
            (heatmap.x_labels.clone(), heatmap.y_labels.clone())
        };

        let state = get_state_mut(&mut reg, figure);
        let total_axes = axes_count(state);
        if axes_index >= total_axes {
            return Err(FigureError::InvalidSubplotIndex {
                rows: state.figure.axes_rows.max(1),
                cols: state.figure.axes_cols.max(1),
                index: axes_index,
            });
        }
        state.active_axes = axes_index;
        state.figure.set_active_axes_index(axes_index);
        // Heatmaps draw their first row at the top, so the y tick labels run bottom-up reversed.
        let mut drawn_y_labels = current_y_labels;
        drawn_y_labels.reverse();
        state
            .figure
            .set_axes_tick_labels(axes_index, Some(current_x_labels), Some(drawn_y_labels));
        state.revision = state.revision.wrapping_add(1);
        state.figure.clone()
    };
    property.notify(figure, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}

pub(crate) fn update_histogram_handle_for_plot_for_property(
    property: &PropertyMutationTarget,

    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    normalization: String,
    raw_counts: Vec<f64>,
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure, plot_index))?;
    let state = reg.find_child_mut(|state| match state {
        PlotChildHandleState::Histogram(hist) => {
            hist.figure == figure && hist.axes_index == axes_index && hist.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Histogram(hist) => {
            hist.normalization = normalization;
            hist.raw_counts = raw_counts;
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_histogram2_handle_for_plot_for_property(
    property: &PropertyMutationTarget,

    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    updater: impl FnOnce(&mut Histogram2HandleState),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure, plot_index))?;
    let state = reg.find_child_mut(|state| match state {
        PlotChildHandleState::Histogram2(hist) => {
            hist.figure == figure && hist.axes_index == axes_index && hist.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Histogram2(hist) => {
            updater(hist);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_image_handle_for_plot_for_property(
    property: &PropertyMutationTarget,

    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    updater: impl FnOnce(&mut ImageHandleState),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure, plot_index))?;
    let state = reg.find_child_mut(|state| match state {
        PlotChildHandleState::Image(image) => {
            image.figure == figure
                && image.axes_index == axes_index
                && image.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Image(image) => {
            updater(image);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn set_histogram_handle_display_name_for_property(
    property: &PropertyMutationTarget,

    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    display_name: Option<String>,
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure, plot_index))?;
    let state = reg.find_child_mut(|state| match state {
        PlotChildHandleState::Histogram(hist) => {
            hist.figure == figure && hist.axes_index == axes_index && hist.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Histogram(hist) => {
            hist.display_name = display_name;
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_histogram_handle_metadata_for_plot_for_property(
    property: &PropertyMutationTarget,

    figure: FigureHandle,
    axes_index: usize,
    plot_index: usize,
    updater: impl FnOnce(&mut HistogramHandleMetadata),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure, plot_index))?;
    let state = reg.find_child_mut(|state| match state {
        PlotChildHandleState::Histogram(hist) => {
            hist.figure == figure && hist.axes_index == axes_index && hist.plot_index == plot_index
        }
        _ => false,
    });
    match state.ok_or(FigureError::InvalidPlotObjectHandle)? {
        PlotChildHandleState::Histogram(hist) => {
            updater(&mut hist.metadata);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn set_animated_line_maximum_num_points_for_property(
    property: &PropertyMutationTarget,
    handle: &AnimatedLineHandleState,
    maximum_num_points: Option<usize>,
) -> Result<(), PropertyMutationError> {
    let figure_clone = {
        let mut reg =
            property.registry(PropertyLocation::Plot(handle.figure, handle.plot_index))?;
        let figure_clone = {
            let state = get_state_mut(&mut reg, handle.figure);
            let plot = state
                .figure
                .get_plot_mut(handle.plot_index)
                .ok_or_else(|| "invalid animated line handle".to_string())?;
            match plot {
                PlotElement::Line(line) => {
                    let (mut x, mut y) = line
                        .host_xy_f64()
                        .map_err(|err| format!("failed to read animated line: {err}"))?
                        .ok_or_else(|| "animated line data is unavailable".to_string())?;
                    trim_oldest_xy(maximum_num_points, &mut x, &mut y);
                    line.update_data(x, y)
                        .map_err(|err| format!("failed to update animated line: {err}"))?;
                }
                PlotElement::Line3(line) => {
                    let (mut x, mut y, mut z) = line
                        .host_xyz_f64()
                        .map_err(|err| format!("failed to read animated line: {err}"))?
                        .ok_or_else(|| "animated line data is unavailable".to_string())?;
                    trim_oldest_xyz(maximum_num_points, &mut x, &mut y, &mut z);
                    line.update_data(x, y, z)
                        .map_err(|err| format!("failed to update animated line: {err}"))?;
                }
                _ => return Err("invalid animated line handle".to_string().into()),
            }
            state.revision = state.revision.wrapping_add(1);
            state.figure.clone()
        };

        reg.for_each_child_mut_where(
            |state| {
                matches!(state, PlotChildHandleState::AnimatedLine(chart)
                if chart.figure == handle.figure && chart.plot_index == handle.plot_index)
            },
            |state| {
                if let PlotChildHandleState::AnimatedLine(chart) = state {
                    chart.maximum_num_points = maximum_num_points;
                }
            },
        );
        figure_clone
    };
    property.notify(handle.figure, &figure_clone, FigureEventKind::Updated)?;
    Ok(())
}
