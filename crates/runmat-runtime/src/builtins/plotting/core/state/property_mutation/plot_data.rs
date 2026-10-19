//! Existing property algorithms at the selected registry transaction boundary.

use super::super::*;

pub(crate) fn update_errorbar_plot_for_property(
    property: &PropertyMutationTarget,

    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::ErrorBar),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure_handle, plot_index))?;
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::ErrorBar(errorbar) => {
            updater(errorbar);
            state.revision = state.revision.wrapping_add(1);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_histogram_plot_data_for_property(
    property: &PropertyMutationTarget,

    figure_handle: FigureHandle,
    plot_index: usize,
    labels: Vec<String>,
    values: Vec<f64>,
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure_handle, plot_index))?;
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Bar(bar) => {
            bar.set_data(labels, values)
                .map_err(|_| FigureError::InvalidPlotObjectHandle)?;
            state.revision = state.revision.wrapping_add(1);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_stem_plot_for_property(
    property: &PropertyMutationTarget,

    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::StemPlot),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure_handle, plot_index))?;
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Stem(stem) => {
            updater(stem);
            state.revision = state.revision.wrapping_add(1);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_quiver_plot_for_property(
    property: &PropertyMutationTarget,

    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::QuiverPlot),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure_handle, plot_index))?;
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Quiver(quiver) => {
            updater(quiver);
            state.revision = state.revision.wrapping_add(1);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_image_plot_for_property(
    property: &PropertyMutationTarget,

    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::SurfacePlot),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure_handle, plot_index))?;
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Surface(surface) if surface.image_mode => {
            updater(surface);
            state.revision = state.revision.wrapping_add(1);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_area_plot_for_property(
    property: &PropertyMutationTarget,

    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::AreaPlot),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure_handle, plot_index))?;
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    match plot {
        runmat_plot::plots::figure::PlotElement::Area(area) => {
            updater(area);
            state.revision = state.revision.wrapping_add(1);
            Ok(())
        }
        _ => Err(FigureError::InvalidPlotObjectHandle),
    }
}

pub(crate) fn update_plot_element_for_property(
    property: &PropertyMutationTarget,

    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::figure::PlotElement),
) -> Result<(), FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure_handle, plot_index))?;
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    updater(plot);
    state.revision = state.revision.wrapping_add(1);
    Ok(())
}

pub(crate) fn try_update_plot_element_for_property<E>(
    property: &PropertyMutationTarget,
    figure_handle: FigureHandle,
    plot_index: usize,
    updater: impl FnOnce(&mut runmat_plot::plots::figure::PlotElement) -> Result<(), E>,
) -> Result<Result<(), E>, FigureError> {
    let mut reg = property.registry(PropertyLocation::Plot(figure_handle, plot_index))?;
    let state = get_state_mut(&mut reg, figure_handle);
    let plot = state
        .figure
        .get_plot_mut(plot_index)
        .ok_or(FigureError::InvalidPlotObjectHandle)?;
    let result = updater(plot);
    if result.is_ok() {
        state.revision = state.revision.wrapping_add(1);
    }
    Ok(result)
}
