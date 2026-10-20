//! Surface presentation and chart-handle publication.
use super::*;
use crate::builtins::plotting::state::{
    register_binscatter_handle, render_active_plot, PlotRenderOptions,
};
use runmat_plot::plots::{ColorMap, ShadingMode};

pub(super) fn build_surface_from_counts(
    counts: &[f64],
    x_edges: &[f64],
    y_edges: &[f64],
    num_bins: [usize; 2],
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<&str>,
    color_limits: Option<(f64, f64)>,
) -> BuiltinResult<SurfacePlot> {
    let mut z_grid = vec![vec![0.0; num_bins[1]]; num_bins[0]];
    for (ix, row) in z_grid.iter_mut().enumerate().take(num_bins[0]) {
        for (iy, cell) in row.iter_mut().enumerate().take(num_bins[1]) {
            let mut value = counts[ix + iy * num_bins[0]];
            if !show_empty_bins && value == 0.0 {
                value = f64::NAN;
            }
            *cell = value;
        }
    }
    let mut surface = SurfacePlot::new(bin_centers(x_edges), bin_centers(y_edges), z_grid)
        .map_err(|err| internal(format!("binscatter: {err}")))?
        .with_flatten_z(true)
        .with_image_mode(true)
        .with_colormap(ColorMap::Parula)
        .with_shading(ShadingMode::None)
        .with_alpha(face_alpha as f32);
    if let Some(display_name) = display_name {
        surface = surface.with_label(display_name.to_string());
    }
    if color_limits.is_some() {
        surface = surface.with_color_limits(color_limits);
    }
    Ok(surface)
}

fn bin_centers(edges: &[f64]) -> Vec<f64> {
    edges
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect()
}

pub(super) fn render_binscatter(
    inputs: ParsedInputs,
    options: BinscatterOptions,
    num_bins: [usize; 2],
    chart: BinscatterChart,
) -> BuiltinResult<f64> {
    let mut surface = Some(chart.surface);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result =
        render_active_plot(NAME, PlotRenderOptions::default(), move |figure, axes| {
            let plot_index = figure.add_surface_plot_on_axes(
                surface.take().expect("binscatter plot consumed once"),
                axes,
            );
            figure.set_axes_colorbar_enabled(axes, true);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = register_binscatter_handle(
        figure_handle,
        axes,
        plot_index,
        chart.values,
        chart.x_bin_edges,
        chart.y_bin_edges,
        inputs.x_data,
        inputs.y_data,
        num_bins,
        options.num_bins.is_none(),
        options.x_limits,
        options.y_limits,
        options.show_empty_bins,
        options.face_alpha,
        options.display_name,
    );
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(crate::builtins::plotting::state::with_render_cause(
            internal(err.message().to_string()),
            err,
        ));
    }
    Ok(handle)
}
