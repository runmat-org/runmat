//! Quantile plot line construction, presentation and handle publication.
use super::*;
use crate::builtins::plotting::state::{
    register_line_handle, render_active_plot, PlotRenderOptions,
};
use glam::Vec4;
use runmat_plot::plots::line::LineMarkerAppearance;
use runmat_plot::plots::scatter::MarkerStyle;
use runmat_plot::plots::{LinePlot, LineStyle};

pub(super) fn render_series(eval: QqplotEvaluation) -> BuiltinResult<Vec<f64>> {
    let mut plots = Vec::new();
    for (idx, series) in eval.series.iter().enumerate() {
        plots.push(data_plot(series, idx)?);
        if !series.quartile_x.is_empty() {
            plots.push(reference_plot(series, idx, true)?);
        }
        if !series.reference_x.is_empty() {
            plots.push(reference_plot(series, idx, false)?);
        }
    }

    let mut plots_opt = Some(plots);
    let plot_indices_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        NAME,
        PlotRenderOptions {
            title: "Q-Q Plot",
            x_label: match eval.mode {
                PlotMode::Normal => "Standard Normal Quantiles",
                PlotMode::TwoSample => "Quantiles of X",
            },
            y_label: match eval.mode {
                PlotMode::Normal => "Quantiles of Input Sample",
                PlotMode::TwoSample => "Quantiles of Y",
            },
            ..Default::default()
        },
        move |figure, axes_index| {
            let plots = plots_opt
                .take()
                .expect("qqplot series consumed exactly once");
            for plot in plots {
                let plot_index = figure.add_line_plot_on_axes(plot, axes_index);
                plot_indices_slot
                    .borrow_mut()
                    .push((axes_index, plot_index));
            }
            Ok(())
        },
    );

    let handles = plot_indices_out
        .borrow()
        .iter()
        .map(|(axes_index, plot_index)| {
            register_line_handle(figure_handle, *axes_index, *plot_index)
        })
        .collect::<Vec<_>>();
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handles);
        }
        return Err(crate::builtins::plotting::state::with_render_cause(
            internal(err.message().to_string()),
            err,
        ));
    }
    Ok(handles)
}

fn data_plot(series: &Series, idx: usize) -> BuiltinResult<LinePlot> {
    let color = palette(idx);
    let mut plot = LinePlot::new(series.theoretical.clone(), series.observed.clone())
        .map_err(|err| internal(format!("qqplot: {err}")))?
        .with_style(color, 1.0, LineStyle::None)
        .with_label(format!("Data {}", idx + 1));
    plot.set_marker(Some(LineMarkerAppearance {
        kind: MarkerStyle::Plus,
        size: 7.0,
        edge_color: color,
        face_color: color,
        filled: false,
    }));
    Ok(plot)
}

fn reference_plot(series: &Series, idx: usize, quartile: bool) -> BuiltinResult<LinePlot> {
    let (x, y, label, style) = if quartile {
        (
            series.quartile_x.clone(),
            series.quartile_y.clone(),
            format!("Quartile {}", idx + 1),
            LineStyle::Solid,
        )
    } else {
        (
            series.reference_x.clone(),
            series.reference_y.clone(),
            format!("Reference {}", idx + 1),
            LineStyle::Dashed,
        )
    };
    LinePlot::new(x, y)
        .map_err(|err| internal(format!("qqplot: {err}")))
        .map(|plot| plot.with_style(palette(idx), 1.0, style).with_label(label))
}

fn palette(idx: usize) -> Vec4 {
    const COLORS: [Vec4; 6] = [
        Vec4::new(0.000, 0.447, 0.741, 1.0),
        Vec4::new(0.850, 0.325, 0.098, 1.0),
        Vec4::new(0.929, 0.694, 0.125, 1.0),
        Vec4::new(0.494, 0.184, 0.556, 1.0),
        Vec4::new(0.466, 0.674, 0.188, 1.0),
        Vec4::new(0.301, 0.745, 0.933, 1.0),
    ];
    COLORS[idx % COLORS.len()]
}
