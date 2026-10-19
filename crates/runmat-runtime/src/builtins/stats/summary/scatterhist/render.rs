//! Scatterhist render domain behavior.
use super::*;

pub(super) fn render_scatterhist(
    series: &[GroupSeries],
    options: &ScatterhistOptions,
) -> BuiltinResult<[f64; 3]> {
    let x_samples = series
        .iter()
        .flat_map(|s| s.marginal_x.iter().copied())
        .collect::<Vec<_>>();
    let y_samples = series
        .iter()
        .flat_map(|s| s.marginal_y.iter().copied())
        .collect::<Vec<_>>();
    let x_limits = combined_limits(x_samples.iter().copied())?;
    let y_limits = combined_limits(y_samples.iter().copied())?;
    let bins = options.bins.unwrap_or([
        auto_bin_count(&x_samples, x_limits),
        auto_bin_count(&y_samples, y_limits),
    ]);
    let x_edges = bin_edges(x_limits, bins[0])?;
    let y_edges = bin_edges(y_limits, bins[1])?;
    let layout = axes_for_layout(options.location, options.direction);
    let x_marginals = prepare_marginal_plots(
        series,
        options,
        &x_edges,
        MarginalAxis::X,
        layout.x_marginal_scale,
    )?;
    let y_marginals = prepare_marginal_plots(
        series,
        options,
        &y_edges,
        MarginalAxis::Y,
        layout.y_marginal_scale,
    )?;

    let figure = options
        .parent
        .unwrap_or_else(|| current_axes_state().handle);
    select_axes_for_figure(figure, 0).map_err(|err| internal(format!("scatterhist: {err}")))?;

    for axes in [layout.empty, layout.xhist, layout.scatter, layout.yhist] {
        configure_subplot(2, 2, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
        select_axes_for_figure(figure, axes)
            .map_err(|err| internal(format!("scatterhist: {err}")))?;
        handle_plot_render_result(render_active_plot(
            NAME,
            PlotRenderOptions::default(),
            |_figure, _axes| Ok(()),
        ))?;
    }

    render_scatter_axes(figure, layout.scatter, series)?;
    render_x_marginal(figure, layout.xhist, x_marginals)?;
    render_y_marginal(figure, layout.yhist, y_marginals)?;

    if options
        .legend
        .unwrap_or(options.group.is_some() && options.plot_group)
    {
        set_legend_for_axes(figure, layout.scatter, true, None, None)
            .map_err(|err| internal(format!("scatterhist: {err}")))?;
    }

    Ok([
        encode_axes_handle(figure, layout.scatter),
        encode_axes_handle(figure, layout.xhist),
        encode_axes_handle(figure, layout.yhist),
    ])
}

pub(super) fn axes_for_layout(location: Location, direction: Direction) -> LayoutAxes {
    let (scatter, xhist, yhist, empty, xhist_is_south, yhist_is_west) = match location {
        // Location names the marginal histogram corner. For example, SouthWest
        // places the x histogram below and y histogram left of the scatter axes.
        Location::NorthEast => (2, 0, 3, 1, false, false),
        Location::SouthEast => (0, 2, 1, 3, true, false),
        Location::SouthWest => (1, 3, 0, 2, true, true),
        Location::NorthWest => (3, 1, 2, 0, false, true),
    };
    let x_marginal_scale = directional_scale(direction, xhist_is_south);
    let y_marginal_scale = directional_scale(direction, yhist_is_west);
    LayoutAxes {
        scatter,
        xhist,
        yhist,
        empty,
        x_marginal_scale,
        y_marginal_scale,
    }
}

pub(super) fn directional_scale(direction: Direction, positive_points_toward_scatter: bool) -> f64 {
    match (direction, positive_points_toward_scatter) {
        (Direction::In, true) | (Direction::Out, false) => 1.0,
        (Direction::In, false) | (Direction::Out, true) => -1.0,
    }
}

pub(super) fn render_scatter_axes(
    figure: crate::builtins::plotting::FigureHandle,
    axes: usize,
    series: &[GroupSeries],
) -> BuiltinResult<()> {
    select_axes_for_figure(figure, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
    handle_plot_render_result(render_active_plot(
        NAME,
        PlotRenderOptions::default(),
        |figure, axes| {
            for item in series {
                if item.scatter_x.is_empty() {
                    continue;
                }
                let mut plot = ScatterPlot::new(item.scatter_x.clone(), item.scatter_y.clone())
                    .map_err(|err| internal(format!("scatterhist: {err}")))?;
                plot.color = item.style.color;
                plot.edge_color = item.style.color;
                plot.marker_size = item.style.marker_size;
                plot.marker_style = item.style.marker;
                plot.label = Some(item.label.clone());
                figure.add_scatter_plot_on_axes(plot, axes);
            }
            Ok(())
        },
    ))
}

pub(super) fn render_x_marginal(
    figure: crate::builtins::plotting::FigureHandle,
    axes: usize,
    prepared: Vec<PreparedMarginalPlot>,
) -> BuiltinResult<()> {
    select_axes_for_figure(figure, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
    handle_plot_render_result(render_active_plot(
        NAME,
        PlotRenderOptions::default(),
        move |figure, axes| {
            for plot in prepared {
                match plot {
                    PreparedMarginalPlot::Line(line) => figure.add_line_plot_on_axes(line, axes),
                    PreparedMarginalPlot::Bar(bar) => figure.add_bar_chart_on_axes(bar, axes),
                };
            }
            Ok(())
        },
    ))
}

pub(super) fn render_y_marginal(
    figure: crate::builtins::plotting::FigureHandle,
    axes: usize,
    prepared: Vec<PreparedMarginalPlot>,
) -> BuiltinResult<()> {
    select_axes_for_figure(figure, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
    handle_plot_render_result(render_active_plot(
        NAME,
        PlotRenderOptions::default(),
        move |figure, axes| {
            for plot in prepared {
                match plot {
                    PreparedMarginalPlot::Line(line) => figure.add_line_plot_on_axes(line, axes),
                    PreparedMarginalPlot::Bar(bar) => figure.add_bar_chart_on_axes(bar, axes),
                };
            }
            Ok(())
        },
    ))
}

pub(super) fn resolved_hist_style(
    options: &ScatterhistOptions,
    series: &[GroupSeries],
) -> HistStyle {
    options.hist_style.unwrap_or_else(|| {
        if options.group.is_some() && options.plot_group && series.len() > 1 {
            HistStyle::Stairs
        } else {
            HistStyle::Bar
        }
    })
}

pub(super) fn handle_plot_render_result(result: BuiltinResult<String>) -> BuiltinResult<()> {
    match result {
        Ok(_) => Ok(()),
        Err(err) => {
            if crate::builtins::plotting::state::presentation_unavailable(&err) {
                Ok(())
            } else {
                Err(crate::builtins::plotting::state::with_render_cause(
                    internal(err.message().to_string()),
                    err,
                ))
            }
        }
    }
}
