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

    let figure = options
        .parent
        .unwrap_or_else(|| current_axes_state().handle);
    select_axes_for_figure(figure, 0).map_err(|err| internal(format!("scatterhist: {err}")))?;
    let layout = axes_for_layout(options.location, options.direction);

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
    render_x_marginal(
        figure,
        layout.xhist,
        series,
        options,
        &x_edges,
        layout.x_marginal_scale,
    )?;
    render_y_marginal(
        figure,
        layout.yhist,
        series,
        options,
        &y_edges,
        layout.y_marginal_scale,
    )?;

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
        PlotRenderOptions {
            title: "Scatterhist",
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
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
    series: &[GroupSeries],
    options: &ScatterhistOptions,
    edges: &[f64],
    marginal_scale: f64,
) -> BuiltinResult<()> {
    select_axes_for_figure(figure, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
    let hist_style = resolved_hist_style(options, series);
    handle_plot_render_result(render_active_plot(
        NAME,
        PlotRenderOptions {
            title: "X Marginal",
            x_label: "X",
            y_label: "Count",
            ..Default::default()
        },
        |figure, axes| {
            for (idx, item) in series.iter().enumerate() {
                if matches!(options.kernel, KernelMode::On | KernelMode::Overlay) {
                    let (xs, ys) =
                        kde_line(&item.marginal_x, edges, item.style.bandwidth.map(|b| b[0]));
                    let ys = scale_values(ys, marginal_scale);
                    let line = LinePlot::new(xs, ys)
                        .map_err(|err| internal(format!("scatterhist: {err}")))?
                        .with_style(
                            item.style.color,
                            item.style.line_width,
                            item.style.line_style,
                        )
                        .with_label(item.label.clone());
                    figure.add_line_plot_on_axes(line, axes);
                }
                if matches!(options.kernel, KernelMode::Off | KernelMode::Overlay) {
                    let counts =
                        scale_values(histogram_counts(&item.marginal_x, edges), marginal_scale);
                    match hist_style {
                        HistStyle::Bar => {
                            let mut bar =
                                bar_from_counts(edges, counts, item.style.color, &item.label)?;
                            bar = bar.with_group(idx, series.len());
                            figure.add_bar_chart_on_axes(bar, axes);
                        }
                        HistStyle::Stairs => {
                            let (xs, ys) = stairs_from_counts(edges, &counts, false);
                            let line = LinePlot::new(xs, ys)
                                .map_err(|err| internal(format!("scatterhist: {err}")))?
                                .with_style(
                                    item.style.color,
                                    item.style.line_width,
                                    item.style.line_style,
                                )
                                .with_label(item.label.clone());
                            figure.add_line_plot_on_axes(line, axes);
                        }
                    }
                }
            }
            Ok(())
        },
    ))
}

pub(super) fn render_y_marginal(
    figure: crate::builtins::plotting::FigureHandle,
    axes: usize,
    series: &[GroupSeries],
    options: &ScatterhistOptions,
    edges: &[f64],
    marginal_scale: f64,
) -> BuiltinResult<()> {
    select_axes_for_figure(figure, axes).map_err(|err| internal(format!("scatterhist: {err}")))?;
    let hist_style = resolved_hist_style(options, series);
    handle_plot_render_result(render_active_plot(
        NAME,
        PlotRenderOptions {
            title: "Y Marginal",
            x_label: "Count",
            y_label: "Y",
            ..Default::default()
        },
        |figure, axes| {
            for (idx, item) in series.iter().enumerate() {
                if matches!(options.kernel, KernelMode::On | KernelMode::Overlay) {
                    let (density, ys) =
                        kde_line(&item.marginal_y, edges, item.style.bandwidth.map(|b| b[1]));
                    let density = scale_values(density, marginal_scale);
                    let line = LinePlot::new(density, ys)
                        .map_err(|err| internal(format!("scatterhist: {err}")))?
                        .with_style(
                            item.style.color,
                            item.style.line_width,
                            item.style.line_style,
                        )
                        .with_label(item.label.clone());
                    figure.add_line_plot_on_axes(line, axes);
                }
                if matches!(options.kernel, KernelMode::Off | KernelMode::Overlay) {
                    let counts =
                        scale_values(histogram_counts(&item.marginal_y, edges), marginal_scale);
                    match hist_style {
                        HistStyle::Bar => {
                            let mut bar =
                                bar_from_counts(edges, counts, item.style.color, &item.label)?
                                    .with_orientation(Orientation::Horizontal);
                            bar = bar.with_group(idx, series.len());
                            figure.add_bar_chart_on_axes(bar, axes);
                        }
                        HistStyle::Stairs => {
                            let (xs, ys) = stairs_from_counts(edges, &counts, true);
                            let line = LinePlot::new(xs, ys)
                                .map_err(|err| internal(format!("scatterhist: {err}")))?
                                .with_style(
                                    item.style.color,
                                    item.style.line_width,
                                    item.style.line_style,
                                )
                                .with_label(item.label.clone());
                            figure.add_line_plot_on_axes(line, axes);
                        }
                    }
                }
            }
            Ok(())
        },
    ))
}

pub(super) fn bar_from_counts(
    edges: &[f64],
    counts: Vec<f64>,
    color: Vec4,
    label: &str,
) -> BuiltinResult<BarChart> {
    let labels = edges
        .windows(2)
        .map(|pair| format!("[{:.3}, {:.3})", pair[0], pair[1]))
        .collect::<Vec<_>>();
    let mut chart = BarChart::new(labels, counts)
        .map_err(|err| internal(format!("scatterhist: {err}")))?
        .with_style(color, 0.95)
        .with_label(label.to_string());
    chart.set_histogram_bin_edges(edges.to_vec());
    Ok(chart)
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
