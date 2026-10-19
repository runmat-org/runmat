//! Prepare numerical marginal geometry before mutating the figure.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum MarginalAxis {
    X,
    Y,
}

pub(super) enum PreparedMarginalPlot {
    Line(LinePlot),
    Bar(BarChart),
}

pub(super) fn prepare_marginal_plots(
    series: &[GroupSeries],
    options: &ScatterhistOptions,
    edges: &[f64],
    axis: MarginalAxis,
    scale: f64,
) -> BuiltinResult<Vec<PreparedMarginalPlot>> {
    let mut prepared = Vec::new();
    let capacity = series
        .len()
        .checked_mul(2)
        .ok_or_else(|| invalid("scatterhist: marginal allocation failed"))?;
    prepared
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("scatterhist: marginal allocation failed"))?;
    let hist_style = resolved_hist_style(options, series);
    for (index, item) in series.iter().enumerate() {
        let (samples, bandwidth) = match axis {
            MarginalAxis::X => (&item.marginal_x, item.style.bandwidth.map(|b| b[0])),
            MarginalAxis::Y => (&item.marginal_y, item.style.bandwidth.map(|b| b[1])),
        };
        if matches!(options.kernel, KernelMode::On | KernelMode::Overlay) {
            let (centers, density) = kde_line(samples, edges, bandwidth)?;
            let density = scale_values(density, scale);
            if density.iter().any(|value| !value.is_finite()) {
                return Err(invalid("scatterhist: marginal density is not finite"));
            }
            let (x, y) = match axis {
                MarginalAxis::X => (centers, density),
                MarginalAxis::Y => (density, centers),
            };
            let line = LinePlot::new(x, y)
                .map_err(|err| internal(format!("scatterhist: {err}")))?
                .with_style(
                    item.style.color,
                    item.style.line_width,
                    item.style.line_style,
                )
                .with_label(item.label.clone());
            prepared.push(PreparedMarginalPlot::Line(line));
        }
        if matches!(options.kernel, KernelMode::Off | KernelMode::Overlay) {
            let counts = scale_values(histogram_counts(samples, edges), scale);
            if counts.iter().any(|value| !value.is_finite()) {
                return Err(invalid("scatterhist: marginal count is not finite"));
            }
            match hist_style {
                HistStyle::Bar => {
                    let mut bar = bar_from_counts(edges, counts, item.style.color, &item.label)?;
                    if matches!(axis, MarginalAxis::Y) {
                        bar = bar.with_orientation(Orientation::Horizontal);
                    }
                    prepared.push(PreparedMarginalPlot::Bar(
                        bar.with_group(index, series.len()),
                    ));
                }
                HistStyle::Stairs => {
                    let (x, y) =
                        stairs_from_counts(edges, &counts, matches!(axis, MarginalAxis::Y));
                    let line = LinePlot::new(x, y)
                        .map_err(|err| internal(format!("scatterhist: {err}")))?
                        .with_style(
                            item.style.color,
                            item.style.line_width,
                            item.style.line_style,
                        )
                        .with_label(item.label.clone());
                    prepared.push(PreparedMarginalPlot::Line(line));
                }
            }
        }
    }
    Ok(prepared)
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
