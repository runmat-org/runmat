//! Ultra-simplified plotting functions that work

use crate::PlotOptions;
use plotters::prelude::*;

/// Create a line plot - PNG version
pub fn line_plot_png(
    xs: &[f64],
    ys: &[f64],
    path: &str,
    _options: &PlotOptions,
) -> Result<(), String> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|_| "Failed to fill background")?;

    let x_min = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|_| "Failed to build chart")?;

    // Skip grid and labels to avoid font issues for now

    if _options.line_style != crate::LineStyle::None {
        chart
            .draw_series(LineSeries::new(
                xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)),
                BLUE.stroke_width(2),
            ))
            .map_err(|_| "Failed to draw line series")?;
    }

    root.present().map_err(|_| "Failed to present chart")?;
    Ok(())
}

/// Create a line plot - SVG version
pub fn line_plot_svg(
    xs: &[f64],
    ys: &[f64],
    path: &str,
    _options: &PlotOptions,
) -> Result<(), String> {
    let root = SVGBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|_| "Failed to fill background")?;

    let x_min = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|_| "Failed to build chart")?;

    // Skip grid and labels to avoid font issues for now

    if _options.line_style != crate::LineStyle::None {
        chart
            .draw_series(LineSeries::new(
                xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)),
                BLUE.stroke_width(2),
            ))
            .map_err(|_| "Failed to draw line series")?;
    }

    root.present().map_err(|_| "Failed to present chart")?;
    Ok(())
}

/// Create a scatter plot - PNG version
pub fn scatter_plot_png(
    xs: &[f64],
    ys: &[f64],
    path: &str,
    _options: &PlotOptions,
) -> Result<(), String> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|_| "Failed to fill background")?;

    let x_min = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|_| "Failed to build chart")?;

    // Skip grid and labels to avoid font issues for now

    chart
        .draw_series(
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| Circle::new((x, y), 3, RED.filled())),
        )
        .map_err(|_| "Failed to draw scatter points")?;

    root.present().map_err(|_| "Failed to present chart")?;
    Ok(())
}

/// Create a scatter plot - SVG version
pub fn scatter_plot_svg(
    xs: &[f64],
    ys: &[f64],
    path: &str,
    _options: &PlotOptions,
) -> Result<(), String> {
    let root = SVGBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|_| "Failed to fill background")?;

    let x_min = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(|_| "Failed to build chart")?;

    // Skip grid and labels to avoid font issues for now

    chart
        .draw_series(
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| Circle::new((x, y), 3, RED.filled())),
        )
        .map_err(|_| "Failed to draw scatter points")?;

    root.present().map_err(|_| "Failed to present chart")?;
    Ok(())
}

/// Create a bar chart - PNG version
pub fn bar_chart_png(
    labels: &[String],
    values: &[f64],
    path: &str,
    _options: &PlotOptions,
) -> Result<(), String> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|_| "Failed to fill background")?;

    let y_max = values.iter().fold(0.0f64, |a, &b| a.max(b)) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..(labels.len() as f64), 0f64..y_max)
        .map_err(|_| "Failed to build chart")?;

    // Skip grid and labels to avoid font issues for now

    chart
        .draw_series(
            values.iter().enumerate().map(|(i, &v)| {
                Rectangle::new([(i as f64, 0.0), (i as f64 + 0.8, v)], GREEN.filled())
            }),
        )
        .map_err(|_| "Failed to draw bars")?;

    root.present().map_err(|_| "Failed to present chart")?;
    Ok(())
}

/// Create a histogram - PNG version
pub fn histogram_png(
    values: &[f64],
    bins: usize,
    path: &str,
    _options: &PlotOptions,
) -> Result<(), String> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|_| "Failed to fill background")?;

    // Calculate histogram
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let bin_width = (max_val - min_val) / bins as f64;

    let mut histogram = vec![0; bins];
    for &value in values {
        let bin_index = ((value - min_val) / bin_width).floor() as usize;
        let bin_index = bin_index.min(bins - 1);
        histogram[bin_index] += 1;
    }

    let max_count = histogram.iter().max().unwrap_or(&0);

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_val..max_val, 0f64..(*max_count as f64 * 1.1))
        .map_err(|_| "Failed to build chart")?;

    // Skip grid and labels to avoid font issues for now

    chart
        .draw_series(histogram.iter().enumerate().map(|(i, &count)| {
            let x_start = min_val + i as f64 * bin_width;
            let x_end = x_start + bin_width;
            Rectangle::new([(x_start, 0.0), (x_end, count as f64)], MAGENTA.filled())
        }))
        .map_err(|_| "Failed to draw histogram")?;

    root.present().map_err(|_| "Failed to present chart")?;
    Ok(())
}
