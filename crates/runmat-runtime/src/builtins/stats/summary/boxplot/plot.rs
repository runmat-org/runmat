//! Graphics plans and publication for boxplot output.
use super::*;

pub(super) fn render_boxplot(
    boxes: &[BoxStats],
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<f64>> {
    let plans = build_plot_plans(boxes, options)?;
    let labels = boxes
        .iter()
        .map(|stats| stats.label.clone())
        .collect::<Vec<_>>();
    let limits = axis_limits(boxes, options)?;
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let mut plans = Some(plans);
    let handles_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let handles_slot = std::rc::Rc::clone(&handles_out);
    let opts = PlotRenderOptions {
        title: "",
        x_label: "",
        y_label: "",
        grid: true,
        axis_equal: false,
    };
    let render_result = render_active_plot(NAME, opts, move |figure, axes| {
        match options.orientation {
            Orientation::Vertical => {
                figure.set_axes_tick_labels(axes, Some(labels.clone()), None);
            }
            Orientation::Horizontal => {
                figure.set_axes_tick_labels(axes, None, Some(labels.clone()));
            }
        }
        figure.set_axes_limits(axes, limits.0, limits.1);
        let mut recorded = Vec::new();
        for plan in plans.take().expect("boxplot plans consumed once") {
            match plan {
                PlotPlan::Line(plot) => {
                    let plot_index = figure.add_line_plot_on_axes(plot, axes);
                    recorded.push((HandleKind::Line, axes, plot_index));
                }
                PlotPlan::Scatter(plot) => {
                    let plot_index = figure.add_scatter_plot_on_axes(plot, axes);
                    recorded.push((HandleKind::Scatter, axes, plot_index));
                }
            }
        }
        *handles_slot.borrow_mut() = recorded;
        Ok(())
    });

    let handles = handles_out
        .borrow()
        .iter()
        .map(|(kind, axes, plot_index)| match kind {
            HandleKind::Line => crate::builtins::plotting::state::register_line_handle(
                figure_handle,
                *axes,
                *plot_index,
            ),
            HandleKind::Scatter => crate::builtins::plotting::state::register_scatter_handle(
                figure_handle,
                *axes,
                *plot_index,
            ),
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

pub(super) type AxisLimits = (Option<(f64, f64)>, Option<(f64, f64)>);

pub(super) fn axis_limits(
    boxes: &[BoxStats],
    options: &BoxplotOptions,
) -> BuiltinResult<AxisLimits> {
    let min_pos = boxes
        .iter()
        .map(|stats| stats.position - stats.width)
        .fold(f64::INFINITY, f64::min);
    let max_pos = boxes
        .iter()
        .map(|stats| stats.position + stats.width)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut min_data = f64::INFINITY;
    let mut max_data = f64::NEG_INFINITY;
    for stats in boxes {
        min_data = min_data.min(stats.lower_whisker).min(stats.q1);
        max_data = max_data.max(stats.upper_whisker).max(stats.q3);
        for (idx, value) in stats.outliers.iter().enumerate() {
            let value = display_data_value(*value, idx, stats.outliers.len(), options);
            min_data = min_data.min(value);
            max_data = max_data.max(value);
        }
    }
    if let Some((lo, hi)) = options.data_lim {
        if lo.is_finite() {
            min_data = lo.min(min_data);
        }
        if hi.is_finite() {
            max_data = hi.max(max_data);
        }
    }
    if !min_data.is_finite() || !max_data.is_finite() {
        return Err(invalid("boxplot: plotted data require finite axis limits"));
    }
    if min_data == max_data {
        let lower = min_data - 0.5;
        let upper = max_data + 0.5;
        let next_lower = min_data.next_down();
        let next_upper = max_data.next_up();
        if lower.is_finite() && lower < min_data {
            min_data = lower;
        } else if next_lower.is_finite() {
            min_data = next_lower;
        }
        if upper.is_finite() && upper > max_data {
            max_data = upper;
        } else if next_upper.is_finite() {
            max_data = next_upper;
        }
    } else {
        let span = max_data - min_data;
        if span.is_finite() {
            let pad = span * 0.08;
            let lower = min_data - pad;
            let upper = max_data + pad;
            if lower.is_finite() {
                min_data = lower;
            }
            if upper.is_finite() {
                max_data = upper;
            }
        }
    }
    if !min_pos.is_finite()
        || !max_pos.is_finite()
        || min_pos >= max_pos
        || !min_data.is_finite()
        || !max_data.is_finite()
        || min_data >= max_data
    {
        return Err(invalid(
            "boxplot: plotted axis limits are not representable",
        ));
    }
    let category = Some((min_pos.min(0.5), max_pos.max(boxes.len() as f64 + 0.5)));
    let data = Some((min_data, max_data));
    Ok(match options.orientation {
        Orientation::Vertical => (category, data),
        Orientation::Horizontal => (data, category),
    })
}

fn build_plot_plans(boxes: &[BoxStats], options: &BoxplotOptions) -> BuiltinResult<Vec<PlotPlan>> {
    let mut plans = Vec::new();
    for stats in boxes {
        add_box_lines(&mut plans, stats, options.orientation)?;
        if !stats.outliers.is_empty() {
            if let Some(marker) = options.outlier_marker {
                let (x, y) = outlier_points(stats, options);
                let color = options.outlier_color.unwrap_or(stats.color);
                let mut scatter = ScatterPlot::new(x, y)
                    .map_err(|err| internal(format!("boxplot: {err}")))?
                    .with_style(color, options.outlier_size, marker)
                    .with_label(format!("{} outliers", stats.label));
                scatter.set_edge_color(color);
                scatter.set_filled(false);
                plans.push(PlotPlan::Scatter(scatter));
            }
        }
    }
    Ok(plans)
}

fn add_box_lines(
    plans: &mut Vec<PlotPlan>,
    stats: &BoxStats,
    orientation: Orientation,
) -> BuiltinResult<()> {
    let pos = stats.position;
    let half = stats.width * 0.5;
    let cap_half = stats.width * 0.25;
    let left = pos - half;
    let right = pos + half;
    let cap_left = pos - cap_half;
    let cap_right = pos + cap_half;
    let segments = [
        (
            vec![left, right, right, left, left],
            vec![stats.q1, stats.q1, stats.q3, stats.q3, stats.q1],
            "box",
        ),
        (
            vec![left, right],
            vec![stats.median, stats.median],
            "median",
        ),
        (
            vec![pos, pos],
            vec![stats.q1, stats.lower_whisker],
            "lower whisker",
        ),
        (
            vec![pos, pos],
            vec![stats.q3, stats.upper_whisker],
            "upper whisker",
        ),
        (
            vec![cap_left, cap_right],
            vec![stats.lower_whisker, stats.lower_whisker],
            "lower cap",
        ),
        (
            vec![cap_left, cap_right],
            vec![stats.upper_whisker, stats.upper_whisker],
            "upper cap",
        ),
    ];
    for (x, y, suffix) in segments {
        let (x, y) = orient_points(x, y, orientation);
        let line = LinePlot::new(x, y)
            .map_err(|err| internal(format!("boxplot: {err}")))?
            .with_style(stats.color, 1.5, LineStyle::Solid)
            .with_label(format!("{} {}", stats.label, suffix));
        plans.push(PlotPlan::Line(line));
    }
    Ok(())
}

fn outlier_points(stats: &BoxStats, options: &BoxplotOptions) -> (Vec<f64>, Vec<f64>) {
    let mut category = Vec::with_capacity(stats.outliers.len());
    for idx in 0..stats.outliers.len() {
        let offset = if options.jitter == 0.0 || stats.outliers.len() == 1 {
            0.0
        } else {
            ((idx % 5) as f64 - 2.0) * stats.width * options.jitter * 0.07
        };
        category.push(stats.position + offset);
    }
    let values = stats
        .outliers
        .iter()
        .enumerate()
        .map(|(idx, value)| display_data_value(*value, idx, stats.outliers.len(), options))
        .collect();
    orient_points(category, values, options.orientation)
}

fn orient_points(x: Vec<f64>, y: Vec<f64>, orientation: Orientation) -> (Vec<f64>, Vec<f64>) {
    match orientation {
        Orientation::Vertical => (x, y),
        Orientation::Horizontal => (y, x),
    }
}

pub(super) fn handles_value(handles: Vec<f64>) -> BuiltinResult<Value> {
    let len = handles.len();
    Tensor::new(handles, vec![1, len])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("boxplot: {err}")))
}

pub(super) fn positions_for(count: usize, options: &BoxplotOptions) -> BuiltinResult<Vec<f64>> {
    if let Some(positions) = &options.positions {
        if positions.len() != count {
            return Err(invalid(
                "boxplot: Positions length must match the number of plotted boxes",
            ));
        }
        return Ok(positions.clone());
    }
    Ok((1..=count).map(|idx| idx as f64).collect())
}

pub(super) fn widths_for(count: usize, options: &BoxplotOptions) -> Vec<f64> {
    (0..count)
        .map(|idx| options.widths[idx % options.widths.len()])
        .collect()
}

pub(super) fn color_for(idx: usize, colors: &[Vec4]) -> Vec4 {
    colors
        .get(idx % colors.len().max(1))
        .copied()
        .unwrap_or(Vec4::new(0.0, 0.4470, 0.7410, 1.0))
}
