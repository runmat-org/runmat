//! Graphics plans and publication for boxplot output.
use super::*;

#[cfg(test)]
mod tests;

pub(super) struct PreparedBoxplot {
    pub(super) plans: Vec<PlotPlan>,
    pub(super) labels: Vec<String>,
    pub(super) ticks: Vec<f64>,
    limits: AxisLimits,
}

#[cfg(test)]
pub(super) fn prepare_box_layout(
    layout: &factor_layout::BoxLayout,
    options: &BoxplotOptions,
) -> BuiltinResult<PreparedBoxplot> {
    futures::executor::block_on(prepare_box_layout_cooperative(
        layout,
        options,
        &mut super::cooperation::BoxplotCooperation::new(None),
    ))
}

pub(super) async fn prepare_box_layout_cooperative(
    layout: &factor_layout::BoxLayout,
    options: &BoxplotOptions,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<PreparedBoxplot> {
    let mut plans = build_plot_plans_cooperative(&layout.boxes, options, cooperation).await?;
    let mut labels = Vec::new();
    let mut ticks = Vec::new();
    labels
        .try_reserve_exact(layout.slots.len())
        .map_err(|error| {
            allocation_failure("boxplot: tick labels exceed available memory", error)
        })?;
    ticks
        .try_reserve_exact(layout.slots.len())
        .map_err(|error| allocation_failure("boxplot: ticks exceed available memory", error))?;
    for (index, slot) in layout.slots.iter().enumerate() {
        let previous = index
            .checked_sub(1)
            .and_then(|previous| layout.slots.get(previous))
            .map(|previous| previous.key.as_slice());
        labels.push(super::appearance::tick_label(
            &slot.label,
            &slot.key,
            previous,
            options.label_verbosity,
            options.labels.is_some(),
        ));
        ticks.push(slot.position);
        cooperation.completed(1).await?;
    }
    let limits = axis_limits_for_layout_cooperative(layout, options, cooperation).await?;
    add_factor_separators_cooperative(
        &mut plans,
        &layout.separators,
        limits,
        options.orientation,
        cooperation,
    )
    .await?;
    Ok(PreparedBoxplot {
        plans,
        labels,
        ticks,
        limits,
    })
}

#[cfg(test)]
pub(super) fn render_boxplot(
    boxes: &[BoxStats],
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<f64>> {
    let slots = boxes
        .iter()
        .enumerate()
        .map(|(index, stats)| factor_layout::FactorSlot {
            key: stats.group_key.clone(),
            label: stats.label.clone(),
            position: stats.position,
            box_index: Some(index),
        })
        .collect();
    let layout = factor_layout::BoxLayout {
        boxes: boxes.to_vec(),
        slots,
        separators: Vec::new(),
    };
    publish_boxplot(prepare_box_layout(&layout, options)?, options)
}

pub(super) fn publish_boxplot(
    prepared: PreparedBoxplot,
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<f64>> {
    let PreparedBoxplot {
        plans,
        labels,
        ticks,
        limits,
    } = prepared;
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let mut recorded = Vec::with_capacity(plans.len());
    let opts = PlotRenderOptions {
        title: "",
        x_label: "",
        y_label: "",
        grid: true,
        axis_equal: false,
    };
    let render_result = render_active_plot(NAME, opts, |figure, axes| {
        match options.orientation {
            Orientation::Vertical => {
                figure.set_axes_ticks(axes, Some(ticks), None);
                figure.set_axes_tick_labels(axes, Some(labels), None);
                figure.set_axes_tick_label_rotations(
                    axes,
                    Some(options.label_orientation.angle(options.orientation)),
                    None,
                );
            }
            Orientation::Horizontal => {
                figure.set_axes_ticks(axes, None, Some(ticks));
                figure.set_axes_tick_labels(axes, None, Some(labels));
                figure.set_axes_tick_label_rotations(axes, None, Some(0.0));
            }
        }
        figure.set_axes_limits(axes, limits.0, limits.1);
        for plan in plans {
            match plan {
                PlotPlan::Line(plot) => {
                    let plot_index = figure.add_line_plot_on_axes(plot, axes);
                    recorded.push((HandleKind::Line, axes, plot_index));
                }
                PlotPlan::Scatter(plot) => {
                    let plot_index = figure.add_scatter_plot_on_axes(plot, axes);
                    recorded.push((HandleKind::Scatter, axes, plot_index));
                }
                PlotPlan::Patch(plot) => {
                    let plot_index = figure.add_patch_plot_on_axes(plot, axes);
                    recorded.push((HandleKind::Patch, axes, plot_index));
                }
            }
        }
        Ok(())
    });

    let handles = recorded
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
            HandleKind::Patch => crate::builtins::plotting::state::register_patch_handle(
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

async fn axis_limits_for_layout_cooperative(
    layout: &factor_layout::BoxLayout,
    options: &BoxplotOptions,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<AxisLimits> {
    let limits = axis_limits_cooperative(&layout.boxes, options, cooperation).await?;
    let mut min_pos = f64::INFINITY;
    let mut max_pos = f64::NEG_INFINITY;
    for (index, slot) in layout.slots.iter().enumerate() {
        let width = options.widths[index % options.widths.len()];
        min_pos = min_pos.min(slot.position - width);
        max_pos = max_pos.max(slot.position + width);
        cooperation.completed(1).await?;
    }
    if !min_pos.is_finite() || !max_pos.is_finite() || min_pos >= max_pos {
        return Err(invalid("boxplot: factor slot limits are not representable"));
    }
    let category = Some((
        min_pos.min(0.5),
        max_pos.max(layout.slots.len() as f64 + 0.5),
    ));
    Ok(match options.orientation {
        Orientation::Vertical => (category, limits.1),
        Orientation::Horizontal => (limits.0, category),
    })
}

async fn add_factor_separators_cooperative(
    plans: &mut Vec<PlotPlan>,
    positions: &[f64],
    limits: AxisLimits,
    orientation: Orientation,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<()> {
    let data = match orientation {
        Orientation::Vertical => limits.1,
        Orientation::Horizontal => limits.0,
    }
    .ok_or_else(|| invalid("boxplot: data limits are unavailable"))?;
    for &position in positions {
        if !(position as f32).is_finite()
            || !(data.0 as f32).is_finite()
            || !(data.1 as f32).is_finite()
        {
            return Err(invalid(
                "boxplot: factor separator geometry is not representable",
            ));
        }
        let (x, y) = match orientation {
            Orientation::Vertical => (vec![position, position], vec![data.0, data.1]),
            Orientation::Horizontal => (vec![data.0, data.1], vec![position, position]),
        };
        let line = LinePlot::new(x, y)
            .map_err(|err| internal(format!("boxplot: {err}")))?
            .with_style(
                Vec4::new(0.65, 0.65, 0.65, 1.0),
                1.0,
                runmat_plot::plots::LineStyle::Solid,
            )
            .with_label("factor separator");
        plans.push(PlotPlan::Line(line));
        cooperation.completed(1).await?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn axis_limits(
    boxes: &[BoxStats],
    options: &BoxplotOptions,
) -> BuiltinResult<AxisLimits> {
    futures::executor::block_on(axis_limits_cooperative(
        boxes,
        options,
        &mut super::cooperation::BoxplotCooperation::new(None),
    ))
}

async fn axis_limits_cooperative(
    boxes: &[BoxStats],
    options: &BoxplotOptions,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<AxisLimits> {
    let mut min_pos = f64::INFINITY;
    let mut max_pos = f64::NEG_INFINITY;
    let mut min_data = f64::INFINITY;
    let mut max_data = f64::NEG_INFINITY;
    for stats in boxes {
        min_pos = min_pos.min(stats.position - stats.width);
        max_pos = max_pos.max(stats.position + stats.width);
        min_data = min_data.min(stats.lower_whisker).min(stats.q1);
        max_data = max_data.max(stats.upper_whisker).max(stats.q3);
        if options.notch_style != NotchStyle::Off {
            let (lower, upper) = super::appearance::notch_interval(stats)?;
            min_data = min_data.min(lower);
            max_data = max_data.max(upper);
        }
        for (idx, value) in stats.outliers.iter().enumerate() {
            let value = display_data_value(*value, idx, stats.outliers.len(), options);
            min_data = min_data.min(value);
            max_data = max_data.max(value);
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
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

#[cfg(test)]
pub(super) fn build_plot_plans(
    boxes: &[BoxStats],
    options: &BoxplotOptions,
) -> BuiltinResult<Vec<PlotPlan>> {
    futures::executor::block_on(build_plot_plans_cooperative(
        boxes,
        options,
        &mut super::cooperation::BoxplotCooperation::new(None),
    ))
}

async fn build_plot_plans_cooperative(
    boxes: &[BoxStats],
    options: &BoxplotOptions,
    cooperation: &mut super::cooperation::BoxplotCooperation,
) -> BuiltinResult<Vec<PlotPlan>> {
    let mut plans = Vec::new();
    for stats in boxes {
        add_box_lines(&mut plans, stats, options)?;
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
        cooperation.completed(1).await?;
    }
    Ok(plans)
}

fn add_box_lines(
    plans: &mut Vec<PlotPlan>,
    stats: &BoxStats,
    options: &BoxplotOptions,
) -> BuiltinResult<()> {
    let pos = stats.position;
    let half = stats.width * 0.5;
    let cap_half = stats.width * 0.25;
    let left = pos - half;
    let right = pos + half;
    let cap_left = pos - cap_half;
    let cap_right = pos + cap_half;
    let notch = options.notch_style.effective(options.compact);
    let (box_x, box_y) = super::appearance::box_outline(stats, notch)?;
    if options.box_style == BoxStyle::Filled {
        let vertices = box_x
            .iter()
            .zip(&box_y)
            .take(box_x.len() - 1)
            .map(|(&category, &data)| {
                let (x, y) = match options.orientation {
                    Orientation::Vertical => (category, data),
                    Orientation::Horizontal => (data, category),
                };
                if !x.is_finite()
                    || !y.is_finite()
                    || !(x as f32).is_finite()
                    || !(y as f32).is_finite()
                {
                    return Err(invalid("boxplot: filled box geometry is not representable"));
                }
                Ok(glam::Vec3::new(x as f32, y as f32, 0.0))
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let count = vertices.len();
        let mut patch = PatchPlot::new(vertices, vec![(0..count).collect()])
            .map_err(|err| internal(format!("boxplot: {err}")))?;
        patch.set_face_color(stats.color);
        patch.set_edge_color(stats.color);
        patch.set_label(Some(format!("{} box", stats.label)));
        plans.push(PlotPlan::Patch(patch));
    }
    use super::appearance::SegmentKind;
    let mut segments = vec![
        (box_x, box_y, SegmentKind::Box),
        (
            vec![pos, pos],
            vec![stats.q1, stats.lower_whisker],
            SegmentKind::LowerWhisker,
        ),
        (
            vec![pos, pos],
            vec![stats.q3, stats.upper_whisker],
            SegmentKind::UpperWhisker,
        ),
        (
            vec![cap_left, cap_right],
            vec![stats.lower_whisker, stats.lower_whisker],
            SegmentKind::LowerCap,
        ),
        (
            vec![cap_left, cap_right],
            vec![stats.upper_whisker, stats.upper_whisker],
            SegmentKind::UpperCap,
        ),
    ];
    if options.median_style == MedianStyle::Line {
        segments.push((
            vec![left, right],
            vec![stats.median, stats.median],
            SegmentKind::Median,
        ));
    } else {
        let (x, y) = orient_points(vec![pos], vec![stats.median], options.orientation);
        let mut ring = ScatterPlot::new(x.clone(), y.clone())
            .map_err(|err| internal(format!("boxplot: {err}")))?
            .with_style(Vec4::ONE, 8.0, MarkerStyle::Circle)
            .with_label(format!("{} median target", stats.label));
        ring.set_edge_color(Vec4::new(0.0, 0.0, 0.0, 1.0));
        ring.set_filled(true);
        plans.push(PlotPlan::Scatter(ring));
        let mut center = ScatterPlot::new(x, y)
            .map_err(|err| internal(format!("boxplot: {err}")))?
            .with_style(Vec4::new(0.0, 0.0, 0.0, 1.0), 3.0, MarkerStyle::Circle)
            .with_label(format!("{} median center", stats.label));
        center.set_edge_color(Vec4::new(0.0, 0.0, 0.0, 1.0));
        center.set_filled(true);
        plans.push(PlotPlan::Scatter(center));
    }
    if notch == NotchStyle::Marker {
        let (lower, upper) = super::appearance::notch_interval(stats)?;
        for (value, suffix) in [(lower, "lower notch"), (upper, "upper notch")] {
            let (x, y) = orient_points(vec![pos], vec![value], options.orientation);
            let mut marker = ScatterPlot::new(x, y)
                .map_err(|err| internal(format!("boxplot: {err}")))?
                .with_style(stats.color, 7.0, MarkerStyle::Triangle)
                .with_label(format!("{} {suffix}", stats.label));
            marker.set_filled(true);
            plans.push(PlotPlan::Scatter(marker));
        }
    }
    for (x, y, kind) in segments {
        let (x, y) = orient_points(x, y, options.orientation);
        let line = LinePlot::new(x, y)
            .map_err(|err| internal(format!("boxplot: {err}")))?
            .with_style(stats.color, 1.5, kind.line_style(options.box_style))
            .with_label(format!("{} {}", stats.label, kind.label()));
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

pub(super) fn color_for(idx: usize, colors: &[Vec4]) -> Vec4 {
    colors
        .get(idx % colors.len().max(1))
        .copied()
        .unwrap_or(Vec4::new(0.0, 0.4470, 0.7410, 1.0))
}
