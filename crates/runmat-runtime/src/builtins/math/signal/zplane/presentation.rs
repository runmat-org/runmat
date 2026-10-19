use super::*;

pub(super) fn render_zero_pole_plot(data: ZeroPoleData) -> BuiltinResult<Value> {
    let mut plots = build_zplane_plots(&data)?;
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let handles_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::<(bool, usize, usize)>::new()));
    let handles_slot = std::rc::Rc::clone(&handles_out);
    let opts = PlotRenderOptions {
        // MATLAB labels zplane axes but adds no title (DEF-31).
        title: "",
        x_label: "Real Part",
        y_label: "Imaginary Part",
        grid: true,
        axis_equal: true,
    };
    let limits = zplane_limits(&data);
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        figure.set_axes_limits(axes, Some(limits), Some(limits));
        for plot in plots.lines.drain(..) {
            let plot_index = figure.add_line_plot_on_axes(plot, axes);
            handles_slot.borrow_mut().push((true, axes, plot_index));
        }
        for plot in plots.scatters.drain(..) {
            let plot_index = figure.add_scatter_plot_on_axes(plot, axes);
            handles_slot.borrow_mut().push((false, axes, plot_index));
        }
        Ok(())
    });

    let handles = handles_out
        .borrow()
        .iter()
        .map(|(is_line, axes, plot_index)| {
            if *is_line {
                crate::builtins::plotting::state::register_line_handle(
                    figure_handle,
                    *axes,
                    *plot_index,
                )
            } else {
                crate::builtins::plotting::state::register_scatter_handle(
                    figure_handle,
                    *axes,
                    *plot_index,
                )
            }
        })
        .collect::<Vec<_>>();
    if let Err(err) = render_result {
        if !crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Err(map_internal(err));
        }
    }
    handles_value(handles)
}

struct ZPlanePlots {
    lines: Vec<LinePlot>,
    scatters: Vec<ScatterPlot>,
}

fn build_zplane_plots(data: &ZeroPoleData) -> BuiltinResult<ZPlanePlots> {
    let limits = zplane_limits(data);
    let mut lines = vec![
        line_plot(
            unit_circle_x(),
            unit_circle_y(),
            GUIDE_COLOR,
            LineStyle::Dotted,
            "Unit Circle",
        )?,
        line_plot(
            vec![limits.0, limits.1],
            vec![0.0, 0.0],
            GUIDE_COLOR,
            LineStyle::Solid,
            "Real Axis",
        )?,
        line_plot(
            vec![0.0, 0.0],
            vec![limits.0, limits.1],
            GUIDE_COLOR,
            LineStyle::Solid,
            "Imaginary Axis",
        )?,
    ];
    for line in &mut lines[1..] {
        line.set_line_width(0.75);
    }
    let mut scatters = Vec::new();
    if !data.zeros.is_empty() {
        scatters.push(scatter_plot(
            &data.zeros,
            ZERO_COLOR,
            MarkerStyle::Circle,
            "Zeros",
            false,
        )?);
    }
    if !data.poles.is_empty() {
        scatters.push(scatter_plot(
            &data.poles,
            POLE_COLOR,
            MarkerStyle::Cross,
            "Poles",
            false,
        )?);
    }
    Ok(ZPlanePlots { lines, scatters })
}

fn line_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    color: Vec4,
    style: LineStyle,
    label: &str,
) -> BuiltinResult<LinePlot> {
    LinePlot::new(x, y)
        .map(|plot| plot.with_style(color, 1.25, style).with_label(label))
        .map_err(|err| map_internal(plotting_error(BUILTIN_NAME, format!("zplane: {err}"))))
}

fn scatter_plot(
    points: &[Complex<f64>],
    color: Vec4,
    marker: MarkerStyle,
    label: &str,
    filled: bool,
) -> BuiltinResult<ScatterPlot> {
    let x = points.iter().map(|z| z.re).collect::<Vec<_>>();
    let y = points.iter().map(|z| z.im).collect::<Vec<_>>();
    let mut plot = ScatterPlot::new(x, y)
        .map(|plot| plot.with_style(color, 9.0, marker).with_label(label))
        .map_err(|err| map_internal(plotting_error(BUILTIN_NAME, format!("zplane: {err}"))))?;
    plot.filled = filled;
    plot.set_edge_color(color);
    Ok(plot)
}

fn unit_circle_x() -> Vec<f64> {
    (0..UNIT_CIRCLE_POINTS)
        .map(|idx| {
            let theta = std::f64::consts::TAU * idx as f64 / (UNIT_CIRCLE_POINTS - 1) as f64;
            theta.cos()
        })
        .collect()
}

fn unit_circle_y() -> Vec<f64> {
    (0..UNIT_CIRCLE_POINTS)
        .map(|idx| {
            let theta = std::f64::consts::TAU * idx as f64 / (UNIT_CIRCLE_POINTS - 1) as f64;
            theta.sin()
        })
        .collect()
}

pub(super) fn zplane_limits(data: &ZeroPoleData) -> (f64, f64) {
    let extent = data
        .zeros
        .iter()
        .chain(data.poles.iter())
        .flat_map(|z| [z.re.abs(), z.im.abs()])
        .filter(|v| v.is_finite())
        .fold(1.0_f64, f64::max);
    let pad = (extent * 0.12).max(0.15);
    let limit = (extent + pad).max(1.15);
    (-limit, limit)
}

fn handles_value(handles: Vec<f64>) -> BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![handles_tensor(handles)?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![handles_tensor(handles)?],
        ));
    }
    handles_tensor(handles)
}

fn handles_tensor(handles: Vec<f64>) -> BuiltinResult<Value> {
    Tensor::new(handles.clone(), vec![1, handles.len()])
        .map(Value::Tensor)
        .map_err(|e| zplane_error(&ZPLANE_ERROR_INTERNAL, e))
}
