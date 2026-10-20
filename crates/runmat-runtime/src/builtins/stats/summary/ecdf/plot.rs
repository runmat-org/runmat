//! Host figure scene and CDF-plot summary publication.
use super::*;
use std::cmp::Ordering;
pub(super) fn plot_ecdf_line(
    builtin: EmpiricalBuiltin,
    eval: &EcdfEvaluation,
) -> BuiltinResult<f64> {
    let (x, y) = step_line_points(&eval.x, &eval.f);
    let label = eval.function.label();
    let mut plots = vec![line_plot(builtin, x, y, label)?];
    if eval.bounds_on {
        let (x, y) = step_line_points(&eval.x, &eval.flo);
        plots.push(line_plot(builtin, x, y, "Lower confidence bound")?);
        let (x, y) = step_line_points(&eval.x, &eval.fup);
        plots.push(line_plot(builtin, x, y, "Upper confidence bound")?);
    }
    let opts = PlotRenderOptions {
        title: if builtin == EmpiricalBuiltin::Cdfplot {
            "Empirical CDF"
        } else {
            ""
        },
        x_label: "x",
        y_label: eval.function.y_label(),
        ..Default::default()
    };
    let mut plots_opt = Some(plots);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(builtin.name(), opts, move |figure, axes_index| {
        let plots = plots_opt.take().expect("ecdf plots consumed exactly once");
        for (idx, plot) in plots.into_iter().enumerate() {
            let plot_index = figure.add_line_plot_on_axes(plot, axes_index);
            if idx == 0 {
                *plot_index_slot.borrow_mut() = Some((axes_index, plot_index));
            }
        }
        Ok(())
    });
    let Some((axes_index, plot_index)) = *plot_index_out.borrow() else {
        return render_result
            .map(|_| f64::NAN)
            .map_err(|err| internal_with_cause(builtin, err));
    };
    let handle = crate::builtins::plotting::state::register_line_handle(
        figure_handle,
        axes_index,
        plot_index,
    );
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(internal_with_cause(builtin, err));
    }
    Ok(handle)
}

fn internal_with_cause(builtin: EmpiricalBuiltin, cause: RuntimeError) -> RuntimeError {
    let mut owner = internal_error(builtin, format!("{builtin}: {cause}"));
    owner.span = cause.span;
    owner.gpu_gather_retry = cause.gpu_gather_retry;
    owner.source = Some(Box::new(cause));
    owner
}

fn line_plot(
    builtin: EmpiricalBuiltin,
    x: Vec<f64>,
    y: Vec<f64>,
    label: &str,
) -> BuiltinResult<LinePlot> {
    LinePlot::new(x, y)
        .map_err(|err| internal_error(builtin, format!("{builtin}: {err}")))
        .map(|plot| {
            plot.with_label(label).with_style(
                [0.0, 0.4470, 0.7410, 1.0].into(),
                1.5,
                LineStyle::Solid,
            )
        })
}

fn step_line_points(x: &[f64], f: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if x.is_empty() || f.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut xs = Vec::with_capacity(x.len().saturating_mul(2));
    let mut ys = Vec::with_capacity(f.len().saturating_mul(2));
    xs.push(x[0]);
    ys.push(f[0]);
    for idx in 1..x.len().min(f.len()) {
        xs.push(x[idx]);
        ys.push(f[idx - 1]);
        xs.push(x[idx]);
        ys.push(f[idx]);
    }
    (xs, ys)
}

pub(super) fn cdfplot_stats(data: &[f64]) -> BuiltinResult<StructValue> {
    let mut values = data
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Err(invalid_argument(
            CDFPLOT_OWNER,
            "cdfplot: sample data must contain at least one non-NaN value",
        ));
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let median = if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    };
    let std = if n < 2 {
        0.0
    } else {
        let var = values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f64>()
            / (n - 1) as f64;
        var.sqrt()
    };
    let mut stats = runmat_value::UnboundedStructValueBuilder::new();
    stats
        .fields
        .insert("min".to_string(), Value::Num(values[0]));
    stats
        .fields
        .insert("max".to_string(), Value::Num(values[n - 1]));
    stats.fields.insert("mean".to_string(), Value::Num(mean));
    stats
        .fields
        .insert("median".to_string(), Value::Num(median));
    stats.fields.insert("std".to_string(), Value::Num(std));
    Ok(stats.finish())
}
