use std::cell::RefCell;
use std::rc::Rc;

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::AreaPlot;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_line_style_args, value_as_f64, LineStyleParseOptions};

const BUILTIN_NAME: &str = "area";
const MATLAB_COLOR_ORDER: [glam::Vec4; 7] = [
    glam::Vec4::new(0.0, 0.447, 0.741, 0.4),
    glam::Vec4::new(0.85, 0.325, 0.098, 0.4),
    glam::Vec4::new(0.929, 0.694, 0.125, 0.4),
    glam::Vec4::new(0.494, 0.184, 0.556, 0.4),
    glam::Vec4::new(0.466, 0.674, 0.188, 0.4),
    glam::Vec4::new(0.301, 0.745, 0.933, 0.4),
    glam::Vec4::new(0.635, 0.078, 0.184, 0.4),
];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::area")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "area",
    op_kind: GpuOpKind::Custom("plot-render"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "area currently gathers inputs to the host before rendering and terminates fusion graphs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::area")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "area",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "area performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "area",
    category = "plotting",
    summary = "Render MATLAB-compatible area plots.",
    keywords = "area,plotting,stacked,fill",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::area"
)]
pub fn area_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target_axes, x_value, y_value, rest) = parse_area_args(args)?;
    let x_tensor = Tensor::try_from(&x_value)
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("area: {e}")))?;
    let y_tensor = Tensor::try_from(&y_value)
        .map_err(|e| plotting_error(BUILTIN_NAME, format!("area: {e}")))?;
    let x = vector_from_tensor(&x_tensor, BUILTIN_NAME)?;
    let series = area_series_from_tensor(x.clone(), &y_tensor, BUILTIN_NAME)?;
    let parsed = parse_area_style_args(&rest)?;

    let plot_handles = Rc::new(RefCell::new(Vec::new()));
    let plot_handles_slot = Rc::clone(&plot_handles);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Area",
            x_label: "X",
            y_label: "Y",
            ..Default::default()
        },
        move |figure, axes| {
            let axes = target_axes.unwrap_or(axes);
            for (idx, (upper, lower)) in series.iter().enumerate() {
                let mut plot = AreaPlot::new(x.clone(), upper.clone())
                    .map_err(|e| plotting_error(BUILTIN_NAME, format!("area: {e}")))?;
                if let Some(lower) = lower.clone() {
                    plot = plot.with_lower_curve(lower);
                } else {
                    plot.baseline = parsed.base_value;
                }
                let color = parsed
                    .color
                    .unwrap_or(MATLAB_COLOR_ORDER[idx % MATLAB_COLOR_ORDER.len()]);
                plot.color = color;
                plot.label = Some(
                    parsed
                        .label
                        .clone()
                        .unwrap_or_else(|| format!("Series {}", idx + 1)),
                );
                let plot_index = figure.add_area_plot_on_axes(plot, axes);
                plot_handles_slot.borrow_mut().push((axes, plot_index));
            }
            Ok(())
        },
    );
    let first = plot_handles.borrow().first().copied();
    let Some((axes, plot_index)) = first else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_area_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

struct ParsedAreaStyle {
    color: Option<glam::Vec4>,
    label: Option<String>,
    base_value: f64,
}

fn parse_area_style_args(args: &[Value]) -> crate::BuiltinResult<ParsedAreaStyle> {
    let mut filtered = Vec::new();
    let mut base_value = 0.0;
    let mut idx = 0usize;
    while idx < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]) {
            if key.trim().eq_ignore_ascii_case("BaseValue") && idx + 1 < args.len() {
                base_value = value_as_f64(&args[idx + 1]).ok_or_else(|| {
                    plotting_error(BUILTIN_NAME, "area: BaseValue must be numeric")
                })?;
                idx += 2;
                continue;
            }
        }
        filtered.push(args[idx].clone());
        idx += 1;
    }
    let parsed = parse_line_style_args(&filtered, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    Ok(ParsedAreaStyle {
        color: Some(parsed.appearance.color),
        label: parsed.label,
        base_value,
    })
}

fn parse_area_args(
    args: Vec<Value>,
) -> crate::BuiltinResult<(Option<usize>, Value, Value, Vec<Value>)> {
    if args.is_empty() {
        return Err(plotting_error(
            BUILTIN_NAME,
            "area: expected Y or X,Y inputs",
        ));
    }
    let mut it = args.into_iter();
    let mut target_axes = None;
    let first = it.next().unwrap();
    let first = if let Ok(handle) =
        crate::builtins::plotting::properties::resolve_plot_handle(&first, BUILTIN_NAME)
    {
        if let crate::builtins::plotting::properties::PlotHandle::Axes(_, axes) = handle {
            target_axes = Some(axes);
            it.next().ok_or_else(|| {
                plotting_error(BUILTIN_NAME, "area: expected data after axes handle")
            })?
        } else {
            first
        }
    } else {
        first
    };
    let Some(second) = it.next() else {
        let y = Tensor::try_from(&first)
            .map_err(|e| plotting_error(BUILTIN_NAME, format!("area: {e}")))?;
        let rows = y.rows.max(1);
        let x = Tensor {
            data: (1..=rows).map(|i| i as f64).collect(),
            shape: vec![rows],
            rows,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        };
        return Ok((target_axes, Value::Tensor(x), first, Vec::new()));
    };
    if matches!(second, Value::String(_) | Value::CharArray(_)) {
        let y = Tensor::try_from(&first)
            .map_err(|e| plotting_error(BUILTIN_NAME, format!("area: {e}")))?;
        let rows = y.rows.max(1);
        let x = Tensor {
            data: (1..=rows).map(|i| i as f64).collect(),
            shape: vec![rows],
            rows,
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        };
        let mut rest = vec![second];
        rest.extend(it);
        return Ok((target_axes, Value::Tensor(x), first, rest));
    }
    Ok((target_axes, first, second, it.collect()))
}

fn vector_from_tensor(tensor: &Tensor, builtin: &'static str) -> crate::BuiltinResult<Vec<f64>> {
    if !(tensor.rows == 1 || tensor.cols == 1 || tensor.shape.len() <= 1) {
        return Err(plotting_error(
            builtin,
            "area: X input must be a vector matching the row count of Y",
        ));
    }
    Ok(tensor.data.clone())
}

fn area_series_from_tensor(
    x: Vec<f64>,
    y: &Tensor,
    builtin: &'static str,
) -> crate::BuiltinResult<Vec<(Vec<f64>, Option<Vec<f64>>)>> {
    let rows = y.rows.max(1);
    let cols = y.cols.max(1);
    if rows != x.len() {
        return Err(plotting_error(
            builtin,
            "area: X length must match the number of rows in Y",
        ));
    }
    let mut out: Vec<(Vec<f64>, Option<Vec<f64>>)> = Vec::with_capacity(cols);
    let mut cumulative = vec![0.0; rows];
    for col in 0..cols {
        let mut top = Vec::with_capacity(rows);
        for row in 0..rows {
            let idx = col * rows + row;
            cumulative[row] += y.data.get(idx).copied().unwrap_or(0.0);
            top.push(cumulative[row]);
        }
        let lower = if col == 0 {
            None
        } else {
            Some(out[col - 1].0.clone())
        };
        out.push((top, lower));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    fn matrix_tensor(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor {
            data,
            shape: vec![rows, cols],
            rows,
            cols,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    #[test]
    fn area_builds_stacked_series_from_matrix() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = area_builtin(vec![Value::Tensor(matrix_tensor(
            vec![1.0, 2.0, 0.5, 0.5],
            2,
            2,
        ))])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plots().count(), 2);
        let PlotElement::Area(first) = fig.plots().next().unwrap() else {
            panic!("expected area")
        };
        let PlotElement::Area(second) = fig.plots().nth(1).unwrap() else {
            panic!("expected area")
        };
        assert_eq!(first.y, vec![1.0, 2.0]);
        assert_eq!(second.lower_y, Some(vec![1.0, 2.0]));
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Type".into())]).unwrap(),
            Value::String("area".into())
        );
    }
}
