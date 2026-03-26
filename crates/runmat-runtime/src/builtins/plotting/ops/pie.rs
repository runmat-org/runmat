use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::PieChart;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

use super::common::gather_tensor_from_gpu_async;
use super::op_common::value_as_text_string;
use super::state::{render_active_plot, PlotRenderOptions};

const BUILTIN_NAME: &str = "pie";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::pie")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pie",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "pie is a plotting sink; GPU inputs may remain on device until host fallback is needed for pie geometry generation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::pie")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pie",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "pie terminates fusion graphs and performs rendering.",
};

#[runtime_builtin(
    name = "pie",
    category = "plotting",
    summary = "Render a MATLAB-compatible pie chart.",
    keywords = "pie,plotting,chart",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::pie"
)]
pub async fn pie_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
    let (target_axes, args) = parse_axes_target(args)?;
    let (values, explode, labels) = parse_pie_args(args).await?;
    let mut chart = PieChart::new(values, None)
        .map_err(|e| crate::builtins::plotting::plotting_error(BUILTIN_NAME, e))?;
    if let Some(explode) = explode {
        chart = chart.with_explode(explode);
    }
    if let Some(labels) = labels {
        match labels {
            PieLabelsArg::Explicit(labels) => {
                chart = chart.with_slice_labels(labels);
            }
            PieLabelsArg::Format(fmt) => {
                chart = chart.with_label_format(fmt);
            }
        }
    }
    let mut chart = Some(chart);
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Pie Chart",
            axis_equal: true,
            grid: false,
            x_label: "",
            y_label: "",
        },
        move |figure, axes| {
            let axes = target_axes.unwrap_or(axes);
            let plot_index =
                figure.add_pie_chart_on_axes(chart.take().expect("pie consumed once"), axes);
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_pie_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

enum PieLabelsArg {
    Explicit(Vec<String>),
    Format(String),
}

async fn parse_pie_args(
    args: Vec<Value>,
) -> crate::BuiltinResult<(Vec<f64>, Option<Vec<bool>>, Option<PieLabelsArg>)> {
    if args.is_empty() {
        return Err(crate::builtins::plotting::plotting_error(
            BUILTIN_NAME,
            "pie: expected values input",
        ));
    }
    let values = tensor_from_value(args[0].clone()).await?;
    let values = values.data;
    if values.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(crate::builtins::plotting::plotting_error(
            BUILTIN_NAME,
            "pie: values must be finite and nonnegative",
        ));
    }
    let mut explode: Option<Vec<bool>> = None;
    let mut labels: Option<PieLabelsArg> = None;
    for arg in args.into_iter().skip(1) {
        if explode.is_none() {
            if let Ok(t) = tensor_from_value(arg.clone()).await {
                if t.data.len() == values.len() && t.data.iter().all(|v| v.is_finite()) {
                    explode = Some(t.data.into_iter().map(|v| v != 0.0).collect());
                    continue;
                }
            }
        }
        labels = Some(parse_labels(arg, values.len())?);
    }
    if let Some(explode) = explode.as_ref() {
        if explode.len() != values.len() {
            return Err(crate::builtins::plotting::plotting_error(
                BUILTIN_NAME,
                "pie: explode vector must match values length",
            ));
        }
    }
    if let Some(PieLabelsArg::Explicit(labels)) = labels.as_ref() {
        if labels.len() != values.len() {
            return Err(crate::builtins::plotting::plotting_error(
                BUILTIN_NAME,
                "pie: labels must match values length",
            ));
        }
    }
    Ok((values, explode, labels))
}

fn parse_axes_target(args: Vec<Value>) -> crate::BuiltinResult<(Option<usize>, Vec<Value>)> {
    if args.is_empty() {
        return Ok((None, args));
    }
    if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(_, axes)) =
        crate::builtins::plotting::properties::resolve_plot_handle(&args[0], BUILTIN_NAME)
    {
        return Ok((Some(axes), args.into_iter().skip(1).collect()));
    }
    Ok((None, args))
}

async fn tensor_from_value(value: Value) -> crate::BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gather_tensor_from_gpu_async(handle, BUILTIN_NAME).await,
        other => Tensor::try_from(&other).map_err(|e| {
            crate::builtins::plotting::plotting_error(BUILTIN_NAME, format!("pie: {e}"))
        }),
    }
}

fn parse_labels(value: Value, value_len: usize) -> crate::BuiltinResult<PieLabelsArg> {
    match value {
        Value::StringArray(arr) => Ok(PieLabelsArg::Explicit(arr.data)),
        Value::Cell(cell) => {
            let mut labels = Vec::new();
            for row in 0..cell.rows {
                for col in 0..cell.cols {
                    let v = cell.get(row, col).map_err(|e| {
                        crate::builtins::plotting::plotting_error(BUILTIN_NAME, format!("pie: {e}"))
                    })?;
                    labels.push(value_as_text_string(&v).ok_or_else(|| {
                        crate::builtins::plotting::plotting_error(
                            BUILTIN_NAME,
                            "pie: labels must be strings",
                        )
                    })?);
                }
            }
            Ok(PieLabelsArg::Explicit(labels))
        }
        other => {
            let text = value_as_text_string(&other).ok_or_else(|| {
                crate::builtins::plotting::plotting_error(
                    BUILTIN_NAME,
                    "pie: labels must be strings",
                )
            })?;
            if value_len > 1 && text.contains('%') {
                Ok(PieLabelsArg::Format(text))
            } else {
                Ok(PieLabelsArg::Explicit(vec![text]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    #[test]
    fn pie_builds_chart_with_labels_and_explode() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(pie_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0, 3.0])),
            Value::Tensor(vec_tensor(&[0.0, 1.0, 0.0])),
            Value::StringArray(runmat_builtins::StringArray {
                data: vec!["A".into(), "B".into(), "C".into()],
                shape: vec![1, 3],
                rows: 1,
                cols: 3,
            }),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Pie(pie) = fig.plots().next().unwrap() else {
            panic!("expected pie");
        };
        assert_eq!(pie.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(pie.slice_labels, vec!["A", "B", "C"]);
        assert_eq!(pie.explode, vec![false, true, false]);
    }

    #[test]
    fn pie_supports_axes_target_and_validates_lengths() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        configure_subplot(1, 2, 1).unwrap();
        let ax = Value::Num(crate::builtins::plotting::state::encode_axes_handle(
            current_figure_handle(),
            1,
        ));

        let _ = futures::executor::block_on(pie_builtin(vec![
            ax,
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::StringArray(runmat_builtins::StringArray {
                data: vec!["Left".into(), "Right".into()],
                shape: vec![1, 2],
                rows: 1,
                cols: 2,
            }),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Pie(_)));
        assert_eq!(fig.plot_axes_indices()[0], 1);

        let err = futures::executor::block_on(pie_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::StringArray(runmat_builtins::StringArray {
                data: vec!["Only".into()],
                shape: vec![1, 1],
                rows: 1,
                cols: 1,
            }),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("labels must match values length"));
    }

    #[test]
    fn pie_rejects_negative_values() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let err =
            futures::executor::block_on(pie_builtin(vec![Value::Tensor(vec_tensor(&[1.0, -1.0]))]))
                .unwrap_err();
        assert!(err.to_string().contains("nonnegative"));
    }

    #[test]
    fn pie_supports_format_string_labels_and_nonbinary_explode() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let _ = futures::executor::block_on(pie_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[0.0, 3.0])),
            Value::String("%.1f%%".into()),
        ]));
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Pie(pie) = fig.plots().next().unwrap() else {
            panic!("expected pie");
        };
        assert_eq!(pie.explode, vec![false, true]);
        assert_eq!(pie.label_format.as_deref(), Some("%.1f%%"));
        let labels = pie
            .slice_meta()
            .into_iter()
            .map(|s| s.label)
            .collect::<Vec<_>>();
        assert_eq!(labels, vec!["33.3%", "66.7%"]);
    }
}
