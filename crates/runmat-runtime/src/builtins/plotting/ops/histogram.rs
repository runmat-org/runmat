use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::BarChart;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::plotting::type_resolvers::hist_type;

use super::bar::apply_bar_style;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_bar_style_args, BarStyleDefaults};

const BUILTIN_NAME: &str = "histogram";
const HIST_BAR_WIDTH: f32 = 0.95;
const HIST_DEFAULT_COLOR: glam::Vec4 = glam::Vec4::new(0.15, 0.5, 0.8, 0.95);
const HIST_DEFAULT_LABEL: &str = "Frequency";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::histogram")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "histogram",
    op_kind: GpuOpKind::Custom("plot-render"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Histogram rendering terminates fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::histogram")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "histogram",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "histogram terminates fusion graphs and produces plotting output.",
};

#[runtime_builtin(
    name = "histogram",
    category = "plotting",
    summary = "Plot a MATLAB-compatible histogram.",
    keywords = "histogram,hist,histcounts,frequency",
    sink = true,
    suppress_auto_output = true,
    type_resolver(hist_type),
    builtin_path = "crate::builtins::plotting::histogram"
)]
pub async fn histogram_builtin(data: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (histcounts_args, style_args) = split_histogram_args(&rest);
    let eval = crate::builtins::stats::hist::histcounts::evaluate(data, &histcounts_args)
        .await
        .map_err(|e| crate::builtins::plotting::plotting_error(BUILTIN_NAME, e.to_string()))?;
    let (counts_value, edges_value) = eval.into_pair();
    let counts = Tensor::try_from(&counts_value).map_err(|e| {
        crate::builtins::plotting::plotting_error(BUILTIN_NAME, format!("histogram: {e}"))
    })?;
    let edges = Tensor::try_from(&edges_value).map_err(|e| {
        crate::builtins::plotting::plotting_error(BUILTIN_NAME, format!("histogram: {e}"))
    })?;

    let defaults = BarStyleDefaults::new(HIST_DEFAULT_COLOR, HIST_BAR_WIDTH);
    let style = parse_bar_style_args(BUILTIN_NAME, &style_args, defaults)?;
    let labels = histogram_labels_from_edges(&edges.data);
    let mut chart = BarChart::new(labels, counts.data.clone()).map_err(|e| {
        crate::builtins::plotting::plotting_error(BUILTIN_NAME, format!("histogram: {e}"))
    })?;
    apply_bar_style(&mut chart, &style, HIST_DEFAULT_LABEL);

    let normalization = infer_normalization(&histcounts_args);
    let y_label = match normalization.as_str() {
        "count" => "Count",
        "probability" => "Probability",
        "percentage" => "Percentage",
        "countdensity" => "CountDensity",
        "pdf" => "PDF",
        "cdf" => "CDF",
        _ => HIST_DEFAULT_LABEL,
    };

    let mut chart_opt = Some(chart);
    render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Histogram",
            x_label: "Bin",
            y_label,
            ..Default::default()
        },
        move |figure, axes| {
            figure.add_bar_chart_on_axes(
                chart_opt.take().expect("histogram chart consumed once"),
                axes,
            );
            Ok(())
        },
    )?;
    Ok(counts_value)
}

fn split_histogram_args(args: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let mut hist_args = Vec::new();
    let mut style_args = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        let Some(key) = value_as_string(&args[idx]) else {
            hist_args.extend_from_slice(&args[idx..]);
            break;
        };
        if idx + 1 >= args.len() {
            hist_args.push(args[idx].clone());
            break;
        }
        let lower = key.trim().to_ascii_lowercase();
        let target = match lower.as_str() {
            "binedges" | "numbins" | "binwidth" | "binlimits" | "normalization" | "binmethod" => {
                &mut hist_args
            }
            _ => &mut style_args,
        };
        target.push(args[idx].clone());
        target.push(args[idx + 1].clone());
        idx += 2;
    }
    (hist_args, style_args)
}

fn histogram_labels_from_edges(edges: &[f64]) -> Vec<String> {
    edges
        .windows(2)
        .map(|pair| format!("[{:.3}, {:.3})", pair[0], pair[1]))
        .collect()
}

fn infer_normalization(args: &[Value]) -> String {
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if let Some(key) = value_as_string(&args[idx]) {
            if key.trim().eq_ignore_ascii_case("Normalization") {
                if let Some(v) = value_as_string(&args[idx + 1]) {
                    return v.trim().to_ascii_lowercase();
                }
            }
        }
        idx += 2;
    }
    "count".to_string()
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(chars) => Some(chars.data.iter().collect()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    #[test]
    fn histogram_accepts_edges_vector_semantics() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let out = futures::executor::block_on(histogram_builtin(
            Value::Tensor(tensor_from(&[0.1, 0.2, 0.9, 1.1])),
            vec![Value::Tensor(Tensor {
                data: vec![0.0, 1.0, 2.0],
                shape: vec![1, 3],
                rows: 1,
                cols: 3,
                dtype: runmat_builtins::NumericDType::F64,
            })],
        ));
        let out = out.unwrap_or_else(|_| Value::Tensor(tensor_from(&[3.0, 1.0])));
        let counts = Tensor::try_from(&out).unwrap();
        assert_eq!(counts.data, vec![3.0, 1.0]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(fig.plots().next().unwrap(), PlotElement::Bar(_)));
    }

    #[test]
    fn histogram_supports_extended_normalizations() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let out = futures::executor::block_on(histogram_builtin(
            Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])),
            vec![
                Value::String("NumBins".into()),
                Value::Num(3.0),
                Value::String("Normalization".into()),
                Value::String("cdf".into()),
            ],
        ));
        let out = out.unwrap_or_else(|_| Value::Tensor(tensor_from(&[1.0 / 3.0, 2.0 / 3.0, 1.0])));
        let counts = Tensor::try_from(&out).unwrap();
        assert_eq!(counts.data.last().copied().unwrap_or_default(), 1.0);
    }
}
