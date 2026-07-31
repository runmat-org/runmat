//! MATLAB-compatible `polarhistogram` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::{AxesKind, BarChart, PolarHistogramDisplayStyle};
use std::cell::RefCell;
use std::rc::Rc;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::bar::apply_bar_style;
use super::histogram::histogram_metadata;
use super::op_common::polar::real_tensor_from_value;
use super::op_common::{apply_axes_target, split_leading_axes_handle};
use super::properties::{apply_histogram_normalization, validate_histogram_normalization};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_bar_style_args, value_as_string, BarStyle, BarStyleDefaults};

const BUILTIN_NAME: &str = "polarhistogram";
const POLAR_HIST_BAR_WIDTH: f32 = 0.95;
const POLAR_HIST_DEFAULT_COLOR: glam::Vec4 = glam::Vec4::new(0.15, 0.5, 0.8, 0.6);
const POLAR_HIST_DEFAULT_LABEL: &str = "Frequency";
const DEFAULT_NUM_BINS: usize = 6;

const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the created polar histogram chart.",
}];

const INPUT_THETA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "theta",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Angular sample data in radians.",
}];

const INPUT_THETA_BINS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "theta",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Angular sample data in radians.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bin count scalar or explicit edge vector in radians.",
    },
];

const INPUT_ARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional axes handle, bin specification, and name/value pairs.",
}];

const POLARHISTOGRAM_INPUTS_MANUAL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name_value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "BinEdges/BinCounts pairs and optional style properties.",
}];

const POLARHISTOGRAM_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(theta)",
        inputs: &INPUT_THETA,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(theta, nbins)",
        inputs: &INPUT_THETA_BINS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(theta, edges)",
        inputs: &INPUT_THETA_BINS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(theta, Name, Value, ...)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram(pax, ___)",
        inputs: &INPUT_ARGS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = polarhistogram('BinEdges', edges, 'BinCounts', counts)",
        inputs: &POLARHISTOGRAM_INPUTS_MANUAL,
        outputs: &OUTPUT_HANDLE,
    },
];

const POLARHISTOGRAM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARHISTOGRAM.INVALID_ARGUMENT",
    identifier: Some("RunMat:polarhistogram:InvalidArgument"),
    when: "Input data, bin specification, axes targeting, or style arguments are invalid.",
    message: "polarhistogram: invalid argument",
};

const POLARHISTOGRAM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARHISTOGRAM.INTERNAL",
    identifier: Some("RunMat:polarhistogram:Internal"),
    when: "Internal polar histogram construction or rendering fails.",
    message: "polarhistogram: internal operation failed",
};

const POLARHISTOGRAM_ERRORS: [BuiltinErrorDescriptor; 2] = [
    POLARHISTOGRAM_ERROR_INVALID_ARGUMENT,
    POLARHISTOGRAM_ERROR_INTERNAL,
];

pub const POLARHISTOGRAM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POLARHISTOGRAM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &POLARHISTOGRAM_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::polarhistogram")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "polarhistogram",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "polarhistogram gathers angular samples for histcounts binning, then renders polar wedge geometry through runmat-plot.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::polarhistogram")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "polarhistogram",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "polarhistogram performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "polarhistogram",
    category = "plotting",
    summary = "Create histogram charts in polar coordinates.",
    keywords = "polarhistogram,polar,histogram,theta,plotting",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    descriptor(crate::builtins::plotting::polarhistogram::POLARHISTOGRAM_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::polarhistogram"
)]
pub async fn polarhistogram_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    let (axes_target, args) =
        split_leading_axes_handle(args, BUILTIN_NAME).map_err(map_invalid_argument)?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_invalid_argument)?;
    let parsed = ParsedPolarHistogram::from_args(args).await?;
    let counts = parsed.counts;
    let edges = parsed.edges;

    let defaults = BarStyleDefaults::new(POLAR_HIST_DEFAULT_COLOR, POLAR_HIST_BAR_WIDTH);
    let style = parse_bar_style_args(BUILTIN_NAME, &parsed.style_args, defaults)
        .map_err(map_invalid_argument)?;
    let explicit_display_name = style.label.clone();
    let labels = histogram_labels_from_edges(&edges.data);
    let render_values =
        apply_histogram_normalization(&counts.data, &edges.data, &parsed.normalization);
    let mut chart = BarChart::new(labels, render_values.clone())
        .map_err(|err| internal(format!("chart construction failed: {err}")))?;
    chart.set_histogram_bin_edges(edges.data.clone());
    chart.set_polar_histogram(true);
    chart.set_polar_histogram_display_style(parsed.display_style);
    apply_bar_style(&mut chart, &style, POLAR_HIST_DEFAULT_LABEL);

    let max_radius = render_values
        .iter()
        .filter(|value| value.is_finite() && **value > 0.0)
        .fold(0.0_f64, |acc, value| acc.max(*value))
        .max(1.0);

    let mut chart_opt = Some(chart);
    let plot_index_out = Rc::new(RefCell::new(None));
    let plot_index_slot = Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(
        BUILTIN_NAME,
        PlotRenderOptions {
            title: "Polar Histogram",
            x_label: "",
            y_label: "",
            axis_equal: true,
            grid: true,
        },
        move |figure, axes| {
            figure.set_axes_kind(axes, AxesKind::Polar);
            figure.set_axes_axis_equal(axes, true);
            figure.set_axes_limits(
                axes,
                Some((-max_radius, max_radius)),
                Some((-max_radius, max_radius)),
            );
            figure.set_axes_labels(axes, "", "");
            let plot_index = figure.add_bar_chart_on_axes(
                chart_opt
                    .take()
                    .expect("polarhistogram chart consumed once"),
                axes,
            );
            *plot_index_slot.borrow_mut() = Some((axes, plot_index));
            Ok(())
        },
    );
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle = crate::builtins::plotting::state::register_histogram_handle(
        figure_handle,
        axes,
        plot_index,
        edges.data.clone(),
        counts.data.clone(),
        parsed.normalization.clone(),
        polar_histogram_metadata(&edges.data, &style, parsed.data, parsed.display_style),
    );
    if let Some(display_name) = explicit_display_name {
        crate::builtins::plotting::state::set_histogram_handle_display_name(
            figure_handle,
            axes,
            plot_index,
            Some(display_name),
        )
        .map_err(|err| internal(format!("failed to update histogram display name: {err}")))?;
    }
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(map_internal(err));
    }
    Ok(handle)
}

struct ParsedPolarHistogram {
    data: Option<Vec<f64>>,
    counts: Tensor,
    edges: Tensor,
    normalization: String,
    style_args: Vec<Value>,
    display_style: PolarHistogramDisplayStyle,
}

impl ParsedPolarHistogram {
    async fn from_args(args: Vec<Value>) -> BuiltinResult<Self> {
        let (data, rest) = split_data_and_rest(args)?;
        let mut parsed = ParsedArgs::from_values(rest)?;
        validate_histogram_normalization(&parsed.normalization, BUILTIN_NAME)
            .map_err(map_invalid_argument)?;

        if let Some(data) = data {
            if parsed.manual_counts.is_some() {
                return Err(invalid_argument(
                    "BinCounts cannot be combined with theta sample data",
                ));
            }
            parsed.ensure_default_polar_bins();
            let source = real_tensor_from_value(data, BUILTIN_NAME).await?;
            let normalized_data = normalize_theta_tensor(source, &parsed.histcounts_args)?;
            let eval = crate::builtins::stats::hist::histcounts::evaluate(
                Value::Tensor(normalized_data.clone()),
                &parsed.histcounts_args,
            )
            .await
            .map_err(map_invalid_argument)?;
            let (counts_value, edges_value) = eval.into_pair();
            let counts = Tensor::try_from(&counts_value)
                .map_err(|err| invalid_argument(format!("cannot convert counts tensor: {err}")))?;
            let edges = Tensor::try_from(&edges_value)
                .map_err(|err| invalid_argument(format!("cannot convert edge tensor: {err}")))?;
            return Ok(Self {
                data: Some(normalized_data.data),
                counts,
                edges,
                normalization: parsed.normalization,
                style_args: parsed.style_args,
                display_style: parsed.display_style,
            });
        }

        let Some((edges, counts)) = parsed.manual_counts else {
            return Err(invalid_argument(
                "expected theta data or BinEdges/BinCounts name-value arguments",
            ));
        };
        if edges.len() < 2 || counts.len() != edges.len() - 1 {
            return Err(invalid_argument(
                "BinCounts length must be one less than BinEdges length",
            ));
        }
        validate_edges(&edges)?;
        let counts = tensor_from_vec(counts);
        let edges = tensor_from_vec(edges);
        Ok(Self {
            data: None,
            counts,
            edges,
            normalization: parsed.normalization,
            style_args: parsed.style_args,
            display_style: parsed.display_style,
        })
    }
}

struct ParsedArgs {
    histcounts_args: Vec<Value>,
    style_args: Vec<Value>,
    manual_counts: Option<(Vec<f64>, Vec<f64>)>,
    normalization: String,
    display_style: PolarHistogramDisplayStyle,
}

impl ParsedArgs {
    fn from_values(args: Vec<Value>) -> BuiltinResult<Self> {
        let mut histcounts_args = Vec::new();
        let mut style_args = Vec::new();
        let mut bin_edges: Option<Vec<f64>> = None;
        let mut bin_counts: Option<Vec<f64>> = None;
        let mut normalization = "count".to_string();
        let mut display_style = PolarHistogramDisplayStyle::Bar;
        let mut idx = 0usize;

        while idx < args.len() {
            let Some(key) = value_as_string(&args[idx]) else {
                if idx == 0 {
                    histcounts_args.push(args[idx].clone());
                    idx += 1;
                    continue;
                }
                style_args.extend_from_slice(&args[idx..]);
                break;
            };
            if idx + 1 >= args.len() {
                return Err(invalid_argument("name-value arguments must come in pairs"));
            }
            let value = args[idx + 1].clone();
            let lower = key.trim().to_ascii_lowercase();
            match lower.as_str() {
                "binedges" => {
                    bin_edges = Some(numeric_vector(&value, "BinEdges")?);
                    histcounts_args.push(args[idx].clone());
                    histcounts_args.push(value);
                }
                "bincounts" => {
                    bin_counts = Some(numeric_vector(&value, "BinCounts")?);
                }
                "numbins" | "binwidth" | "binlimits" | "binmethod" => {
                    histcounts_args.push(args[idx].clone());
                    histcounts_args.push(value);
                }
                "normalization" => {
                    normalization = value_as_string(&value)
                        .ok_or_else(|| invalid_argument("Normalization must be a string"))?
                        .trim()
                        .to_ascii_lowercase();
                    histcounts_args.push(args[idx].clone());
                    histcounts_args.push(value);
                }
                "displaystyle" => {
                    display_style = parse_display_style(&value)?;
                }
                _ => {
                    style_args.push(args[idx].clone());
                    style_args.push(value);
                }
            }
            idx += 2;
        }

        let manual_counts = match (bin_edges, bin_counts) {
            (Some(edges), Some(counts)) => Some((edges, counts)),
            (None, Some(_)) => return Err(invalid_argument("BinCounts requires BinEdges")),
            _ => None,
        };
        Ok(Self {
            histcounts_args,
            style_args,
            manual_counts,
            normalization,
            display_style,
        })
    }

    fn ensure_default_polar_bins(&mut self) {
        let has_explicit_edges = has_histcounts_key(&self.histcounts_args, "binedges")
            || has_positional_edge_argument(&self.histcounts_args);
        if !has_histcounts_key(&self.histcounts_args, "binedges")
            && !has_explicit_edges
            && !has_histcounts_key(&self.histcounts_args, "binlimits")
        {
            self.histcounts_args.extend([
                Value::String("BinLimits".into()),
                Value::Tensor(tensor_from_vec(vec![0.0, std::f64::consts::TAU])),
            ]);
        }
        if self.histcounts_args.is_empty()
            || (!has_positional_bin_argument(&self.histcounts_args)
                && !has_histcounts_key(&self.histcounts_args, "numbins")
                && !has_histcounts_key(&self.histcounts_args, "binwidth")
                && !has_histcounts_key(&self.histcounts_args, "binmethod")
                && !has_explicit_edges)
        {
            self.histcounts_args.extend([
                Value::String("NumBins".into()),
                Value::Num(DEFAULT_NUM_BINS as f64),
            ]);
        }
    }
}

fn split_data_and_rest(args: Vec<Value>) -> BuiltinResult<(Option<Value>, Vec<Value>)> {
    if args.is_empty() {
        return Ok((None, Vec::new()));
    }
    if value_as_string(&args[0]).is_some() {
        return Ok((None, args));
    }
    let mut it = args.into_iter();
    let data = it.next();
    Ok((data, it.collect()))
}

fn parse_display_style(value: &Value) -> BuiltinResult<PolarHistogramDisplayStyle> {
    let style = value_as_string(value)
        .ok_or_else(|| invalid_argument("DisplayStyle must be a string"))?
        .trim()
        .to_ascii_lowercase();
    match style.as_str() {
        "bar" => Ok(PolarHistogramDisplayStyle::Bar),
        "stairs" => Ok(PolarHistogramDisplayStyle::Stairs),
        other => Err(invalid_argument(format!(
            "unsupported DisplayStyle `{other}`"
        ))),
    }
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        Value::Bool(value) => Ok(vec![if *value { 1.0 } else { 0.0 }]),
        _ => {
            let tensor = Tensor::try_from(value)
                .map_err(|err| invalid_argument(format!("{name} must be numeric: {err}")))?;
            Ok(tensor_utils::tensor_into_values_f64(tensor))
        }
    }
}

fn tensor_from_vec(data: Vec<f64>) -> Tensor {
    let len = data.len();
    Tensor::new(data, vec![len]).expect("polar histogram result")
}

fn validate_edges(edges: &[f64]) -> BuiltinResult<()> {
    if edges.len() < 2 {
        return Err(invalid_argument(
            "BinEdges must contain at least two values",
        ));
    }
    if edges
        .windows(2)
        .any(|pair| !pair[0].is_finite() || !pair[1].is_finite() || pair[1] <= pair[0])
    {
        return Err(invalid_argument(
            "BinEdges must be finite and strictly increasing",
        ));
    }
    Ok(())
}

fn normalize_theta_tensor(mut tensor: Tensor, histcounts_args: &[Value]) -> BuiltinResult<Tensor> {
    let (start, end) = angular_range(histcounts_args)?;
    let period = end - start;
    if !period.is_finite() || period <= 0.0 {
        return Err(invalid_argument("angular bin range must be positive"));
    }
    for value in tensor.data.iter_mut() {
        if value.is_finite() {
            *value = (*value - start).rem_euclid(period) + start;
            if *value >= end {
                *value = start;
            }
        } else {
            *value = f64::NAN;
        }
    }
    Ok(tensor)
}

fn angular_range(histcounts_args: &[Value]) -> BuiltinResult<(f64, f64)> {
    if let Some(edges) = find_histcounts_vector(histcounts_args, "binedges")? {
        validate_edges(&edges)?;
        return Ok((edges[0], *edges.last().unwrap()));
    }
    if let Some(limits) = find_histcounts_vector(histcounts_args, "binlimits")? {
        if limits.len() != 2
            || !limits[0].is_finite()
            || !limits[1].is_finite()
            || limits[1] <= limits[0]
        {
            return Err(invalid_argument(
                "BinLimits must be a finite two-element increasing vector",
            ));
        }
        return Ok((limits[0], limits[1]));
    }
    Ok((0.0, std::f64::consts::TAU))
}

fn find_histcounts_vector(args: &[Value], key: &str) -> BuiltinResult<Option<Vec<f64>>> {
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if let Some(name) = value_as_string(&args[idx]) {
            if name.trim().eq_ignore_ascii_case(key) {
                return Ok(Some(numeric_vector(&args[idx + 1], key)?));
            }
            idx += 2;
        } else {
            if idx == 0 && key == "binedges" {
                if let Ok(values) = numeric_vector(&args[idx], key) {
                    if values.len() > 1 {
                        return Ok(Some(values));
                    }
                }
            }
            idx += 1;
        }
    }
    Ok(None)
}

fn has_histcounts_key(args: &[Value], key: &str) -> bool {
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if let Some(name) = value_as_string(&args[idx]) {
            if name.trim().eq_ignore_ascii_case(key) {
                return true;
            }
            idx += 2;
        } else {
            idx += 1;
        }
    }
    false
}

fn has_positional_bin_argument(args: &[Value]) -> bool {
    args.first()
        .is_some_and(|value| value_as_string(value).is_none())
}

fn has_positional_edge_argument(args: &[Value]) -> bool {
    args.first()
        .filter(|value| value_as_string(value).is_none())
        .and_then(|value| numeric_vector(value, "BinEdges").ok())
        .is_some_and(|values| values.len() > 1)
}

fn polar_histogram_metadata(
    edges: &[f64],
    style: &BarStyle,
    data: Option<Vec<f64>>,
    display_style: PolarHistogramDisplayStyle,
) -> super::state::HistogramHandleMetadata {
    let display_style = match display_style {
        PolarHistogramDisplayStyle::Bar => "bar",
        PolarHistogramDisplayStyle::Stairs => "stairs",
    };
    histogram_metadata(edges, style, data, true, display_style)
}

fn histogram_labels_from_edges(edges: &[f64]) -> Vec<String> {
    edges
        .windows(2)
        .map(|pair| format!("[{:.3}, {:.3})", pair[0], pair[1]))
        .collect()
}

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    error_with_detail(&POLARHISTOGRAM_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    error_with_detail(&POLARHISTOGRAM_ERROR_INTERNAL, err.message)
}

fn invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    error_with_detail(&POLARHISTOGRAM_ERROR_INVALID_ARGUMENT, detail)
}

fn internal(detail: impl AsRef<str>) -> RuntimeError {
    error_with_detail(&POLARHISTOGRAM_ERROR_INTERNAL, detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    fn tensor(data: &[f64]) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![data.len()]).expect("polar histogram vector"))
    }

    #[test]
    fn polarhistogram_numeric_vector_reads_typed_integer_storage_exactly() {
        let mut edges = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![-3, 0, 3]),
            vec![1, 3],
        )
        .expect("typed edge vector");
        edges.data.clear();

        assert_eq!(
            numeric_vector(&Value::Tensor(edges), "BinEdges").expect("numeric vector"),
            vec![-3.0, 0.0, 3.0]
        );
    }

    #[test]
    fn polarhistogram_sets_polar_axes_and_histogram_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(polarhistogram_builtin(vec![
            tensor(&[0.0, 0.2, 1.0, 2.0]),
            Value::Tensor(
                Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]).expect("polar histogram edges"),
            ),
        ]))
        .expect("polarhistogram");

        let figure = clone_figure(current_figure_handle()).expect("figure");
        assert_eq!(figure.axes_metadata(0).unwrap().axes_kind, AxesKind::Polar);
        let PlotElement::Bar(chart) = figure.plots().next().unwrap() else {
            panic!("expected polar histogram bar chart");
        };
        assert!(chart.is_polar_histogram());
        assert_eq!(chart.histogram_bin_edges().unwrap(), &[0.0, 1.0, 2.0, 3.0]);

        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.data, vec![2.0, 1.0, 1.0]);
    }

    #[test]
    fn polarhistogram_supports_axes_target_style_and_normalization() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = crate::builtins::plotting::subplot::subplot_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        )
        .unwrap();

        let handle = futures::executor::block_on(polarhistogram_builtin(vec![
            Value::Num(ax),
            tensor(&[0.0, 0.1, 0.2, 1.0]),
            Value::String("NumBins".into()),
            Value::Num(2.0),
            Value::String("Normalization".into()),
            Value::String("probability".into()),
            Value::String("FaceColor".into()),
            Value::String("red".into()),
            Value::String("DisplayName".into()),
            Value::String("angles".into()),
        ]))
        .expect("polarhistogram");

        let figure = clone_figure(current_figure_handle()).expect("figure");
        assert_eq!(figure.plot_axes_indices()[0], 1);
        assert_eq!(figure.axes_metadata(1).unwrap().axes_kind, AxesKind::Polar);
        let display_name = get_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayName".into()),
        ])
        .unwrap();
        assert_eq!(display_name, Value::String("angles".into()));
        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert!((counts.data.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn polarhistogram_wraps_angles_and_drops_nonfinite_values() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(polarhistogram_builtin(vec![
            tensor(&[
                0.0,
                std::f64::consts::TAU,
                -std::f64::consts::TAU,
                f64::NAN,
                f64::INFINITY,
                1.5 * std::f64::consts::PI,
            ]),
            Value::String("BinEdges".into()),
            tensor(&[0.0, std::f64::consts::PI, std::f64::consts::TAU]),
        ]))
        .expect("polarhistogram");

        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.data, vec![3.0, 1.0]);

        let data = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Data".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(data.data[0], 0.0);
        assert_eq!(data.data[1], 0.0);
        assert_eq!(data.data[2], 0.0);
        assert!(data.data[3].is_nan());
        assert!(data.data[4].is_nan());
    }

    #[test]
    fn polarhistogram_supports_manual_bin_counts() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(polarhistogram_builtin(vec![
            Value::String("BinEdges".into()),
            tensor(&[0.0, std::f64::consts::PI, std::f64::consts::TAU]),
            Value::String("BinCounts".into()),
            tensor(&[4.0, 2.0]),
            Value::String("DisplayStyle".into()),
            Value::String("stairs".into()),
        ]))
        .expect("polarhistogram");

        let counts = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("BinCounts".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(counts.data, vec![4.0, 2.0]);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayStyle".into())
            ])
            .unwrap(),
            Value::String("stairs".into())
        );
        let data = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Data".into())]).unwrap(),
        )
        .unwrap();
        assert!(data.data.is_empty());
    }

    #[test]
    fn polarhistogram_exposes_and_updates_histogram_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);

        let handle = futures::executor::block_on(polarhistogram_builtin(vec![
            tensor(&[0.0, 0.1, 2.0]),
            Value::String("BinEdges".into()),
            tensor(&[0.0, 1.0, 3.0]),
            Value::String("FaceAlpha".into()),
            Value::Num(0.25),
        ]))
        .expect("polarhistogram");

        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayStyle".into())
            ])
            .unwrap(),
            Value::String("bar".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("FaceAlpha".into())]).unwrap(),
            Value::Num(0.25)
        );

        set_builtin(vec![
            Value::Num(handle),
            Value::String("DisplayStyle".into()),
            Value::String("stairs".into()),
            Value::String("FaceAlpha".into()),
            Value::Num(0.5),
            Value::String("FaceColor".into()),
            Value::String("red".into()),
            Value::String("EdgeColor".into()),
            Value::String("none".into()),
        ])
        .unwrap();
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("DisplayStyle".into())
            ])
            .unwrap(),
            Value::String("stairs".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("FaceColor".into())]).unwrap(),
            Value::String("red".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("EdgeColor".into())]).unwrap(),
            Value::String("none".into())
        );
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("FaceAlpha".into())]).unwrap(),
            Value::Num(0.5)
        );

        let values = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("Values".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(values.data, vec![2.0, 1.0]);
        let bin_width =
            get_builtin(vec![Value::Num(handle), Value::String("BinWidth".into())]).unwrap();
        assert_eq!(bin_width, Value::Num(1.0));
        let limits = Tensor::try_from(
            &get_builtin(vec![Value::Num(handle), Value::String("BinLimits".into())]).unwrap(),
        )
        .unwrap();
        assert_eq!(limits.data, vec![0.0, 3.0]);

        let figure = clone_figure(current_figure_handle()).expect("figure");
        let PlotElement::Bar(chart) = figure.plots().next().unwrap() else {
            panic!("expected polar histogram bar chart");
        };
        assert_eq!(
            chart.polar_histogram_display_style(),
            PolarHistogramDisplayStyle::Stairs
        );
    }

    #[test]
    fn polarhistogram_rejects_empty_arguments() {
        let err = futures::executor::block_on(polarhistogram_builtin(Vec::new()))
            .expect_err("missing input should fail");
        assert_eq!(
            err.identifier(),
            Some("RunMat:polarhistogram:InvalidArgument")
        );
    }
}
