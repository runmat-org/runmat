//! MATLAB-compatible `hist` builtin.
#[path = "hist/integer_centers.rs"]
mod integer_centers;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    HIST_ERROR_INTERNAL, HIST_ERROR_INVALID_ARGUMENT, HIST_INTEGER_CENTERS_EXTENSION,
    HIST_INTEGER_DATA_EXTENSION, HIST_MODERN_OPTIONS_EXTENSION,
};

use glam::{Vec3, Vec4};
use runmat_accelerate_api::{self, GpuTensorHandle};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::bar::{BarGpuInputs, BarGpuParams, BarLayoutMode, BarOrientation};
use runmat_plot::gpu::histogram::{histogram_counts, histogram_render_values, ExactHistogramEdges};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::BarChart;
use runmat_value::{IntValue, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;

use super::bar::apply_bar_style;
use super::common::{numeric_vector, value_as_f64};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_bar_style_args, BarStyle, BarStyleDefaults};
use crate::builtins::plotting::gpu_helpers::gather_tensor_from_gpu_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::hist")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "hist",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    // Plotting is a sink, but can consume gpuArray inputs zero-copy when a shared WGPU context exists.
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Histogram rendering terminates fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::hist")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "hist",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "hist terminates fusion graphs and produces I/O.",
};

const BUILTIN_NAME: &str = "hist";
const HIST_BAR_WIDTH: f32 = 0.95;
const HIST_DEFAULT_COLOR: Vec4 = Vec4::new(0.15, 0.5, 0.8, 0.95);
const HIST_DEFAULT_LABEL: &str = "Frequency";

fn hist_descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    detail: Option<impl AsRef<str>>,
) -> RuntimeError {
    let message = match detail {
        Some(detail) => {
            let raw = detail.as_ref().trim();
            let normalized = raw.strip_prefix("hist:").map(str::trim).unwrap_or(raw);
            if normalized.is_empty() {
                error.message.to_string()
            } else {
                format!("{}: {}", error.message, normalized)
            }
        }
        None => error.message.to_string(),
    };
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn hist_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    hist_descriptor_error(&HIST_ERROR_INVALID_ARGUMENT, Some(detail))
}

fn hist_internal(detail: impl AsRef<str>) -> RuntimeError {
    hist_descriptor_error(&HIST_ERROR_INTERNAL, Some(detail))
}

fn hist_err(message: impl Into<String>) -> RuntimeError {
    hist_invalid_argument(message.into())
}

struct HistComputation {
    counts: Vec<f64>,
    centers: Vec<f64>,
    output_shape: Vec<usize>,
    charts: Vec<BarChart>,
}

/// Captures the evaluated histogram so both the renderer and MATLAB outputs share the same data.
pub struct HistEvaluation {
    counts: Tensor,
    #[allow(dead_code)]
    centers: Tensor,
    charts: Vec<BarChart>,
}

impl HistEvaluation {
    fn new(
        counts: Vec<f64>,
        centers: Vec<f64>,
        output_shape: Vec<usize>,
        charts: Vec<BarChart>,
    ) -> BuiltinResult<Self> {
        if counts.len() != centers.len() {
            return Err(hist_internal("mismatch between counts and bin centers"));
        }
        let counts_tensor = Tensor::new(counts, output_shape.clone())?;
        let centers_tensor = Tensor::new(centers, output_shape)?;
        Ok(Self {
            counts: counts_tensor,
            centers: centers_tensor,
            charts,
        })
    }

    pub fn counts_value(&self) -> Value {
        Value::Tensor(self.counts.clone())
    }

    #[allow(dead_code)]
    pub fn centers_value(&self) -> Value {
        Value::Tensor(self.centers.clone())
    }

    pub fn render_plot(&self) -> BuiltinResult<()> {
        let mut charts = Some(self.charts.clone());
        let opts = PlotRenderOptions::default();
        render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
            let charts = charts
                .take()
                .expect("hist charts consumed exactly once at render time");
            for chart in charts {
                figure.add_bar_chart_on_axes(chart, axes);
            }
            Ok(())
        })?;
        Ok(())
    }
}

impl HistComputation {
    fn into_evaluation(self) -> BuiltinResult<HistEvaluation> {
        HistEvaluation::new(self.counts, self.centers, self.output_shape, self.charts)
    }
}

#[derive(Clone)]
enum HistBinSpec {
    Auto,
    Count(usize),
    Centers(Vec<f64>),
    IntegerCenters(integer_centers::IntegerCenterBins),
    Edges(Vec<f64>),
}

#[derive(Clone)]
struct HistBinOptions {
    spec: HistBinSpec,
    bin_width: Option<f64>,
    bin_limits: Option<(f64, f64)>,
    bin_method: Option<HistBinMethod>,
}

impl HistBinOptions {
    fn new(spec: HistBinSpec) -> Self {
        Self {
            spec,
            bin_width: None,
            bin_limits: None,
            bin_method: None,
        }
    }
}

#[derive(Clone, Copy)]
enum HistBinMethod {
    Sqrt,
    Sturges,
    Integers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum HistNormalization {
    #[default]
    Count,
    Probability,
    Pdf,
}

#[derive(Clone)]
enum HistWeightsInput {
    None,
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl HistWeightsInput {
    fn from_value(value: Value, expected_len: usize) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => {
                let len =
                    super::gpu_helpers::GpuBufferExports::element_count(&handle, BUILTIN_NAME)?;
                if len != expected_len {
                    return Err(hist_err(format!(
                        "hist: Weights must contain {expected_len} elements (got {len})"
                    )));
                }
                Ok(HistWeightsInput::Gpu(handle))
            }
            other => {
                let tensor = tensor_utils::value_into_tensor_for("hist Weights", other)
                    .map_err(|e| hist_err(format!("hist: Weights {e}")))?;
                let len = tensor_utils::tensor_element_len(&tensor);
                if len != expected_len {
                    return Err(hist_err(format!(
                        "hist: Weights must contain {expected_len} elements (got {})",
                        len
                    )));
                }
                Ok(HistWeightsInput::Host(tensor))
            }
        }
    }

    async fn resolve_for_cpu_async(
        &self,
        context: &'static str,
        sample_len: usize,
    ) -> BuiltinResult<(Option<Vec<f64>>, f64)> {
        match self {
            HistWeightsInput::None => Ok((None, sample_len as f64)),
            HistWeightsInput::Host(tensor) => {
                let values = numeric_vector(tensor.clone());
                let total = values.iter().copied().sum::<f64>();
                Ok((Some(values), total))
            }
            HistWeightsInput::Gpu(handle) => {
                let tensor = gather_tensor_from_gpu_async(handle.clone(), context).await?;
                let values = numeric_vector(tensor);
                let total = values.iter().copied().sum::<f64>();
                Ok((Some(values), total))
            }
        }
    }
}

#[runtime_builtin(
    name = "hist",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::hist"
)]
pub async fn hist_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 2) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:plotting:TooManyOutputs",
            "hist: too many output arguments",
        ));
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "hist")?;
    let _arguments = args.fork();
    work.finish(hist_impl(args, &work).await)
}
async fn hist_impl(
    mut args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<Value> {
    let data = args
        .pop_front()
        .ok_or_else(|| hist_err("hist: expected data input"))?;
    let rest = args.view();
    if value_has_integer_storage(&data) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HIST_INTEGER_DATA_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest
        .iter()
        .any(|value| matches!(value, Value::String(_) | Value::CharArray(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HIST_MODERN_OPTIONS_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let evaluation = evaluate_async(data, rest).await?;
    work.check()?;
    evaluation.render_plot()?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![evaluation.counts_value()])),
        Some(2) => Ok(Value::output_list(vec![
            evaluation.counts_value(),
            evaluation.centers_value(),
        ])),
        Some(_) => Err(hist_err("hist: too many output arguments")),
        None => Ok(evaluation.counts_value()),
    }
}

fn value_has_integer_storage(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

/// Evaluate the histogram inputs once so renderers and MATLAB outputs share the same data.
pub async fn evaluate_async(
    data: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<HistEvaluation> {
    let mut input = Some(HistInput::from_value(data)?);
    let sample_len = input.as_ref().map(|value| value.len()).unwrap_or(0);
    let (bin_options, normalization, style_args, weights_value) =
        parse_hist_arguments(sample_len, rest)?;
    let defaults = BarStyleDefaults::new(HIST_DEFAULT_COLOR, HIST_BAR_WIDTH);
    let bar_style = parse_bar_style_args("hist", &style_args, defaults)?;
    let weights_input = if let Some(value) = weights_value {
        HistWeightsInput::from_value(value, sample_len)?
    } else {
        HistWeightsInput::None
    };

    let mut prepared_bins = None;
    let mut continued_effect = false;
    let mut host_statistics = false;
    let computation = if !bar_style.requires_cpu_path() {
        if let Some(handle) = input.as_ref().and_then(|value| value.vector_gpu_handle()) {
            match build_histogram_gpu_chart_async(
                handle,
                &bin_options,
                sample_len,
                normalization,
                &bar_style,
                &weights_input,
            )
            .await?
            {
                Some(HistogramGpuPreparation::Complete(chart)) => Some(chart),
                Some(HistogramGpuPreparation::HostBins {
                    bins,
                    completed_effect,
                }) => {
                    prepared_bins = Some(bins);
                    continued_effect = completed_effect;
                    None
                }
                Some(HistogramGpuPreparation::HostStatistics) => {
                    host_statistics = true;
                    continued_effect = true;
                    None
                }
                None => None,
            }
        } else {
            None
        }
    } else {
        None
    };

    let computation = match computation {
        Some(chart) => chart,
        None => {
            let continuing_prepared_work = continued_effect;
            let result = async {
                let data_arg = input.take().expect("hist input consumed once");
                let tensor = match data_arg {
                    HistInput::Host(tensor) => tensor,
                    HistInput::Gpu(handle) => gather_tensor_from_gpu_async(handle, "hist").await?,
                };
                let tensor_shape = tensor.shape().to_vec();
                let (weight_values, total_weight) = weights_input
                    .resolve_for_cpu_async("hist weights", sample_len)
                    .await?;
                if let HistBinSpec::IntegerCenters(bins) = &bin_options.spec {
                    return bins.evaluate(
                        &tensor,
                        normalization,
                        weight_values.as_deref(),
                        total_weight,
                    );
                }
                let samples = numeric_vector(tensor);
                Ok(if let Some(bins) = prepared_bins {
                    // This is continuation of successful bounds/bin preparation,
                    // not replay of an unavailable or failed GPU operation.
                    build_histogram_chart_with_bins(
                        &samples,
                        &bins,
                        normalization,
                        weight_values.as_deref(),
                        total_weight,
                    )?
                } else if host_statistics {
                    build_histogram_chart(
                        samples,
                        &bin_options,
                        normalization,
                        weight_values.as_deref(),
                        total_weight,
                    )?
                } else {
                    build_histogram_charts(
                        samples,
                        &tensor_shape,
                        &bin_options,
                        normalization,
                        weight_values.as_deref(),
                        total_weight,
                    )?
                })
            }
            .await;
            if continuing_prepared_work {
                result.map_err(|error| {
                    super::gpu_helpers::preparation_after_effect(error, BUILTIN_NAME)
                })?
            } else {
                result?
            }
        }
    };

    let mut evaluation = computation.into_evaluation()?;
    let chart_count = evaluation.charts.len();
    for (index, chart) in evaluation.charts.iter_mut().enumerate() {
        apply_bar_style(chart, &bar_style, HIST_DEFAULT_LABEL);
        if chart_count > 1 {
            chart.group_index = index;
            chart.group_count = chart_count;
        }
    }
    Ok(evaluation)
}

fn parse_hist_arguments(
    sample_len: usize,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(HistBinOptions, HistNormalization, Vec<Value>, Option<Value>)> {
    let mut idx = 0usize;
    let mut bin_options = HistBinOptions::new(HistBinSpec::Auto);
    let mut bin_set = false;
    let mut normalization = HistNormalization::Count;
    let mut norm_set = false;
    let mut style_args = Vec::new();
    let mut weights_value: Option<Value> = None;

    while idx < args.len() {
        let arg = &args[idx];
        if !bin_set && is_bin_candidate(arg) {
            let spec = parse_hist_bins(Some(arg.clone()), sample_len)?;
            ensure_spec_compatible(&spec, &bin_options, "bin argument")?;
            bin_options.spec = spec;
            bin_set = true;
            idx += 1;
            continue;
        }

        if !norm_set {
            if let Some(result) = try_parse_norm_literal(arg) {
                normalization = result?;
                norm_set = true;
                idx += 1;
                continue;
            }
        }

        let Some(key) = value_as_string(arg).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "hist",
                source.to_string(),
                source,
            )
        })?
        else {
            style_args.extend(args.range(idx..)?.iter().cloned());
            break;
        };
        if idx + 1 >= args.len() {
            return Err(hist_err(format!("hist: missing value for '{key}' option")));
        }
        let value = args[idx + 1].clone();
        let lower = key.trim().to_ascii_lowercase();
        match lower.as_str() {
            "normalization" => {
                normalization = parse_hist_normalization(Some(value))?;
                norm_set = true;
            }
            "binedges" => {
                if bin_set {
                    return Err(hist_err(
                        "hist: specify either bins argument or 'BinEdges', not both",
                    ));
                }
                let edges = parse_bin_edges_value(value)?;
                ensure_spec_compatible(
                    &HistBinSpec::Edges(edges.clone()),
                    &bin_options,
                    "BinEdges",
                )?;
                bin_options.spec = HistBinSpec::Edges(edges);
                bin_set = true;
            }
            "numbins" => {
                if bin_set {
                    return Err(hist_err(
                        "hist: NumBins cannot be combined with explicit bins",
                    ));
                }
                let count = parse_num_bins_value(&value)?;
                ensure_spec_compatible(&HistBinSpec::Count(count), &bin_options, "NumBins")?;
                bin_options.spec = HistBinSpec::Count(count);
                bin_set = true;
            }
            "binwidth" => {
                if bin_set {
                    return Err(hist_err(
                        "hist: BinWidth cannot be combined with explicit bins",
                    ));
                }
                ensure_no_explicit_bins(&bin_options, "BinWidth")?;
                if bin_options.bin_width.is_some() {
                    return Err(hist_err("hist: BinWidth specified more than once"));
                }
                let width = parse_positive_scalar(
                    &value,
                    "hist: BinWidth must be a positive finite scalar",
                )?;
                bin_options.bin_width = Some(width);
            }
            "binlimits" => {
                ensure_no_explicit_bins(&bin_options, "BinLimits")?;
                if bin_options.bin_limits.is_some() {
                    return Err(hist_err("hist: BinLimits specified more than once"));
                }
                let limits = parse_bin_limits_value(value)?;
                bin_options.bin_limits = Some(limits);
            }
            "binmethod" => {
                if bin_options.bin_width.is_some() {
                    return Err(hist_err("hist: BinMethod cannot be combined with BinWidth"));
                }
                ensure_no_explicit_bins(&bin_options, "BinMethod")?;
                if bin_options.bin_method.is_some() {
                    return Err(hist_err("hist: BinMethod specified more than once"));
                }
                let method = parse_hist_bin_method(&value)?;
                bin_options.bin_method = Some(method);
            }
            "weights" => {
                if weights_value.is_some() {
                    return Err(hist_err("hist: Weights specified more than once"));
                }
                weights_value = Some(value);
            }
            _ => {
                style_args.push(arg.clone());
                style_args.push(value);
            }
        }
        idx += 2;
    }

    Ok((bin_options, normalization, style_args, weights_value))
}

fn parse_hist_bins(arg: Option<Value>, sample_len: usize) -> BuiltinResult<HistBinSpec> {
    if arg.as_ref().is_some_and(|value| {
        matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some() && !tensor_utils::is_scalar_tensor(tensor))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HIST_INTEGER_CENTERS_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let spec = match arg {
        None => HistBinSpec::Auto,
        Some(Value::Tensor(tensor)) => parse_center_vector(tensor)?,
        Some(Value::Int(value)) => parse_integer_bin_count(&value)?,
        Some(Value::GpuTensor(_)) => {
            return Err(hist_err("hist: bin definitions must reside on the host"))
        }
        Some(other) => {
            if let Some(numeric) = value_as_f64(&other) {
                parse_bin_count_value(numeric)?
            } else {
                return Err(hist_err(
                    "hist: bin argument must be a scalar count or a vector of centers",
                ));
            }
        }
    };
    Ok(match spec {
        HistBinSpec::Count(0) => HistBinSpec::Count(default_bin_count(sample_len)),
        other => other,
    })
}

#[derive(Clone, Copy)]
struct HistDataStats {
    min: Option<f64>,
    max: Option<f64>,
}

impl HistDataStats {
    fn from_samples(samples: &[f64]) -> Self {
        let mut min: Option<f64> = None;
        let mut max: Option<f64> = None;
        for &value in samples {
            if value.is_nan() {
                continue;
            }
            min = Some(match min {
                Some(current) => current.min(value),
                None => value,
            });
            max = Some(match max {
                Some(current) => current.max(value),
                None => value,
            });
        }
        Self { min, max }
    }
}

struct RealizedBins {
    edges: Vec<f64>,
    widths: Vec<f64>,
    labels: Vec<String>,
    centers: Vec<f64>,
}

impl RealizedBins {
    fn from_edges(edges: Vec<f64>) -> BuiltinResult<Self> {
        if edges.len() < 2 {
            return Err(hist_err(
                "hist: bin definitions must contain at least two edges",
            ));
        }
        let widths = widths_from_edges(&edges);
        let labels = histogram_labels_from_edges(&edges);
        let centers = centers_from_edges(&edges);
        Ok(Self {
            edges,
            widths,
            labels,
            centers,
        })
    }

    fn bin_count(&self) -> usize {
        self.widths.len()
    }
}

fn realize_bins(
    options: &HistBinOptions,
    sample_len: usize,
    stats: Option<&HistDataStats>,
    fallback_value: Option<f64>,
) -> BuiltinResult<RealizedBins> {
    match &options.spec {
        HistBinSpec::Centers(centers) => {
            let edges = edges_from_centers(centers)?;
            RealizedBins::from_edges(edges)
        }
        HistBinSpec::IntegerCenters(bins) => Ok(bins.realize()),
        HistBinSpec::Edges(edges) => RealizedBins::from_edges(edges.clone()),
        _ => {
            if matches!(options.bin_method, Some(HistBinMethod::Integers)) {
                let edges = integer_edges(options, stats, fallback_value)?;
                return RealizedBins::from_edges(edges);
            }
            let edges = uniform_edges_from_options(options, sample_len, stats, fallback_value)?;
            RealizedBins::from_edges(edges)
        }
    }
}

fn integer_edges(
    options: &HistBinOptions,
    stats: Option<&HistDataStats>,
    fallback_value: Option<f64>,
) -> BuiltinResult<Vec<f64>> {
    let (lower, upper) = determine_limits(options, stats, fallback_value)?;
    let start = lower.floor();
    let mut end = upper.ceil();
    if approx_equal(start, end) {
        end = start + 1.0;
    }
    if end <= start {
        end = start + 1.0;
    }
    let mut edges = Vec::new();
    let mut current = start;
    while current <= end {
        edges.push(current);
        current += 1.0;
    }
    if edges.len() < 2 {
        edges.push(edges[0] + 1.0);
    }
    Ok(edges)
}

fn uniform_edges_from_options(
    options: &HistBinOptions,
    sample_len: usize,
    stats: Option<&HistDataStats>,
    fallback_value: Option<f64>,
) -> BuiltinResult<Vec<f64>> {
    let (mut lower, mut upper) = determine_limits(options, stats, fallback_value)?;
    if !lower.is_finite() || !upper.is_finite() {
        lower = -0.5;
        upper = 0.5;
    }
    if approx_equal(lower, upper) {
        upper = lower + 1.0;
    }
    if let Some(width) = options.bin_width {
        let bins = ((upper - lower) / width).ceil().max(1.0) as usize;
        let mut edges = Vec::with_capacity(bins + 1);
        for i in 0..=bins {
            edges.push(lower + width * i as f64);
        }
        if let Some(last) = edges.last_mut() {
            *last = upper;
        }
        return Ok(edges);
    }
    let span = (upper - lower).abs();
    let bin_count = determine_bin_count(options, sample_len)?;
    let mut edges = Vec::with_capacity(bin_count + 1);
    let step = if bin_count == 0 {
        1.0
    } else {
        span / bin_count as f64
    };
    for i in 0..=bin_count {
        edges.push(lower + step * i as f64);
    }
    if let Some(last) = edges.last_mut() {
        *last = upper;
    }
    Ok(edges)
}

fn widths_from_edges(edges: &[f64]) -> Vec<f64> {
    edges
        .windows(2)
        .map(|pair| {
            let width = pair[1] - pair[0];
            if width > 0.0 {
                width
            } else {
                f64::MIN_POSITIVE
            }
        })
        .collect()
}

fn determine_limits(
    options: &HistBinOptions,
    stats: Option<&HistDataStats>,
    fallback_value: Option<f64>,
) -> BuiltinResult<(f64, f64)> {
    if let Some((lo, hi)) = options.bin_limits {
        if hi <= lo {
            return Err(hist_err("hist: BinLimits must be increasing"));
        }
        return Ok((lo, hi));
    }
    if let Some(stats) = stats {
        if let (Some(min), Some(max)) = (stats.min, stats.max) {
            if approx_equal(min, max) {
                let span = options.bin_width.unwrap_or(1.0);
                return Ok((min - span * 0.5, min + span * 0.5));
            }
            return Ok((min, max));
        }
    }
    let center = fallback_value.unwrap_or(0.0);
    let span = options.bin_width.unwrap_or(1.0);
    Ok((center - span * 0.5, center + span * 0.5))
}

fn determine_bin_count(options: &HistBinOptions, sample_len: usize) -> BuiltinResult<usize> {
    if let HistBinSpec::Count(count) = options.spec {
        return Ok(count.max(1));
    }
    if let Some(method) = options.bin_method {
        return Ok(match method {
            HistBinMethod::Sqrt => sqrt_bin_count(sample_len),
            HistBinMethod::Sturges => sturges_bin_count(sample_len),
            HistBinMethod::Integers => {
                return Err(hist_internal("internal integer bin method misuse"))
            }
        });
    }
    Ok(default_bin_count(sample_len))
}

fn sqrt_bin_count(sample_len: usize) -> usize {
    ((sample_len as f64).sqrt().ceil() as usize).max(1)
}

fn sturges_bin_count(sample_len: usize) -> usize {
    let n = sample_len.max(1) as f64;
    ((n.log2().ceil() + 1.0) as usize).max(1)
}

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9
}

fn ensure_spec_compatible(
    new_spec: &HistBinSpec,
    options: &HistBinOptions,
    source: &str,
) -> BuiltinResult<()> {
    if matches!(
        new_spec,
        HistBinSpec::Centers(_) | HistBinSpec::IntegerCenters(_) | HistBinSpec::Edges(_)
    ) && (options.bin_width.is_some()
        || options.bin_method.is_some()
        || options.bin_limits.is_some())
    {
        return Err(hist_err(format!(
            "hist: {source} cannot be combined with BinWidth, BinLimits, or BinMethod"
        )));
    }
    Ok(())
}

fn ensure_no_explicit_bins(options: &HistBinOptions, source: &str) -> BuiltinResult<()> {
    if matches!(
        options.spec,
        HistBinSpec::Centers(_) | HistBinSpec::IntegerCenters(_) | HistBinSpec::Edges(_)
    ) {
        return Err(hist_err(format!(
            "hist: {source} cannot be combined with explicit bin centers or edges"
        )));
    }
    Ok(())
}

fn parse_num_bins_value(value: &Value) -> BuiltinResult<usize> {
    if let Some(count) = exact_integer_scalar(value) {
        return parse_integer_num_bins(&count);
    }
    let Some(scalar) = value_as_f64(value) else {
        return Err(hist_err("hist: NumBins must be a numeric scalar"));
    };
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(hist_err("hist: NumBins must be a positive finite scalar"));
    }
    let rounded = scalar.round();
    if (scalar - rounded).abs() > 1e-9 {
        return Err(hist_err("hist: NumBins must be an integer"));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(hist_err("hist: NumBins is too large"));
    }
    Ok(rounded as usize)
}

fn parse_positive_scalar(value: &Value, err: &str) -> BuiltinResult<f64> {
    let Some(scalar) = value_as_f64(value) else {
        return Err(hist_err(err));
    };
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(hist_err(err));
    }

    Ok(scalar)
}

fn parse_bin_limits_value(value: Value) -> BuiltinResult<(f64, f64)> {
    let tensor = Tensor::try_from(&value)
        .map_err(|_| hist_err("hist: BinLimits must be provided as a numeric vector"))?;
    let values = numeric_vector(tensor);
    if values.len() != 2 {
        return Err(hist_err(
            "hist: BinLimits must contain exactly two elements",
        ));
    }
    let lo = values[0];
    let hi = values[1];
    if !lo.is_finite() || !hi.is_finite() {
        return Err(hist_err("hist: BinLimits must be finite"));
    }
    if hi <= lo {
        return Err(hist_err("hist: BinLimits must be increasing"));
    }
    Ok((lo, hi))
}

fn parse_hist_bin_method(value: &Value) -> BuiltinResult<HistBinMethod> {
    let Some(text) = value_as_string(value).map_err(|source| {
        crate::builtins::plotting::plotting_error_with_source("hist", source.to_string(), source)
    })?
    else {
        return Err(hist_err("hist: BinMethod must be a string"));
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "sqrt" => Ok(HistBinMethod::Sqrt),
        "sturges" => Ok(HistBinMethod::Sturges),
        "integers" => Ok(HistBinMethod::Integers),
        other => Err(hist_err(format!(
            "hist: BinMethod '{other}' is not supported yet (supported: 'sqrt', 'sturges', 'integers')"
        ))),
    }
}

fn parse_center_vector(tensor: Tensor) -> BuiltinResult<HistBinSpec> {
    let len = tensor_utils::tensor_element_len(&tensor);
    if len == 0 {
        return Err(hist_err("hist: bin center array cannot be empty"));
    }
    if len == 1 {
        if let Some(value) = tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0))
        {
            return parse_integer_bin_count(&value);
        }
        return parse_bin_count_value(tensor_utils::tensor_value_f64(&tensor, 0));
    }
    if let Some(storage) = tensor.integer_storage() {
        return integer_centers::IntegerCenterBins::new(storage).map(HistBinSpec::IntegerCenters);
    }
    let values = numeric_vector(tensor);
    validate_monotonic(&values)?;
    ensure_uniform_spacing(&values)?;
    Ok(HistBinSpec::Centers(values))
}

fn parse_bin_count_value(value: f64) -> BuiltinResult<HistBinSpec> {
    if !value.is_finite() || value <= 0.0 {
        return Err(hist_err("hist: bin count must be positive"));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-9 {
        return Err(hist_err("hist: bin count must be an integer"));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(hist_err("hist: bin count is too large"));
    }
    Ok(HistBinSpec::Count(rounded as usize))
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn parse_integer_num_bins(value: &IntValue) -> BuiltinResult<usize> {
    let Some(count) = value.try_to_usize() else {
        return Err(hist_err("hist: NumBins must be a positive finite scalar"));
    };
    if count == 0 {
        return Err(hist_err("hist: NumBins must be a positive finite scalar"));
    }
    Ok(count)
}

fn parse_integer_bin_count(value: &IntValue) -> BuiltinResult<HistBinSpec> {
    let Some(count) = value.try_to_usize() else {
        return Err(hist_err("hist: bin count must be positive"));
    };
    if count == 0 {
        return Err(hist_err("hist: bin count must be positive"));
    }
    Ok(HistBinSpec::Count(count))
}

fn is_bin_candidate(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

fn try_parse_norm_literal(value: &Value) -> Option<BuiltinResult<HistNormalization>> {
    match value {
        Value::String(_) | Value::CharArray(_) => {
            let cloned = value.clone();
            match parse_hist_normalization(Some(cloned)) {
                Ok(norm) => Some(Ok(norm)),
                Err(_) => None,
            }
        }
        _ => None,
    }
}

fn parse_bin_edges_value(value: Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            let edges = numeric_vector(tensor);
            if edges.len() < 2 {
                return Err(hist_err(
                    "hist: 'BinEdges' must contain at least two elements",
                ));
            }
            validate_monotonic(&edges)?;
            Ok(edges)
        }
        Value::GpuTensor(_) => Err(hist_err("hist: 'BinEdges' must be provided on the host")),
        _ => Err(hist_err("hist: 'BinEdges' expects a numeric vector")),
    }
}

fn ensure_uniform_spacing(values: &[f64]) -> BuiltinResult<()> {
    if values.len() <= 2 {
        return Ok(());
    }
    let mut diffs = values.windows(2).map(|pair| pair[1] - pair[0]);
    let first = diffs.next().unwrap();
    if first <= 0.0 || !first.is_finite() {
        return Err(hist_err("hist: bin centers must be strictly increasing"));
    }
    // Relative to the spacing, not an absolute unit-sized floor: a tiny but
    // materially uneven interval must not be admitted as uniform geometry.
    let tol = first.abs() * 1e-6;
    for diff in diffs {
        if (diff - first).abs() > tol {
            return Err(hist_err("hist: bin centers must be evenly spaced"));
        }
    }
    Ok(())
}

fn parse_hist_normalization(arg: Option<Value>) -> BuiltinResult<HistNormalization> {
    match arg {
        None => Ok(HistNormalization::Count),
        Some(Value::String(runmat_value::StringElement::Text(s))) => parse_norm_string(&s),
        Some(Value::CharArray(chars)) => {
            let text: String =
                runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                    crate::builtins::plotting::plotting_error_with_source(
                        "hist",
                        source.to_string(),
                        source,
                    )
                })?;
            parse_norm_string(&text)
        }
        Some(value) => {
            if let Some(text) = value_as_string(&value).map_err(|source| {
                crate::builtins::plotting::plotting_error_with_source(
                    "hist",
                    source.to_string(),
                    source,
                )
            })? {
                parse_norm_string(&text)
            } else {
                Err(hist_err(
                    "hist: normalization must be 'count', 'probability', or 'pdf'",
                ))
            }
        }
    }
}

fn parse_norm_string(text: &str) -> BuiltinResult<HistNormalization> {
    match text.trim().to_ascii_lowercase().as_str() {
        "count" | "counts" => Ok(HistNormalization::Count),
        "probability" | "prob" => Ok(HistNormalization::Probability),
        "pdf" => Ok(HistNormalization::Pdf),
        other => Err(hist_err(format!(
            "hist: unsupported normalization '{other}' (expected 'count', 'probability', or 'pdf')"
        ))),
    }
}

fn value_as_string(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::CharArray(chars) => Some(runmat_value::decode_utf16_code_units(chars.data())?),
        _ => None,
    })
}

fn default_bin_count(sample_len: usize) -> usize {
    let _ = sample_len;
    10
}

fn build_histogram_charts(
    data: Vec<f64>,
    shape: &[usize],
    bin_options: &HistBinOptions,
    normalization: HistNormalization,
    weights: Option<&[f64]>,
    total_weight: f64,
) -> BuiltinResult<HistComputation> {
    let rows = shape.first().copied().unwrap_or(data.len());
    let columns = if shape.len() >= 2 {
        shape[1..].iter().copied().product::<usize>()
    } else {
        1
    };
    if columns <= 1 || (shape.len() == 2 && shape[0] == 1) {
        return build_histogram_chart(data, bin_options, normalization, weights, total_weight);
    }

    let column_local_bins = matches!(bin_options.spec, HistBinSpec::Auto | HistBinSpec::Count(_))
        && bin_options.bin_width.is_none()
        && bin_options.bin_limits.is_none()
        && bin_options.bin_method.is_none();
    let shared_bins = if column_local_bins {
        None
    } else {
        let stats = HistDataStats::from_samples(&data);
        Some(realize_bins(
            bin_options,
            rows,
            Some(&stats),
            data.first().copied(),
        )?)
    };
    let expected_bin_count = shared_bins
        .as_ref()
        .map(RealizedBins::bin_count)
        .unwrap_or_else(|| match bin_options.spec {
            HistBinSpec::Count(count) => count.max(1),
            _ => default_bin_count(rows),
        });
    let mut all_counts = Vec::with_capacity(expected_bin_count * columns);
    let mut all_centers = Vec::with_capacity(expected_bin_count * columns);
    let mut charts = Vec::with_capacity(columns);
    for column in 0..columns {
        let start = column
            .checked_mul(rows)
            .ok_or_else(|| hist_internal("matrix column offset overflow"))?;
        let end = start
            .checked_add(rows)
            .ok_or_else(|| hist_internal("matrix column extent overflow"))?;
        let samples = data
            .get(start..end)
            .ok_or_else(|| hist_internal("matrix data does not match its shape"))?;
        let local_bins;
        let bins = if let Some(shared) = shared_bins.as_ref() {
            shared
        } else {
            let stats = HistDataStats::from_samples(samples);
            local_bins = realize_bins(bin_options, rows, Some(&stats), samples.first().copied())?;
            &local_bins
        };
        if bins.bin_count() != expected_bin_count {
            return Err(hist_internal(
                "matrix columns produced incompatible histogram widths",
            ));
        }
        let column_weights = weights.and_then(|values| values.get(start..end));
        let column_total = column_weights
            .map(|values| values.iter().copied().sum())
            .unwrap_or(rows as f64);
        let mut counts = vec![0.0; bins.bin_count()];
        for (sample_index, value) in samples.iter().enumerate() {
            let bin_index = find_bin_index(&bins.edges, *value);
            counts[bin_index] += column_weights
                .and_then(|values| values.get(sample_index).copied())
                .unwrap_or(1.0);
        }
        apply_normalization(&mut counts, &bins.widths, normalization, column_total);
        let chart = build_hist_cpu_chart(bins, counts.clone())?.with_group(column, columns);
        all_counts.extend_from_slice(&counts);
        all_centers.extend_from_slice(&bins.centers);
        charts.push(chart);
    }
    Ok(HistComputation {
        counts: all_counts,
        centers: all_centers,
        output_shape: vec![expected_bin_count, columns],
        charts,
    })
}

fn build_histogram_chart(
    data: Vec<f64>,
    bin_options: &HistBinOptions,
    normalization: HistNormalization,
    weights: Option<&[f64]>,
    total_weight: f64,
) -> BuiltinResult<HistComputation> {
    let sample_len = data.len();
    if sample_len == 0 {
        return build_empty_histogram_chart(bin_options, normalization, 0, total_weight);
    }
    let stats = HistDataStats::from_samples(&data);
    let fallback = data.first().copied();
    let bins = realize_bins(bin_options, sample_len, Some(&stats), fallback)?;
    build_histogram_chart_with_bins(&data, &bins, normalization, weights, total_weight)
}

fn build_histogram_chart_with_bins(
    data: &[f64],
    bins: &RealizedBins,
    normalization: HistNormalization,
    weights: Option<&[f64]>,
    total_weight: f64,
) -> BuiltinResult<HistComputation> {
    let weight_for_sample = |sample_idx: usize| -> f64 {
        weights
            .and_then(|slice| slice.get(sample_idx).copied())
            .unwrap_or(1.0)
    };
    let mut counts = vec![0f64; bins.bin_count()];
    for (sample_idx, value) in data.iter().enumerate() {
        let bin_idx = find_bin_index(&bins.edges, *value);
        counts[bin_idx] += weight_for_sample(sample_idx);
    }
    apply_normalization(&mut counts, &bins.widths, normalization, total_weight);
    build_hist_cpu_result(bins, counts)
}

fn build_empty_histogram_chart(
    bin_options: &HistBinOptions,
    _normalization: HistNormalization,
    sample_len: usize,
    _total_weight: f64,
) -> BuiltinResult<HistComputation> {
    let bins = realize_bins(bin_options, sample_len, None, None)?;
    let counts = vec![0.0; bins.bin_count()];
    build_hist_cpu_result(&bins, counts)
}

fn build_hist_cpu_result(bins: &RealizedBins, counts: Vec<f64>) -> BuiltinResult<HistComputation> {
    let bar = build_hist_cpu_chart(bins, counts.clone())?;
    Ok(HistComputation {
        counts,
        centers: bins.centers.clone(),
        output_shape: vec![1, bins.bin_count()],
        charts: vec![bar],
    })
}

fn build_hist_cpu_chart(bins: &RealizedBins, counts: Vec<f64>) -> BuiltinResult<BarChart> {
    let mut bar = BarChart::new(bins.labels.clone(), counts)
        .map_err(|err| hist_err(format!("hist: {err}")))?;
    bar.label = Some(HIST_DEFAULT_LABEL.to_string());
    Ok(bar)
}

fn validate_monotonic(values: &[f64]) -> BuiltinResult<()> {
    if values.windows(2).all(|w| w[0] < w[1]) {
        Ok(())
    } else {
        Err(hist_err("hist: values must be strictly increasing"))
    }
}

fn find_bin_index(edges: &[f64], value: f64) -> usize {
    if value <= edges[0] {
        return 0;
    }
    let last = edges.len() - 2;
    for i in 0..=last {
        if value < edges[i + 1] || i == last {
            return i;
        }
    }
    last
}

fn edges_from_centers(centers: &[f64]) -> BuiltinResult<Vec<f64>> {
    if centers.is_empty() {
        return Err(hist_err(
            "hist: bin centers must contain at least one element",
        ));
    }
    if centers.len() == 1 {
        let half = 0.5;
        return Ok(vec![centers[0] - half, centers[0] + half]);
    }
    validate_monotonic(centers)?;
    let mut edges = Vec::with_capacity(centers.len() + 1);
    edges.push(centers[0] - (centers[1] - centers[0]) * 0.5);
    for pair in centers.windows(2) {
        edges.push(stable_midpoint(pair[0], pair[1]));
    }
    edges.push(
        centers[centers.len() - 1]
            + (centers[centers.len() - 1] - centers[centers.len() - 2]) * 0.5,
    );
    Ok(edges)
}

fn stable_midpoint(left: f64, right: f64) -> f64 {
    left.midpoint(right)
}

fn histogram_labels_from_edges(edges: &[f64]) -> Vec<String> {
    edges
        .windows(2)
        .map(|pair| {
            let start = pair[0];
            let end = pair[1];
            format!("[{start:.3}, {end:.3})")
        })
        .collect()
}

fn centers_from_edges(edges: &[f64]) -> Vec<f64> {
    edges
        .windows(2)
        .map(|pair| stable_midpoint(pair[0], pair[1]))
        .collect()
}

fn apply_normalization(
    counts: &mut [f64],
    widths: &[f64],
    normalization: HistNormalization,
    total_weight: f64,
) {
    match normalization {
        HistNormalization::Count => {}
        HistNormalization::Probability => {
            let total = if total_weight > 0.0 {
                total_weight
            } else {
                f64::EPSILON
            };
            for count in counts {
                *count /= total;
            }
        }
        HistNormalization::Pdf => {
            let total = if total_weight > 0.0 {
                total_weight
            } else {
                f64::EPSILON
            };
            for (count, width) in counts.iter_mut().zip(widths.iter()) {
                let w = if *width > 0.0 {
                    *width
                } else {
                    f64::MIN_POSITIVE
                };
                // Dividing in stages avoids underflowing a valid tiny total
                // times a valid tiny bin width to zero first.
                *count = (*count / total) / w;
            }
        }
    }
}

/// A completed bounds stage may discover nonuniform data-dependent bins.
/// Retain those exact bins for the existing host count algorithm.
enum HistogramGpuPreparation {
    Complete(HistComputation),
    HostBins {
        bins: RealizedBins,
        completed_effect: bool,
    },
    HostStatistics,
}

async fn build_histogram_gpu_chart_async(
    values: &GpuTensorHandle,
    bin_options: &HistBinOptions,
    sample_len: usize,
    normalization: HistNormalization,
    style: &BarStyle,
    weights: &HistWeightsInput,
) -> BuiltinResult<Option<HistogramGpuPreparation>> {
    if super::gpu_helpers::GpuBufferExports::element_count(values, BUILTIN_NAME)? != sample_len {
        return Err(hist_err(
            "hist: GPU sample shape does not match its admitted length",
        ));
    }
    if u32::try_from(sample_len).is_err() {
        return Ok(None);
    }
    if let HistWeightsInput::Gpu(input) = weights {
        if super::gpu_helpers::GpuBufferExports::element_count(input, BUILTIN_NAME)? != sample_len {
            return Err(hist_err("hist: GPU weights must match sample length"));
        }
    }
    // Exact integer-center geometry is counted by its typed host owner after one
    // gather; exporting or preparing floating bounds would discard its midpoints.
    if matches!(bin_options.spec, HistBinSpec::IntegerCenters(_)) {
        return Ok(None);
    }
    // Explicit edge/center geometry does not depend on samples; no bounds pass
    // may replace it with a rendering approximation.
    let mut known_bins = match &bin_options.spec {
        HistBinSpec::Edges(_) | HistBinSpec::Centers(_) => {
            Some(realize_bins(bin_options, sample_len, None, None)?)
        }
        _ => None,
    };
    if let Some(bins) = &known_bins {
        if u32::try_from(bins.bin_count()).is_err()
            || !bins.edges.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Ok(None);
        }
    }
    if !matches!(weights, HistWeightsInput::None) {
        if let Some(bins) = known_bins.take() {
            return Ok(Some(HistogramGpuPreparation::HostBins {
                bins,
                completed_effect: false,
            }));
        }
    }
    let Some((context, mut gpu_exports)) =
        super::gpu_helpers::GpuBufferExports::prepare(&[values], BUILTIN_NAME)?
    else {
        return Ok(None);
    };
    let result = async {
        let exported = gpu_exports
            .export(values, BUILTIN_NAME)?
            .ok_or_else(|| hist_internal("unable to export GPU data"))?;
        if exported.len == 0 {
            return build_empty_histogram_chart(bin_options, normalization, 0, 0.0)
                .map(HistogramGpuPreparation::Complete);
        }
        let bins = if let Some(bins) = known_bins {
            bins
        } else {
            let bounds = {
                let mut protected = gpu_exports.protected_handles();
                if let HistWeightsInput::Gpu(input) = weights {
                    protected.push(input);
                }
                super::gpu_helpers::numeric_bounds_protecting(values, &protected, BUILTIN_NAME)
                    .await?
            };
            gpu_exports.record_completed_effect();
            // Generic extrema may include NaN; histogram statistics omit NaNs.
            // This successful-prefix continuation uses the existing CPU stats
            // owner, not a failed hook or a second GPU reduction.
            if bounds.min.is_nan() || bounds.max.is_nan() {
                return Ok(HistogramGpuPreparation::HostStatistics);
            }
            realize_bins(
                bin_options,
                sample_len,
                Some(&HistDataStats {
                    min: Some(bounds.min),
                    max: Some(bounds.max),
                }),
                Some(bounds.min),
            )?
        };
        if !matches!(weights, HistWeightsInput::None)
            || u32::try_from(bins.bin_count()).is_err()
            || !bins.edges.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Ok(HistogramGpuPreparation::HostBins {
                bins,
                completed_effect: gpu_exports.has_completed_effect(),
            });
        }
        let edges = ExactHistogramEdges::new(&bins.edges).map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                hist_internal("invalid numerical edges"),
                error,
            )
        })?;
        let evidence = histogram_counts(&context.access, exported, &edges)
            .await
            .map_err(|error| {
                crate::runtime_error::with_provider_execution_source(
                    hist_internal("GPU numerical counting failed"),
                    error,
                )
            })?;
        gpu_exports.record_completed_effect();
        let mut counts = evidence.into_double_counts();
        apply_normalization(&mut counts, &bins.widths, normalization, sample_len as f64);
        let values_buffer = histogram_render_values(&context.access, &counts).map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                hist_internal("render-value projection failed"),
                error,
            )
        })?;
        let bin_count_u32 = u32::try_from(bins.bin_count())
            .map_err(|_| hist_internal("bin count exceeds GPU range"))?;
        let bar_inputs = BarGpuInputs {
            values_buffer: values_buffer.into(),
            row_count: bin_count_u32,
            scalar: ScalarType::F32,
        };
        let bar_params = BarGpuParams {
            color: style.face_rgba(),
            bar_width: style.bar_width,
            series_index: 0,
            series_count: 1,
            source_row_count: bin_count_u32,
            transpose_source: false,
            group_index: 0,
            group_count: 1,
            orientation: BarOrientation::Vertical,
            layout: BarLayoutMode::Grouped,
        };
        let gpu_vertices = context
            .access
            .with_shared(|emission| {
                runmat_plot::gpu::bar::pack_vertices_from_values(emission, &bar_inputs, &bar_params)
            })
            .map_err(|source| {
                super::plotting_error_with_source(
                    BUILTIN_NAME,
                    format!("hist: GPU emission unavailable: {source}"),
                    source,
                )
            })?
            .map_err(|error| hist_internal(format!("failed to build GPU vertices: {error}")))?;
        let bounds = histogram_bar_bounds(&counts, style.bar_width);
        let vertex_count = gpu_vertices.vertex_count;
        let mut bar = BarChart::from_gpu_buffer(
            bins.labels.clone(),
            bins.bin_count(),
            gpu_vertices,
            vertex_count,
            bounds,
            style.face_rgba(),
            style.bar_width,
        )
        .with_gpu_source(bar_inputs, 0, 1);
        bar.label = Some(HIST_DEFAULT_LABEL.to_string());
        let bin_count = bins.bin_count();
        Ok(HistogramGpuPreparation::Complete(HistComputation {
            counts,
            centers: bins.centers,
            output_shape: vec![1, bin_count],
            charts: vec![bar],
        }))
    }
    .await;
    gpu_exports.complete(result, BUILTIN_NAME).map(Some)
}

fn histogram_bar_bounds(counts: &[f64], bar_width: f32) -> BoundingBox {
    let max_y = counts.iter().copied().fold(0.0_f64, f64::max) as f32;
    let max_y = if max_y.is_finite() && max_y > 0.0 {
        max_y
    } else {
        1.0
    };
    BoundingBox::new(
        Vec3::new(1.0 - bar_width * 0.5, 0.0, 0.0),
        Vec3::new(counts.len() as f32 + bar_width * 0.5, max_y, 0.0),
    )
}

enum HistInput {
    Host(Tensor),
    Gpu(GpuTensorHandle),
}

impl HistInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = tensor_utils::value_into_tensor_for("hist", other)
                    .map_err(|e| hist_err(format!("hist: {e}")))?;
                Ok(Self::Host(tensor))
            }
        }
    }

    fn vector_gpu_handle(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Gpu(handle)
                if handle.shape().iter().filter(|&&dim| dim > 1).count() <= 1
                    && runmat_accelerate_api::handle_integer_type(handle).is_none()
                    && !runmat_accelerate_api::handle_is_logical(handle) =>
            {
                Some(handle)
            }
            Self::Host(_) => None,
            Self::Gpu(_) => None,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Host(tensor) => tensor_utils::tensor_element_len(tensor),
            Self::Gpu(handle) => handle.shape().iter().product(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    async fn evaluate_async(data: Value, rest: &[Value]) -> BuiltinResult<HistEvaluation> {
        super::evaluate_async(data, crate::call::FunctionArgumentView::from_slice(rest)).await
    }

    async fn hist_builtin(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        let args = std::iter::once(first).chain(rest).collect();
        super::hist_builtin(
            crate::context::legacy::active(),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }

    use crate::builtins::common::test_support;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::NumericDType;

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("hist test vector")
    }

    fn int_tensor(data: Vec<i16>) -> Tensor {
        Tensor::new_integer(
            runmat_value::IntegerStorage::I16(data.clone()),
            vec![data.len()],
        )
        .expect("integer tensor")
    }

    fn assert_plotting_unavailable(err: &RuntimeError) {
        assert!(
            crate::builtins::plotting::state::presentation_unavailable(err),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_respects_bin_argument() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[1.0, 2.0, 3.0, 4.0]));
        let bins = vec![Value::from(2.0)];
        let result = block_on(hist_builtin(data, bins));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[test]
    fn hist_legacy_default_is_ten_bins() {
        assert_eq!(default_bin_count(0), 10);
        assert_eq!(default_bin_count(4), 10);
        assert_eq!(default_bin_count(10_000), 10);
    }

    #[test]
    fn hist_integer_samples_are_strictly_extension_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(hist_builtin(
            Value::Tensor(int_tensor(vec![1, 2, 3])),
            Vec::new(),
        ))
        .expect_err("integer samples require RunMat mode");
        assert_eq!(
            error.identifier(),
            HIST_INTEGER_DATA_EXTENSION.error_identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_bin_centers_vector() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0, 1.5]));
        let centers = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0]));
        let result = block_on(hist_builtin(data, vec![centers]));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_bin_counts_read_typed_integer_tensors_exactly() {
        let exact = 9_007_199_254_740_993_u64;
        let scalar_count = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![exact]),
            vec![1, 1],
        )
        .expect("typed bin count");
        match parse_hist_bins(Some(Value::Tensor(scalar_count)), 10).unwrap() {
            HistBinSpec::Count(count) => assert_eq!(count, exact as usize),
            _ => panic!("expected count bin spec"),
        }

        let num_bins = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![exact]),
            vec![1, 1],
        )
        .expect("typed NumBins");
        assert_eq!(
            parse_num_bins_value(&Value::Tensor(num_bins)).unwrap(),
            exact as usize
        );

        let negative = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![-1]),
            vec![1, 1],
        )
        .expect("negative bin count");
        assert!(parse_hist_bins(Some(Value::Tensor(negative)), 10).is_err());

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(parse_num_bins_value(&Value::Num(boundary)).is_err());
        assert!(parse_hist_bins(Some(Value::Num(boundary)), 10).is_err());
        assert!(parse_hist_bins(Some(Value::Num(2.5)), 10).is_err());
    }

    #[test]
    fn hist_weights_read_typed_integer_storage_exactly() {
        let weights = HistWeightsInput::from_value(Value::Tensor(int_tensor(vec![1, 2, 3])), 3)
            .expect("weights");

        let (values, total) = block_on(weights.resolve_for_cpu_async("hist weights", 3)).unwrap();
        assert_eq!(values, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(total, 6.0);
    }

    #[test]
    fn hist_gpu_weights_preserve_native_single_class() {
        let tensor =
            Tensor::new_with_dtype(vec![1.0, 2.0, 3.0], vec![1, 3], NumericDType::F32).unwrap();
        let weights =
            HistWeightsInput::from_value(Value::Tensor(tensor), 3).expect("single weights");

        let HistWeightsInput::Host(tensor) = &weights else {
            panic!("host Single weights");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        let (values, total) = block_on(weights.resolve_for_cpu_async("hist weights", 3)).unwrap();
        assert_eq!(values, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(total, 6.0);
    }

    #[test]
    fn resident_integer_hist_uses_compatibility_gate_before_owner_gather() {
        test_support::with_test_provider(|provider| {
            let input = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U8(&[1, 2, 2, 3]),
                    shape: &[1, 4],
                })
                .expect("resident uint8 histogram input");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(hist_builtin(Value::GpuTensor(input), Vec::new()))
                .expect_err("strict compatibility rejects resident integer hist data");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:HistIntegerDataExtension")
            );
        });
    }

    #[test]
    fn resident_integer_hist_gathers_exactly_in_runmat_mode() {
        test_support::with_test_provider(|provider| {
            ensure_plot_test_env();
            let input = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[9_007_199_254_740_993, 9_007_199_254_740_994]),
                    shape: &[1, 2],
                })
                .expect("resident wide histogram input");
            let centers = Value::Tensor(
                Tensor::new_integer(
                    runmat_value::IntegerStorage::U64(vec![
                        9_007_199_254_740_993,
                        9_007_199_254_740_994,
                    ]),
                    vec![1, 2],
                )
                .unwrap(),
            );
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let output = block_on(hist_builtin(Value::GpuTensor(input), vec![centers]))
                .expect("resident wide histogram");
            let counts = Tensor::try_from(&output).expect("histogram counts");
            let counts = counts.as_f64_slice().expect("double counts");
            assert_eq!(counts.iter().sum::<f64>(), 2.0);
            assert_eq!(counts.iter().filter(|&&count| count == 1.0).count(), 2);
        });
    }

    #[test]
    fn resident_logical_hist_data_and_weights_gather_through_owner() {
        test_support::with_test_provider(|provider| {
            ensure_plot_test_env();
            let data = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[0, 1, 1, 0],
                    shape: &[1, 4],
                })
                .expect("resident logical data");
            let weights = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1, 0, 1, 0],
                    shape: &[1, 4],
                })
                .expect("resident logical weights");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let output = block_on(hist_builtin(
                Value::GpuTensor(data),
                vec![
                    Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap()),
                    Value::String("Weights".into()),
                    Value::GpuTensor(weights),
                ],
            ))
            .expect("resident logical histogram");
            let counts = Tensor::try_from(&output).expect("histogram counts");
            assert_eq!(counts.as_f64_slice().unwrap().iter().sum::<f64>(), 2.0);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_probability_normalization() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = block_on(hist_builtin(
            data,
            vec![Value::from(3.0), Value::String("probability".into())],
        ));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_string_only_normalization() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = block_on(hist_builtin(data, vec![Value::String("pdf".into())]));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_normalization_name_value_pair() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = block_on(hist_builtin(
            data,
            vec![
                Value::String("Normalization".into()),
                Value::String("probability".into()),
            ],
        ));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_bin_edges_option() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.1, 0.4, 0.7]));
        let edges = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = block_on(hist_builtin(
            data,
            vec![Value::String("BinEdges".into()), edges],
        ));
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_evaluate_returns_counts_and_centers() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.2, 0.8, 1.0]));
        let eval = block_on(evaluate_async(data, &[])).expect("hist evaluate");
        let counts = match eval.counts_value() {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("unexpected value: {other:?}"),
        };
        assert_eq!(counts.len(), 10);
        let centers = match eval.centers_value() {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("unexpected centers: {other:?}"),
        };
        assert_eq!(centers.len(), 10);
    }

    #[test]
    fn hist_matrix_input_returns_one_count_column_per_data_column() {
        setup_plot_tests();
        let data = Value::Tensor(
            Tensor::new(vec![1.0, 1.0, 2.0, 10.0, 10.0, 20.0], vec![3, 2]).expect("hist matrix"),
        );
        let bins = [Value::Tensor(
            Tensor::new(vec![1.0, 10.0], vec![2]).expect("hist centers"),
        )];
        let eval = block_on(evaluate_async(data, &bins)).expect("matrix hist evaluation");
        let counts = Tensor::try_from(&eval.counts_value()).expect("matrix counts");
        let centers = Tensor::try_from(&eval.centers_value()).expect("matrix centers");
        assert_eq!(counts.shape(), &[2, 2]);
        assert_eq!(counts.materialize_f64(), vec![3.0, 0.0, 0.0, 3.0]);
        assert_eq!(centers.shape(), &[2, 2]);
        assert_eq!(centers.materialize_f64(), vec![1.0, 10.0, 1.0, 10.0]);
        assert_eq!(eval.charts.len(), 2);
        assert_eq!(
            (eval.charts[0].group_index, eval.charts[0].group_count),
            (0, 2)
        );
        assert_eq!(
            (eval.charts[1].group_index, eval.charts[1].group_count),
            (1, 2)
        );
        // Ordinary row and column vectors are one distribution, not one
        // single-sample chart per row-vector element. Keep the matrix witness
        // above to distinguish actual column expansion from vector treatment.
        for shape in [vec![1, 6], vec![6, 1]] {
            let source = Tensor::new(vec![1.0, 1.0, 2.0, 10.0, 10.0, 20.0], shape).unwrap();
            let vector = block_on(evaluate_async(Value::Tensor(source), &bins)).unwrap();
            let counts = Tensor::try_from(&vector.counts_value()).unwrap();
            assert_eq!(counts.shape(), &[1, 2]);
            assert_eq!(counts.materialize_f64(), vec![3.0, 3.0]);
            assert_eq!(vector.charts.len(), 1);
        }
    }

    #[test]
    fn hist_matrix_automatic_bins_are_selected_per_column() {
        setup_plot_tests();
        let data = Value::Tensor(
            Tensor::new(vec![0.0, 1.0, 2.0, 100.0, 110.0, 120.0], vec![3, 2]).expect("hist matrix"),
        );
        let eval = block_on(evaluate_async(data, &[])).expect("matrix hist evaluation");
        let centers = Tensor::try_from(&eval.centers_value()).expect("matrix centers");
        assert_eq!(centers.shape(), &[10, 2]);
        let values = centers.materialize_f64();
        assert!(values[0] < 1.0);
        assert!(values[10] > 100.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_supports_numbins_option() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0, 1.5]));
        let args = vec![Value::String("NumBins".into()), Value::Num(4.0)];
        let eval = block_on(evaluate_async(data, &args)).expect("hist evaluate");
        let centers = match eval.centers_value() {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("unexpected centers: {other:?}"),
        };
        assert_eq!(centers.len(), 4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_supports_binwidth_and_limits() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.1, 0.2, 0.6, 0.8]));
        let args = vec![
            Value::String("BinWidth".into()),
            Value::Num(0.5),
            Value::String("BinLimits".into()),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
        ];
        let eval = block_on(evaluate_async(data, &args)).expect("hist evaluate");
        let centers = match eval.centers_value() {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("unexpected centers: {other:?}"),
        };
        assert_eq!(centers.len(), 2);
        assert!((centers[0] - 0.25).abs() < 1e-9);

        // A successful bounds stage can realize a clipped final bin. Continue
        // counting with those exact retained bins, not fresh host-data bounds.
        let mut options = HistBinOptions::new(HistBinSpec::Auto);
        options.bin_width = Some(0.6);
        let bins = realize_bins(
            &options,
            4,
            Some(&HistDataStats {
                min: Some(0.0),
                max: Some(1.0),
            }),
            Some(0.0),
        )
        .unwrap();
        assert_ne!(bins.widths[0], bins.widths[1]);
        let continued = build_histogram_chart_with_bins(
            &[0.1, 0.2, 0.6, 0.8],
            &bins,
            HistNormalization::Count,
            Some(&[1.0, 2.0, 3.0, 4.0]),
            10.0,
        )
        .unwrap();
        assert_eq!(continued.counts, vec![3.0, 7.0]);
        assert_eq!(&continued.centers, &bins.centers);
        assert_eq!(continued.output_shape, vec![1, 2]);
        for centers in [vec![0.0, 1.0, 4.0], vec![0.0, 1.0, 2.0]] {
            let bins = realize_bins(
                &HistBinOptions::new(HistBinSpec::Centers(centers)),
                4,
                None,
                None,
            )
            .unwrap();
            assert!(
                ExactHistogramEdges::new(&bins.edges).is_ok(),
                "both uniform and nonuniform actual edges are valid numerical inputs"
            );
        }
        let tiny_weight = build_histogram_chart_with_bins(
            &[0.1],
            &bins,
            HistNormalization::Probability,
            Some(&[1.0e-10]),
            1.0e-10,
        )
        .unwrap();
        assert_eq!(tiny_weight.counts, vec![1.0, 0.0]);
        let exact_weight = 1.0 + 2.0f64.powi(-30);
        let precise = build_histogram_chart_with_bins(
            &[0.1],
            &bins,
            HistNormalization::Count,
            Some(&[exact_weight]),
            exact_weight,
        )
        .unwrap();
        assert_eq!(precise.counts, vec![exact_weight, 0.0]);

        // A unit-sized absolute tolerance used to admit these materially
        // different tiny intervals as uniformly spaced centers.
        assert!(ensure_uniform_spacing(&[0.0, 1.0e-10, 2.1e-10]).is_err());
        assert!(ensure_uniform_spacing(&[0.1, 0.2, 0.3]).is_ok());
        let tiny_edges = RealizedBins::from_edges(vec![0.0, f64::from_bits(1), 1.0]).unwrap();
        assert_eq!(tiny_edges.widths[0], f64::from_bits(1));
        let mut tiny_pdf = vec![f64::from_bits(1)];
        apply_normalization(
            &mut tiny_pdf,
            &tiny_edges.widths[..1],
            HistNormalization::Pdf,
            f64::from_bits(1),
        );
        assert_eq!(tiny_pdf[0], f64::INFINITY);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_supports_sqrt_binmethod() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.2, 0.4, 0.6, 0.8]));
        let args = vec![
            Value::String("BinMethod".into()),
            Value::String("sqrt".into()),
        ];
        let eval = block_on(evaluate_async(data, &args)).expect("hist evaluate");
        let centers = match eval.centers_value() {
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("unexpected centers: {other:?}"),
        };
        assert!(centers.len() >= 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn apply_normalization_handles_weighted_probability() {
        setup_plot_tests();
        let mut counts = vec![2.0, 4.0];
        let widths = vec![1.0, 1.0];
        apply_normalization(&mut counts, &widths, HistNormalization::Probability, 6.0);
        assert!((counts[0] - 2.0 / 6.0).abs() < 1e-12);
        assert!((counts[1] - 4.0 / 6.0).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn apply_normalization_handles_weighted_pdf() {
        setup_plot_tests();
        let mut counts = vec![5.0];
        let widths = vec![0.5];
        apply_normalization(&mut counts, &widths, HistNormalization::Pdf, 10.0);
        // PDF height = weight / (total_weight * bin_width) = 5 / (10 * 0.5) = 1
        assert!((counts[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hist_type_defaults_to_row_vector() {
        use runmat_types::*;
        for (dims, expected) in [
            (vec![Some(1), Some(8)], vec![Some(1), Some(10)]),
            (vec![Some(8), Some(1)], vec![Some(1), Some(10)]),
            (vec![Some(8), Some(2)], vec![Some(10), Some(2)]),
        ] {
            let input = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(dims),
                StorageFact::Dense,
            );
            let request = CallRequest {
                arguments: vec![input],
                literals: LiteralContext::new(vec![]),
                outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
            };
            let result =
                runmat_builtins::infer_catalog_call(&runmat_builtins::HIST_CATALOG_ENTRY, &request);
            assert!(result.diagnostics.is_empty());
            assert_eq!(result.outputs.len(), 2);
            assert!(result
                .outputs
                .iter()
                .all(|fact| fact.shape == ShapeFact::from(expected.clone())
                    && fact.residency == ResidencyFact::Host));
        }
    }

    #[test]
    fn hist_type_uses_bin_centers_length() {
        use runmat_types::*;
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(1), Some(8)]),
            StorageFact::Dense,
        );
        let centers = ValueFact::proven(
            input.kind.clone(),
            ShapeFact::from(vec![Some(1), Some(5)]),
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![input, centers],
            literals: LiteralContext::new(vec![]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        let result =
            runmat_builtins::infer_catalog_call(&runmat_builtins::HIST_CATALOG_ENTRY, &request);
        assert!(result.diagnostics.is_empty());
        assert!(result
            .outputs
            .iter()
            .all(|fact| fact.shape == ShapeFact::from(vec![Some(1), Some(5)])));
    }

    #[test]
    fn hist_descriptor_includes_core_signatures() {
        let labels: Vec<&str> = runmat_builtins::HIST_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"N = hist(X)"));
        assert!(labels.contains(&"N = hist(X, bins)"));
        assert!(labels.contains(&"N = hist(X, Name, Value, ...)"));
    }

    #[test]
    fn hist_missing_option_value_uses_stable_identifier() {
        let result = block_on(evaluate_async(
            Value::Tensor(tensor_from(&[1.0, 2.0, 3.0])),
            &[Value::String("BinEdges".into())],
        ));
        let err = match result {
            Ok(_) => panic!("expected histogram parse failure"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), HIST_ERROR_INVALID_ARGUMENT.identifier);
    }
}
