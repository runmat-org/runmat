//! MATLAB-compatible `hist` builtin.

use glam::{Vec3, Vec4};
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::bar::{BarGpuInputs, BarGpuParams, BarLayoutMode, BarOrientation};
use runmat_plot::gpu::histogram::{
    HistogramGpuInputs, HistogramGpuOutput, HistogramGpuParams, HistogramGpuWeights,
    HistogramNormalizationMode,
};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::BarChart;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::bar::apply_bar_style;
use super::common::{gather_tensor_from_gpu, numeric_vector, value_as_f64};
use super::gpu_helpers::axis_bounds;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_bar_style_args, BarStyle, BarStyleDefaults};

use crate::{BuiltinResult, RuntimeError};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "hist",
        builtin_path = "crate::builtins::plotting::hist"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
const BUILTIN_NAME: &str = "hist";

pub const DOC_MD: &str = r#"---
title: "hist"
category: "plotting"
keywords: ["hist", "histogram", "frequency", "gpuArray"]
summary: "Render MATLAB-compatible histograms using automatic or user-specified bin counts."
references:
  - https://www.mathworks.com/help/matlab/ref/hist.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["single"]
  broadcasting: "none"
  notes: "Single-precision gpuArray inputs stay on the device and feed the shared WebGPU renderer; other data gathers automatically."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::plotting::hist::tests"
---

# What does `hist` do?
`hist(data, nbins)` counts the number of observations that fall into equally spaced bins spanning
the data range. When `nbins` is omitted, MATLAB (and RunMat) default to `floor(sqrt(numel(data)))`
bins. The resulting frequencies are displayed as a bar chart.

## Behaviour highlights
- Inputs must be numeric vectors. Empty inputs return a figure with zero-height bins.
- `nbins` may be provided as a numeric scalar or a vector of equally spaced bin centers. Non-positive
  or non-finite values raise MATLAB-style errors.
- A `'BinEdges'` name-value pair accepts a strictly increasing vector of edges (uniform or
  non-uniform). When the edges are evenly spaced the gpuArray path remains zero-copy; other cases
  fall back to the CPU implementation automatically.
- Normalization can be supplied either as the second positional argument or via the
  `'Normalization'` name-value pair. Supported values are `"count"`, `"probability"`, and `"pdf"`.
- Single-precision gpuArray inputs stay on the device: bin counts are computed by a compute shader and
  bar vertices are emitted directly into the shared WebGPU context whenever the bins are uniform.
  Other cases gather automatically (full normalization semantics still apply in either path).

## Examples
```matlab
data = randn(1, 1000);
hist(data, 20);
```
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::hist")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "hist",
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
    notes: "Single-precision gpuArray vectors render zero-copy when the shared renderer is active; other contexts gather first.",
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

const HIST_BAR_WIDTH: f32 = 0.95;
const HIST_DEFAULT_COLOR: Vec4 = Vec4::new(0.15, 0.5, 0.8, 0.95);
const HIST_DEFAULT_LABEL: &str = "Frequency";

fn hist_err(message: impl Into<String>) -> RuntimeError {
    plotting_error(BUILTIN_NAME, message)
}

struct HistComputation {
    counts: Vec<f64>,
    centers: Vec<f64>,
    chart: BarChart,
}

/// Captures the evaluated histogram so both the renderer and MATLAB outputs share the same data.
pub struct HistEvaluation {
    counts: Tensor,
    #[allow(dead_code)]
    centers: Tensor,
    chart: BarChart,
    normalization: HistNormalization,
}

impl HistEvaluation {
    fn new(
        counts: Vec<f64>,
        centers: Vec<f64>,
        chart: BarChart,
        normalization: HistNormalization,
    ) -> BuiltinResult<Self> {
        if counts.len() != centers.len() {
            return Err(hist_err("hist: mismatch between counts and bin centers"));
        }
        let cols = counts.len();
        let shape = vec![1, cols];
        let counts_tensor = Tensor::new(counts, shape.clone())?;
        let centers_tensor = Tensor::new(centers, shape)?;
        Ok(Self {
            counts: counts_tensor,
            centers: centers_tensor,
            chart,
            normalization,
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
        let y_label = match self.normalization {
            HistNormalization::Count => "Count",
            HistNormalization::Probability => "Probability",
            HistNormalization::Pdf => "PDF",
        };
        let mut chart_opt = Some(self.chart.clone());
        let opts = PlotRenderOptions {
            title: "Histogram",
            x_label: "Bin",
            y_label,
            ..Default::default()
        };
        render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
            let chart = chart_opt
                .take()
                .expect("hist chart consumed exactly once at render time");
            figure.add_bar_chart_on_axes(chart, axes);
            Ok(())
        })?;
        Ok(())
    }
}

impl HistComputation {
    fn into_evaluation(self, normalization: HistNormalization) -> BuiltinResult<HistEvaluation> {
        HistEvaluation::new(self.counts, self.centers, self.chart, normalization)
    }
}

#[derive(Clone)]
enum HistBinSpec {
    Auto,
    Count(usize),
    Centers(Vec<f64>),
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

    fn is_uniform(&self) -> bool {
        match &self.spec {
            HistBinSpec::Edges(edges) => uniform_edge_width(edges).is_some(),
            _ => true,
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
                let len: usize = handle.shape.iter().product();
                if len != expected_len {
                    return Err(hist_err(format!(
                        "hist: Weights must contain {expected_len} elements (got {len})"
                    )));
                }
                Ok(HistWeightsInput::Gpu(handle))
            }
            other => {
                let tensor = Tensor::try_from(&other).map_err(|e| hist_err(format!("hist: Weights {e}")))?;
                if tensor.data.len() != expected_len {
                    return Err(hist_err(format!(
                        "hist: Weights must contain {expected_len} elements (got {})",
                        tensor.data.len()
                    )));
                }
                Ok(HistWeightsInput::Host(tensor))
            }
        }
    }

    fn resolve_for_cpu(
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
                let tensor = gather_tensor_from_gpu(handle.clone(), context)?;
                let values = numeric_vector(tensor);
                let total = values.iter().copied().sum::<f64>();
                Ok((Some(values), total))
            }
        }
    }

    fn total_weight_hint(&self, sample_len: usize) -> Option<f64> {
        match self {
            HistWeightsInput::None => Some(sample_len as f64),
            HistWeightsInput::Host(tensor) => {
                let values = numeric_vector(tensor.clone());
                Some(values.iter().copied().sum::<f64>())
            }
            HistWeightsInput::Gpu(_) => None,
        }
    }

    fn to_gpu_weights(&self, sample_len: usize) -> BuiltinResult<HistogramGpuWeights> {
        match self {
            HistWeightsInput::None => Ok(HistogramGpuWeights::Uniform {
                total_weight: sample_len as f32,
            }),
            HistWeightsInput::Host(tensor) => {
                let values = numeric_vector(tensor.clone());
                let total = values.iter().copied().sum::<f64>() as f32;
                match tensor.dtype {
                    NumericDType::F32 => {
                        let data: Vec<f32> = values.iter().map(|v| *v as f32).collect();
                        Ok(HistogramGpuWeights::HostF32 {
                            data,
                            total_weight: total,
                        })
                    }
                    NumericDType::F64 => Ok(HistogramGpuWeights::HostF64 {
                        data: values,
                        total_weight: total,
                    }),
                }
            }
            HistWeightsInput::Gpu(handle) => {
                let exported = runmat_accelerate_api::export_wgpu_buffer(handle)
                    .ok_or_else(|| hist_err("hist: unable to export GPU weights"))?;
                match exported.precision {
                    ProviderPrecision::F32 => Ok(HistogramGpuWeights::GpuF32 {
                        buffer: exported.buffer.clone(),
                    }),
                    ProviderPrecision::F64 => Ok(HistogramGpuWeights::GpuF64 {
                        buffer: exported.buffer.clone(),
                    }),
                }
            }
        }
    }
}

#[runtime_builtin(
    name = "hist",
    category = "plotting",
    summary = "Plot a histogram with MATLAB-compatible defaults.",
    keywords = "hist,histogram,frequency",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::hist"
)]
pub fn hist_builtin(data: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let evaluation = evaluate(data, &rest)?;
    evaluation.render_plot()?;
    Ok(evaluation.counts_value())
}

/// Evaluate the histogram inputs once so renderers and MATLAB outputs share the same data.
pub fn evaluate(data: Value, rest: &[Value]) -> BuiltinResult<HistEvaluation> {
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

    let computation = if !bar_style.requires_cpu_path() {
        if let Some(handle) = input.as_ref().and_then(|value| value.gpu_handle()) {
            if bin_options.is_uniform() {
                match build_histogram_gpu_chart(
                    handle,
                    &bin_options,
                    sample_len,
                    normalization,
                    &bar_style,
                    &weights_input,
                ) {
                    Ok(chart) => Some(chart),
                    Err(err) => {
                        warn!("hist GPU path unavailable: {err}");
                        None
                    }
                }
            } else {
                None
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
            let data_arg = input.take().expect("hist input consumed once");
            let tensor = data_arg.into_tensor("hist")?;
            let samples = numeric_vector(tensor);
            let (weight_values, total_weight) =
                weights_input.resolve_for_cpu("hist weights", sample_len)?;
            build_histogram_chart(
                samples,
                &bin_options,
                normalization,
                weight_values.as_deref(),
                total_weight,
            )?
        }
    };

    let mut evaluation = computation.into_evaluation(normalization)?;
    apply_bar_style(&mut evaluation.chart, &bar_style, HIST_DEFAULT_LABEL);
    Ok(evaluation)
}

fn parse_hist_arguments(
    sample_len: usize,
    args: &[Value],
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

        let Some(key) = value_as_string(arg) else {
            style_args.extend_from_slice(&args[idx..]);
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
                    return Err(hist_err("hist: specify either bins argument or 'BinEdges', not both"));
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
                    return Err(hist_err("hist: NumBins cannot be combined with explicit bins"));
                }
                let count = parse_num_bins_value(&value)?;
                ensure_spec_compatible(&HistBinSpec::Count(count), &bin_options, "NumBins")?;
                bin_options.spec = HistBinSpec::Count(count);
                bin_set = true;
            }
            "binwidth" => {
                if bin_set {
                    return Err(hist_err("hist: BinWidth cannot be combined with explicit bins"));
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
    let spec = match arg {
        None => HistBinSpec::Auto,
        Some(Value::Tensor(tensor)) => parse_center_vector(tensor)?,
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
    uniform_width: Option<f64>,
}

impl RealizedBins {
    fn from_edges(edges: Vec<f64>) -> BuiltinResult<Self> {
        if edges.len() < 2 {
            return Err(hist_err("hist: bin definitions must contain at least two edges"));
        }
        let widths = widths_from_edges(&edges);
        let labels = histogram_labels_from_edges(&edges);
        let centers = centers_from_edges(&edges);
        let uniform_width = if widths.iter().all(|w| approx_equal(*w, widths[0])) {
            Some(widths[0])
        } else {
            None
        };
        Ok(Self {
            edges,
            widths,
            labels,
            centers,
            uniform_width,
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
        .map(|pair| (pair[1] - pair[0]).max(f64::MIN_POSITIVE))
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
                return Err(hist_err("hist: internal integer bin method misuse"))
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
    if matches!(new_spec, HistBinSpec::Centers(_) | HistBinSpec::Edges(_))
        && (options.bin_width.is_some()
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
        HistBinSpec::Centers(_) | HistBinSpec::Edges(_)
    ) {
        return Err(hist_err(format!(
            "hist: {source} cannot be combined with explicit bin centers or edges"
        )));
    }
    Ok(())
}

fn parse_num_bins_value(value: &Value) -> BuiltinResult<usize> {
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
        return Err(hist_err("hist: BinLimits must contain exactly two elements"));
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
    let Some(text) = value_as_string(value) else {
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
    let values = numeric_vector(tensor);
    if values.is_empty() {
        return Err(hist_err("hist: bin center array cannot be empty"));
    }
    if values.len() == 1 {
        return parse_bin_count_value(values[0]);
    }
    validate_monotonic(&values)?;
    ensure_uniform_spacing(&values)?;
    Ok(HistBinSpec::Centers(values))
}

fn parse_bin_count_value(value: f64) -> BuiltinResult<HistBinSpec> {
    if value.is_finite() && value > 0.0 {
        Ok(HistBinSpec::Count(value.round() as usize))
    } else {
        Err(hist_err("hist: bin count must be positive"))
    }
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
                return Err(hist_err("hist: 'BinEdges' must contain at least two elements"));
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
    let tol = first.abs().max(1.0) * 1e-6;
    for diff in diffs {
        if (diff - first).abs() > tol {
            return Err(hist_err("hist: bin centers must be evenly spaced"));
        }
    }
    Ok(())
}

fn uniform_edge_width(edges: &[f64]) -> Option<f64> {
    if edges.len() < 2 {
        return None;
    }
    let mut diffs = edges.windows(2).map(|pair| pair[1] - pair[0]);
    let first = diffs.next().unwrap();
    if first <= 0.0 || !first.is_finite() {
        return None;
    }
    let tol = first.abs().max(1.0) * 1e-5;
    for diff in diffs {
        if diff <= 0.0 || !diff.is_finite() {
            return None;
        }
        if (diff - first).abs() > tol {
            return None;
        }
    }
    Some(first)
}

fn parse_hist_normalization(arg: Option<Value>) -> BuiltinResult<HistNormalization> {
    match arg {
        None => Ok(HistNormalization::Count),
        Some(Value::String(s)) => parse_norm_string(&s),
        Some(Value::CharArray(chars)) => {
            let text: String = chars.data.iter().collect();
            parse_norm_string(&text)
        }
        Some(value) => {
            if let Some(text) = value_as_string(&value) {
                parse_norm_string(&text)
            } else {
                Err(hist_err("hist: normalization must be 'count', 'probability', or 'pdf'"))
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

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(chars) => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn default_bin_count(sample_len: usize) -> usize {
    ((sample_len as f64).sqrt().floor() as usize).max(1)
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
    build_hist_cpu_result(&bins, counts)
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
    let mut bar =
        BarChart::new(bins.labels.clone(), counts.clone()).map_err(|err| hist_err(format!("hist: {err}")))?;
    bar.label = Some(HIST_DEFAULT_LABEL.to_string());
    Ok(HistComputation {
        counts,
        centers: bins.centers.clone(),
        chart: bar,
    })
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
        return Err(hist_err("hist: bin centers must contain at least one element"));
    }
    if centers.len() == 1 {
        let half = 0.5;
        return Ok(vec![centers[0] - half, centers[0] + half]);
    }
    validate_monotonic(centers)?;
    let mut edges = Vec::with_capacity(centers.len() + 1);
    edges.push(centers[0] - (centers[1] - centers[0]) * 0.5);
    for pair in centers.windows(2) {
        edges.push((pair[0] + pair[1]) * 0.5);
    }
    edges.push(
        centers[centers.len() - 1]
            + (centers[centers.len() - 1] - centers[centers.len() - 2]) * 0.5,
    );
    Ok(edges)
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
        .map(|pair| (pair[0] + pair[1]) * 0.5)
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
            let total = total_weight.max(f64::EPSILON);
            for count in counts {
                *count /= total;
            }
        }
        HistNormalization::Pdf => {
            let total = total_weight.max(f64::EPSILON);
            for (count, width) in counts.iter_mut().zip(widths.iter()) {
                let w = width.max(f64::MIN_POSITIVE);
                *count /= total * w;
            }
        }
    }
}

fn build_histogram_gpu_chart(
    values: &GpuTensorHandle,
    bin_options: &HistBinOptions,
    sample_len: usize,
    normalization: HistNormalization,
    style: &BarStyle,
    weights: &HistWeightsInput,
) -> BuiltinResult<HistComputation> {
    let context = runmat_plot::shared_wgpu_context()
        .ok_or_else(|| hist_err("hist: plotting GPU context unavailable"))?;
    let exported = runmat_accelerate_api::export_wgpu_buffer(values)
        .ok_or_else(|| hist_err("hist: unable to export GPU data"))?;
    if exported.len == 0 {
        let total_hint = weights
            .total_weight_hint(sample_len)
            .unwrap_or(sample_len as f64);
        return build_empty_histogram_chart(bin_options, normalization, sample_len, total_hint);
    }

    let sample_count_u32 = u32::try_from(exported.len)
        .map_err(|_| hist_err("hist: sample count exceeds supported range"))?;
    let gpu_weights = weights.to_gpu_weights(sample_len)?;
    let (min_value_f32, max_value_f32) = axis_bounds(values, "hist")?;
    let stats = HistDataStats {
        min: Some(min_value_f32 as f64),
        max: Some(max_value_f32 as f64),
    };
    let bins = realize_bins(
        bin_options,
        sample_len,
        Some(&stats),
        Some(min_value_f32 as f64),
    )?;
    let Some(uniform_width_f64) = bins.uniform_width else {
        return Err(hist_err("hist: GPU rendering currently requires uniform bin edges"));
    };
    let uniform_width = uniform_width_f64 as f32;
    let bin_count_u32 = u32::try_from(bins.bin_count())
        .map_err(|_| hist_err("hist: bin count exceeds supported range for GPU execution"))?;

    let histogram_inputs = HistogramGpuInputs {
        samples: exported.buffer.clone(),
        sample_count: sample_count_u32,
        scalar: ScalarType::from_is_f64(exported.precision == ProviderPrecision::F64),
        weights: gpu_weights,
    };
    let histogram_params = HistogramGpuParams {
        min_value: bins.edges[0] as f32,
        inv_bin_width: 1.0 / uniform_width,
        bin_count: bin_count_u32,
    };
    let normalization_mode = match normalization {
        HistNormalization::Count => HistogramNormalizationMode::Count,
        HistNormalization::Probability => HistogramNormalizationMode::Probability,
        HistNormalization::Pdf => HistogramNormalizationMode::Pdf {
            bin_width: uniform_width.max(f32::MIN_POSITIVE),
        },
    };

    let histogram_output = runmat_plot::gpu::histogram::histogram_values_buffer(
        &context.device,
        &context.queue,
        histogram_inputs,
        &histogram_params,
        normalization_mode,
    )
    .map_err(|e| hist_err(format!("hist: failed to build GPU histogram counts: {e}")))?;

    let HistogramGpuOutput {
        values_buffer,
        total_weight,
    } = histogram_output;

    let bar_inputs = BarGpuInputs {
        values_buffer,
        row_count: bin_count_u32,
        scalar: ScalarType::F32,
    };
    let bar_params = BarGpuParams {
        color: style.face_rgba(),
        bar_width: style.bar_width,
        series_index: 0,
        series_count: 1,
        group_index: 0,
        group_count: 1,
        orientation: BarOrientation::Vertical,
        layout: BarLayoutMode::Grouped,
    };

    let gpu_vertices = runmat_plot::gpu::bar::pack_vertices_from_values(
        &context.device,
        &context.queue,
        &bar_inputs,
        &bar_params,
    )
    .map_err(|e| hist_err(format!("hist: failed to build GPU vertices: {e}")))?;

    let bin_count = bins.bin_count();
    let normalization_scale = match normalization {
        HistNormalization::Count => 1.0,
        HistNormalization::Probability => {
            if total_weight <= f32::EPSILON {
                0.0
            } else {
                1.0 / total_weight
            }
        }
        HistNormalization::Pdf => {
            if total_weight <= f32::EPSILON {
                0.0
            } else {
                1.0 / (total_weight * uniform_width)
            }
        }
    };
    let bounds = histogram_bar_bounds(
        bin_count,
        total_weight,
        normalization_scale,
        style.bar_width,
    );
    let vertex_count = gpu_vertices.vertex_count;
    let mut bar = BarChart::from_gpu_buffer(
        bins.labels.clone(),
        bin_count,
        gpu_vertices,
        vertex_count,
        bounds,
        style.face_rgba(),
        style.bar_width,
    );
    bar.label = Some(HIST_DEFAULT_LABEL.to_string());
    let counts_f32 = runmat_plot::gpu::util::readback_f32_buffer(
        &context.device,
        bar_inputs.values_buffer.as_ref(),
        bin_count,
    )
    .map_err(|e| hist_err(format!("hist: failed to read GPU histogram counts: {e}")))?;
    let counts: Vec<f64> = counts_f32.iter().map(|v| *v as f64).collect();

    Ok(HistComputation {
        counts,
        centers: bins.centers.clone(),
        chart: bar,
    })
}

fn histogram_bar_bounds(
    bins: usize,
    total_weight: f32,
    normalization_scale: f32,
    bar_width: f32,
) -> BoundingBox {
    let min_x = 1.0 - bar_width * 0.5;
    let max_x = bins as f32 + bar_width * 0.5;
    let max_y = total_weight * normalization_scale;
    let max_y = if max_y.is_finite() && max_y > 0.0 {
        max_y
    } else {
        1.0
    };
    BoundingBox::new(Vec3::new(min_x, 0.0, 0.0), Vec3::new(max_x, max_y, 0.0))
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
                let tensor = Tensor::try_from(&other).map_err(|e| hist_err(format!("hist: {e}")))?;
                Ok(Self::Host(tensor))
            }
        }
    }

    fn gpu_handle(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Gpu(handle) => Some(handle),
            Self::Host(_) => None,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Host(tensor) => tensor.data.len(),
            Self::Gpu(handle) => handle.shape.iter().product(),
        }
    }

    fn into_tensor(self, context: &'static str) -> BuiltinResult<Tensor> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Gpu(handle) => gather_tensor_from_gpu(handle, context),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use crate::RuntimeError;

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor {
            data: data.to_vec(),
            shape: vec![data.len()],
            rows: data.len(),
            cols: 1,
            dtype: runmat_builtins::NumericDType::F64,
        }
    }

    fn assert_plotting_unavailable(err: &RuntimeError) {
        let lower = err.to_string().to_lowercase();
        assert!(
            lower.contains("plotting is unavailable") || lower.contains("non-main thread"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_respects_bin_argument() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[1.0, 2.0, 3.0, 4.0]));
        let bins = vec![Value::from(2.0)];
        let result = hist_builtin(data, bins);
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_bin_centers_vector() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0, 1.5]));
        let centers = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0]));
        let result = hist_builtin(data, vec![centers]);
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_probability_normalization() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = hist_builtin(
            data,
            vec![Value::from(3.0), Value::String("probability".into())],
        );
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_string_only_normalization() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = hist_builtin(data, vec![Value::String("pdf".into())]);
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_normalization_name_value_pair() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = hist_builtin(
            data,
            vec![
                Value::String("Normalization".into()),
                Value::String("probability".into()),
            ],
        );
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_accepts_bin_edges_option() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.1, 0.4, 0.7]));
        let edges = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = hist_builtin(data, vec![Value::String("BinEdges".into()), edges]);
        if let Err(flow) = result {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_evaluate_returns_counts_and_centers() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.2, 0.8, 1.0]));
        let eval = evaluate(data, &[]).expect("hist evaluate");
        let counts = match eval.counts_value() {
            Value::Tensor(tensor) => tensor.data,
            other => panic!("unexpected value: {other:?}"),
        };
        assert_eq!(counts.len(), 2);
        let centers = match eval.centers_value() {
            Value::Tensor(tensor) => tensor.data,
            other => panic!("unexpected centers: {other:?}"),
        };
        assert_eq!(centers.len(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hist_supports_numbins_option() {
        setup_plot_tests();
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0, 1.5]));
        let args = vec![Value::String("NumBins".into()), Value::Num(4.0)];
        let eval = evaluate(data, &args).expect("hist evaluate");
        let centers = match eval.centers_value() {
            Value::Tensor(tensor) => tensor.data,
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
        let eval = evaluate(data, &args).expect("hist evaluate");
        let centers = match eval.centers_value() {
            Value::Tensor(tensor) => tensor.data,
            other => panic!("unexpected centers: {other:?}"),
        };
        assert_eq!(centers.len(), 2);
        assert!((centers[0] - 0.25).abs() < 1e-9);
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
        let eval = evaluate(data, &args).expect("hist evaluate");
        let centers = match eval.centers_value() {
            Value::Tensor(tensor) => tensor.data,
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
}
