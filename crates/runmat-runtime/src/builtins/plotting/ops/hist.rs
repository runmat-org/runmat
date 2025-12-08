//! MATLAB-compatible `hist` builtin.

use glam::{Vec3, Vec4};
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::bar::{BarGpuInputs, BarGpuParams, BarOrientation};
use runmat_plot::gpu::histogram::{HistogramGpuInputs, HistogramGpuParams};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::BarChart;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

use super::bar::apply_bar_style;
use super::common::{gather_tensor_from_gpu, numeric_vector, value_as_f64};
use super::gpu_helpers::axis_bounds;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{parse_bar_style_args, BarStyle, BarStyleDefaults};

#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;

#[cfg(feature = "doc_export")]
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

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "hist",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "hist terminates fusion graphs and produces I/O.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("hist", DOC_MD);

const HIST_BAR_WIDTH: f32 = 0.95;
const HIST_DEFAULT_COLOR: Vec4 = Vec4::new(0.15, 0.5, 0.8, 0.95);
const HIST_DEFAULT_LABEL: &str = "Frequency";

#[derive(Clone)]
enum HistBinSpec {
    Auto,
    Count(usize),
    Centers(Vec<f64>),
    Edges(Vec<f64>),
}

impl HistBinSpec {
    fn bin_count(&self, sample_len: usize) -> usize {
        match self {
            HistBinSpec::Auto => default_bin_count(sample_len),
            HistBinSpec::Count(count) => (*count).max(1),
            HistBinSpec::Centers(centers) => centers.len().max(1),
            HistBinSpec::Edges(edges) => edges.len().saturating_sub(1).max(1),
        }
    }

    fn is_uniform(&self) -> bool {
        match self {
            HistBinSpec::Edges(edges) => uniform_edge_width(edges).is_some(),
            _ => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HistNormalization {
    Count,
    Probability,
    Pdf,
}

impl Default for HistNormalization {
    fn default() -> Self {
        HistNormalization::Count
    }
}

#[runtime_builtin(
    name = "hist",
    category = "plotting",
    summary = "Plot a histogram with MATLAB-compatible defaults.",
    keywords = "hist,histogram,frequency",
    sink = true
)]
pub fn hist_builtin(data: Value, rest: Vec<Value>) -> Result<String, String> {
    let mut input = Some(HistInput::from_value(data)?);
    let sample_len = input.as_ref().map(|value| value.len()).unwrap_or(0);
    let (bin_spec, normalization, style_args) = parse_hist_arguments(sample_len, &rest)?;
    let defaults = BarStyleDefaults::new(HIST_DEFAULT_COLOR, HIST_BAR_WIDTH);
    let bar_style = parse_bar_style_args("hist", &style_args, defaults)?;
    let y_label = match normalization {
        HistNormalization::Count => "Count",
        HistNormalization::Probability => "Probability",
        HistNormalization::Pdf => "PDF",
    };
    let opts = PlotRenderOptions {
        title: "Histogram",
        x_label: "Bin",
        y_label,
        ..Default::default()
    };
    render_active_plot(opts, move |figure, axes| {
        let data_arg = input.take().expect("hist input consumed once");
        if !bar_style.requires_cpu_path() {
            if let Some(handle) = data_arg.gpu_handle() {
                if bin_spec.is_uniform() {
                    match build_histogram_gpu_chart(
                        handle,
                        &bin_spec,
                        sample_len,
                        normalization,
                        &bar_style,
                    ) {
                        Ok(mut bar) => {
                            apply_bar_style(&mut bar, &bar_style, HIST_DEFAULT_LABEL);
                            figure.add_bar_chart_on_axes(bar, axes);
                            return Ok(());
                        }
                        Err(err) => warn!("hist GPU path unavailable: {err}"),
                    }
                }
            }
        }
        let tensor = data_arg.into_tensor("hist")?;
        let samples = numeric_vector(tensor);
        let mut bar = build_histogram_chart(samples, &bin_spec, normalization)?;
        apply_bar_style(&mut bar, &bar_style, HIST_DEFAULT_LABEL);
        figure.add_bar_chart_on_axes(bar, axes);
        Ok(())
    })
}

fn parse_hist_arguments(
    sample_len: usize,
    args: &[Value],
) -> Result<(HistBinSpec, HistNormalization, Vec<Value>), String> {
    let mut idx = 0usize;
    let mut bin_spec = HistBinSpec::Auto;
    let mut normalization = HistNormalization::Count;
    let mut bin_set = false;
    let mut norm_set = false;
    let mut style_args = Vec::new();

    while idx < args.len() {
        let arg = &args[idx];
        if !bin_set && is_bin_candidate(arg) {
            bin_spec = parse_hist_bins(Some(arg.clone()), sample_len)?;
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
            return Err(format!("hist: missing value for '{key}' option"));
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
                    return Err(
                        "hist: specify either bins argument or 'BinEdges', not both".to_string()
                    );
                }
                let edges = parse_bin_edges_value(value)?;
                bin_spec = HistBinSpec::Edges(edges);
                bin_set = true;
            }
            _ => {
                style_args.push(arg.clone());
                style_args.push(value);
            }
        }
        idx += 2;
    }

    Ok((bin_spec, normalization, style_args))
}

fn parse_hist_bins(arg: Option<Value>, sample_len: usize) -> Result<HistBinSpec, String> {
    let spec = match arg {
        None => HistBinSpec::Auto,
        Some(Value::Tensor(tensor)) => parse_center_vector(tensor)?,
        Some(Value::GpuTensor(_)) => {
            return Err("hist: bin definitions must reside on the host".to_string())
        }
        Some(other) => {
            if let Some(numeric) = value_as_f64(&other) {
                parse_bin_count_value(numeric)?
            } else {
                return Err(
                    "hist: bin argument must be a scalar count or a vector of centers".to_string(),
                );
            }
        }
    };
    Ok(match spec {
        HistBinSpec::Count(0) => HistBinSpec::Count(default_bin_count(sample_len)),
        other => other,
    })
}

fn parse_center_vector(tensor: Tensor) -> Result<HistBinSpec, String> {
    let values = numeric_vector(tensor);
    if values.is_empty() {
        return Err("hist: bin center array cannot be empty".to_string());
    }
    if values.len() == 1 {
        return parse_bin_count_value(values[0]);
    }
    validate_monotonic(&values)?;
    ensure_uniform_spacing(&values)?;
    Ok(HistBinSpec::Centers(values))
}

fn parse_bin_count_value(value: f64) -> Result<HistBinSpec, String> {
    if value.is_finite() && value > 0.0 {
        Ok(HistBinSpec::Count(value.round() as usize))
    } else {
        Err("hist: bin count must be positive".to_string())
    }
}

fn is_bin_candidate(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
    )
}

fn try_parse_norm_literal(value: &Value) -> Option<Result<HistNormalization, String>> {
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

fn parse_bin_edges_value(value: Value) -> Result<Vec<f64>, String> {
    match value {
        Value::Tensor(tensor) => {
            let edges = numeric_vector(tensor);
            if edges.len() < 2 {
                return Err("hist: 'BinEdges' must contain at least two elements".to_string());
            }
            validate_monotonic(&edges)?;
            Ok(edges)
        }
        Value::GpuTensor(_) => Err("hist: 'BinEdges' must be provided on the host".to_string()),
        _ => Err("hist: 'BinEdges' expects a numeric vector".to_string()),
    }
}

fn ensure_uniform_spacing(values: &[f64]) -> Result<(), String> {
    if values.len() <= 2 {
        return Ok(());
    }
    let mut diffs = values.windows(2).map(|pair| pair[1] - pair[0]);
    let first = diffs.next().unwrap();
    if first <= 0.0 || !first.is_finite() {
        return Err("hist: bin centers must be strictly increasing".to_string());
    }
    let tol = first.abs().max(1.0) * 1e-6;
    for diff in diffs {
        if (diff - first).abs() > tol {
            return Err("hist: bin centers must be evenly spaced".to_string());
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

fn widths_from_edges(edges: &[f64]) -> Vec<f64> {
    edges
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).max(f64::MIN_POSITIVE))
        .collect()
}

fn parse_hist_normalization(arg: Option<Value>) -> Result<HistNormalization, String> {
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
                Err("hist: normalization must be 'count', 'probability', or 'pdf'".to_string())
            }
        }
    }
}

fn parse_norm_string(text: &str) -> Result<HistNormalization, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "count" | "counts" => Ok(HistNormalization::Count),
        "probability" | "prob" => Ok(HistNormalization::Probability),
        "pdf" => Ok(HistNormalization::Pdf),
        other => Err(format!(
            "hist: unsupported normalization '{other}' (expected 'count', 'probability', or 'pdf')"
        )),
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
    bin_spec: &HistBinSpec,
    normalization: HistNormalization,
) -> Result<BarChart, String> {
    let sample_len = data.len();
    if sample_len == 0 {
        return build_empty_histogram_chart(bin_spec, normalization, 0);
    }
    match bin_spec {
        HistBinSpec::Centers(centers) => {
            let edges = edges_from_centers(centers)?;
            let mut counts = vec![0f64; centers.len()];
            for value in data {
                let idx = find_bin_index(&edges, value);
                counts[idx] += 1.0;
            }
            let widths = widths_from_edges(&edges);
            apply_normalization(&mut counts, &widths, normalization, sample_len);
            let labels = histogram_labels_from_edges(&edges);
            build_bar(labels, counts)
        }
        HistBinSpec::Edges(edges) => {
            let mut counts = vec![0f64; edges.len().saturating_sub(1)];
            for value in data {
                let idx = find_bin_index(edges, value);
                counts[idx] += 1.0;
            }
            let widths = widths_from_edges(edges);
            apply_normalization(&mut counts, &widths, normalization, sample_len);
            let labels = histogram_labels_from_edges(edges);
            build_bar(labels, counts)
        }
        HistBinSpec::Auto | HistBinSpec::Count(_) => {
            let bin_count = bin_spec.bin_count(sample_len);
            let min = data
                .iter()
                .copied()
                .fold(f64::INFINITY, |acc, v| acc.min(v));
            let max = data
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |acc, v| acc.max(v));
            if (max - min).abs() < f64::EPSILON {
                return build_degenerate_histogram(sample_len, bin_count, min);
            }
            let edges = uniform_edges(min, max, bin_count)?;
            let mut counts = vec![0f64; bin_count];
            for value in data {
                let idx = find_bin_index(&edges, value);
                counts[idx] += 1.0;
            }
            let widths = vec![(max - min) / bin_count as f64; bin_count];
            apply_normalization(&mut counts, &widths, normalization, sample_len);
            let labels = histogram_labels_from_edges(&edges);
            build_bar(labels, counts)
        }
    }
}

fn build_empty_histogram_chart(
    bin_spec: &HistBinSpec,
    _normalization: HistNormalization,
    sample_len: usize,
) -> Result<BarChart, String> {
    match bin_spec {
        HistBinSpec::Centers(centers) => {
            let labels: Vec<String> = centers.iter().map(|c| format!("{c:.3}")).collect();
            let heights = vec![0.0; centers.len()];
            build_bar(labels, heights)
        }
        HistBinSpec::Edges(edges) => {
            let labels = histogram_labels_from_edges(edges);
            let heights = vec![0.0; edges.len().saturating_sub(1)];
            build_bar(labels, heights)
        }
        _ => {
            let bin_count = bin_spec.bin_count(sample_len);
            let labels = (0..bin_count).map(|i| format!("Bin {i}")).collect();
            let heights = vec![0.0; bin_count];
            build_bar(labels, heights)
        }
    }
}

fn build_bar(labels: Vec<String>, heights: Vec<f64>) -> Result<BarChart, String> {
    let mut bar = BarChart::new(labels, heights).map_err(|err| format!("hist: {err}"))?;
    bar.label = Some(HIST_DEFAULT_LABEL.to_string());
    Ok(bar)
}

fn validate_monotonic(values: &[f64]) -> Result<(), String> {
    if values.windows(2).all(|w| w[0] < w[1]) {
        Ok(())
    } else {
        Err("hist: values must be strictly increasing".to_string())
    }
}

fn find_bin_index(edges: &[f64], value: f64) -> usize {
    let last = edges.len() - 2;
    for i in 0..=last {
        if value < edges[i + 1] || i == last {
            return i;
        }
    }
    last
}

fn edges_from_centers(centers: &[f64]) -> Result<Vec<f64>, String> {
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

fn uniform_edges(min: f64, max: f64, bin_count: usize) -> Result<Vec<f64>, String> {
    if bin_count == 0 {
        return Err("hist: bin count must be positive".to_string());
    }
    let width = (max - min) / bin_count as f64;
    if !width.is_finite() || width <= 0.0 {
        return Err("hist: invalid bin width computed".to_string());
    }
    Ok((0..=bin_count).map(|i| min + width * i as f64).collect())
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

fn apply_normalization(
    counts: &mut [f64],
    widths: &[f64],
    normalization: HistNormalization,
    total_samples: usize,
) {
    match normalization {
        HistNormalization::Count => {}
        HistNormalization::Probability => {
            let total = total_samples.max(1) as f64;
            for count in counts {
                *count /= total;
            }
        }
        HistNormalization::Pdf => {
            let total = total_samples.max(1) as f64;
            for (count, width) in counts.iter_mut().zip(widths.iter()) {
                let w = width.max(f64::MIN_POSITIVE);
                *count = *count / (total * w);
            }
        }
    }
}

fn build_histogram_gpu_chart(
    values: &GpuTensorHandle,
    bin_spec: &HistBinSpec,
    sample_len: usize,
    normalization: HistNormalization,
    style: &BarStyle,
) -> Result<BarChart, String> {
    let context = runmat_plot::shared_wgpu_context()
        .ok_or_else(|| "hist: plotting GPU context unavailable".to_string())?;
    let exported = runmat_accelerate_api::export_wgpu_buffer(values)
        .ok_or_else(|| "hist: unable to export GPU data".to_string())?;

    if exported.len == 0 {
        return build_empty_histogram_chart(bin_spec, HistNormalization::Count, sample_len);
    }

    let sample_count_u32 = u32::try_from(exported.len)
        .map_err(|_| "hist: sample count exceeds supported range".to_string())?;
    let (min_value, bin_width, bin_count_u32, labels) = match bin_spec {
        HistBinSpec::Auto | HistBinSpec::Count(_) => {
            let bin_count = bin_spec.bin_count(sample_len);
            let bin_count_u32 = u32::try_from(bin_count)
                .map_err(|_| "hist: bin count exceeds supported range".to_string())?;
            let (min_value, max_value) = axis_bounds(values, "hist")?;
            if (max_value - min_value).abs() < 1e-6 {
                return build_degenerate_histogram(exported.len, bin_count, min_value as f64);
            }
            let bin_width = (max_value - min_value) / bin_count_u32 as f32;
            if !bin_width.is_finite() || bin_width <= 0.0 {
                return Err("hist: invalid bin width computed for GPU path".to_string());
            }
            let edges = uniform_edges(min_value as f64, max_value as f64, bin_count)?;
            let labels = histogram_labels_from_edges(&edges);
            (min_value, bin_width, bin_count_u32, labels)
        }
        HistBinSpec::Centers(centers) => {
            let edges = edges_from_centers(centers)?;
            if edges.len() < 2 {
                return Err("hist: bin centers must describe at least one bin".to_string());
            }
            let width = (edges[1] - edges[0]) as f32;
            if !width.is_finite() || width <= 0.0 {
                return Err("hist: invalid bin width computed from centers".to_string());
            }
            let bin_count = edges.len() - 1;
            let bin_count_u32 = u32::try_from(bin_count)
                .map_err(|_| "hist: bin count exceeds supported range".to_string())?;
            let labels = histogram_labels_from_edges(&edges);
            (edges[0] as f32, width, bin_count_u32, labels)
        }
        HistBinSpec::Edges(edges) => {
            let width = uniform_edge_width(edges).ok_or_else(|| {
                "hist: GPU rendering currently requires uniform bin edges".to_string()
            })?;
            let bin_count = edges.len().saturating_sub(1);
            let bin_count_u32 = u32::try_from(bin_count)
                .map_err(|_| "hist: bin count exceeds supported range".to_string())?;
            let labels = histogram_labels_from_edges(edges);
            (edges[0] as f32, width as f32, bin_count_u32, labels)
        }
    };

    let histogram_inputs = HistogramGpuInputs {
        samples: exported.buffer.clone(),
        sample_count: sample_count_u32,
        scalar: ScalarType::from_is_f64(exported.precision == ProviderPrecision::F64),
    };
    let histogram_params = HistogramGpuParams {
        min_value,
        inv_bin_width: 1.0 / bin_width,
        bin_count: bin_count_u32,
    };
    let normalization_scale = match normalization {
        HistNormalization::Count => 1.0,
        HistNormalization::Probability => {
            if sample_len == 0 {
                0.0
            } else {
                1.0 / sample_len as f32
            }
        }
        HistNormalization::Pdf => {
            if sample_len == 0 {
                0.0
            } else {
                1.0 / (sample_len as f32 * bin_width)
            }
        }
    };

    let values_buffer = runmat_plot::gpu::histogram::histogram_values_buffer(
        &context.device,
        &context.queue,
        &histogram_inputs,
        &histogram_params,
        normalization_scale,
    )
    .map_err(|e| format!("hist: failed to build GPU histogram counts: {e}"))?;

    let bar_inputs = BarGpuInputs {
        values_buffer,
        len: bin_count_u32,
        scalar: ScalarType::F32,
    };
    let bar_params = BarGpuParams {
        color: style.face_rgba(),
        bar_width: style.bar_width,
        group_index: 0,
        group_count: 1,
        orientation: BarOrientation::Vertical,
    };

    let gpu_vertices = runmat_plot::gpu::bar::pack_vertices_from_values(
        &context.device,
        &context.queue,
        &bar_inputs,
        &bar_params,
    )
    .map_err(|e| format!("hist: failed to build GPU vertices: {e}"))?;

    let bin_count = labels.len();
    let max_height = sample_len as f32 * normalization_scale;
    let bounds = histogram_bar_bounds(bin_count, max_height, style.bar_width);
    let vertex_count = gpu_vertices.vertex_count;

    let mut bar = BarChart::from_gpu_buffer(
        labels,
        bin_count,
        gpu_vertices,
        vertex_count,
        bounds,
        style.face_rgba(),
        style.bar_width,
    );
    bar.label = Some(HIST_DEFAULT_LABEL.to_string());
    Ok(bar)
}

fn build_degenerate_histogram(
    sample_count: usize,
    bins: usize,
    value: f64,
) -> Result<BarChart, String> {
    if bins == 0 {
        return Err("hist: bin count must be positive".to_string());
    }
    let mut heights = vec![0.0; bins];
    if bins > 0 {
        heights[bins.saturating_sub(1)] = sample_count as f64;
    }
    let labels = vec![format!("{value:.3}"); bins];
    let mut bar = BarChart::new(labels, heights).map_err(|err| format!("hist: {err}"))?;
    bar.label = Some(HIST_DEFAULT_LABEL.to_string());
    Ok(bar)
}

fn histogram_bar_bounds(bins: usize, max_height: f32, bar_width: f32) -> BoundingBox {
    let min_x = 1.0 - bar_width * 0.5;
    let max_x = bins as f32 + bar_width * 0.5;
    let max_y = if max_height.is_finite() && max_height > 0.0 {
        max_height
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
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            other => {
                let tensor = Tensor::try_from(&other).map_err(|e| format!("hist: {e}"))?;
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

    fn into_tensor(self, context: &str) -> Result<Tensor, String> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            Self::Gpu(handle) => gather_tensor_from_gpu(handle, context),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn hist_respects_bin_argument() {
        let data = Value::Tensor(tensor_from(&[1.0, 2.0, 3.0, 4.0]));
        let bins = vec![Value::from(2.0)];
        let result = hist_builtin(data, bins);
        if let Err(msg) = result {
            assert!(
                msg.contains("Plotting is unavailable"),
                "unexpected error: {msg}"
            );
        }
    }

    #[test]
    fn hist_accepts_bin_centers_vector() {
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0, 1.5]));
        let centers = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0]));
        let result = hist_builtin(data, vec![centers]);
        if let Err(msg) = result {
            assert!(
                msg.contains("Plotting is unavailable"),
                "unexpected error: {msg}"
            );
        }
    }

    #[test]
    fn hist_accepts_probability_normalization() {
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = hist_builtin(
            data,
            vec![Value::from(3.0), Value::String("probability".into())],
        );
        if let Err(msg) = result {
            assert!(
                msg.contains("Plotting is unavailable"),
                "unexpected error: {msg}"
            );
        }
    }

    #[test]
    fn hist_accepts_string_only_normalization() {
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = hist_builtin(data, vec![Value::String("pdf".into())]);
        if let Err(msg) = result {
            assert!(
                msg.contains("Plotting is unavailable"),
                "unexpected error: {msg}"
            );
        }
    }

    #[test]
    fn hist_accepts_normalization_name_value_pair() {
        let data = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = hist_builtin(
            data,
            vec![
                Value::String("Normalization".into()),
                Value::String("probability".into()),
            ],
        );
        if let Err(msg) = result {
            assert!(
                msg.contains("Plotting is unavailable"),
                "unexpected error: {msg}"
            );
        }
    }

    #[test]
    fn hist_accepts_bin_edges_option() {
        let data = Value::Tensor(tensor_from(&[0.1, 0.4, 0.7]));
        let edges = Value::Tensor(tensor_from(&[0.0, 0.5, 1.0]));
        let result = hist_builtin(data, vec![Value::String("BinEdges".into()), edges]);
        if let Err(msg) = result {
            assert!(
                msg.contains("Plotting is unavailable"),
                "unexpected error: {msg}"
            );
        }
    }
}
