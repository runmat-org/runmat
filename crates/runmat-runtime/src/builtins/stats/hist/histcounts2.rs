//! MATLAB-compatible `histcounts2` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

const NAME: &str = "histcounts2";
const DEFAULT_BIN_COUNT: usize = 10;
const RANGE_EPS: f64 = 1.0e-12;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "histcounts2",
        builtin_path = "crate::builtins::stats::hist::histcounts2"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "histcounts2"
category: "stats/hist"
keywords: ["histcounts2", "2d histogram", "joint distribution", "binning", "probability", "gpu"]
summary: "Count paired observations into two-dimensional histogram bins."
references:
  - https://www.mathworks.com/help/matlab/ref/histcounts2.html
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Falls back to host execution today; providers can attach a `histcounts2` hook for device kernels."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::stats::hist::histcounts2::tests"
  integration: "builtins::stats::hist::histcounts2::tests::histcounts2_gpu_roundtrip"
  gpu: "builtins::stats::hist::histcounts2::tests::histcounts2_wgpu_roundtrip"
---

# What does the `histcounts2` function do in MATLAB / RunMat?
`histcounts2(X, Y)` bins paired observations from `X` and `Y` into a rectangular two-dimensional
histogram. Each bin counts the number of pairs whose `X` component lies in a horizontal interval
and whose `Y` component lies in a vertical interval, mirroring MathWorks MATLAB behaviour.

## How does the `histcounts2` function behave in MATLAB / RunMat?
- `histcounts2(X, Y)` flattens both inputs column-major, ensuring they contain the same number of
  elements, and fills a `numel(Xedges) - 1` by `numel(Yedges) - 1` matrix of bin counts.
- Bins are half-open on the right except for the last column and row, which include their upper
  edges so the maximum finite values are counted.
- Optional arguments let you specify bin counts, explicit edges, bin limits, bin widths, and
  automatic binning heuristics independently for the `X` and `Y` axes.
- Name/value pairs such as `'NumBins'`, `'XBinEdges'`, `'YBinEdges'`, `'XBinWidth'`, `'YBinWidth'`,
  `'BinMethod'`, and `'Normalization'` follow MATLAB precedence and validation rules.
- Pairs containing `NaN` in either coordinate are ignored. Infinite values participate when the
  chosen edges include them.

## `histcounts2` Function GPU Execution Behaviour
When either input is a `gpuArray`, RunMat gathers the samples back to host memory, performs the
reference CPU implementation, and returns dense CPU tensors for the histogram and edges. The
acceleration layer exposes a `histcounts2` provider hook; once kernels land, the runtime will
automatically keep residency on the GPU and skip gathering. The builtin is registered as a sink, so
fusion plans flush GPU residency before histogramming and the current implementation always yields
host-resident outputs.

## Examples of using the `histcounts2` function in MATLAB / RunMat

### Counting paired values with explicit edges
```matlab
X = [0.5 1.5 2.5 3.5];
Y = [0.2 0.9 1.4 2.8];
[N, Xedges, Yedges] = histcounts2(X, Y, [0 1 2 3 4], [0 1 2 3]);
```
Expected output:
```matlab
N = [1 0 0; 1 0 0; 0 1 0; 0 0 1];
Xedges = [0 1 2 3 4];
Yedges = [0 1 2 3];
```

### Specifying separate bin counts for each axis
```matlab
X = [1 2 3 4];
Y = [1 2 3 4];
N = histcounts2(X, Y, 2, 4);
```
Expected output:
```matlab
size(N) = [2 4];
sum(N, "all") = 4;
```

### Using different bin widths for X and Y
```matlab
X = [1 1.5 2.4 3.7];
Y = [2 2.2 2.9 3.1];
[N, Xedges, Yedges] = histcounts2(X, Y, 'XBinWidth', 1, 'YBinWidth', 0.5);
```
Expected output:
```matlab
diff(Xedges) = [1 1 1];
diff(Yedges(1:3)) = [0.5 0.5];
sum(N, "all") = 4;
```

### Normalizing a 2-D histogram to probabilities
```matlab
X = [0.2 0.4 1.1 1.5];
Y = [0.1 0.8 1.2 1.9];
N = histcounts2(X, Y, [0 1 2], [0 1 2], 'Normalization', 'probability');
```
Expected output:
```matlab
N = [0.5 0.0; 0.0 0.5];
```

### Ignoring NaN values in paired data
```matlab
X = [1 2 NaN 3];
Y = [2 2 2 NaN];
N = histcounts2(X, Y, [0 1 2 3], [0 1 2 3]);
```
Expected output:
```matlab
sum(N, "all") = 2;
```

### Histogramming gpuArray inputs without manual gather
```matlab
Gx = gpuArray([0.5 1.5 2.5]);
Gy = gpuArray([1.0 1.1 2.9]);
[counts, Xedges, Yedges] = histcounts2(Gx, Gy, [0 1 2 3], [0 2 3]);
```
Expected output:
```matlab
isa(counts, 'double')      % counts are returned on the CPU
counts = [1 0; 1 0; 0 1];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
As with other RunMat histogram routines, you do not need to call `gpuArray` explicitly just to
obtain GPU execution. Once providers implement the `histcounts2` hook, the fusion planner will keep
residency on the device automatically. Until then, the builtin gathers data to the host and returns
CPU tensors even when inputs originate on the GPU, matching MATLAB semantics after an explicit
`gather`.

## FAQ

### Do `X` and `Y` need the same size?
Yes. `histcounts2` requires both inputs to contain the same number of elements. RunMat mirrors
MATLAB by raising an error when the sizes do not match.

### How are the bin edges interpreted?
All interior bins are `[left, right)`, while the last row and column are `[left, right]`, so the
largest finite values are counted.

### What happens to `NaN`, `Inf`, or `-Inf` values?
Pairs containing `NaN` in either coordinate are ignored. Infinite values participate when the
specified edges include them; otherwise, they are excluded just like MATLAB.

### Can I mix explicit edges on one axis with automatic binning on the other?
Yes. You can supply `'XBinEdges'` while leaving the `Y` axis to be determined by `'NumBins'`,
`'YBinWidth'`, or the default heuristics.

### Which normalisation modes are supported?
`'count'`, `'probability'`, `'countdensity'`, `'pdf'`, `'cumcount'`, and `'cdf'` are implemented.
`'cdf'` and `'cumcount'` operate in column-major order so the result matches MATLAB's cumulative
behaviour.

### How do I request integer-aligned bins?
Use `'BinMethod', 'integers'` or `'XBinMethod'/'YBinMethod'` with the value `'integers'`. RunMat
ensures the resulting edges align with integer boundaries, respecting any supplied bin limits.

## See Also
[histcounts](./histcounts), [accumarray](../../array/accumarray), [sum](../../math/reduction/sum), [gpuArray](../../acceleration/gpu/gpuArray)

## Source & Feedback
- The full source code for this builtin lives at `crates/runmat-runtime/src/builtins/stats/hist/histcounts2.rs`.
- Found a discrepancy? Please open an issue with a minimal reproduction example.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::hist::histcounts2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "histcounts2",
    op_kind: GpuOpKind::Custom("histcounts2"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("histcounts2")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Omit,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may install a custom 2-D histogram kernel; current builds gather to host memory.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::hist::histcounts2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "histcounts2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Histogram binning terminates fusion chains and materialises host-resident outputs.",
};

#[runtime_builtin(
    name = "histcounts2",
    category = "stats/hist",
    summary = "Count paired observations into two-dimensional histogram bins.",
    keywords = "histcounts2,2d histogram,joint distribution,binning,probability,gpu",
    accel = "reduction",
    sink = true,
    builtin_path = "crate::builtins::stats::hist::histcounts2"
)]
fn histcounts2_builtin(x: Value, y: Value, rest: Vec<Value>) -> Result<Value, String> {
    evaluate(x, y, &rest).map(|eval| eval.into_counts_value())
}

/// Evaluate `histcounts2` once and surface all outputs.
pub fn evaluate(x: Value, y: Value, rest: &[Value]) -> Result<Histcounts2Evaluation, String> {
    let options = parse_options(rest)?;
    let x_tensor = match x {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor(&handle)?,
        other => tensor::value_into_tensor_for(NAME, other)?,
    };
    let y_tensor = match y {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor(&handle)?,
        other => tensor::value_into_tensor_for(NAME, other)?,
    };
    histcounts2_from_tensors(x_tensor, y_tensor, &options)
}

fn histcounts2_from_tensors(
    x_tensor: Tensor,
    y_tensor: Tensor,
    options: &Histcounts2Options,
) -> Result<Histcounts2Evaluation, String> {
    if x_tensor.data.len() != y_tensor.data.len() {
        return Err(format!(
            "{NAME}: X and Y must contain the same number of elements"
        ));
    }
    if x_tensor.shape != y_tensor.shape {
        return Err(format!("{NAME}: X and Y must be the same size"));
    }

    let x_axis = collect_axis_data(&x_tensor.data);
    let y_axis = collect_axis_data(&y_tensor.data);

    let x_edges = compute_edges_for_axis(&x_axis, &options.x, "X")?;
    let y_edges = compute_edges_for_axis(&y_axis, &options.y, "Y")?;

    let counts = compute_histogram_counts(&x_tensor.data, &y_tensor.data, &x_edges, &y_edges);
    let normalised = apply_normalization_2d(&counts, &x_edges, &y_edges, options.normalization);

    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    let counts_tensor =
        Tensor::new(normalised, vec![x_bins, y_bins]).map_err(|e| format!("{NAME}: {e}"))?;
    let x_edges_tensor =
        Tensor::new(x_edges.clone(), vec![1, x_edges.len()]).map_err(|e| format!("{NAME}: {e}"))?;
    let y_edges_tensor =
        Tensor::new(y_edges.clone(), vec![1, y_edges.len()]).map_err(|e| format!("{NAME}: {e}"))?;

    Ok(Histcounts2Evaluation::new(
        counts_tensor,
        x_edges_tensor,
        y_edges_tensor,
    ))
}

fn compute_histogram_counts(
    x_values: &[f64],
    y_values: &[f64],
    x_edges: &[f64],
    y_edges: &[f64],
) -> Vec<f64> {
    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    let mut counts = vec![0.0f64; x_bins * y_bins];

    for (x, y) in x_values.iter().zip(y_values.iter()) {
        if x.is_nan() || y.is_nan() {
            continue;
        }
        if let (Some(ix), Some(iy)) = (find_bin_index(*x, x_edges), find_bin_index(*y, y_edges)) {
            let idx = ix + iy * x_bins;
            if idx < counts.len() {
                counts[idx] += 1.0;
            }
        }
    }

    counts
}

#[derive(Clone)]
struct AxisData {
    values: Vec<f64>,
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
}

fn collect_axis_data(values: &[f64]) -> AxisData {
    let mut filtered = Vec::new();
    let mut min_val: Option<f64> = None;
    let mut max_val: Option<f64> = None;

    for &sample in values {
        if sample.is_nan() {
            continue;
        }
        filtered.push(sample);
        if sample.is_finite() {
            min_val = Some(match min_val {
                Some(current) if sample >= current => current,
                Some(_) => sample,
                None => sample,
            });
            max_val = Some(match max_val {
                Some(current) if sample <= current => current,
                Some(_) => sample,
                None => sample,
            });
        }
    }

    let original_range_zero = match (min_val, max_val) {
        (Some(minimum), Some(maximum)) => approx_equal(minimum, maximum),
        _ => false,
    };

    AxisData {
        values: filtered,
        min_val,
        max_val,
        original_range_zero,
    }
}

fn compute_edges_for_axis(
    data: &AxisData,
    options: &AxisOptions,
    axis: &str,
) -> Result<Vec<f64>, String> {
    if let Some(edges) = &options.explicit_edges {
        validate_edges(edges, axis)?;
        return Ok(edges.clone());
    }

    if let Some(method) = options.bin_method {
        return compute_edges_with_method(
            &data.values,
            data.min_val,
            data.max_val,
            data.original_range_zero,
            method,
            options,
            axis,
        );
    }

    compute_edges_standard(
        data.min_val,
        data.max_val,
        data.original_range_zero,
        options,
        axis,
    )
}

fn compute_edges_standard(
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    options: &AxisOptions,
    axis: &str,
) -> Result<Vec<f64>, String> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(format!(
            "{NAME}: data range for {axis} must be finite; specify {axis}BinLimits or {axis}BinEdges"
        ));
    }

    if upper < lower {
        return Err(format!("{NAME}: {axis} bin limits must be increasing"));
    }

    if options.bin_limits.is_some() && approx_equal(lower, upper) {
        return Err(format!(
            "{NAME}: {axis}BinLimits must specify a non-zero width"
        ));
    }

    if let Some(width) = options.bin_width {
        if !width.is_finite() || width <= 0.0 {
            return Err(format!(
                "{NAME}: {axis}BinWidth must be a positive finite scalar"
            ));
        }

        if original_range_zero && options.bin_limits.is_none() {
            let centre = min_val.unwrap_or(0.0);
            lower = centre - width / 2.0;
            upper = centre + width / 2.0;
        } else if options.bin_limits.is_none() {
            if let (Some(minimum), Some(maximum)) = (min_val, max_val) {
                lower = minimum;
                upper = maximum;
            }
        }

        if approx_equal(lower, upper) {
            upper = lower + width;
        }

        let bins = ((upper - lower) / width).ceil().max(1.0) as usize;
        let mut edges = Vec::with_capacity(bins + 1);
        for i in 0..=bins {
            edges.push(lower + width * i as f64);
        }
        if let Some(last) = edges.last_mut() {
            *last = upper;
        }
        validate_edges(&edges, axis)?;
        return Ok(edges);
    }

    let mut num_bins = options.num_bins.unwrap_or(DEFAULT_BIN_COUNT);
    if num_bins == 0 {
        return Err(format!("{NAME}: NumBins must be a positive integer"));
    }

    if original_range_zero {
        let centre = min_val.unwrap_or(0.0);
        if options.num_bins.is_none() && options.bin_limits.is_none() && options.bin_width.is_none()
        {
            num_bins = 1;
        }
        if options.bin_limits.is_none() && options.bin_width.is_none() {
            lower = centre - 0.5;
            upper = centre + 0.5;
        }
    } else if options.bin_limits.is_none() {
        if let (Some(minimum), Some(maximum)) = (min_val, max_val) {
            lower = lower.min(minimum);
            upper = upper.max(maximum);
        }
    }

    if approx_equal(lower, upper) {
        upper = lower + 1.0;
    }

    let edges = linspace(lower, upper, num_bins + 1);
    validate_edges(&edges, axis)?;
    Ok(edges)
}

fn compute_edges_with_method(
    values: &[f64],
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    method: BinMethod,
    options: &AxisOptions,
    axis: &str,
) -> Result<Vec<f64>, String> {
    if values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options, axis);
    }

    if matches!(method, BinMethod::Integers) {
        let edges = compute_integer_edges(min_val, max_val, options, axis)?;
        validate_edges(&edges, axis)?;
        return Ok(edges);
    }

    let (lower, upper) = derive_initial_limits(min_val, max_val, options.bin_limits);
    if !lower.is_finite() || !upper.is_finite() {
        return Err(format!(
            "{NAME}: {axis} data range must be finite for BinMethod"
        ));
    }

    if approx_equal(lower, upper) {
        if options.bin_limits.is_some() {
            return Err(format!(
                "{NAME}: {axis}BinLimits must specify a non-zero width"
            ));
        }
        return compute_edges_standard(min_val, max_val, true, options, axis);
    }

    let finite_values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options, axis);
    }

    let range = upper - lower;
    if range <= 0.0 {
        return compute_edges_standard(min_val, max_val, true, options, axis);
    }

    let mut sorted = finite_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let width = match method {
        BinMethod::Auto => {
            let fd = freedman_diaconis_width(&sorted);
            if fd > 0.0 {
                fd
            } else {
                let scott = scott_width(&finite_values);
                if scott > 0.0 {
                    scott
                } else {
                    range / sturges_bin_count(sorted.len()) as f64
                }
            }
        }
        BinMethod::Scott => scott_width(&finite_values),
        BinMethod::Fd => freedman_diaconis_width(&sorted),
        BinMethod::Sturges => range / sturges_bin_count(sorted.len()) as f64,
        BinMethod::Sqrt => range / sqrt_bin_count(sorted.len()) as f64,
        BinMethod::Integers => unreachable!(),
    };

    if !width.is_finite() || width <= 0.0 {
        return compute_edges_standard(min_val, max_val, original_range_zero, options, axis);
    }

    let bins = ((range / width).ceil().max(1.0)) as usize;
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(lower + width * i as f64);
    }
    if let Some(last) = edges.last_mut() {
        *last = upper;
    }
    validate_edges(&edges, axis)?;
    Ok(edges)
}

fn compute_integer_edges(
    min_val: Option<f64>,
    max_val: Option<f64>,
    options: &AxisOptions,
    axis: &str,
) -> Result<Vec<f64>, String> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(format!(
            "{NAME}: {axis}BinLimits must be finite for 'integers' BinMethod"
        ));
    }

    if approx_equal(lower, upper) {
        let centre = min_val.or(max_val).unwrap_or(lower);
        lower = centre - 0.5;
        upper = centre + 0.5;
    }

    if let Some((lo, hi)) = options.bin_limits {
        lower = lo;
        upper = hi;
    } else {
        lower = lower.floor();
        upper = upper.ceil();
    }

    if upper <= lower {
        upper = lower + 1.0;
    }

    let mut edges = Vec::new();
    edges.push(lower);

    if let Some((lo, hi)) = options.bin_limits {
        let mut k = lo.ceil() as i64;
        while (k as f64) < hi {
            let candidate = k as f64;
            if candidate > lo + RANGE_EPS {
                edges.push(candidate);
            }
            k += 1;
        }
    } else {
        let mut current = lower.floor() as i64 + 1;
        let end = upper.ceil() as i64;
        while current < end {
            edges.push(current as f64);
            current += 1;
        }
    }

    if !approx_equal(*edges.last().unwrap(), upper) {
        edges.push(upper);
    }

    if edges.len() < 2 {
        edges.push(upper + 1.0);
    }

    Ok(edges)
}

fn derive_initial_limits(
    min_val: Option<f64>,
    max_val: Option<f64>,
    bin_limits: Option<(f64, f64)>,
) -> (f64, f64) {
    if let Some((lo, hi)) = bin_limits {
        (lo, hi)
    } else if let (Some(minimum), Some(maximum)) = (min_val, max_val) {
        (minimum, maximum)
    } else {
        (0.0, 1.0)
    }
}

fn find_bin_index(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value > edges[edges.len() - 1] {
        return None;
    }
    if value == edges[edges.len() - 1] {
        return Some(edges.len() - 2);
    }
    match edges.binary_search_by(|edge| edge.partial_cmp(&value).unwrap_or(Ordering::Less)) {
        Ok(index) => {
            if index == 0 {
                Some(0)
            } else if index < edges.len() - 1 {
                Some(index)
            } else {
                Some(edges.len() - 2)
            }
        }
        Err(index) => {
            if index == 0 || index > edges.len() - 1 {
                None
            } else {
                Some(index - 1)
            }
        }
    }
}

fn apply_normalization_2d(
    counts: &[f64],
    x_edges: &[f64],
    y_edges: &[f64],
    mode: HistogramNormalization,
) -> Vec<f64> {
    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    let total: f64 = counts.iter().sum();
    let x_widths: Vec<f64> = x_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let y_widths: Vec<f64> = y_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();

    match mode {
        HistogramNormalization::Count => counts.to_vec(),
        HistogramNormalization::Probability => {
            if total > 0.0 {
                counts.iter().map(|&c| c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CountDensity => {
            let mut out = vec![0.0; counts.len()];
            for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
                for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
                    let idx = ix + iy * x_bins;
                    let area = x_width * y_width;
                    out[idx] = if area > 0.0 { counts[idx] / area } else { 0.0 };
                }
            }
            out
        }
        HistogramNormalization::Pdf => {
            if total > 0.0 {
                let mut out = vec![0.0; counts.len()];
                for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
                    for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
                        let idx = ix + iy * x_bins;
                        let area = x_width * y_width;
                        out[idx] = if area > 0.0 {
                            counts[idx] / (total * area)
                        } else {
                            0.0
                        };
                    }
                }
                out
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CumCount => {
            let mut acc = 0.0;
            counts
                .iter()
                .map(|&c| {
                    acc += c;
                    acc
                })
                .collect()
        }
        HistogramNormalization::Cdf => {
            if total > 0.0 {
                let mut acc = 0.0;
                counts
                    .iter()
                    .map(|&c| {
                        acc += c;
                        acc / total
                    })
                    .collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
    }
}

fn validate_edges(edges: &[f64], axis: &str) -> Result<(), String> {
    if edges.len() < 2 {
        return Err(format!(
            "{NAME}: {axis}BinEdges must contain at least two elements"
        ));
    }
    for pair in edges.windows(2) {
        if pair[0].is_nan() || pair[1].is_nan() {
            return Err(format!(
                "{NAME}: {axis}BinEdges must contain finite numbers"
            ));
        }
        if pair[1] <= pair[0] {
            return Err(format!(
                "{NAME}: {axis}BinEdges must be strictly increasing"
            ));
        }
    }
    Ok(())
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f64;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(start + step * i as f64);
    }
    if let Some(last) = out.last_mut() {
        *last = stop;
    }
    out
}

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= RANGE_EPS * (a.abs().max(b.abs()).max(1.0))
}

fn scott_width(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let sigma = standard_deviation(values);
    if sigma <= 0.0 {
        return 0.0;
    }
    3.5 * sigma / (n as f64).powf(1.0 / 3.0)
}

fn freedman_diaconis_width(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 2 {
        return 0.0;
    }
    let q1 = percentile(sorted, 0.25);
    let q3 = percentile(sorted, 0.75);
    let iqr = q3 - q1;
    if iqr <= 0.0 {
        return 0.0;
    }
    2.0 * iqr / (n as f64).powf(1.0 / 3.0)
}

fn sturges_bin_count(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        ((n as f64).log2() + 1.0).ceil().max(1.0) as usize
    }
}

fn sqrt_bin_count(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (n as f64).sqrt().ceil().max(1.0) as usize
    }
}

fn standard_deviation(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut acc = 0.0;
    for &value in values {
        let diff = value - mean;
        acc += diff * diff;
    }
    (acc / (n as f64 - 1.0).max(1.0)).sqrt()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let clamped = p.clamp(0.0, 1.0);
    let rank = clamped * (n as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = rank - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

#[derive(Clone, Default)]
struct Histcounts2Options {
    x: AxisOptions,
    y: AxisOptions,
    normalization: HistogramNormalization,
}

impl Histcounts2Options {
    fn validate(&self) -> Result<(), String> {
        self.x.validate("X")?;
        self.y.validate("Y")?;
        Ok(())
    }
}

#[derive(Clone, Default)]
struct AxisOptions {
    explicit_edges: Option<Vec<f64>>,
    num_bins: Option<usize>,
    bin_width: Option<f64>,
    bin_limits: Option<(f64, f64)>,
    bin_method: Option<BinMethod>,
}

impl AxisOptions {
    fn validate(&self, axis: &str) -> Result<(), String> {
        if self.explicit_edges.is_some()
            && (self.num_bins.is_some() || self.bin_width.is_some() || self.bin_limits.is_some())
        {
            return Err(format!(
                "{NAME}: {axis}BinEdges cannot be combined with NumBins, {axis}BinWidth, or {axis}BinLimits"
            ));
        }
        if self.bin_method.is_some()
            && (self.explicit_edges.is_some()
                || self.bin_width.is_some()
                || self.num_bins.is_some())
        {
            return Err(format!(
                "{NAME}: {axis}BinMethod cannot be combined with {axis}BinEdges, NumBins, or {axis}BinWidth"
            ));
        }
        if self.num_bins.is_some() && self.bin_width.is_some() {
            return Err(format!(
                "{NAME}: specify only one of NumBins or {axis}BinWidth"
            ));
        }
        if let Some((lo, hi)) = self.bin_limits {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(format!("{NAME}: {axis}BinLimits must be finite"));
            }
            if hi < lo {
                return Err(format!("{NAME}: {axis}BinLimits must be increasing"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinMethod {
    Auto,
    Scott,
    Fd,
    Sturges,
    Sqrt,
    Integers,
}

#[derive(Clone, Copy, Debug, Default)]
enum HistogramNormalization {
    #[default]
    Count,
    Probability,
    CountDensity,
    Pdf,
    CumCount,
    Cdf,
}

#[derive(Clone)]
pub struct Histcounts2Evaluation {
    counts: Tensor,
    x_edges: Tensor,
    y_edges: Tensor,
}

impl Histcounts2Evaluation {
    fn new(counts: Tensor, x_edges: Tensor, y_edges: Tensor) -> Self {
        Self {
            counts,
            x_edges,
            y_edges,
        }
    }

    pub fn into_counts_value(self) -> Value {
        Value::Tensor(self.counts)
    }

    pub fn into_pair(self) -> (Value, Value) {
        let counts = self.counts;
        let x_edges = self.x_edges;
        (Value::Tensor(counts), Value::Tensor(x_edges))
    }

    pub fn into_triple(self) -> (Value, Value, Value) {
        let counts = self.counts;
        let x_edges = self.x_edges;
        let y_edges = self.y_edges;
        (
            Value::Tensor(counts),
            Value::Tensor(x_edges),
            Value::Tensor(y_edges),
        )
    }
}

fn parse_options(args: &[Value]) -> Result<Histcounts2Options, String> {
    let mut options = Histcounts2Options::default();
    let mut index = 0;

    if index < args.len() && !is_option_key(&args[index]) {
        if index + 1 < args.len() && !is_option_key(&args[index + 1]) {
            let x_vec = numeric_vector(&args[index], NAME, "positional X argument")?;
            let y_vec = numeric_vector(&args[index + 1], NAME, "positional Y argument")?;
            if x_vec.len() >= 2 && y_vec.len() >= 2 {
                validate_edges(&x_vec, "X")?;
                validate_edges(&y_vec, "Y")?;
                options.x.explicit_edges = Some(x_vec);
                options.y.explicit_edges = Some(y_vec);
            } else if x_vec.len() == 1 && y_vec.len() == 1 {
                let nx = positive_usize(&args[index], NAME, "NumBins")?;
                let ny = positive_usize(&args[index + 1], NAME, "NumBins")?;
                options.x.num_bins = Some(nx);
                options.y.num_bins = Some(ny);
            } else {
                return Err(format!(
                    "{NAME}: positional bin arguments must either specify two edge vectors or two scalar bin counts"
                ));
            }
            index += 2;
        } else {
            let bins = numeric_vector(&args[index], NAME, "NumBins")?;
            match bins.len() {
                1 => {
                    let n = positive_usize_from_f64(bins[0], "NumBins")?;
                    options.x.num_bins = Some(n);
                    options.y.num_bins = Some(n);
                }
                2 => {
                    let nx = positive_usize_from_f64(bins[0], "NumBins")?;
                    let ny = positive_usize_from_f64(bins[1], "NumBins")?;
                    options.x.num_bins = Some(nx);
                    options.y.num_bins = Some(ny);
                }
                _ => {
                    return Err(format!(
                        "{NAME}: NumBins must be a scalar or two-element vector"
                    ))
                }
            }
            index += 1;
        }
    }

    while index < args.len() {
        let key = tensor::value_to_string(&args[index])
            .ok_or_else(|| format!("{NAME}: expected name/value pair arguments"))?;
        index += 1;
        if index >= args.len() {
            return Err(format!("{NAME}: missing value for option '{key}'"));
        }
        let value = &args[index];
        index += 1;
        let lowered = key.trim().to_ascii_lowercase();

        match lowered.as_str() {
            "numbins" => {
                let bins = numeric_vector(value, NAME, "NumBins")?;
                match bins.len() {
                    1 => {
                        let n = positive_usize_from_f64(bins[0], "NumBins")?;
                        options.x.num_bins = Some(n);
                        options.y.num_bins = Some(n);
                    }
                    2 => {
                        let nx = positive_usize_from_f64(bins[0], "NumBins")?;
                        let ny = positive_usize_from_f64(bins[1], "NumBins")?;
                        options.x.num_bins = Some(nx);
                        options.y.num_bins = Some(ny);
                    }
                    _ => {
                        return Err(format!(
                            "{NAME}: NumBins must be a scalar or two-element vector"
                        ))
                    }
                }
            }
            "xbinwidth" => {
                let width = positive_scalar(value, NAME, "XBinWidth")?;
                options.x.bin_width = Some(width);
            }
            "ybinwidth" => {
                let width = positive_scalar(value, NAME, "YBinWidth")?;
                options.y.bin_width = Some(width);
            }
            "binwidth" => {
                let widths = numeric_vector(value, NAME, "BinWidth")?;
                match widths.len() {
                    1 => {
                        let width = positive_scalar_from_f64(widths[0], "BinWidth")?;
                        options.x.bin_width = Some(width);
                        options.y.bin_width = Some(width);
                    }
                    2 => {
                        let wx = positive_scalar_from_f64(widths[0], "BinWidth")?;
                        let wy = positive_scalar_from_f64(widths[1], "BinWidth")?;
                        options.x.bin_width = Some(wx);
                        options.y.bin_width = Some(wy);
                    }
                    _ => {
                        return Err(format!(
                            "{NAME}: BinWidth must be a scalar or two-element vector"
                        ))
                    }
                }
            }
            "xbinlimits" => {
                let limits = numeric_vector(value, NAME, "XBinLimits")?;
                if limits.len() != 2 {
                    return Err(format!(
                        "{NAME}: XBinLimits must contain exactly two elements"
                    ));
                }
                let lo = limits[0];
                let hi = limits[1];
                if hi < lo {
                    return Err(format!("{NAME}: XBinLimits must be increasing"));
                }
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(format!("{NAME}: XBinLimits must be finite"));
                }
                options.x.bin_limits = Some((lo, hi));
            }
            "ybinlimits" => {
                let limits = numeric_vector(value, NAME, "YBinLimits")?;
                if limits.len() != 2 {
                    return Err(format!(
                        "{NAME}: YBinLimits must contain exactly two elements"
                    ));
                }
                let lo = limits[0];
                let hi = limits[1];
                if hi < lo {
                    return Err(format!("{NAME}: YBinLimits must be increasing"));
                }
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(format!("{NAME}: YBinLimits must be finite"));
                }
                options.y.bin_limits = Some((lo, hi));
            }
            "binlimits" => {
                let limits = numeric_vector(value, NAME, "BinLimits")?;
                match limits.len() {
                    2 => {
                        let lo = limits[0];
                        let hi = limits[1];
                        if hi < lo {
                            return Err(format!("{NAME}: BinLimits must be increasing"));
                        }
                        if !lo.is_finite() || !hi.is_finite() {
                            return Err(format!("{NAME}: BinLimits must be finite"));
                        }
                        options.x.bin_limits = Some((lo, hi));
                        options.y.bin_limits = Some((lo, hi));
                    }
                    4 => {
                        let x_lo = limits[0];
                        let x_hi = limits[1];
                        let y_lo = limits[2];
                        let y_hi = limits[3];
                        if x_hi < x_lo || y_hi < y_lo {
                            return Err(format!("{NAME}: BinLimits must be increasing"));
                        }
                        if !x_lo.is_finite()
                            || !x_hi.is_finite()
                            || !y_lo.is_finite()
                            || !y_hi.is_finite()
                        {
                            return Err(format!("{NAME}: BinLimits must be finite"));
                        }
                        options.x.bin_limits = Some((x_lo, x_hi));
                        options.y.bin_limits = Some((y_lo, y_hi));
                    }
                    _ => {
                        return Err(format!(
                            "{NAME}: BinLimits must contain two or four elements"
                        ))
                    }
                }
            }
            "xbinedges" => {
                let edges = numeric_vector(value, NAME, "XBinEdges")?;
                validate_edges(&edges, "X")?;
                options.x.explicit_edges = Some(edges);
            }
            "ybinedges" => {
                let edges = numeric_vector(value, NAME, "YBinEdges")?;
                validate_edges(&edges, "Y")?;
                options.y.explicit_edges = Some(edges);
            }
            "binmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| format!("{NAME}: BinMethod must be a string"))?;
                let method = parse_bin_method(&text)?;
                options.x.bin_method = Some(method);
                options.y.bin_method = Some(method);
            }
            "xbinmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| format!("{NAME}: XBinMethod must be a string"))?;
                options.x.bin_method = Some(parse_bin_method(&text)?);
            }
            "ybinmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| format!("{NAME}: YBinMethod must be a string"))?;
                options.y.bin_method = Some(parse_bin_method(&text)?);
            }
            "normalization" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| format!("{NAME}: Normalization must be a string"))?;
                options.normalization = parse_normalization(&text)?;
            }
            other => {
                return Err(format!("{NAME}: unrecognised option '{other}'"));
            }
        }
    }

    options.validate()?;
    Ok(options)
}

fn is_option_key(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn numeric_vector(value: &Value, name: &str, option: &str) -> Result<Vec<f64>, String> {
    let tensor =
        tensor::value_to_tensor(value).map_err(|_| format!("{name}: {option} must be numeric"))?;
    Ok(tensor.data)
}

fn positive_usize(value: &Value, name: &str, option: &str) -> Result<usize, String> {
    let scalar = scalar_value(value, name, option)?;
    if scalar <= 0.0 || !scalar.is_finite() {
        return Err(format!("{name}: {option} must be a positive finite scalar"));
    }
    let rounded = scalar.round();
    if (scalar - rounded).abs() > f64::EPSILON {
        return Err(format!("{name}: {option} must be an integer"));
    }
    Ok(rounded as usize)
}

fn positive_scalar(value: &Value, name: &str, option: &str) -> Result<f64, String> {
    let scalar = scalar_value(value, name, option)?;
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(format!("{name}: {option} must be a positive finite scalar"));
    }
    Ok(scalar)
}

fn scalar_value(value: &Value, name: &str, option: &str) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(format!("{name}: {option} must be a scalar"));
            }
            Ok(tensor.data[0])
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != 1 {
                return Err(format!("{name}: {option} must be a scalar"));
            }
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        other => Err(format!("{name}: {option} must be numeric, got {:?}", other)),
    }
}

fn positive_usize_from_f64(value: f64, option: &str) -> Result<usize, String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{NAME}: {option} must be a positive finite scalar"));
    }
    let rounded = value.round();
    if (value - rounded).abs() > f64::EPSILON {
        return Err(format!("{NAME}: {option} must be an integer"));
    }
    Ok(rounded as usize)
}

fn positive_scalar_from_f64(value: f64, option: &str) -> Result<f64, String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{NAME}: {option} must be a positive finite scalar"));
    }
    Ok(value)
}

fn parse_bin_method(text: &str) -> Result<BinMethod, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(BinMethod::Auto),
        "scott" => Ok(BinMethod::Scott),
        "fd" | "freedmandiaconis" => Ok(BinMethod::Fd),
        "sturges" => Ok(BinMethod::Sturges),
        "sqrt" => Ok(BinMethod::Sqrt),
        "integers" => Ok(BinMethod::Integers),
        other => Err(format!("{NAME}: unrecognised BinMethod value '{other}'")),
    }
}

fn parse_normalization(text: &str) -> Result<HistogramNormalization, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "count" => Ok(HistogramNormalization::Count),
        "probability" => Ok(HistogramNormalization::Probability),
        "countdensity" => Ok(HistogramNormalization::CountDensity),
        "pdf" | "probabilitydensity" => Ok(HistogramNormalization::Pdf),
        "cumcount" => Ok(HistogramNormalization::CumCount),
        "cdf" => Ok(HistogramNormalization::Cdf),
        other => Err(format!(
            "{NAME}: unrecognised Normalization value '{other}'"
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    fn tensor_from_value(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("expected tensor value, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_basic_counts() {
        let x = Tensor::new(vec![0.5, 1.5, 2.5, 3.5], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![0.2, 0.9, 1.4, 2.8], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
            ],
        )
        .expect("histcounts2");
        let (counts, xedges, yedges) = eval.into_triple();
        let counts = tensor_from_value(counts);
        assert_eq!(counts.shape, vec![4, 3]);
        let xedges = tensor_from_value(xedges);
        let yedges = tensor_from_value(yedges);
        assert_eq!(xedges.data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(yedges.data, vec![0.0, 1.0, 2.0, 3.0]);
        let rows = counts.shape[0];
        let count = |ix: usize, iy: usize| counts.data[ix + iy * rows];
        assert_eq!(count(0, 0), 1.0);
        assert_eq!(count(1, 0), 1.0);
        assert_eq!(count(2, 1), 1.0);
        assert_eq!(count(3, 2), 1.0);
        for iy in 0..counts.shape[1] {
            for ix in 0..rows {
                let keep = matches!((ix, iy), (0, 0) | (1, 0) | (2, 1) | (3, 2));
                if !keep {
                    assert_eq!(count(ix, iy), 0.0);
                }
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_probability_normalization() {
        let x = Tensor::new(vec![0.2, 0.4, 1.1, 1.5], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![0.1, 0.8, 1.2, 1.9], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
                Value::from("Normalization"),
                Value::from("probability"),
            ],
        )
        .expect("histcounts2");
        let counts = tensor_from_value(eval.into_counts_value());
        assert_eq!(counts.shape, vec![2, 2]);
        assert_eq!(counts.data, vec![0.5, 0.0, 0.0, 0.5]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_nan_pairs_excluded() {
        let x = Tensor::new(vec![1.0, 2.0, f64::NAN, 3.0], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![2.0, 2.0, 2.0, f64::NAN], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
            ],
        )
        .expect("histcounts2");
        let counts = tensor_from_value(eval.into_counts_value());
        assert_eq!(counts.data.iter().sum::<f64>(), 2.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_num_bins_vector() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::from("NumBins"),
                Value::Tensor(Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap()),
            ],
        )
        .expect("histcounts2");
        let counts = tensor_from_value(eval.into_counts_value());
        assert_eq!(counts.shape, vec![2, 4]);
        assert_eq!(counts.data.iter().sum::<f64>(), 4.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_bin_width_and_limits() {
        let x_tensor = Tensor::new(vec![0.2, 1.2, 2.4, 2.6], vec![4, 1]).unwrap();
        let y_tensor = Tensor::new(vec![0.1, 0.6, 1.4, 2.2], vec![4, 1]).unwrap();
        let bin_limits = Tensor::new(vec![0.0, 3.0, 0.0, 2.5], vec![4, 1]).unwrap();

        let eval = evaluate(
            Value::Tensor(x_tensor.clone()),
            Value::Tensor(y_tensor.clone()),
            &[
                Value::from("XBinWidth"),
                Value::Num(1.0),
                Value::from("YBinWidth"),
                Value::Num(0.5),
                Value::from("BinLimits"),
                Value::Tensor(bin_limits.clone()),
            ],
        )
        .expect("histcounts2");
        let (counts_v, xedges_v, yedges_v) = eval.into_triple();
        let counts = tensor_from_value(counts_v);
        let xedges = tensor_from_value(xedges_v);
        let yedges = tensor_from_value(yedges_v);

        assert_eq!(xedges.data, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(yedges.data, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        assert_eq!(counts.shape, vec![3, 5]);
        assert_eq!(counts.data.iter().sum::<f64>(), 4.0);

        let density_eval = evaluate(
            Value::Tensor(x_tensor),
            Value::Tensor(y_tensor),
            &[
                Value::from("XBinWidth"),
                Value::Num(1.0),
                Value::from("YBinWidth"),
                Value::Num(0.5),
                Value::from("BinLimits"),
                Value::Tensor(bin_limits),
                Value::from("Normalization"),
                Value::from("countdensity"),
            ],
        )
        .expect("histcounts2 countdensity");
        let density = tensor_from_value(density_eval.into_counts_value());
        let positives: Vec<f64> = density.data.iter().copied().filter(|v| *v > 0.0).collect();
        assert!(!positives.is_empty());
        for value in positives {
            assert!((value - 2.0).abs() < 1e-12);
        }
        assert_eq!(density.data[0], 2.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_cdf_normalization() {
        let x = Tensor::new(vec![0.1, 0.9, 1.2, 1.8], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![0.2, 0.7, 1.4, 1.6], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
                Value::from("Normalization"),
                Value::from("cdf"),
            ],
        )
        .expect("histcounts2");
        let cdf = tensor_from_value(eval.into_counts_value());
        assert_eq!(cdf.shape, vec![2, 2]);
        assert_eq!(cdf.data.last().copied().unwrap(), 1.0);
        assert!(cdf.data.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_integer_bin_method() {
        let x = Tensor::new(vec![0.2, 1.7, 2.1], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.2, 1.8, 2.9], vec![3, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::from("XBinMethod"),
                Value::from("integers"),
                Value::from("YBinMethod"),
                Value::from("integers"),
                Value::from("BinLimits"),
                Value::Tensor(Tensor::new(vec![0.0, 3.0, 1.0, 3.0], vec![4, 1]).unwrap()),
            ],
        )
        .expect("histcounts2");
        let (counts_v, xedges_v, yedges_v) = eval.into_triple();
        let counts = tensor_from_value(counts_v);
        let xedges = tensor_from_value(xedges_v);
        let yedges = tensor_from_value(yedges_v);
        assert_eq!(xedges.data, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(yedges.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(counts.data.iter().sum::<f64>(), 3.0);
        assert!(xedges
            .data
            .windows(2)
            .all(|w| (w[1] - w[0] - 1.0).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_num_bins_zero_errors() {
        let x = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[Value::from("NumBins"), Value::Num(0.0)],
        );
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_binmethod_conflict_errors() {
        let x = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let result = evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::from("XBinEdges"),
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
                Value::from("XBinMethod"),
                Value::from("auto"),
            ],
        );
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let x = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
            let y = Tensor::new(vec![1.0, 1.1, 2.9], vec![3, 1]).unwrap();
            let x_view = runmat_accelerate_api::HostTensorView {
                data: &x.data,
                shape: &x.shape,
            };
            let y_view = runmat_accelerate_api::HostTensorView {
                data: &y.data,
                shape: &y.shape,
            };
            let x_handle = provider.upload(&x_view).expect("upload X");
            let y_handle = provider.upload(&y_view).expect("upload Y");
            let eval = evaluate(
                Value::GpuTensor(x_handle),
                Value::GpuTensor(y_handle),
                &[
                    Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                    Value::Tensor(Tensor::new(vec![0.0, 2.0, 3.0], vec![3, 1]).unwrap()),
                ],
            )
            .expect("histcounts2");
            let counts = tensor_from_value(eval.into_counts_value());
            assert_eq!(counts.shape, vec![3, 2]);
            let rows = counts.shape[0];
            let count = |ix: usize, iy: usize| counts.data[ix + iy * rows];
            assert_eq!(count(0, 0), 1.0);
            assert_eq!(count(1, 0), 1.0);
            assert_eq!(count(2, 1), 1.0);
            assert_eq!(count(2, 0), 0.0);
            assert_eq!(count(0, 1), 0.0);
            assert_eq!(count(1, 1), 0.0);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn histcounts2_wgpu_roundtrip() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let x = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 1.1, 2.9], vec![3, 1]).unwrap();

        let x_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &x.data,
                shape: &x.shape,
            })
            .expect("upload x");
        let y_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &y.data,
                shape: &y.shape,
            })
            .expect("upload y");

        let eval = evaluate(
            Value::GpuTensor(x_handle),
            Value::GpuTensor(y_handle),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            ],
        )
        .expect("histcounts2");

        let (counts_v, xedges_v, yedges_v) = eval.into_triple();
        let counts = tensor_from_value(counts_v);
        let xedges = tensor_from_value(xedges_v);
        let yedges = tensor_from_value(yedges_v);

        assert_eq!(xedges.data, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(yedges.data, vec![0.0, 2.0, 3.0]);

        assert_eq!(counts.shape, vec![3, 2]);
        let rows = counts.shape[0];
        let count = |ix: usize, iy: usize| counts.data[ix + iy * rows];
        assert_eq!(count(0, 0), 1.0);
        assert_eq!(count(1, 0), 1.0);
        assert_eq!(count(2, 1), 1.0);
        assert_eq!(count(2, 0), 0.0);
        assert_eq!(count(0, 1), 0.0);
        assert_eq!(count(1, 1), 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
