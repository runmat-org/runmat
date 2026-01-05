//! MATLAB-compatible `histcounts` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

const DEFAULT_BIN_COUNT: usize = 10;
const RANGE_EPS: f64 = 1.0e-12;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "histcounts",
        builtin_path = "crate::builtins::stats::hist::histcounts"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "histcounts"
category: "stats/hist"
keywords: ["histcounts", "histogram", "binning", "normalization", "probability", "cdf", "gpu"]
summary: "Count observations in numeric arrays using configurable histogram bins."
references:
  - https://www.mathworks.com/help/matlab/ref/histcounts.html
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Falls back to host execution today; providers can implement a custom histogram kernel via the `histcounts` hook."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::stats::hist::histcounts::tests"
  integration: "builtins::stats::hist::histcounts::tests::histcounts_gpu_roundtrip"
  gpu: "builtins::stats::hist::histcounts::tests::histcounts_wgpu_roundtrip"
---

# What does the `histcounts` function do in MATLAB / RunMat?
`histcounts` tallies the number of elements that fall within each histogram bin.
Bins can be specified explicitly, derived from a target bin width, or chosen by
the default heuristics so that both simple and advanced workflows mirror
MathWorks MATLAB semantics.

## How does the `histcounts` function behave in MATLAB / RunMat?
- `histcounts(X)` flattens numeric or logical inputs column-major and returns a
  row vector of counts spread across ten equal-width bins spanning the data range.
- `histcounts(X, N)` partitions the data into `N` equally spaced bins.
- `histcounts(X, edges)` counts observations using the supplied bin edges.
- Name/value pairs such as `'BinWidth'`, `'BinLimits'`, `'NumBins'`, `'BinEdges'`,
  `'BinMethod'`, and `'Normalization'` follow MATLAB's precedence rules and validation logic.
- Values outside the bin limits are excluded. The last bin includes its upper
  edge while all other bins are half-open on the right.
- `NaN` values are ignored; `Inf` and `-Inf` participate when the edges cover them.

## `histcounts` Function GPU Execution Behaviour
When the input arrives as a `gpuArray`, RunMat gathers the samples to host
memory, executes the CPU reference implementation, and materialises the results
as ordinary tensors. The builtin is registered as a sink, so fusion plans flush
residency before histogramming and the outputs always live on the host today.
The acceleration layer exposes a `histcounts` provider hook; once GPU kernels
are implemented, existing code will pick up device-side execution
automatically.

## Examples of using the `histcounts` function in MATLAB / RunMat

### Counting values with custom bin counts
```matlab
data = [1 2 2 4 5 7];
[counts, edges] = histcounts(data, 3);
```
Expected output:
```matlab
counts = [3 1 2];
edges  = [1 3 5 7];
```

### Using explicit bin edges
```matlab
edges = [0 1 2 3];
counts = histcounts([0.1 0.5 0.9 1.2 1.8 2.1], edges);
```
Expected output:
```matlab
counts = [3 2 1];
```

### Setting bin width and limits
```matlab
[counts, edges] = histcounts([5 7 8 10 12], 'BinWidth', 2, 'BinLimits', [4 12]);
```
Expected output:
```matlab
counts = [1 1 1 2];
edges  = [4 6 8 10 12];
```

### Choosing an automatic binning method
```matlab
[counts, edges] = histcounts(randn(1, 500), 'BinMethod', 'sturges');
```
Expected output (counts will vary; edges obey Sturges' rule):
```matlab
numel(counts) = ceil(log2(500) + 1);   % 10 bins
```

### Normalising counts to probabilities
```matlab
counts = histcounts([0.2 0.4 1.1 1.4 1.8 2.5], [0 1 2 3], 'Normalization', 'probability');
```
Expected output:
```matlab
counts = [0.3333 0.5000 0.1667];
```

### Building a cumulative distribution
```matlab
counts = histcounts([1 2 2 3], [0 1 2 3], 'Normalization', 'cdf');
```
Expected output:
```matlab
counts = [0 0.25 1];
```

### Counting values stored on a GPU array
```matlab
G = gpuArray([0.5 1.5 2.5]);
[counts, edges] = histcounts(G, [0 1 2 3]);   % counts/edges return as CPU arrays
```
Expected output:
```matlab
counts = [1 1 1];
edges  = [0 1 2 3];
```

## FAQ

### Why does the last bin include its upper edge?
To match MATLAB semantics each bin is `[left, right)` except for the final bin,
which is `[left, right]`. This ensures the maximum finite value is always
counted.

### How are `NaN` values handled?
They are ignored entirely and do not contribute to any bin count. Infinite
values participate as long as the bin edges include them.

### What happens when all observations are identical?
RunMat mirrors MATLAB by collapsing the histogram to a single bin centred on the
shared value unless you explicitly supply edges, limits, or a bin width.

### Does `histcounts` support non-double inputs?
Yes. Logical inputs are promoted to doubles, integer types are converted to
`double`, and gpuArray inputs are gathered to host memory in this release.

### Can I request both `'BinEdges'` and `'BinWidth'`?
No. Bin specifications are mutually exclusive—choose one of `'BinEdges'`,
`'BinWidth'`, or `'NumBins'`, optionally constrained by `'BinLimits'`.

### How do probability and PDF normalisations differ?
`'probability'` scales counts so that they sum to one. `'pdf'` divides by both
bin width and the total count, matching MATLAB's probability-density definition.

### Do outputs stay on the GPU when the input is a `gpuArray`?
Until specialised provider hooks land, RunMat gathers GPU data to the CPU and
returns host-resident outputs. Use `gather` only for clarity; the values are
already in host memory.

## See Also
[linspace](./linspace), [sum](./sum), [mean](./mean), [rand](./rand)

## Source & Feedback
- The full source code lives at `crates/runmat-runtime/src/builtins/stats/hist/histcounts.rs`.
- Found a discrepancy? Please open an issue with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::hist::histcounts")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "histcounts",
    op_kind: GpuOpKind::Custom("histcounts"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("histcounts")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Omit,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement device-side histogramming via the custom hook; current builds gather to host memory.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::hist::histcounts")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "histcounts",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Histogram binning materialises counts on the host and terminates fusion chains.",
};

#[runtime_builtin(
    name = "histcounts",
    category = "stats/hist",
    summary = "Count observations in numeric arrays using configurable histogram bins.",
    keywords = "histcounts,histogram,binning,normalization,probability,cdf,gpu",
    accel = "reduction",
    sink = true,
    builtin_path = "crate::builtins::stats::hist::histcounts"
)]
fn histcounts_builtin(data: Value, rest: Vec<Value>) -> Result<Value, String> {
    evaluate(data, &rest).map(|eval| eval.into_counts_value())
}

/// Evaluate `histcounts` once and surface both primary outputs.
pub fn evaluate(data: Value, rest: &[Value]) -> Result<HistcountsEvaluation, String> {
    let options = parse_options(rest)?;
    match data {
        Value::GpuTensor(handle) => histcounts_gpu(handle, &options),
        other => histcounts_host(other, &options),
    }
}

fn histcounts_gpu(
    handle: GpuTensorHandle,
    options: &HistcountsOptions,
) -> Result<HistcountsEvaluation, String> {
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    histcounts_from_tensor(tensor, options)
}

fn histcounts_host(
    value: Value,
    options: &HistcountsOptions,
) -> Result<HistcountsEvaluation, String> {
    let tensor = tensor::value_into_tensor_for("histcounts", value)?;
    histcounts_from_tensor(tensor, options)
}

fn histcounts_from_tensor(
    tensor: Tensor,
    options: &HistcountsOptions,
) -> Result<HistcountsEvaluation, String> {
    let mut values = Vec::new();
    let mut min_val: Option<f64> = None;
    let mut max_val: Option<f64> = None;

    for &sample in &tensor.data {
        if sample.is_nan() {
            continue;
        }
        values.push(sample);
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

    let edges = compute_edges(&values, min_val, max_val, original_range_zero, options)?;
    let mut counts = vec![0.0f64; edges.len() - 1];

    for value in values {
        if value < edges[0] || value > edges[edges.len() - 1] {
            continue;
        }
        if value == edges[edges.len() - 1] {
            if let Some(last) = counts.last_mut() {
                *last += 1.0;
            }
            continue;
        }

        match edges.binary_search_by(|edge| match edge.partial_cmp(&value) {
            Some(order) => order,
            None => Ordering::Less,
        }) {
            Ok(index) => {
                if index == 0 {
                    counts[0] += 1.0;
                } else if index < counts.len() {
                    counts[index] += 1.0;
                } else if let Some(last) = counts.last_mut() {
                    *last += 1.0;
                }
            }
            Err(index) => {
                if index == 0 || index > counts.len() {
                    continue;
                }
                counts[index - 1] += 1.0;
            }
        }
    }

    let normalised = apply_normalization(&counts, &edges, options.normalization);
    let counts_tensor =
        Tensor::new(normalised, vec![1, counts.len()]).map_err(|e| format!("histcounts: {e}"))?;
    let edges_tensor =
        Tensor::new(edges.clone(), vec![1, edges.len()]).map_err(|e| format!("histcounts: {e}"))?;

    Ok(HistcountsEvaluation::new(counts_tensor, edges_tensor))
}

fn compute_edges(
    values: &[f64],
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    options: &HistcountsOptions,
) -> Result<Vec<f64>, String> {
    if let Some(edges) = &options.explicit_edges {
        validate_edges(edges)?;
        return Ok(edges.clone());
    }

    if let Some(method) = options.bin_method {
        return compute_edges_with_method(
            values,
            min_val,
            max_val,
            original_range_zero,
            method,
            options,
        );
    }

    compute_edges_standard(min_val, max_val, original_range_zero, options)
}

fn compute_edges_standard(
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    options: &HistcountsOptions,
) -> Result<Vec<f64>, String> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(
            "histcounts: data range must be finite; specify BinLimits or BinEdges".to_string(),
        );
    }

    if upper < lower {
        return Err("histcounts: bin limits must be increasing".to_string());
    }

    if options.bin_limits.is_some() && approx_equal(lower, upper) {
        return Err("histcounts: BinLimits must specify a non-zero width".to_string());
    }

    if let Some(width) = options.bin_width {
        if !width.is_finite() || width <= 0.0 {
            return Err("histcounts: BinWidth must be a positive finite scalar".to_string());
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
        validate_edges(&edges)?;
        return Ok(edges);
    }

    let mut num_bins = options.num_bins.unwrap_or(DEFAULT_BIN_COUNT);
    if num_bins == 0 {
        return Err("histcounts: NumBins must be a positive integer".to_string());
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
    validate_edges(&edges)?;
    Ok(edges)
}

fn compute_edges_with_method(
    values: &[f64],
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    method: BinMethod,
    options: &HistcountsOptions,
) -> Result<Vec<f64>, String> {
    if values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options);
    }

    if matches!(method, BinMethod::Integers) {
        let edges = compute_integer_edges(min_val, max_val, options)?;
        validate_edges(&edges)?;
        return Ok(edges);
    }

    let (lower, upper) = derive_initial_limits(min_val, max_val, options.bin_limits);
    if !lower.is_finite() || !upper.is_finite() {
        return Err("histcounts: data range must be finite for BinMethod".to_string());
    }

    if approx_equal(lower, upper) {
        if options.bin_limits.is_some() {
            return Err("histcounts: BinLimits must specify a non-zero width".to_string());
        }
        return compute_edges_standard(min_val, max_val, true, options);
    }

    let finite_values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options);
    }

    let range = upper - lower;
    if range <= 0.0 {
        return compute_edges_standard(min_val, max_val, true, options);
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
        return compute_edges_standard(min_val, max_val, original_range_zero, options);
    }

    let bins = ((range / width).ceil().max(1.0)) as usize;
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(lower + width * i as f64);
    }
    if let Some(last) = edges.last_mut() {
        *last = upper;
    }
    validate_edges(&edges)?;
    Ok(edges)
}

fn compute_integer_edges(
    min_val: Option<f64>,
    max_val: Option<f64>,
    options: &HistcountsOptions,
) -> Result<Vec<f64>, String> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err("histcounts: BinLimits must be finite for 'integers' BinMethod".to_string());
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

fn apply_normalization(counts: &[f64], edges: &[f64], mode: HistogramNormalization) -> Vec<f64> {
    let widths: Vec<f64> = edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let total: f64 = counts.iter().sum();

    match mode {
        HistogramNormalization::Count => counts.to_vec(),
        HistogramNormalization::Probability => {
            if total > 0.0 {
                counts.iter().map(|&c| c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CountDensity => counts
            .iter()
            .zip(widths.iter())
            .map(|(&c, &w)| if w > 0.0 { c / w } else { 0.0 })
            .collect(),
        HistogramNormalization::Pdf => {
            if total > 0.0 {
                counts
                    .iter()
                    .zip(widths.iter())
                    .map(|(&c, &w)| if w > 0.0 { c / (total * w) } else { 0.0 })
                    .collect()
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

fn validate_edges(edges: &[f64]) -> Result<(), String> {
    if edges.len() < 2 {
        return Err("histcounts: bin edges must contain at least two elements".to_string());
    }
    for pair in edges.windows(2) {
        if pair[0].is_nan() || pair[1].is_nan() {
            return Err("histcounts: bin edges must be finite numbers".to_string());
        }
        if pair[1] <= pair[0] {
            return Err("histcounts: bin edges must be strictly increasing".to_string());
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

#[derive(Clone, Default)]
struct HistcountsOptions {
    explicit_edges: Option<Vec<f64>>,
    num_bins: Option<usize>,
    bin_width: Option<f64>,
    bin_limits: Option<(f64, f64)>,
    bin_method: Option<BinMethod>,
    normalization: HistogramNormalization,
}

impl HistcountsOptions {
    fn validate(&self) -> Result<(), String> {
        if self.explicit_edges.is_some()
            && (self.num_bins.is_some() || self.bin_width.is_some() || self.bin_limits.is_some())
        {
            return Err(
                "histcounts: BinEdges cannot be combined with NumBins, BinWidth, or BinLimits"
                    .to_string(),
            );
        }
        if self.bin_method.is_some()
            && (self.explicit_edges.is_some()
                || self.bin_width.is_some()
                || self.num_bins.is_some())
        {
            return Err(
                "histcounts: BinMethod cannot be combined with BinEdges, NumBins, or BinWidth"
                    .to_string(),
            );
        }
        if self.num_bins.is_some() && self.bin_width.is_some() {
            return Err("histcounts: specify only one of NumBins or BinWidth".to_string());
        }
        if let Some((lo, hi)) = self.bin_limits {
            if !lo.is_finite() || !hi.is_finite() {
                return Err("histcounts: BinLimits must be finite".to_string());
            }
            if hi < lo {
                return Err("histcounts: BinLimits must be increasing".to_string());
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
pub struct HistcountsEvaluation {
    counts: Tensor,
    edges: Tensor,
}

impl HistcountsEvaluation {
    fn new(counts: Tensor, edges: Tensor) -> Self {
        Self { counts, edges }
    }

    pub fn into_counts_value(self) -> Value {
        Value::Tensor(self.counts)
    }

    pub fn into_pair(self) -> (Value, Value) {
        let counts = self.counts;
        let edges = self.edges;
        (Value::Tensor(counts), Value::Tensor(edges))
    }
}

fn parse_options(args: &[Value]) -> Result<HistcountsOptions, String> {
    let mut options = HistcountsOptions::default();
    let mut index = 0;

    if index < args.len() && !is_option_key(&args[index]) {
        match classify_bin_argument(&args[index])? {
            BinArgument::NumBins(n) => {
                options.num_bins = Some(n);
            }
            BinArgument::Edges(edges) => {
                validate_edges(&edges)?;
                options.explicit_edges = Some(edges);
            }
        }
        index += 1;
    }

    while index < args.len() {
        let key = tensor::value_to_string(&args[index])
            .ok_or_else(|| "histcounts: expected name/value pair arguments".to_string())?;
        index += 1;
        if index >= args.len() {
            return Err(format!("histcounts: missing value for option '{key}'"));
        }
        let lowered = key.trim().to_ascii_lowercase();
        let value = &args[index];
        index += 1;

        match lowered.as_str() {
            "binedges" => {
                let edges = numeric_vector(value, "histcounts", "BinEdges")?;
                validate_edges(&edges)?;
                options.explicit_edges = Some(edges);
            }
            "numbins" => {
                let scalar = positive_usize(value, "histcounts", "NumBins")?;
                options.num_bins = Some(scalar);
            }
            "binwidth" => {
                let width = positive_scalar(value, "histcounts", "BinWidth")?;
                options.bin_width = Some(width);
            }
            "binlimits" => {
                let limits = numeric_vector(value, "histcounts", "BinLimits")?;
                if limits.len() != 2 {
                    return Err(
                        "histcounts: BinLimits must contain exactly two elements".to_string()
                    );
                }
                let lo = limits[0];
                let hi = limits[1];
                if hi < lo {
                    return Err("histcounts: BinLimits must be increasing".to_string());
                }
                if !lo.is_finite() || !hi.is_finite() {
                    return Err("histcounts: BinLimits must be finite".to_string());
                }
                options.bin_limits = Some((lo, hi));
            }
            "normalization" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| "histcounts: Normalization must be a string".to_string())?;
                options.normalization = parse_normalization(&text)?;
            }
            "binmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| "histcounts: BinMethod must be a string".to_string())?;
                options.bin_method = Some(parse_bin_method(&text)?);
            }
            other => {
                return Err(format!("histcounts: unrecognised option '{other}'"));
            }
        }
    }

    options.validate()?;
    Ok(options)
}

#[derive(Debug)]
enum BinArgument {
    NumBins(usize),
    Edges(Vec<f64>),
}

fn classify_bin_argument(value: &Value) -> Result<BinArgument, String> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let n = positive_usize(value, "histcounts", "NumBins")?;
            Ok(BinArgument::NumBins(n))
        }
        Value::Tensor(tensor) => {
            if tensor.data.len() == 1 {
                let scalar_value = tensor.data[0];
                if !scalar_value.is_finite() || scalar_value <= 0.0 {
                    return Err("histcounts: NumBins must be a positive finite scalar".to_string());
                }
                let rounded = scalar_value.round();
                if (scalar_value - rounded).abs() > f64::EPSILON {
                    return Err("histcounts: NumBins must be an integer".to_string());
                }
                Ok(BinArgument::NumBins(rounded as usize))
            } else {
                let edges = tensor.data.clone();
                Ok(BinArgument::Edges(edges))
            }
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)?;
            if tensor.data.len() == 1 {
                let n = tensor.data[0];
                if n <= 0.0 || !n.is_finite() {
                    return Err("histcounts: NumBins must be a positive finite scalar".to_string());
                }
                let rounded = n.round();
                if (n - rounded).abs() > f64::EPSILON {
                    return Err("histcounts: NumBins must be an integer".to_string());
                }
                Ok(BinArgument::NumBins(rounded as usize))
            } else {
                Ok(BinArgument::Edges(tensor.data))
            }
        }
        Value::GpuTensor(_) => {
            Err("histcounts: bin specification cannot be a gpuArray".to_string())
        }
        other => Err(format!(
            "histcounts: unsupported bin specification {:?}",
            other
        )),
    }
}

fn is_option_key(value: &Value) -> bool {
    if let Some(text) = tensor::value_to_string(value) {
        let lowered = text.trim().to_ascii_lowercase();
        matches!(
            lowered.as_str(),
            "binedges" | "numbins" | "binwidth" | "binlimits" | "normalization" | "binmethod"
        )
    } else {
        false
    }
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

fn parse_bin_method(text: &str) -> Result<BinMethod, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(BinMethod::Auto),
        "scott" => Ok(BinMethod::Scott),
        "fd" | "freedmandiaconis" => Ok(BinMethod::Fd),
        "sturges" => Ok(BinMethod::Sturges),
        "sqrt" => Ok(BinMethod::Sqrt),
        "integers" => Ok(BinMethod::Integers),
        other => Err(format!(
            "histcounts: unrecognised BinMethod value '{other}'"
        )),
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
            "histcounts: unrecognised Normalization value '{other}'"
        )),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor, Value};

    fn values_from_tensor(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(t) => t.data,
            Value::Num(n) => vec![n],
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_basic_numbins() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0, 5.0, 7.0], vec![6, 1]).unwrap();
        let eval =
            evaluate(Value::Tensor(tensor), &[Value::Int(IntValue::I32(3))]).expect("histcounts");
        let (counts_val, edges_val) = eval.into_pair();
        assert_eq!(values_from_tensor(counts_val), vec![3.0, 1.0, 2.0]);
        assert_eq!(values_from_tensor(edges_val), vec![1.0, 3.0, 5.0, 7.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_binwidth_and_limits() {
        let tensor = Tensor::new(vec![5.0, 7.0, 8.0, 10.0, 12.0], vec![5, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[
                Value::from("BinWidth"),
                Value::Num(2.0),
                Value::from("BinLimits"),
                Value::Tensor(Tensor::new(vec![4.0, 12.0], vec![2, 1]).unwrap()),
            ],
        )
        .expect("histcounts");
        let (counts_val, edges_val) = eval.into_pair();
        assert_eq!(values_from_tensor(counts_val), vec![1.0, 1.0, 1.0, 2.0]);
        assert_eq!(
            values_from_tensor(edges_val),
            vec![4.0, 6.0, 8.0, 10.0, 12.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_probability_normalization() {
        let data = Tensor::new(vec![0.2, 0.4, 1.1, 1.4, 1.8, 2.5], vec![6, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(data),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                Value::from("Normalization"),
                Value::from("probability"),
            ],
        )
        .expect("histcounts");
        let (counts_val, _) = eval.into_pair();
        let counts = values_from_tensor(counts_val);
        assert!((counts[0] - 0.3333).abs() < 5e-4);
        assert!((counts[1] - 0.5).abs() < 5e-4);
        assert!((counts[2] - 0.1667).abs() < 5e-4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_cdf_normalization() {
        let data = Tensor::new(vec![1.0, 2.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(data),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                Value::from("Normalization"),
                Value::from("cdf"),
            ],
        )
        .expect("histcounts");
        let (counts_val, _) = eval.into_pair();
        let counts = values_from_tensor(counts_val);
        assert_eq!(counts, vec![0.0, 0.25, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_handles_nan() {
        let data = Tensor::new(vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0], vec![5, 1]).unwrap();
        let eval =
            evaluate(Value::Tensor(data), &[Value::Int(IntValue::I32(3))]).expect("histcounts");
        let (counts_val, _) = eval.into_pair();
        assert_eq!(values_from_tensor(counts_val), vec![1.0, 1.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_constant_data_single_bin() {
        let data = Tensor::new(vec![4.0, 4.0, 4.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(data), &[]).expect("histcounts");
        let (counts_val, edges_val) = eval.into_pair();
        assert_eq!(values_from_tensor(counts_val), vec![3.0]);
        assert_eq!(values_from_tensor(edges_val), vec![3.5, 4.5]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_binmethod_sqrt() {
        let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
        let tensor = Tensor::new(data, vec![16, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[Value::from("BinMethod"), Value::from("sqrt")],
        )
        .expect("histcounts");
        let (counts_val, edges_val) = eval.into_pair();
        assert_eq!(values_from_tensor(counts_val).len(), 4);
        assert_eq!(values_from_tensor(edges_val).len(), 5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_binmethod_integers_with_limits() {
        let tensor = Tensor::new(vec![2.2, 2.8, 3.4, 3.9], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[
                Value::from("BinMethod"),
                Value::from("integers"),
                Value::from("BinLimits"),
                Value::Tensor(Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap()),
            ],
        )
        .expect("histcounts");
        let (counts_val, edges_val) = eval.into_pair();
        let counts = values_from_tensor(counts_val);
        let edges = values_from_tensor(edges_val);
        assert_eq!(counts, vec![2.0, 2.0]);
        assert_eq!(edges, vec![2.0, 3.0, 4.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_binmethod_conflict_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = evaluate(
            Value::Tensor(tensor),
            &[
                Value::from("BinMethod"),
                Value::from("auto"),
                Value::from("NumBins"),
                Value::Num(5.0),
            ],
        );
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_invalid_binwidth_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = evaluate(
            Value::Tensor(tensor),
            &[Value::from("BinWidth"), Value::Num(0.0)],
        );
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(
                Value::GpuTensor(handle),
                &[Value::Tensor(
                    Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap(),
                )],
            )
            .expect("histcounts");
            let (counts_val, edges_val) = eval.into_pair();
            assert_eq!(values_from_tensor(counts_val), vec![1.0, 1.0, 1.0]);
            assert_eq!(values_from_tensor(edges_val), vec![0.0, 1.0, 2.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn histcounts_wgpu_roundtrip() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let eval = evaluate(
            Value::GpuTensor(handle),
            &[Value::Tensor(
                Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap(),
            )],
        )
        .expect("histcounts");
        let (counts_val, edges_val) = eval.into_pair();
        assert_eq!(values_from_tensor(counts_val), vec![1.0, 1.0, 1.0]);
        assert_eq!(values_from_tensor(edges_val), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
