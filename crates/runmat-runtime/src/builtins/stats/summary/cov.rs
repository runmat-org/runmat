//! MATLAB-compatible `cov` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{CovNormalization, CovRows, CovarianceOptions};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "cov",
        builtin_path = "crate::builtins::stats::summary::cov"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "cov"
category: "stats/summary"
keywords: ["cov", "covariance", "statistics", "weighted covariance", "gpu"]
summary: "Compute covariance matrices for vectors, matrices, or paired data sets."
references:
  - https://www.mathworks.com/help/matlab/ref/cov.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Runs on the GPU when rows='all' and no weight vector is supplied; other modes transparently fall back to the CPU reference path."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::stats::summary::cov::tests"
  integration: "builtins::stats::summary::cov::tests::cov_gpu_roundtrip"
---

# What does the `cov` function do in MATLAB / RunMat?
`cov` returns covariance matrices for numeric data. Columns represent variables and rows are
observations. You can pass in one matrix, two matching data sets, or supply observation weights
and row-handling options that mirror MATLAB.

## How does the `cov` function behave in MATLAB / RunMat?
- `cov(X)` treats each column of `X` as a variable and returns a square covariance matrix.
- `cov(X, Y)` concatenates `X` and `Y` column-wise (they must have the same number of rows) before computing the covariance.
- The second argument can be the normalization flag `0` (default) or `1`, matching MATLAB's unbiased and biased estimators.
- You can pass a weight vector to obtain frequency-weighted covariance.
- `'omitrows'` drops rows containing `NaN` or `Inf` before the covariance is computed.
- `'partialrows'` performs pairwise deletion: each covariance entry uses only the rows that contain finite values for that column pair.

## `cov` Function GPU Execution Behaviour
RunMat invokes provider-specific GPU kernels when:

1. All inputs already reside on the GPU;
2. No weight vector is supplied;
3. The rows option is `'all'`; and
4. The active provider exposes the custom `covariance` hook.

If any of these conditions is not met, RunMat gathers the data to the host, evaluates the
reference implementation, and returns a dense host tensor. This guarantees MATLAB-compatible
behaviour regardless of GPU support.

## Examples of using the `cov` function in MATLAB / RunMat

### Computing covariance of columns in a matrix
```matlab
X = [4.0 2.0 0.60;
     4.2 2.1 0.59;
     3.9 2.0 0.58;
     4.3 2.1 0.62;
     4.1 2.2 0.63];
C = cov(X);
```
Expected output:
```matlab
C =
    0.0250    0.0075    0.0018
    0.0075    0.0070    0.0014
    0.0018    0.0014    0.0004
```

### Covariance between two vectors
```matlab
x = [1 2 3 4]';
y = [10 11 9 12]';
C = cov(x, y);
```
Expected output:
```matlab
C =
    1.6667    1.5000
    1.5000    1.6667
```

### Weighted covariance with observation weights
```matlab
X = [4.0 2.0;
     4.2 2.1;
     3.9 2.0;
     4.3 2.1;
     4.1 2.2];
w = [1 1 1 2 2];
Cw = cov(X, w);
```
Expected output:
```matlab
Cw =
    0.0224    0.0050
    0.0050    0.0067
```

### Ignoring rows that contain missing values
```matlab
X = [1   NaN 2;
     3   4   5;
     NaN 6   7;
     8   9   10];
C = cov(X, 'omitrows');
```
Expected output:
```matlab
C =
   12.5000   12.5000   12.5000
   12.5000   12.5000   12.5000
   12.5000   12.5000   12.5000
```

### Pairwise covariance with staggered NaNs
```matlab
X = [ 1   2   NaN;
      4   NaN 6;
      7   8   9];
C = cov(X, 'partialrows');
```
Expected output:
```matlab
C =
    9.0000   18.0000    4.5000
   18.0000   18.0000       NaN
    4.5000       NaN    4.5000
```

### Running covariance on `gpuArray` inputs
```matlab
G = gpuArray(X);        % reuse matrix from earlier examples
CG = cov(G);
CG_host = gather(CG);
```
Expected output:
```matlab
CG_host =
    0.0250    0.0075    0.0018
    0.0075    0.0070    0.0014
    0.0018    0.0014    0.0004
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray`. Expressions such as `cov(sin(X))` keep temporary
results on the GPU as long as the active provider handles the operation. The builtin gathers to
the CPU only when weights, `'omitrows'`, or `'partialrows'` are requested, or when the provider
does not implement the covariance hook. Explicitly calling `gpuArray` remains supported for
MATLAB compatibility and to seed GPU residency when you are unsure about planner decisions.

## FAQ

### Does `cov` support biased and unbiased estimators?
Yes. The default is the unbiased estimator (divide by *N - 1*). Passing `1` as the second argument
switches to the biased estimator (divide by *N*), matching MATLAB.

### How do I provide observation weights?
Supply a weight vector whose length equals the number of observations. The covariance is frequency-weighted using the MATLAB formula. Weighted covariance currently falls back to the CPU implementation when running on the GPU.

### What happens when columns contain constant values?
The diagonal entries become zero, and off-diagonal entries involving the constant column are zero. Any slight negative values caused by floating-point noise are clamped to zero.

### How are `NaN` and `Inf` handled?
By default (`'all'`), non-finite values propagate `NaN` into the affected covariance entries.
`'omitrows'` drops rows containing non-finite values, while `'partialrows'` recomputes each
covariance entry using only rows that are finite for the relevant column pair.

### Can I call `cov` on logical inputs?
Yes. Logical arrays are converted to double precision (`true → 1.0`, `false → 0.0`) before the
covariance is computed, matching MATLAB's behaviour.

## See Also
[corrcoef](./corrcoef), [mean](../../math/reduction/mean), [sum](../../math/reduction/sum), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `cov` function is available at: [`crates/runmat-runtime/src/builtins/stats/summary/cov.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/cov.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::summary::cov")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cov",
    op_kind: GpuOpKind::Custom("summary-stats"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("covariance")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "GPU execution is available when rows='all' and no weight vector is supplied; other cases fall back to the CPU path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::summary::cov")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cov",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "The covariance builtin is treated as a fusion boundary and executes via dedicated kernels or the host reference.",
};

#[runtime_builtin(
    name = "cov",
    category = "stats/summary",
    summary = "Compute covariance matrices for vectors, matrices, or paired data sets.",
    keywords = "cov,covariance,statistics,weights,gpu",
    accel = "reduction",
    builtin_path = "crate::builtins::stats::summary::cov"
)]
fn cov_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let args = CovArgs::parse(value, rest)?;
    if let Some(result) = cov_try_gpu(&args)? {
        return Ok(result);
    }
    cov_host(args)
}

/// Public entry point for providers that need the reference implementation.
pub fn cov_from_tensors(
    left: Tensor,
    right: Option<Tensor>,
    rows: CovRows,
    weight: CovWeightSpec,
) -> Result<Tensor, String> {
    let matrix = combine_tensors(left, right)?;
    if let CovWeightSpec::Vector(ref vec) = weight {
        if matrix.rows != vec.len() {
            return Err(format!(
                "cov: weight vector must contain {} elements",
                matrix.rows
            ));
        }
    }
    match rows {
        CovRows::All => covariance_dense(&matrix, &weight),
        CovRows::OmitRows => {
            let (filtered, filtered_weight) = filter_complete_rows(&matrix, weight);
            covariance_dense(&filtered, &filtered_weight)
        }
        CovRows::PartialRows => covariance_pairwise(&matrix, &weight),
    }
}

#[derive(Debug)]
struct CovArgs {
    first: Value,
    second: Option<Value>,
    normalization: CovNormalization,
    rows: CovRows,
    weight_vector: Option<Value>,
}

impl CovArgs {
    fn parse(first: Value, rest: Vec<Value>) -> Result<Self, String> {
        let mut second_candidate: Option<Value> = None;
        let mut weight_candidate: Option<Value> = None;
        let mut normalization = CovNormalization::Unbiased;
        let mut normalization_explicit = false;
        let mut rows = CovRows::All;

        let iter = rest.into_iter();
        for arg in iter {
            match arg {
                Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                    let key = tensor::value_to_string(&arg)
                        .ok_or_else(|| "cov: expected string option".to_string())?;
                    let lowered = key.trim().to_ascii_lowercase();
                    rows = parse_rows_option(&lowered)?;
                }
                Value::Tensor(_) | Value::LogicalArray(_) | Value::GpuTensor(_) => {
                    if second_candidate.is_none() {
                        second_candidate = Some(arg);
                    } else if weight_candidate.is_none() {
                        weight_candidate = Some(arg);
                    } else {
                        return Err("cov: too many array arguments".to_string());
                    }
                }
                Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
                    if normalization_explicit || weight_candidate.is_some() {
                        return Err("cov: normalization flag specified more than once".to_string());
                    }
                    normalization = parse_normalization(arg)?;
                    normalization_explicit = true;
                }
                Value::ComplexTensor(_) => {
                    return Err("cov: complex inputs are not supported yet".to_string());
                }
                other => {
                    return Err(format!("cov: unsupported argument type {:?}", other));
                }
            }
        }

        if let Some(weight_array) = weight_candidate {
            // Explicit weight vector always takes precedence over dataset detection.
            return Ok(Self {
                first,
                second: second_candidate,
                normalization,
                rows,
                weight_vector: Some(weight_array),
            });
        }

        let mut second = second_candidate;
        let mut weight_vector: Option<Value> = None;

        if let Some(candidate) = second.take() {
            if should_treat_as_weight(&first, &candidate, normalization_explicit, rows)? {
                weight_vector = Some(candidate);
            } else {
                second = Some(candidate);
            }
        }

        Ok(Self {
            first,
            second,
            normalization,
            rows,
            weight_vector,
        })
    }
}

#[derive(Debug, Clone)]
pub enum CovWeightSpec {
    Scalar(CovNormalization),
    Vector(Vec<f64>),
}

fn cov_try_gpu(args: &CovArgs) -> Result<Option<Value>, String> {
    if args.rows != CovRows::All || args.weight_vector.is_some() {
        return Ok(None);
    }

    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let first_handle = match &args.first {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };

    let maybe_second_handle = match &args.second {
        Some(Value::GpuTensor(handle)) => Some(handle),
        Some(_) => return Ok(None),
        None => None,
    };

    let options = CovarianceOptions {
        normalization: args.normalization,
        rows: args.rows,
        has_weight_vector: false,
    };

    match provider.covariance(first_handle, maybe_second_handle, None, &options) {
        Ok(result) => Ok(Some(Value::GpuTensor(result))),
        Err(_) => Ok(None),
    }
}

fn cov_host(args: CovArgs) -> Result<Value, String> {
    let CovArgs {
        first,
        second,
        normalization,
        rows,
        weight_vector,
    } = args;

    let left = value_to_tensor_gather(first)?;
    let right = match second {
        Some(value) => Some(value_to_tensor_gather(value)?),
        None => None,
    };

    let weight_spec = if let Some(weight_value) = weight_vector {
        let vector = value_to_weight_vector(weight_value, left.rows())?;
        CovWeightSpec::Vector(vector)
    } else {
        CovWeightSpec::Scalar(normalization)
    };

    let tensor = cov_from_tensors(left, right, rows, weight_spec)?;
    Ok(Value::Tensor(tensor))
}

fn value_to_tensor_gather(value: Value) -> Result<Tensor, String> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor(&handle),
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical),
        other => tensor::value_into_tensor_for("cov", other),
    }
}

fn value_to_weight_vector(value: Value, expected_rows: usize) -> Result<Vec<f64>, String> {
    let tensor = match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor(&handle)?,
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)?,
        other => tensor::value_into_tensor_for("cov", other)?,
    };
    if tensor.shape.len() > 2 {
        return Err("cov: weight vector must be one-dimensional".to_string());
    }
    if tensor.rows() != expected_rows && tensor.cols() != expected_rows {
        return Err(format!(
            "cov: weight vector must contain {} elements",
            expected_rows
        ));
    }
    for (idx, weight) in tensor.data.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(format!(
                "cov: weights must be non-negative finite values (index {idx})"
            ));
        }
    }
    if tensor.data.is_empty() {
        return Err("cov: weight vector cannot be empty".to_string());
    }
    Ok(tensor.data)
}

fn parse_rows_option(value: &str) -> Result<CovRows, String> {
    match value {
        "all" => Ok(CovRows::All),
        "omitrows" | "omit" => Ok(CovRows::OmitRows),
        "partialrows" | "partial" | "pairwise" => Ok(CovRows::PartialRows),
        other => Err(format!("cov: unknown rows option '{other}'")),
    }
}

fn parse_normalization(value: Value) -> Result<CovNormalization, String> {
    match value {
        Value::Int(i) => match i.to_i64() {
            0 => Ok(CovNormalization::Unbiased),
            1 => Ok(CovNormalization::Biased),
            other => Err(format!(
                "cov: normalization flag must be 0 or 1, received {other}"
            )),
        },
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("cov: normalization flag must be finite".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > 1.0e-12 {
                return Err("cov: normalization flag must be an integer".to_string());
            }
            match rounded as i64 {
                0 => Ok(CovNormalization::Unbiased),
                1 => Ok(CovNormalization::Biased),
                other => Err(format!(
                    "cov: normalization flag must be 0 or 1, received {other}"
                )),
            }
        }
        Value::Bool(flag) => {
            if flag {
                Ok(CovNormalization::Biased)
            } else {
                Ok(CovNormalization::Unbiased)
            }
        }
        other => Err(format!(
            "cov: normalization flag must be numeric, received {:?}",
            other
        )),
    }
}

fn should_treat_as_weight(
    first: &Value,
    candidate: &Value,
    normalization_explicit: bool,
    rows_option: CovRows,
) -> Result<bool, String> {
    let (rows_first, cols_first) = value_rows_cols(first)?;
    let (rows_candidate, cols_candidate) = value_rows_cols(candidate)?;

    let is_vector = rows_candidate == 1
        || cols_candidate == 1
        || rows_candidate * cols_candidate == rows_candidate
            && (rows_candidate == rows_first || cols_candidate == rows_first);

    if !is_vector {
        return Ok(false);
    }

    if rows_candidate != rows_first && cols_candidate != rows_first {
        // Length mismatch, treat as dataset so the later validation emits the proper error.
        return Ok(false);
    }

    if cols_first == 1 && !normalization_explicit && matches!(rows_option, CovRows::All) {
        // Ambiguous `cov(x, y)` case – prefer dataset semantics for compatibility.
        return Ok(false);
    }

    Ok(true)
}

fn value_rows_cols(value: &Value) -> Result<(usize, usize), String> {
    match value {
        Value::Tensor(tensor) => Ok((tensor.rows(), tensor.cols())),
        Value::LogicalArray(array) => {
            if array.shape.len() > 2 {
                return Err("cov: inputs must be 2-D matrices or vectors".to_string());
            }
            let rows = if array.shape.is_empty() {
                1
            } else {
                array.shape[0]
            };
            let cols = if array.shape.len() >= 2 {
                array.shape[1]
            } else {
                1
            };
            Ok((rows, cols))
        }
        Value::GpuTensor(handle) => {
            if handle.shape.len() > 2 {
                return Err("cov: inputs must be 2-D matrices or vectors".to_string());
            }
            let rows = if handle.shape.is_empty() {
                1
            } else {
                handle.shape[0]
            };
            let cols = if handle.shape.len() >= 2 {
                handle.shape[1]
            } else {
                1
            };
            Ok((rows, cols))
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok((1, 1)),
        other => Err(format!(
            "cov: unsupported input type for shape inspection: {:?}",
            other
        )),
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn from_tensor(name: &str, tensor: Tensor) -> Result<Self, String> {
        if tensor.shape.len() > 2 {
            return Err(format!("{name}: inputs must be 2-D matrices or vectors"));
        }
        Ok(Self {
            rows: tensor.rows(),
            cols: tensor.cols(),
            data: tensor.data,
        })
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }

    #[inline]
    fn column(&self, col: usize) -> &[f64] {
        let start = col * self.rows;
        let end = start + self.rows;
        &self.data[start..end]
    }
}

fn combine_tensors(left: Tensor, right: Option<Tensor>) -> Result<Matrix, String> {
    let mut matrix = Matrix::from_tensor("cov", left)?;
    if let Some(second) = right {
        let right_matrix = Matrix::from_tensor("cov", second)?;
        if matrix.rows != right_matrix.rows {
            return Err("cov: inputs must have the same number of rows".to_string());
        }
        matrix.cols += right_matrix.cols;
        matrix
            .data
            .extend_from_slice(&right_matrix.data[..right_matrix.rows * right_matrix.cols]);
    }
    Ok(matrix)
}

fn covariance_dense(matrix: &Matrix, weight: &CovWeightSpec) -> Result<Tensor, String> {
    let cols = matrix.cols;
    let rows = matrix.rows;

    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("cov: {e}"));
    }

    let mut result = vec![f64::NAN; cols * cols];

    match weight {
        CovWeightSpec::Scalar(normalization) => {
            let denom = match normalization {
                CovNormalization::Unbiased => (rows as f64) - 1.0,
                CovNormalization::Biased => rows as f64,
            };
            if denom <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(|e| format!("cov: {e}"));
            }

            let mut means = vec![0.0; cols];
            for (col, mean_slot) in means.iter_mut().enumerate() {
                let column = matrix.column(col);
                let mut sum = 0.0;
                let mut valid = true;
                for &value in column {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    sum += value;
                }
                *mean_slot = if valid { sum / (rows as f64) } else { f64::NAN };
            }

            for i in 0..cols {
                for j in i..cols {
                    let value = covariance_unweighted_pair(matrix, i, j, &means, denom);
                    set_entry(&mut result, cols, i, j, sanitize_covariance(i == j, value));
                }
            }
        }
        CovWeightSpec::Vector(weights) => {
            if weights.len() != rows {
                return Err(format!("cov: weight vector must contain {} elements", rows));
            }
            let sum_w: f64 = weights.iter().sum();
            if sum_w <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(|e| format!("cov: {e}"));
            }
            let denom = sum_w - 1.0;
            if denom <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(|e| format!("cov: {e}"));
            }

            let mut means = vec![0.0; cols];
            for (col, mean_slot) in means.iter_mut().enumerate() {
                let column = matrix.column(col);
                let mut weighted_sum = 0.0;
                let mut valid = true;
                for (row, &value) in column.iter().enumerate() {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    weighted_sum += weights[row] * value;
                }
                *mean_slot = if valid {
                    weighted_sum / sum_w
                } else {
                    f64::NAN
                };
            }

            for i in 0..cols {
                for j in i..cols {
                    let value = covariance_weighted_pair(matrix, i, j, weights, &means, denom);
                    set_entry(&mut result, cols, i, j, sanitize_covariance(i == j, value));
                }
            }
        }
    }

    Tensor::new(result, vec![cols, cols]).map_err(|e| format!("cov: {e}"))
}

fn filter_complete_rows(matrix: &Matrix, weight: CovWeightSpec) -> (Matrix, CovWeightSpec) {
    if matrix.rows == 0 {
        return (
            Matrix {
                data: Vec::new(),
                rows: 0,
                cols: matrix.cols,
            },
            weight,
        );
    }

    let mut valid_rows = Vec::new();
    for row in 0..matrix.rows {
        let mut is_valid = true;
        for col in 0..matrix.cols {
            if !matrix.get(row, col).is_finite() {
                is_valid = false;
                break;
            }
        }
        if is_valid {
            valid_rows.push(row);
        }
    }

    if valid_rows.len() == matrix.rows {
        // No filtering required.
        return (matrix.clone(), weight);
    }

    let mut data = Vec::with_capacity(valid_rows.len() * matrix.cols);
    for col in 0..matrix.cols {
        for &row in &valid_rows {
            data.push(matrix.get(row, col));
        }
    }

    let filtered_matrix = Matrix {
        data,
        rows: valid_rows.len(),
        cols: matrix.cols,
    };

    let filtered_weight = match weight {
        CovWeightSpec::Scalar(norm) => CovWeightSpec::Scalar(norm),
        CovWeightSpec::Vector(vec) => {
            let mut filtered = Vec::with_capacity(valid_rows.len());
            for &row in &valid_rows {
                filtered.push(vec[row]);
            }
            CovWeightSpec::Vector(filtered)
        }
    };

    (filtered_matrix, filtered_weight)
}

fn covariance_pairwise(matrix: &Matrix, weight: &CovWeightSpec) -> Result<Tensor, String> {
    let cols = matrix.cols;
    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("cov: {e}"));
    }
    let mut result = vec![f64::NAN; cols * cols];
    for i in 0..cols {
        let variance = covariance_pair(matrix, i, i, weight);
        set_entry(&mut result, cols, i, i, sanitize_covariance(true, variance));
        for j in (i + 1)..cols {
            let value = covariance_pair(matrix, i, j, weight);
            set_entry(&mut result, cols, i, j, sanitize_covariance(false, value));
        }
    }
    Tensor::new(result, vec![cols, cols]).map_err(|e| format!("cov: {e}"))
}

fn covariance_unweighted_pair(
    matrix: &Matrix,
    lhs: usize,
    rhs: usize,
    means: &[f64],
    denom: f64,
) -> f64 {
    if !means[lhs].is_finite() || !means[rhs].is_finite() {
        return f64::NAN;
    }
    let mut accumulator = 0.0;
    for row in 0..matrix.rows {
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        accumulator += (x - means[lhs]) * (y - means[rhs]);
    }
    accumulator / denom
}

fn covariance_weighted_pair(
    matrix: &Matrix,
    lhs: usize,
    rhs: usize,
    weights: &[f64],
    means: &[f64],
    denom: f64,
) -> f64 {
    if !means[lhs].is_finite() || !means[rhs].is_finite() {
        return f64::NAN;
    }
    let mut accumulator = 0.0;
    for (row, &weight) in weights.iter().enumerate().take(matrix.rows) {
        if weight == 0.0 {
            continue;
        }
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        accumulator += weight * (x - means[lhs]) * (y - means[rhs]);
    }
    accumulator / denom
}

fn covariance_pair(matrix: &Matrix, lhs: usize, rhs: usize, weight: &CovWeightSpec) -> f64 {
    match weight {
        CovWeightSpec::Scalar(normalization) => {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for row in 0..matrix.rows {
                let x = matrix.get(row, lhs);
                let y = matrix.get(row, rhs);
                if x.is_finite() && y.is_finite() {
                    xs.push(x);
                    ys.push(y);
                }
            }
            covariance_unweighted_slice(&xs, &ys, *normalization)
        }
        CovWeightSpec::Vector(weights) => {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut ws = Vec::new();
            for (row, &weight) in weights.iter().enumerate().take(matrix.rows) {
                let x = matrix.get(row, lhs);
                let y = matrix.get(row, rhs);
                if x.is_finite() && y.is_finite() {
                    xs.push(x);
                    ys.push(y);
                    ws.push(weight);
                }
            }
            covariance_weighted_slice(&xs, &ys, &ws)
        }
    }
}

fn covariance_unweighted_slice(xs: &[f64], ys: &[f64], normalization: CovNormalization) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return f64::NAN;
    }
    let n = xs.len().min(ys.len());
    if n == 0 {
        return f64::NAN;
    }
    let denom = match normalization {
        CovNormalization::Unbiased => (n as f64) - 1.0,
        CovNormalization::Biased => n as f64,
    };
    if denom <= 0.0 {
        return f64::NAN;
    }
    let sum_x: f64 = xs.iter().take(n).sum();
    let sum_y: f64 = ys.iter().take(n).sum();
    let mean_x = sum_x / (n as f64);
    let mean_y = sum_y / (n as f64);
    let mut accumulator = 0.0;
    for idx in 0..n {
        accumulator += (xs[idx] - mean_x) * (ys[idx] - mean_y);
    }
    accumulator / denom
}

fn covariance_weighted_slice(xs: &[f64], ys: &[f64], weights: &[f64]) -> f64 {
    if xs.is_empty() || ys.is_empty() || weights.is_empty() {
        return f64::NAN;
    }
    let n = xs.len().min(ys.len()).min(weights.len());
    if n == 0 {
        return f64::NAN;
    }
    let sum_w: f64 = weights.iter().take(n).sum();
    if sum_w <= 0.0 {
        return f64::NAN;
    }
    let denom = sum_w - 1.0;
    if denom <= 0.0 {
        return f64::NAN;
    }
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for idx in 0..n {
        mean_x += weights[idx] * xs[idx];
        mean_y += weights[idx] * ys[idx];
    }
    mean_x /= sum_w;
    mean_y /= sum_w;
    let mut accumulator = 0.0;
    for idx in 0..n {
        accumulator += weights[idx] * (xs[idx] - mean_x) * (ys[idx] - mean_y);
    }
    accumulator / denom
}

fn sanitize_covariance(is_diag: bool, value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    if is_diag && value < 0.0 && value > -1.0e-12 {
        0.0
    } else {
        value
    }
}

fn set_entry(buffer: &mut [f64], dim: usize, row: usize, col: usize, value: f64) {
    let idx = row + col * dim;
    buffer[idx] = value;
    if row != col {
        let symmetrical = col + row * dim;
        buffer[symmetrical] = value;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::Tensor;

    fn assert_tensor_close(actual: &Tensor, expected: &[f64], tol: f64) {
        let dim = (expected.len() as f64).sqrt() as usize;
        assert_eq!(actual.shape, vec![dim, dim], "unexpected tensor shape");
        for (idx, (&got, &want)) in actual.data.iter().zip(expected.iter()).enumerate() {
            if want.is_nan() {
                assert!(
                    got.is_nan(),
                    "expected NaN at linear index {idx}, found {got}"
                );
            } else {
                assert!(
                    (got - want).abs() <= tol,
                    "mismatch at linear index {idx}: got {got}, expected {want}"
                );
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_matrix_basic() {
        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2, //
                0.60, 0.59, 0.58, 0.62, 0.63,
            ],
            vec![5, 3],
        )
        .unwrap();
        let result = cov_builtin(Value::Tensor(tensor), Vec::new()).expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            0.0250, 0.0075, 0.00175, //
            0.0075, 0.0070, 0.00135, //
            0.00175, 0.00135, 0.00043,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_two_vectors() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![10.0, 11.0, 9.0, 12.0], vec![4, 1]).unwrap();
        let result = cov_builtin(Value::Tensor(x), vec![Value::Tensor(y)]).expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            1.6666666666666667,
            0.6666666666666666, //
            0.6666666666666666,
            1.6666666666666667,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_weighted_vector() {
        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();
        let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![5, 1]).unwrap();
        let result = cov_builtin(Value::Tensor(tensor), vec![Value::Tensor(weights)]).expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            0.022380952380952376,
            0.004999999999999994, //
            0.004999999999999994,
            0.006666666666666678,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_omitrows() {
        let tensor = Tensor::new(
            vec![
                1.0,
                3.0,
                f64::NAN,
                8.0, //
                f64::NAN,
                4.0,
                6.0,
                9.0, //
                2.0,
                5.0,
                7.0,
                10.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let result =
            cov_builtin(Value::Tensor(tensor), vec![Value::from("omitrows")]).expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            12.5, 12.5, 12.5, //
            12.5, 12.5, 12.5, //
            12.5, 12.5, 12.5,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_partialrows() {
        let tensor = Tensor::new(
            vec![
                1.0,
                4.0,
                7.0, //
                2.0,
                f64::NAN,
                8.0, //
                f64::NAN,
                6.0,
                9.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let result =
            cov_builtin(Value::Tensor(tensor), vec![Value::from("partialrows")]).expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            9.0,
            18.0,
            4.5, //
            18.0,
            18.0,
            f64::NAN, //
            4.5,
            f64::NAN,
            4.5,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![
                    4.0, 4.2, 3.9, 4.3, 4.1, //
                    2.0, 2.1, 2.0, 2.1, 2.2,
                ],
                vec![5, 2],
            )
            .unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cov_builtin(Value::GpuTensor(handle), Vec::new()).expect("cov");
            let gathered = test_support::gather(result).expect("gather");
            let expected = [
                0.0250, 0.0075, //
                0.0075, 0.0070,
            ];
            assert_tensor_close(&gathered, &expected, 1.0e-6);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();

        let cpu_result = cov_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cov");
        let cpu_tensor = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = cov_builtin(Value::GpuTensor(handle), Vec::new()).expect("cov");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_tensor_close(&gathered, &cpu_tensor.data, 1.0e-6);
    }
}
