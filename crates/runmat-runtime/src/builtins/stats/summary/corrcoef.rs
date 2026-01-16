//! MATLAB-compatible `corrcoef` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{
    CorrcoefNormalization, CorrcoefOptions, CorrcoefRows, GpuTensorHandle,
};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor::{self, value_to_string};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::summary::corrcoef")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "corrcoef",
    op_kind: GpuOpKind::Custom("summary-stats"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("corrcoef")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses provider-side corrcoef kernels when rows='all'; other cases fall back to host execution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::summary::corrcoef")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "corrcoef",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion planner treats corrcoef as a non-fusible boundary; GPU execution is provided via a custom provider hook.",
};

#[runtime_builtin(
    name = "corrcoef",
    category = "stats/summary",
    summary = "Compute Pearson correlation coefficients for the columns of matrices or paired data sets.",
    keywords = "corrcoef,correlation,statistics,rows,normalization,gpu",
    accel = "reduction",
    builtin_path = "crate::builtins::stats::summary::corrcoef"
)]
fn corrcoef_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let args = CorrcoefArgs::parse(value, rest)?;
    if let Some(result) = corrcoef_try_gpu(&args)? {
        return Ok(result);
    }
    corrcoef_host(args)
}

/// Exposed for acceleration providers that need the host reference implementation.
pub fn corrcoef_from_tensors(
    left: Tensor,
    right: Option<Tensor>,
    normalization: CorrcoefNormalization,
    rows: CorrcoefRows,
) -> Result<Tensor, String> {
    let matrix = combine_tensors(left, right)?;
    match rows {
        CorrcoefRows::All => corrcoef_dense(&matrix, normalization),
        CorrcoefRows::Complete => {
            let filtered = filter_complete_rows(&matrix);
            corrcoef_dense(&filtered, normalization)
        }
        CorrcoefRows::Pairwise => corrcoef_pairwise(&matrix, normalization),
    }
}

#[derive(Debug)]
struct CorrcoefArgs {
    first: Value,
    second: Option<Value>,
    normalization: CorrcoefNormalization,
    rows: CorrcoefRows,
}

impl CorrcoefArgs {
    fn parse(first: Value, rest: Vec<Value>) -> Result<Self, String> {
        let mut second: Option<Value> = None;
        let mut normalization: Option<CorrcoefNormalization> = None;
        let mut rows = CorrcoefRows::All;
        let mut iter = rest.into_iter().peekable();

        while let Some(arg) = iter.next() {
            match arg {
                Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                    let key = value_to_string(&arg)
                        .ok_or_else(|| "corrcoef: expected string argument".to_string())?
                        .to_ascii_lowercase();
                    match key.as_str() {
                        "rows" => {
                            let option = iter.next().ok_or_else(|| {
                                "corrcoef: expected a rows option after 'rows'".to_string()
                            })?;
                            let choice = value_to_string(&option)
                                .ok_or_else(|| {
                                    "corrcoef: rows option must be a string value".to_string()
                                })?
                                .to_ascii_lowercase();
                            rows = parse_rows_option(&choice)?;
                        }
                        _ => return Err(format!("corrcoef: unknown option '{key}'")),
                    }
                }
                Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
                    if normalization.is_some() {
                        return Err(
                            "corrcoef: normalization flag specified more than once".to_string()
                        );
                    }
                    normalization = Some(parse_normalization(arg)?);
                }
                Value::Tensor(_) | Value::LogicalArray(_) | Value::GpuTensor(_) => {
                    if second.is_some() {
                        return Err("corrcoef: too many input arrays".to_string());
                    }
                    second = Some(arg);
                }
                Value::ComplexTensor(_) => {
                    return Err("corrcoef: complex inputs are not supported yet".to_string());
                }
                other => return Err(format!("corrcoef: unsupported argument type {:?}", other)),
            }
        }

        Ok(Self {
            first,
            second,
            normalization: normalization.unwrap_or(CorrcoefNormalization::Unbiased),
            rows,
        })
    }
}

fn corrcoef_try_gpu(args: &CorrcoefArgs) -> Result<Option<Value>, String> {
    if args.rows != CorrcoefRows::All {
        return Ok(None);
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let first_handle = match &args.first {
        Value::GpuTensor(handle) => handle.clone(),
        _ => return Ok(None),
    };
    let maybe_second_handle = match &args.second {
        Some(Value::GpuTensor(handle)) => Some(handle.clone()),
        Some(_) => return Ok(None),
        None => None,
    };

    let mut owned_concat: Option<GpuTensorHandle> = None;
    let matrix_handle = if let Some(second) = maybe_second_handle {
        let handles = [first_handle.clone(), second];
        match provider.cat(2, &handles) {
            Ok(concat) => {
                owned_concat = Some(concat.clone());
                concat
            }
            Err(_) => return Ok(None),
        }
    } else {
        first_handle
    };

    let options = CorrcoefOptions {
        normalization: args.normalization,
        rows: args.rows,
    };

    match provider.corrcoef(&matrix_handle, &options) {
        Ok(result) => {
            if let Some(temp) = owned_concat {
                let _ = provider.free(&temp);
            }
            Ok(Some(Value::GpuTensor(result)))
        }
        Err(_) => {
            if let Some(temp) = owned_concat {
                let _ = provider.free(&temp);
            }
            Ok(None)
        }
    }
}

fn corrcoef_host(args: CorrcoefArgs) -> Result<Value, String> {
    let CorrcoefArgs {
        first,
        second,
        normalization,
        rows,
    } = args;
    let left = value_to_tensor_gather(first)?;
    let right = match second {
        Some(value) => Some(value_to_tensor_gather(value)?),
        None => None,
    };
    let tensor = corrcoef_from_tensors(left, right, normalization, rows)?;
    Ok(Value::Tensor(tensor))
}

fn value_to_tensor_gather(value: Value) -> Result<Tensor, String> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor(&handle),
        other => tensor::value_into_tensor_for("corrcoef", other),
    }
}

fn parse_rows_option(value: &str) -> Result<CorrcoefRows, String> {
    match value {
        "all" => Ok(CorrcoefRows::All),
        "complete" | "completecase" | "completecases" => Ok(CorrcoefRows::Complete),
        "pairwise" | "pairwisecomplete" | "pairwisecompletecase" | "pairwisecompletecases" => {
            Ok(CorrcoefRows::Pairwise)
        }
        other => Err(format!("corrcoef: unknown rows option '{other}'")),
    }
}

fn parse_normalization(value: Value) -> Result<CorrcoefNormalization, String> {
    match value {
        Value::Int(i) => match i.to_i64() {
            0 => Ok(CorrcoefNormalization::Unbiased),
            1 => Ok(CorrcoefNormalization::Biased),
            other => Err(format!(
                "corrcoef: normalization flag must be 0 or 1, received {other}"
            )),
        },
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("corrcoef: normalization flag must be finite".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > 1.0e-12 {
                return Err("corrcoef: normalization flag must be an integer".to_string());
            }
            match rounded as i64 {
                0 => Ok(CorrcoefNormalization::Unbiased),
                1 => Ok(CorrcoefNormalization::Biased),
                other => Err(format!(
                    "corrcoef: normalization flag must be 0 or 1, received {other}"
                )),
            }
        }
        Value::Bool(b) => Ok(if b {
            CorrcoefNormalization::Biased
        } else {
            CorrcoefNormalization::Unbiased
        }),
        other => Err(format!(
            "corrcoef: normalization flag must be numeric or logical, received {other:?}"
        )),
    }
}

fn normalization_denominator(norm: CorrcoefNormalization, len: usize) -> f64 {
    match norm {
        CorrcoefNormalization::Unbiased => (len as f64) - 1.0,
        CorrcoefNormalization::Biased => len as f64,
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn from_tensor(tensor: Tensor) -> Result<Self, String> {
        if tensor.shape.len() > 2 {
            return Err("corrcoef: inputs must be 2-D matrices or vectors".to_string());
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
    let mut matrix = Matrix::from_tensor(left)?;
    if let Some(second) = right {
        let right_matrix = Matrix::from_tensor(second)?;
        if matrix.rows != right_matrix.rows {
            return Err("corrcoef: inputs must have the same number of rows".to_string());
        }
        matrix.cols += right_matrix.cols;
        matrix
            .data
            .extend_from_slice(&right_matrix.data[..right_matrix.rows * right_matrix.cols]);
    }
    Ok(matrix)
}

fn filter_complete_rows(matrix: &Matrix) -> Matrix {
    if matrix.rows == 0 {
        return Matrix {
            data: Vec::new(),
            rows: 0,
            cols: matrix.cols,
        };
    }

    let mut valid_rows = Vec::new();
    for row in 0..matrix.rows {
        let mut is_valid = true;
        for col in 0..matrix.cols {
            let value = matrix.get(row, col);
            if !value.is_finite() {
                is_valid = false;
                break;
            }
        }
        if is_valid {
            valid_rows.push(row);
        }
    }

    let new_rows = valid_rows.len();
    if new_rows == 0 {
        return Matrix {
            data: Vec::new(),
            rows: 0,
            cols: matrix.cols,
        };
    }

    let mut data = Vec::with_capacity(new_rows * matrix.cols);
    for col in 0..matrix.cols {
        for &row in &valid_rows {
            data.push(matrix.get(row, col));
        }
    }

    Matrix {
        data,
        rows: new_rows,
        cols: matrix.cols,
    }
}

fn corrcoef_dense(matrix: &Matrix, normalization: CorrcoefNormalization) -> Result<Tensor, String> {
    let cols = matrix.cols;
    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("corrcoef: {e}"));
    }

    let mut result = vec![f64::NAN; cols * cols];
    let rows = matrix.rows;
    if rows == 0 {
        return Tensor::new(result, vec![cols, cols]).map_err(|e| format!("corrcoef: {e}"));
    }

    let denom = normalization_denominator(normalization, rows);
    if denom <= 0.0 {
        return Tensor::new(result, vec![cols, cols]).map_err(|e| format!("corrcoef: {e}"));
    }

    let mut means = vec![0.0; cols];
    for (col, mean_slot) in means.iter_mut().enumerate() {
        let column = matrix.column(col);
        let mut sum = 0.0;
        let mut count = 0usize;
        for &value in column {
            if value.is_finite() {
                sum += value;
                count += 1;
            }
        }
        *mean_slot = if count > 0 {
            sum / (count as f64)
        } else {
            f64::NAN
        };
    }

    for col in 0..cols {
        let mean = means[col];
        if !mean.is_finite() {
            continue;
        }
        let mut variance = 0.0;
        for row in 0..rows {
            let value = matrix.get(row, col);
            if !value.is_finite() {
                variance = f64::NAN;
                break;
            }
            let dev = value - mean;
            variance += dev * dev;
        }
        if variance.is_nan() {
            continue;
        }
        variance /= denom;
        if variance < 0.0 && variance > -1.0e-12 {
            variance = 0.0;
        }
        let stddev = variance.sqrt();
        let diag = if stddev > 0.0 { 1.0 } else { f64::NAN };
        set_entry(&mut result, cols, col, col, diag);
        for other in (col + 1)..cols {
            let corr = column_pair_corr(matrix, col, other, &means, denom);
            set_entry(&mut result, cols, col, other, corr);
        }
    }

    Tensor::new(result, vec![cols, cols]).map_err(|e| format!("corrcoef: {e}"))
}

fn column_pair_corr(matrix: &Matrix, lhs: usize, rhs: usize, means: &[f64], denom: f64) -> f64 {
    let mean_x = means[lhs];
    let mean_y = means[rhs];
    if !mean_x.is_finite() || !mean_y.is_finite() {
        return f64::NAN;
    }

    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut covariance = 0.0;
    for row in 0..matrix.rows {
        let a = matrix.get(row, lhs);
        let b = matrix.get(row, rhs);
        if !a.is_finite() || !b.is_finite() {
            return f64::NAN;
        }
        let dx = a - mean_x;
        let dy = b - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        covariance += dx * dy;
    }

    var_x /= denom;
    var_y /= denom;
    covariance /= denom;

    clamp_correlation(divide_covariance(var_x, var_y, covariance))
}

fn corrcoef_pairwise(
    matrix: &Matrix,
    normalization: CorrcoefNormalization,
) -> Result<Tensor, String> {
    let cols = matrix.cols;
    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("corrcoef: {e}"));
    }
    let mut result = vec![f64::NAN; cols * cols];
    for col in 0..cols {
        set_entry(&mut result, cols, col, col, 1.0);
        for other in (col + 1)..cols {
            let corr = pairwise_corr(matrix, col, other, normalization);
            set_entry(&mut result, cols, col, other, corr);
        }
    }
    Tensor::new(result, vec![cols, cols]).map_err(|e| format!("corrcoef: {e}"))
}

fn pairwise_corr(
    matrix: &Matrix,
    lhs: usize,
    rhs: usize,
    normalization: CorrcoefNormalization,
) -> f64 {
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for row in 0..matrix.rows {
        let a = matrix.get(row, lhs);
        let b = matrix.get(row, rhs);
        if a.is_finite() && b.is_finite() {
            xs.push(a);
            ys.push(b);
        }
    }

    compute_corr(&xs, &ys, normalization)
}

fn compute_corr(xs: &[f64], ys: &[f64], normalization: CorrcoefNormalization) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return f64::NAN;
    }
    let n = xs.len().min(ys.len());
    let denom = normalization_denominator(normalization, n);
    if denom <= 0.0 {
        return f64::NAN;
    }
    let sum_x: f64 = xs.iter().take(n).sum();
    let sum_y: f64 = ys.iter().take(n).sum();
    let mean_x = sum_x / (n as f64);
    let mean_y = sum_y / (n as f64);

    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut covariance = 0.0;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        covariance += dx * dy;
    }
    var_x /= denom;
    var_y /= denom;
    covariance /= denom;
    if var_x < 0.0 && var_x > -1.0e-12 {
        var_x = 0.0;
    }
    if var_y < 0.0 && var_y > -1.0e-12 {
        var_y = 0.0;
    }
    clamp_correlation(divide_covariance(var_x, var_y, covariance))
}

fn divide_covariance(var_x: f64, var_y: f64, covariance: f64) -> f64 {
    if !var_x.is_finite() || !var_y.is_finite() || var_x <= 0.0 || var_y <= 0.0 {
        return f64::NAN;
    }
    let std_x = var_x.sqrt();
    let std_y = var_y.sqrt();
    if std_x == 0.0 || std_y == 0.0 {
        return f64::NAN;
    }
    covariance / (std_x * std_y)
}

fn clamp_correlation(value: f64) -> f64 {
    if value.is_nan() {
        return value;
    }
    if value > 1.0 {
        if value - 1.0 < 1.0e-12 {
            1.0
        } else {
            value
        }
    } else if value < -1.0 {
        if -1.0 - value < 1.0e-12 {
            -1.0
        } else {
            value
        }
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
    use runmat_builtins::{IntValue, Tensor, Value};

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
    fn corrcoef_matrix_basic() {
        let tensor = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                2.0, 4.0, 6.0, 8.0, //
                4.0, 1.0, -1.0, 0.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let result = corrcoef_builtin(Value::Tensor(tensor), Vec::new()).expect("corrcoef");
        match result {
            Value::Tensor(out) => {
                let expected = [
                    1.0,
                    1.0,
                    -0.836_660_026_534,
                    1.0,
                    1.0,
                    -0.836_660_026_534,
                    -0.836_660_026_534,
                    -0.836_660_026_534,
                    1.0,
                ];
                assert_tensor_close(&out, &expected, 1.0e-10);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_two_inputs_matches_concatenation() {
        let left = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                4.0, 5.0, 6.0, 7.0,
            ],
            vec![4, 2],
        )
        .unwrap();
        let right = Tensor::new(vec![8.0, 6.0, 7.0, 5.0], vec![4, 1]).unwrap();
        let combined = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 6.0, 7.0, 5.0,
            ],
            vec![4, 3],
        )
        .unwrap();

        let via_two = corrcoef_builtin(
            Value::Tensor(left.clone()),
            vec![Value::Tensor(right.clone())],
        )
        .expect("corrcoef");
        let via_combined =
            corrcoef_builtin(Value::Tensor(combined), Vec::new()).expect("corrcoef combined");

        let expected_tensor = match via_combined {
            Value::Tensor(t) => t,
            _ => panic!("expected tensor output"),
        };
        let actual_tensor = match via_two {
            Value::Tensor(t) => t,
            _ => panic!("expected tensor output"),
        };
        assert_tensor_close(&actual_tensor, &expected_tensor.data, 1.0e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_rows_complete_ignores_missing() {
        let tensor = Tensor::new(
            vec![
                1.0,
                f64::NAN,
                3.0,
                4.0, //
                2.0,
                5.0,
                f64::NAN,
                8.0,
            ],
            vec![4, 2],
        )
        .unwrap();
        let result = corrcoef_builtin(
            Value::Tensor(tensor),
            vec![Value::from("rows"), Value::from("complete")],
        )
        .expect("corrcoef");
        match result {
            Value::Tensor(out) => {
                let expected = [
                    1.0, 1.0, //
                    1.0, 1.0,
                ];
                assert_tensor_close(&out, &expected, 1.0e-10);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_rows_pairwise_staggered_missing() {
        let tensor = Tensor::new(
            vec![
                1.0,
                f64::NAN,
                4.0,
                5.0, //
                2.0,
                5.0,
                f64::NAN,
                8.0, //
                3.0,
                1.0,
                6.0,
                f64::NAN,
            ],
            vec![4, 3],
        )
        .unwrap();
        let result = corrcoef_builtin(
            Value::Tensor(tensor),
            vec![Value::from("rows"), Value::from("pairwise")],
        )
        .expect("corrcoef");
        match result {
            Value::Tensor(out) => {
                let expected = [
                    1.0, 1.0, 1.0, //
                    1.0, 1.0, -1.0, //
                    1.0, -1.0, 1.0,
                ];
                assert_tensor_close(&out, &expected, 1.0e-10);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_flag_one_accepted() {
        let tensor = Tensor::new(
            vec![
                1.0, 3.0, 5.0, //
                2.0, 4.0, 6.0,
            ],
            vec![3, 2],
        )
        .unwrap();
        let unbiased =
            corrcoef_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("unbiased");
        let biased = corrcoef_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))])
            .expect("biased");

        let a = match biased {
            Value::Tensor(t) => t,
            _ => panic!("expected tensor"),
        };
        let b = match unbiased {
            Value::Tensor(t) => t,
            _ => panic!("expected tensor"),
        };
        assert_tensor_close(&a, &b.data, 1.0e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![
                    1.0, 2.0, 3.0, 4.0, //
                    2.0, 4.0, 6.0, 8.0, //
                    4.0, 1.0, -1.0, 0.0,
                ],
                vec![4, 3],
            )
            .unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = corrcoef_builtin(Value::GpuTensor(handle), Vec::new()).expect("corrcoef");
            let gathered = test_support::gather(result).expect("gather");
            let expected = [
                1.0,
                1.0,
                -0.836_660_026_534,
                1.0,
                1.0,
                -0.836_660_026_534,
                -0.836_660_026_534,
                -0.836_660_026_534,
                1.0,
            ];
            assert_tensor_close(&gathered, &expected, 1.0e-10);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_mismatched_rows_errors() {
        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = corrcoef_builtin(Value::Tensor(left), vec![Value::Tensor(right)])
            .expect_err("expected mismatch error");
        assert!(
            err.contains("same number of rows"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_invalid_flag_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = corrcoef_builtin(Value::Tensor(tensor), vec![Value::Num(2.5)])
            .expect_err("expected invalid flag error");
        assert!(
            err.contains("normalization flag"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn corrcoef_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                2.0, 5.0, 6.0, 8.0, //
                4.0, 1.0, 7.0, 0.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let cpu = corrcoef_from_tensors(
            tensor.clone(),
            None,
            CorrcoefNormalization::Unbiased,
            CorrcoefRows::All,
        )
        .expect("cpu corrcoef");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider.upload(&view).expect("upload");
        let options = CorrcoefOptions {
            normalization: CorrcoefNormalization::Unbiased,
            rows: CorrcoefRows::All,
        };
        let gpu = provider.corrcoef(&handle, &options).expect("corrcoef");
        let host = provider.download(&gpu).expect("download");
        let gathered =
            Tensor::new(host.data.clone(), host.shape.clone()).expect("tensor reconstruction");
        assert_tensor_close(&gathered, &cpu.data, 1.0e-6);
    }
}
