//! MATLAB-compatible `corr` builtin.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::builtins::stats::summary::distribution_math::{standard_normal_cdf, student_t_cdf};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "corr";

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pairwise correlation coefficients.",
}];

const OUTPUT_R_P: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pairwise correlation coefficients.",
    },
    BuiltinParamDescriptor {
        name: "PValue",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "P-values for testing the hypothesis of no correlation.",
    },
];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input observations by variables.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Second observations-by-variables input.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options including Type and Rows.",
};

const INPUTS_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUTS_X_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_OPTIONS];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];

const SIGNATURES_WITH_P: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "R = corr(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corr(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corr(X, Name, Value)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corr(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X, Name, Value)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, PValue] = corr(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_R_P,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.corr.INVALID_ARGUMENT",
    identifier: Some("RunMat:corr:InvalidArgument"),
    when: "Inputs, row counts, Type, or Rows options are malformed or unsupported.",
    message: "corr: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.corr.INTERNAL",
    identifier: Some("RunMat:corr:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "corr: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const CORR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES_WITH_P,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn corr_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![None, None]),
    }
}

fn corr_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier("RunMat:corr:InvalidArgument")
        .build()
}

fn corr_internal(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(NAME)
        .with_identifier("RunMat:corr:Internal")
        .build()
}

#[derive(Clone, Copy)]
enum CorrType {
    Pearson,
    Spearman,
    Kendall,
}

#[derive(Clone, Copy)]
enum RowsMode {
    All,
    Complete,
    Pairwise,
}

#[derive(Clone, Copy)]
enum Tail {
    Both,
    Right,
    Left,
}

struct CorrArgs {
    left: Tensor,
    right: Option<Tensor>,
    corr_type: CorrType,
    rows: RowsMode,
    tail: Tail,
}

#[runtime_builtin(
    name = "corr",
    category = "stats/summary",
    summary = "Compute pairwise linear or rank correlations between variables.",
    keywords = "corr,correlation,pearson,spearman,kendall,statistics,rows,tail",
    type_resolver(corr_type),
    descriptor(crate::builtins::stats::summary::corr::CORR_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::corr"
)]
async fn corr_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = parse_args(value, rest).await?;
    corr_compute(args)
}

async fn parse_args(value: Value, rest: Vec<Value>) -> BuiltinResult<CorrArgs> {
    let left = value_to_tensor(value).await?;
    let mut right = None;
    let mut corr_type = CorrType::Pearson;
    let mut rows = RowsMode::All;
    let mut tail = Tail::Both;
    let mut idx = 0usize;
    if let Some(first) = rest.first() {
        if keyword_of(first).is_none() {
            right = Some(value_to_tensor(first.clone()).await?);
            idx = 1;
        }
    }
    while idx < rest.len() {
        if idx + 1 >= rest.len() {
            return Err(corr_error(
                "corr: name-value options must be provided in pairs",
            ));
        }
        let name = keyword_of(&rest[idx])
            .ok_or_else(|| corr_error("corr: option names must be string scalars"))?;
        let option = keyword_of(&rest[idx + 1])
            .ok_or_else(|| corr_error("corr: option values must be string scalars"))?;
        match name.to_ascii_lowercase().as_str() {
            "type" => match option.to_ascii_lowercase().as_str() {
                "pearson" => corr_type = CorrType::Pearson,
                "spearman" => corr_type = CorrType::Spearman,
                "kendall" => corr_type = CorrType::Kendall,
                other => return Err(corr_error(format!("corr: unsupported Type '{other}'"))),
            },
            "rows" => match option.to_ascii_lowercase().as_str() {
                "all" => rows = RowsMode::All,
                "complete" => rows = RowsMode::Complete,
                "pairwise" => rows = RowsMode::Pairwise,
                other => return Err(corr_error(format!("corr: unsupported Rows '{other}'"))),
            },
            "tail" => match option.to_ascii_lowercase().as_str() {
                "both" => tail = Tail::Both,
                "right" => tail = Tail::Right,
                "left" => tail = Tail::Left,
                other => return Err(corr_error(format!("corr: unsupported Tail '{other}'"))),
            },
            other => return Err(corr_error(format!("corr: unsupported option '{other}'"))),
        }
        idx += 2;
    }
    Ok(CorrArgs {
        left,
        right,
        corr_type,
        rows,
        tail,
    })
}

async fn value_to_tensor(value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| corr_error(format!("corr: {err}")))?;
    let tensor = tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| corr_error(format!("corr: {err}")))?;
    tensor::integer_tensor_to_f64(tensor).map_err(|err| corr_error(format!("corr: {err}")))
}

fn matrix_shape(tensor: &Tensor) -> (usize, usize) {
    let shape = tensor::default_shape_for(&tensor.shape, tensor.data.len());
    match shape.as_slice() {
        [] => (1, 1),
        [_] => (tensor.data.len(), 1),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn column(tensor: &Tensor, rows: usize, col: usize) -> Vec<f64> {
    let start = col * rows;
    tensor.data[start..start + rows].to_vec()
}

fn corr_compute(args: CorrArgs) -> BuiltinResult<Value> {
    let (left_rows, left_cols) = matrix_shape(&args.left);
    let right = args.right.as_ref().unwrap_or(&args.left);
    let (right_rows, right_cols) = matrix_shape(right);
    if left_rows != right_rows {
        return Err(corr_error(
            "corr: X and Y must have the same number of observations",
        ));
    }
    let mut rho_data = Vec::with_capacity(left_cols * right_cols);
    let mut p_data = Vec::with_capacity(left_cols * right_cols);
    let complete_mask = match args.rows {
        RowsMode::Complete => Some(complete_rows(
            &args.left, right, left_rows, left_cols, right_cols,
        )),
        RowsMode::All | RowsMode::Pairwise => None,
    };
    let all_mask = vec![true; left_rows];
    for right_col in 0..right_cols {
        let y = column(right, right_rows, right_col);
        for left_col in 0..left_cols {
            let x = column(&args.left, left_rows, left_col);
            let pair = match complete_mask.as_ref() {
                Some(mask) => corr_pair_masked(&x, &y, mask, args.corr_type, RowsMode::All),
                None => corr_pair_masked(&x, &y, &all_mask, args.corr_type, args.rows),
            };
            rho_data.push(pair.rho);
            p_data.push(correlation_pvalue(
                pair.rho,
                pair.n,
                args.corr_type,
                args.tail,
            ));
        }
    }
    let shape = vec![left_cols, right_cols];
    let rho = Tensor::new(rho_data, shape.clone())
        .map(tensor::tensor_into_value)
        .map_err(|err| corr_internal(format!("corr: {err}")))?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![rho]));
        }
        let p = Tensor::new(p_data, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| corr_internal(format!("corr: {err}")))?;
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![rho, p],
        ));
    }
    Ok(rho)
}

fn complete_rows(
    left: &Tensor,
    right: &Tensor,
    rows: usize,
    left_cols: usize,
    right_cols: usize,
) -> Vec<bool> {
    let mut mask = vec![true; rows];
    for row in 0..rows {
        for col in 0..left_cols {
            if left.data[col * rows + row].is_nan() {
                mask[row] = false;
            }
        }
        for col in 0..right_cols {
            if right.data[col * rows + row].is_nan() {
                mask[row] = false;
            }
        }
    }
    mask
}

#[derive(Clone, Copy)]
struct CorrPair {
    rho: f64,
    n: usize,
}

fn corr_pair_masked(
    x: &[f64],
    y: &[f64],
    mask: &[bool],
    corr_type: CorrType,
    rows: RowsMode,
) -> CorrPair {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for idx in 0..x.len() {
        if !mask[idx] {
            continue;
        }
        match rows {
            RowsMode::All => {
                xs.push(x[idx]);
                ys.push(y[idx]);
            }
            RowsMode::Complete | RowsMode::Pairwise => {
                if !x[idx].is_nan() && !y[idx].is_nan() {
                    xs.push(x[idx]);
                    ys.push(y[idx]);
                }
            }
        }
    }
    if matches!(rows, RowsMode::All)
        && (xs.iter().any(|v| v.is_nan()) || ys.iter().any(|v| v.is_nan()))
    {
        return CorrPair {
            rho: f64::NAN,
            n: xs.len(),
        };
    }
    if xs.len() < 2 {
        return CorrPair {
            rho: f64::NAN,
            n: xs.len(),
        };
    }
    let rho = match corr_type {
        CorrType::Pearson => pearson(&xs, &ys),
        CorrType::Spearman => {
            let xr = tied_rank(&xs);
            let yr = tied_rank(&ys);
            pearson(&xr, &yr)
        }
        CorrType::Kendall => kendall_tau_b(&xs, &ys),
    };
    CorrPair { rho, n: xs.len() }
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (a, b) in x.iter().zip(y.iter()) {
        let dx = *a - mean_x;
        let dy = *b - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        f64::NAN
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

fn tied_rank(values: &[f64]) -> Vec<f64> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == indexed[start].1 {
            end += 1;
        }
        let rank = (start + 1 + end) as f64 / 2.0;
        for (original, _) in &indexed[start..end] {
            ranks[*original] = rank;
        }
        start = end;
    }
    ranks
}

fn kendall_tau_b(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }
    let mut concordant = 0.0;
    let mut discordant = 0.0;
    let mut ties_x = 0.0;
    let mut ties_y = 0.0;
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            let dx = compare_pair(x[i], x[j]);
            let dy = compare_pair(y[i], y[j]);
            if dx == Ordering::Equal {
                ties_x += 1.0;
            }
            if dy == Ordering::Equal {
                ties_y += 1.0;
            }
            if dx != Ordering::Equal && dy != Ordering::Equal {
                if dx == dy {
                    concordant += 1.0;
                } else {
                    discordant += 1.0;
                }
            }
        }
    }
    let denominator =
        f64::sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y));
    if denominator == 0.0 {
        f64::NAN
    } else {
        (concordant - discordant) / denominator
    }
}

fn compare_pair(a: f64, b: f64) -> Ordering {
    if a < b {
        Ordering::Less
    } else if a > b {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn correlation_pvalue(rho: f64, n: usize, corr_type: CorrType, tail: Tail) -> f64 {
    if rho.is_nan() || n < 2 {
        return f64::NAN;
    }
    let cdf = match corr_type {
        CorrType::Pearson | CorrType::Spearman => {
            if n < 3 {
                return f64::NAN;
            }
            let df = (n - 2) as f64;
            let denom = 1.0 - rho * rho;
            let t = if denom <= 0.0 {
                if rho > 0.0 {
                    f64::INFINITY
                } else if rho < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    0.0
                }
            } else {
                rho * f64::sqrt(df / denom)
            };
            student_t_cdf(t, df)
        }
        CorrType::Kendall => {
            if n < 3 {
                return f64::NAN;
            }
            let n = n as f64;
            let variance = 2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0));
            standard_normal_cdf(rho / f64::sqrt(variance))
        }
    };
    tail_pvalue(cdf, tail)
}

fn tail_pvalue(cdf: f64, tail: Tail) -> f64 {
    if cdf.is_nan() {
        return f64::NAN;
    }
    match tail {
        Tail::Both => (2.0 * cdf.min(1.0 - cdf)).clamp(0.0, 1.0),
        Tail::Right => (1.0 - cdf).clamp(0.0, 1.0),
        Tail::Left => cdf.clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
    }

    #[test]
    fn corr_matrix_defaults_to_columns() {
        let out = block_on(corr_builtin(
            tensor(vec![1.0, 2.0, 1.0, 4.0], vec![2, 2]),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert!((tensor.data[0] - 1.0).abs() < 1e-12);
                assert!((tensor.data[1] - 1.0).abs() < 1e-12);
                assert!((tensor.data[2] - 1.0).abs() < 1e-12);
                assert!((tensor.data[3] - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn corr_accepts_typed_integer_matrix_inputs() {
        let out = block_on(corr_builtin(
            int_tensor(IntegerStorage::I16(vec![1, 2, 1, 4]), vec![2, 2]),
            Vec::new(),
        ))
        .unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert!((tensor.data[0] - 1.0).abs() < 1.0e-12);
                assert!((tensor.data[3] - 1.0).abs() < 1.0e-12);
            }
            other => panic!("expected tensor correlation, got {other:?}"),
        }
    }

    #[test]
    fn corr_xy_returns_cross_correlation_matrix() {
        let x = tensor(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![3, 2]);
        let y = tensor(vec![2.0, 4.0, 6.0], vec![3, 1]);
        let out = block_on(corr_builtin(x, vec![y])).unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 1]);
                assert!((tensor.data[0] - 1.0).abs() < 1e-12);
                assert!((tensor.data[1] + 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn corr_pairwise_omits_nan_pair() {
        let x = tensor(vec![1.0, 2.0, f64::NAN, 4.0], vec![4, 1]);
        let y = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let out = block_on(corr_builtin(
            x,
            vec![y, Value::from("Rows"), Value::from("pairwise")],
        ))
        .unwrap();
        match out {
            Value::Num(value) => assert!((value - 1.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn corr_returns_pvalues_when_requested() {
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let y = tensor(vec![2.0, 4.0, 6.0, 8.0], vec![4, 1]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(corr_builtin(x, vec![y])).unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                assert!(matches!(&values[0], Value::Num(value) if (*value - 1.0).abs() < 1e-12));
                assert!(matches!(&values[1], Value::Num(value) if *value <= 1e-12));
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn corr_supports_kendall_and_tail_option() {
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let y = tensor(vec![1.0, 3.0, 2.0, 4.0], vec![4, 1]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(corr_builtin(
            x,
            vec![
                y,
                Value::from("Type"),
                Value::from("Kendall"),
                Value::from("Tail"),
                Value::from("right"),
            ],
        ))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                assert!(
                    matches!(&values[0], Value::Num(value) if (*value - (2.0 / 3.0)).abs() < 1e-12)
                );
                assert!(matches!(&values[1], Value::Num(value) if *value > 0.0 && *value < 0.2));
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }
}
