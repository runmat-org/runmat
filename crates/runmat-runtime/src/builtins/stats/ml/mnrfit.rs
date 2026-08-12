//! Multinomial logistic regression compatibility helper.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, LogicalArray, StringArray, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::stats::summary::distribution_math;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "mnrfit";
const EPS: f64 = 1.0e-12;
const RIDGE: f64 = 1.0e-8;

const OUTPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Coefficient matrix with one column per non-reference response category.",
};

const OUTPUT_DEV: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dev",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Model deviance.",
};

const OUTPUT_STATS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "stats",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Fit statistics structure containing beta, se, t, p, covb, coeffcorr, and dfe.",
};

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predictor matrix with observations in rows.",
};

const INPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response class vector or two-column binomial counts matrix.",
};

const INPUT_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Name-value options such as Model, Link, Constant, Weights, IterationLimit, Tolerance, and Options.",
};

const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_Y];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_Y, INPUT_OPTIONS];
const OUTPUTS_B: [BuiltinParamDescriptor; 1] = [OUTPUT_B];
const OUTPUTS_B_DEV_STATS: [BuiltinParamDescriptor; 3] = [OUTPUT_B, OUTPUT_DEV, OUTPUT_STATS];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "B = mnrfit(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUTS_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = mnrfit(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUTS_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, dev, stats] = mnrfit(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUTS_B_DEV_STATS,
    },
    BuiltinSignatureDescriptor {
        label: "[B, dev, stats] = mnrfit(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUTS_B_DEV_STATS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MNRFIT.INVALID_ARGUMENT",
    identifier: Some("RunMat:mnrfit:InvalidArgument"),
    when: "Inputs, response values, dimensions, or options are malformed.",
    message: "mnrfit: invalid argument",
};

const ERROR_CONVERGENCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MNRFIT.CONVERGENCE",
    identifier: Some("RunMat:mnrfit:Convergence"),
    when: "The multinomial logistic solver cannot make numerical progress.",
    message: "mnrfit: convergence failure",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MNRFIT.INTERNAL",
    identifier: Some("RunMat:mnrfit:Internal"),
    when: "RunMat cannot allocate or construct mnrfit outputs.",
    message: "mnrfit: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_CONVERGENCE, ERROR_INTERNAL];

pub const MNRFIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn mnrfit_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn error(message: impl Into<String>, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_INVALID_ARGUMENT)
}

fn convergence(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_CONVERGENCE)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Debug)]
struct Options {
    constant: bool,
    weights: Option<Vec<f64>>,
    max_iter: usize,
    tol_x: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            constant: true,
            weights: None,
            max_iter: 100,
            tol_x: 1.0e-6,
        }
    }
}

#[derive(Clone, Debug)]
struct PreparedData {
    design: Vec<f64>,
    rows: usize,
    cols: usize,
    outcomes: Vec<Vec<f64>>,
    weights: Vec<f64>,
    classes: ClassNames,
    observations: f64,
}

type ResponseOutcomes = (Vec<Vec<f64>>, ClassNames, Vec<f64>, f64);

#[derive(Clone, Debug)]
enum ClassNames {
    Numeric(Vec<f64>),
    Text(Vec<String>),
}

impl ClassNames {
    fn len(&self) -> usize {
        match self {
            ClassNames::Numeric(values) => values.len(),
            ClassNames::Text(values) => values.len(),
        }
    }
}

#[derive(Clone, Debug)]
struct FitResult {
    b: Value,
    dev: Value,
    stats: Value,
}

#[runtime_builtin(
    name = "mnrfit",
    category = "stats/ml",
    summary = "Fit nominal multinomial logistic regression coefficients.",
    keywords = "mnrfit,multinomial,logistic regression,nominal,binomial,statistics,machine learning",
    type_resolver(mnrfit_type),
    descriptor(crate::builtins::stats::ml::mnrfit::MNRFIT_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::mnrfit"
)]
pub(crate) async fn mnrfit_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let x = gather_value(x).await?;
    let y = gather_value(y).await?;
    let rest = gather_values(rest).await?;
    let options = parse_options(rest)?;
    let result = fit(x, y, options)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![result.b])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![result.b, result.dev, result.stats],
        )),
        None => Ok(result.b),
    }
}

async fn gather_value(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("mnrfit: {err}")))
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather_value(value).await?);
    }
    Ok(out)
}

fn fit(x: Value, y: Value, options: Options) -> BuiltinResult<FitResult> {
    let x = numeric_tensor("X", x)?;
    if x.rows() == 0 || x.cols() == 0 {
        return Err(invalid("mnrfit: X must be a nonempty numeric matrix"));
    }
    let data = prepare_data(&x, y, &options)?;
    let fit = fit_nominal_logit(&data, options.max_iter, options.tol_x)?;
    let b = coefficients_value(&fit.beta, data.cols, data.outcomes.len())?;
    let stats = stats_value(&fit, &data)?;
    Ok(FitResult {
        b,
        dev: Value::Num(fit.deviance),
        stats,
    })
}

#[derive(Clone, Debug)]
struct NominalFit {
    beta: Vec<f64>,
    covb: Vec<f64>,
    se: Vec<f64>,
    t: Vec<f64>,
    p: Vec<f64>,
    deviance: f64,
    dfe: f64,
}

fn fit_nominal_logit(
    data: &PreparedData,
    max_iter: usize,
    tol_x: f64,
) -> BuiltinResult<NominalFit> {
    let cols = data.cols;
    let logits = data.outcomes.len();
    let dim = cols
        .checked_mul(logits)
        .ok_or_else(|| internal("mnrfit: coefficient dimension overflow"))?;
    let matrix_len = dim
        .checked_mul(dim)
        .ok_or_else(|| internal("mnrfit: information matrix dimension overflow"))?;
    let mut beta = vec![0.0; dim];
    let mut information = vec![0.0; matrix_len];
    let mut gradient = vec![0.0; dim];
    for _ in 0..max_iter {
        information.fill(0.0);
        gradient.fill(0.0);
        for row in 0..data.rows {
            let probs = row_probabilities(&data.design, row, cols, &beta, logits);
            let weight = data.weights[row];
            for j in 0..logits {
                let residual = weight * (data.outcomes[j][row] - probs[j]);
                for a in 0..cols {
                    gradient[j * cols + a] += residual * data.design[row + a * data.rows];
                }
            }
            for j in 0..logits {
                for l in 0..logits {
                    let block = weight * probs[j] * if j == l { 1.0 - probs[l] } else { -probs[l] };
                    for a in 0..cols {
                        let za = data.design[row + a * data.rows];
                        for b in 0..cols {
                            let zb = data.design[row + b * data.rows];
                            information[(j * cols + a) * dim + (l * cols + b)] += block * za * zb;
                        }
                    }
                }
            }
        }
        for idx in 0..dim {
            information[idx * dim + idx] += RIDGE;
        }
        let delta = solve_linear(information.clone(), gradient.clone(), dim)?;
        let max_step = delta.iter().map(|value| value.abs()).fold(0.0, f64::max);
        for (coef, step) in beta.iter_mut().zip(delta.iter()) {
            *coef += *step;
        }
        if max_step <= tol_x.max(EPS) {
            break;
        }
    }
    if beta.iter().any(|value| !value.is_finite()) {
        return Err(convergence(
            "mnrfit: solver produced non-finite coefficients",
        ));
    }
    information.fill(0.0);
    for row in 0..data.rows {
        let probs = row_probabilities(&data.design, row, cols, &beta, logits);
        let weight = data.weights[row];
        for j in 0..logits {
            for l in 0..logits {
                let block = weight * probs[j] * if j == l { 1.0 - probs[l] } else { -probs[l] };
                for a in 0..cols {
                    let za = data.design[row + a * data.rows];
                    for b in 0..cols {
                        let zb = data.design[row + b * data.rows];
                        information[(j * cols + a) * dim + (l * cols + b)] += block * za * zb;
                    }
                }
            }
        }
    }
    for idx in 0..dim {
        information[idx * dim + idx] += RIDGE;
    }
    let covb = invert_matrix(information, dim)?;
    let se = (0..dim)
        .map(|idx| covb[idx * dim + idx].max(0.0).sqrt())
        .collect::<Vec<_>>();
    let t = beta
        .iter()
        .zip(se.iter())
        .map(|(coef, se)| if *se > 0.0 { coef / se } else { f64::NAN })
        .collect::<Vec<_>>();
    let p = t
        .iter()
        .map(|stat| {
            if stat.is_finite() {
                2.0 * (1.0 - distribution_math::standard_normal_cdf(stat.abs()))
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    let loglik = log_likelihood(data, &beta);
    let saturated = saturated_log_likelihood(data);
    let dfe = (data.observations - dim as f64).max(0.0);
    Ok(NominalFit {
        beta,
        covb,
        se,
        t,
        p,
        deviance: (2.0 * (saturated - loglik)).max(0.0),
        dfe,
    })
}

fn row_probabilities(
    design: &[f64],
    row: usize,
    cols: usize,
    beta: &[f64],
    logits: usize,
) -> Vec<f64> {
    let rows = design.len() / cols;
    let mut eta = Vec::with_capacity(logits);
    let mut max_eta = 0.0f64;
    for j in 0..logits {
        let mut value = 0.0;
        for col in 0..cols {
            value += design[row + col * rows] * beta[j * cols + col];
        }
        max_eta = max_eta.max(value);
        eta.push(value);
    }
    let base = (-max_eta).exp();
    let mut denom = base;
    let mut exp_eta = Vec::with_capacity(logits);
    for value in eta {
        let exp = (value - max_eta).exp();
        denom += exp;
        exp_eta.push(exp);
    }
    exp_eta.into_iter().map(|value| value / denom).collect()
}

fn log_likelihood(data: &PreparedData, beta: &[f64]) -> f64 {
    let logits = data.outcomes.len();
    let mut ll = 0.0;
    for row in 0..data.rows {
        let probs = row_probabilities(&data.design, row, data.cols, beta, logits);
        let mut baseline = 1.0 - probs.iter().sum::<f64>();
        baseline = baseline.max(EPS);
        for (j, prob) in probs.iter().enumerate() {
            let count = data.weights[row] * data.outcomes[j][row];
            if count > 0.0 {
                ll += count * prob.max(EPS).ln();
            }
        }
        let baseline_count =
            data.weights[row] * (1.0 - (0..logits).map(|j| data.outcomes[j][row]).sum::<f64>());
        if baseline_count > 0.0 {
            ll += baseline_count * baseline.ln();
        }
    }
    ll
}

fn saturated_log_likelihood(data: &PreparedData) -> f64 {
    let logits = data.outcomes.len();
    let mut ll = 0.0;
    for row in 0..data.rows {
        let weight = data.weights[row];
        for j in 0..logits {
            let p = data.outcomes[j][row];
            if p > 0.0 {
                ll += weight * p * p.max(EPS).ln();
            }
        }
        let baseline_p = (1.0 - (0..logits).map(|j| data.outcomes[j][row]).sum::<f64>()).max(0.0);
        if baseline_p > 0.0 {
            ll += weight * baseline_p * baseline_p.max(EPS).ln();
        }
    }
    ll
}

fn prepare_data(x: &Tensor, y: Value, options: &Options) -> BuiltinResult<PreparedData> {
    let rows = x.rows();
    let mut design = design_matrix(x, options.constant)?;
    let cols = if options.constant {
        x.cols()
            .checked_add(1)
            .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?
    } else {
        x.cols()
    };
    let (mut outcomes, classes, mut weights, _) = response_outcomes(y, rows)?;
    if let Some(option_weights) = &options.weights {
        if option_weights.len() != rows {
            return Err(invalid(
                "mnrfit: Weights vector length must match the number of rows in X",
            ));
        }
        for (weight, option_weight) in weights.iter_mut().zip(option_weights.iter()) {
            *weight *= *option_weight;
        }
    }
    let kept_rows = compact_complete_cases(x, cols, &mut design, &mut outcomes, &mut weights)?;
    if kept_rows == 0 {
        return Err(invalid("mnrfit: no complete observations remain"));
    }
    if weights
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err(invalid(
            "mnrfit: weights must be nonnegative finite values with positive total",
        ));
    }
    let observations = weights.iter().sum::<f64>();
    Ok(PreparedData {
        design,
        rows: kept_rows,
        cols,
        outcomes,
        weights,
        classes,
        observations,
    })
}

fn design_matrix(x: &Tensor, constant: bool) -> BuiltinResult<Vec<f64>> {
    let rows = x.rows();
    let x_cols = x.cols();
    let cols = if constant {
        x_cols
            .checked_add(1)
            .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?
    } else {
        x_cols
    };
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?;
    let mut design = vec![0.0; len];
    let mut out_col = 0usize;
    if constant {
        for row in 0..rows {
            design[row] = 1.0;
        }
        out_col = 1;
    }
    let values = tensor::tensor_values_f64_cow(x);
    for col in 0..x_cols {
        for row in 0..rows {
            design[row + (out_col + col) * rows] = values[row + col * rows];
        }
    }
    Ok(design)
}

fn compact_complete_cases(
    x: &Tensor,
    cols: usize,
    design: &mut Vec<f64>,
    outcomes: &mut [Vec<f64>],
    weights: &mut Vec<f64>,
) -> BuiltinResult<usize> {
    let rows = x.rows();
    let values = tensor::tensor_values_f64_cow(x);
    let keep = (0..rows)
        .filter(|row| {
            let x_complete = (0..x.cols()).all(|col| values[*row + col * rows].is_finite());
            let y_complete = weights[*row].is_finite()
                && weights[*row] > 0.0
                && outcomes.iter().all(|outcome| outcome[*row].is_finite());
            x_complete && y_complete
        })
        .collect::<Vec<_>>();
    if keep.len() == rows {
        return Ok(rows);
    }
    let new_len = keep
        .len()
        .checked_mul(cols)
        .ok_or_else(|| internal("mnrfit: design matrix dimension overflow"))?;
    let mut new_design = vec![0.0; new_len];
    for col in 0..cols {
        for (new_row, old_row) in keep.iter().enumerate() {
            new_design[new_row + col * keep.len()] = design[old_row + col * rows];
        }
    }
    for outcome in outcomes.iter_mut() {
        *outcome = keep.iter().map(|old_row| outcome[*old_row]).collect();
    }
    *weights = keep.iter().map(|old_row| weights[*old_row]).collect();
    *design = new_design;
    Ok(keep.len())
}

fn response_outcomes(value: Value, rows: usize) -> BuiltinResult<ResponseOutcomes> {
    match value {
        Value::Tensor(tensor) if tensor.cols() >= 2 && tensor.rows() == rows => count_outcomes(&tensor),
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            numeric_outcomes(tensor::tensor_values_f64(&tensor), shape, rows)
        }
        Value::LogicalArray(logical) => categorical_outcomes(
            logical.data.iter().map(|value| f64::from(*value != 0)).collect(),
            logical.shape,
            rows,
        ),
        Value::StringArray(array) => text_outcomes(array.data, array.shape, rows),
        Value::String(text) if rows == 1 => text_outcomes(vec![text], vec![1, 1], rows),
        Value::CharArray(array) => {
            let shape = vec![array.rows, 1];
            text_outcomes(char_rows(&array), shape, rows)
        }
        Value::Cell(cell) => cellstr_outcomes(&cell, rows),
        Value::Object(object) if object.is_class("categorical") => {
            let labels = crate::builtins::table::categorical_labels(&Value::Object(object))?;
            text_outcomes(labels, vec![rows, 1], rows)
        }
        Value::Num(value) if rows == 1 => numeric_outcomes(vec![value], vec![1, 1], rows),
        Value::Bool(value) if rows == 1 => {
            categorical_outcomes(vec![f64::from(value)], vec![1, 1], rows)
        }
        other => Err(invalid(format!(
            "mnrfit: Y must be a numeric, logical, text, categorical, cellstr, or grouped counts response, got {other:?}"
        ))),
    }
}

fn count_outcomes(tensor: &Tensor) -> BuiltinResult<ResponseOutcomes> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let data = tensor::tensor_values_f64(tensor);
    let logits = cols - 1;
    let mut outcomes = vec![vec![0.0; rows]; logits];
    let mut weights = Vec::with_capacity(rows);
    let mut observations = 0.0;
    for row in 0..rows {
        let mut total = 0.0;
        let mut missing = false;
        for col in 0..cols {
            let count = data[row + col * rows];
            if count.is_nan() {
                missing = true;
                break;
            }
            if !count.is_finite() || count < 0.0 {
                return Err(invalid(
                    "mnrfit: grouped counts must be nonnegative finite values",
                ));
            }
            total += count;
        }
        if missing || total <= 0.0 {
            weights.push(0.0);
        } else {
            for (col, outcome) in outcomes.iter_mut().enumerate().take(logits) {
                outcome[row] = data[row + col * rows] / total;
            }
            weights.push(total);
            observations += total;
        }
    }
    if observations <= 0.0 {
        return Err(invalid("mnrfit: grouped counts must contain observations"));
    }
    let classes = (1..=cols).map(|idx| idx as f64).collect::<Vec<_>>();
    Ok((
        outcomes,
        ClassNames::Numeric(classes),
        weights,
        observations,
    ))
}

fn numeric_outcomes(
    values: Vec<f64>,
    shape: Vec<usize>,
    rows: usize,
) -> BuiltinResult<ResponseOutcomes> {
    if values.len() != rows || shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid(
            "mnrfit: categorical Y must be a vector with one response per row of X",
        ));
    }
    let mut classes = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    classes.dedup_by(|a, b| (*a - *b).abs() <= EPS);
    if classes.len() < 2 {
        return Err(invalid("mnrfit: Y must contain at least two classes"));
    }
    let logits = classes.len() - 1;
    let mut outcomes = vec![vec![0.0; rows]; logits];
    let mut weights = vec![1.0; rows];
    for (row, value) in values.iter().enumerate() {
        if value.is_nan() {
            weights[row] = 0.0;
            continue;
        }
        if !value.is_finite() {
            return Err(invalid("mnrfit: Y classes must be finite or NaN"));
        }
        let class_idx = classes
            .iter()
            .position(|class| (*class - *value).abs() <= EPS)
            .ok_or_else(|| internal("mnrfit: failed to map response class"))?;
        if class_idx < logits {
            outcomes[class_idx][row] = 1.0;
        }
    }
    let observations = weights.iter().sum::<f64>();
    Ok((
        outcomes,
        ClassNames::Numeric(classes),
        weights,
        observations,
    ))
}

fn categorical_outcomes(
    values: Vec<f64>,
    shape: Vec<usize>,
    rows: usize,
) -> BuiltinResult<ResponseOutcomes> {
    numeric_outcomes(values, shape, rows)
}

fn text_outcomes(
    values: Vec<String>,
    shape: Vec<usize>,
    rows: usize,
) -> BuiltinResult<ResponseOutcomes> {
    if values.len() != rows || shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid(
            "mnrfit: categorical Y must be a vector with one response per row of X",
        ));
    }
    if values.iter().any(|value| value.is_empty()) {
        return Err(invalid("mnrfit: text response classes must be nonempty"));
    }
    let mut classes = values.clone();
    classes.sort();
    classes.dedup();
    if classes.len() < 2 {
        return Err(invalid("mnrfit: Y must contain at least two classes"));
    }
    let logits = classes.len() - 1;
    let mut outcomes = vec![vec![0.0; rows]; logits];
    for (row, value) in values.iter().enumerate() {
        let class_idx = classes
            .iter()
            .position(|class| class == value)
            .ok_or_else(|| internal("mnrfit: failed to map response class"))?;
        if class_idx < logits {
            outcomes[class_idx][row] = 1.0;
        }
    }
    Ok((
        outcomes,
        ClassNames::Text(classes),
        vec![1.0; rows],
        rows as f64,
    ))
}

fn cellstr_outcomes(cell: &CellArray, rows: usize) -> BuiltinResult<ResponseOutcomes> {
    let labels = cell
        .data
        .iter()
        .map(|value| {
            scalar_text(value).ok_or_else(|| invalid("mnrfit: Y cell entries must be text"))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    text_outcomes(labels, cell.shape.clone(), rows)
}

fn parse_options(values: Vec<Value>) -> BuiltinResult<Options> {
    if !values.len().is_multiple_of(2) {
        return Err(invalid("mnrfit: name-value options must be paired"));
    }
    let mut options = Options::default();
    let mut index = 0usize;
    while index < values.len() {
        let name = scalar_text(&values[index])
            .ok_or_else(|| invalid("mnrfit: option name must be text"))?;
        let value = &values[index + 1];
        match name.to_ascii_lowercase().as_str() {
            "model" => {
                let model = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: Model option must be text"))?;
                match model.to_ascii_lowercase().as_str() {
                    "nominal" => {}
                    other => {
                        return Err(invalid(format!(
                            "mnrfit: Model '{other}' is not supported yet"
                        )));
                    }
                }
            }
            "link" => {
                let link = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: Link option must be text"))?;
                match link.to_ascii_lowercase().as_str() {
                    "logit" => {}
                    other => {
                        return Err(invalid(format!(
                            "mnrfit: Link '{other}' is not supported yet"
                        )));
                    }
                }
            }
            "constant" => {
                options.constant = match text_or_logical(value, "Constant")? {
                    BoolOrText::Bool(flag) => flag,
                    BoolOrText::Text(text) => match text.to_ascii_lowercase().as_str() {
                        "on" => true,
                        "off" => false,
                        other => {
                            return Err(invalid(format!(
                                "mnrfit: Constant must be 'on' or 'off', got '{other}'"
                            )));
                        }
                    },
                };
            }
            "interactions" => {
                let text = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: Interactions option must be text"))?;
                if !text.eq_ignore_ascii_case("off") {
                    return Err(invalid(
                        "mnrfit: Interactions values other than 'off' are not supported yet",
                    ));
                }
            }
            "weights" => options.weights = Some(numeric_vector(value.clone(), "Weights")?),
            "maxiter" | "iterationlimit" => {
                options.max_iter = positive_integer(value, "IterationLimit")?
            }
            "tolx" | "tolerance" => options.tol_x = positive_scalar(value, "Tolerance")?,
            "options" => apply_statset_options(value, &mut options)?,
            "display" => {
                let _ = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: Display option must be text"))?;
            }
            "estdisp" => {
                let text = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: EstDisp option must be text"))?;
                if !text.eq_ignore_ascii_case("off") {
                    return Err(invalid(
                        "mnrfit: overdispersion estimation is not supported yet",
                    ));
                }
            }
            other => return Err(invalid(format!("mnrfit: unknown option '{other}'"))),
        }
        index += 2;
    }
    if options.max_iter == 0 {
        return Err(invalid("mnrfit: IterationLimit must be positive"));
    }
    if !options.tol_x.is_finite() || options.tol_x <= 0.0 {
        return Err(invalid("mnrfit: Tolerance must be positive and finite"));
    }
    Ok(options)
}

fn apply_statset_options(value: &Value, options: &mut Options) -> BuiltinResult<()> {
    let Value::Struct(st) = value else {
        return Err(invalid("mnrfit: Options must be a struct"));
    };
    if let Some(value) = st.fields.get("MaxIter") {
        options.max_iter = positive_integer(value, "Options.MaxIter")?;
    }
    if let Some(value) = st.fields.get("IterationLimit") {
        options.max_iter = positive_integer(value, "Options.IterationLimit")?;
    }
    if let Some(value) = st.fields.get("TolX") {
        options.tol_x = positive_scalar(value, "Options.TolX")?;
    }
    if let Some(value) = st.fields.get("Tolerance") {
        options.tol_x = positive_scalar(value, "Options.Tolerance")?;
    }
    Ok(())
}

enum BoolOrText {
    Bool(bool),
    Text(String),
}

fn text_or_logical(value: &Value, label: &str) -> BuiltinResult<BoolOrText> {
    if let Some(text) = scalar_text(value) {
        return Ok(BoolOrText::Text(text));
    }
    match value {
        Value::Bool(flag) => Ok(BoolOrText::Bool(*flag)),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(BoolOrText::Bool(logical.data[0] != 0))
        }
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(BoolOrText::Bool(*value != 0.0)),
        _ => Err(invalid(format!("mnrfit: {label} must be text or logical"))),
    }
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn numeric_tensor(label: &str, value: Value) -> BuiltinResult<Tensor> {
    tensor::value_into_tensor_for(label, value)
        .map_err(|_| invalid(format!("mnrfit: {label} must be numeric")))
}

fn numeric_vector(value: Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = numeric_tensor(label, value)?;
    if tensor.shape.iter().copied().filter(|dim| *dim > 1).count() > 1 {
        return Err(invalid(format!("mnrfit: {label} must be a vector")));
    }
    let values = tensor::tensor_values_f64(&tensor);
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "mnrfit: {label} must contain finite values"
        )));
    }
    Ok(values)
}

fn numeric_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Bool(value) => Ok(f64::from(*value)),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            Ok(f64::from(data[0] != 0))
        }
        _ => Err(invalid(format!("mnrfit: {label} must be a scalar"))),
    }
}

fn positive_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    let scalar = numeric_scalar(value, label)?;
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(invalid(format!(
            "mnrfit: {label} must be positive and finite"
        )));
    }
    Ok(scalar)
}

fn positive_integer(value: &Value, label: &str) -> BuiltinResult<usize> {
    let scalar = positive_scalar(value, label)?;
    if scalar.fract() != 0.0
        || scalar > usize::MAX as f64
        || (usize::BITS == 64 && scalar == usize::MAX as f64)
    {
        return Err(invalid(format!(
            "mnrfit: {label} must be a positive integer"
        )));
    }
    Ok(scalar as usize)
}

fn coefficients_value(beta: &[f64], rows: usize, cols: usize) -> BuiltinResult<Value> {
    Tensor::new(beta.to_vec(), vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("mnrfit: {err}")))
}

fn stats_value(fit: &NominalFit, data: &PreparedData) -> BuiltinResult<Value> {
    let mut st = StructValue::new();
    let class_count = data.classes.len();
    let logits = data.outcomes.len();
    st.insert("beta", coefficients_value(&fit.beta, data.cols, logits)?);
    st.insert("se", coefficients_value(&fit.se, data.cols, logits)?);
    st.insert("t", coefficients_value(&fit.t, data.cols, logits)?);
    st.insert("p", coefficients_value(&fit.p, data.cols, logits)?);
    st.insert(
        "covb",
        matrix_value(&fit.covb, fit.beta.len(), fit.beta.len())?,
    );
    st.insert("coeffcorr", coeffcorr_value(&fit.covb, &fit.se)?);
    st.insert("dfe", Value::Num(fit.dfe));
    st.insert("s", Value::Num(1.0));
    st.insert("estdisp", Value::Bool(false));
    st.insert("classNames", class_names_value(&data.classes, class_count)?);
    Ok(Value::Struct(st))
}

fn matrix_value(data: &[f64], rows: usize, cols: usize) -> BuiltinResult<Value> {
    Tensor::new(data.to_vec(), vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("mnrfit: {err}")))
}

fn row_tensor(values: &[f64]) -> BuiltinResult<Value> {
    Tensor::new(values.to_vec(), vec![1, values.len()])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("mnrfit: {err}")))
}

fn class_names_value(classes: &ClassNames, len: usize) -> BuiltinResult<Value> {
    match classes {
        ClassNames::Numeric(values) => row_tensor(values),
        ClassNames::Text(values) => StringArray::new(values.clone(), vec![1, len])
            .map(Value::StringArray)
            .map_err(|err| internal(format!("mnrfit: {err}"))),
    }
}

fn char_rows(array: &CharArray) -> Vec<String> {
    let mut out = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        out.push(array.data[start..start + array.cols].iter().collect());
    }
    out
}

fn coeffcorr_value(covb: &[f64], se: &[f64]) -> BuiltinResult<Value> {
    let dim = se.len();
    let len = dim
        .checked_mul(dim)
        .ok_or_else(|| internal("mnrfit: coefficient correlation dimension overflow"))?;
    let mut data = vec![0.0; len];
    for col in 0..dim {
        for row in 0..dim {
            let denom = se[row] * se[col];
            data[row + col * dim] = if denom > 0.0 {
                covb[row + col * dim] / denom
            } else {
                f64::NAN
            };
        }
    }
    matrix_value(&data, dim, dim)
}

fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> BuiltinResult<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in col + 1..n {
            let value = a[row * n + col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best <= EPS || !best.is_finite() {
            return Err(convergence(
                "mnrfit: information matrix is singular or ill-conditioned",
            ));
        }
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
            }
            b.swap(col, pivot);
        }
        let diag = a[col * n + col];
        for row in col + 1..n {
            let factor = a[row * n + col] / diag;
            a[row * n + col] = 0.0;
            for j in col + 1..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut rhs = b[row];
        for col in row + 1..n {
            rhs -= a[row * n + col] * x[col];
        }
        x[row] = rhs / a[row * n + row];
    }
    Ok(x)
}

fn invert_matrix(a: Vec<f64>, n: usize) -> BuiltinResult<Vec<f64>> {
    let mut out = vec![0.0; n * n];
    for col in 0..n {
        let mut rhs = vec![0.0; n];
        rhs[col] = 1.0;
        let solved = solve_linear(a.clone(), rhs, n)?;
        for row in 0..n {
            out[row + col * n] = solved[row];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn tensor_value(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
        Value::Tensor(tensor)
    }

    fn output_list(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn mnrfit_binary_logistic_returns_coefficients_stats() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let out = block_on(mnrfit_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0, 4.0], 5, 1),
            tensor_value(vec![1.0, 1.0, 2.0, 2.0, 2.0], 5, 1),
            vec![
                Value::String("MaxIter".into()),
                Value::Num(200.0),
                Value::String("TolX".into()),
                Value::Num(1.0e-8),
            ],
        ))
        .unwrap();
        let values = output_list(out);
        let Value::Tensor(b) = &values[0] else {
            panic!("expected B tensor");
        };
        assert_eq!(b.shape, vec![2, 1]);
        assert!(b.materialize_f64()[1] < 0.0);
        assert!(matches!(&values[1], Value::Num(value) if value.is_finite()));
        let Value::Struct(stats) = &values[2] else {
            panic!("expected stats struct");
        };
        assert!(stats.fields.contains_key("se"));
        assert!(stats.fields.contains_key("covb"));
    }

    #[test]
    fn mnrfit_positive_integer_rejects_unrepresentable_double_bounds() {
        assert_eq!(positive_integer(&Value::Num(5.0), "MaxIter").unwrap(), 5);
        let boundary = positive_integer(&Value::Num(usize::MAX as f64), "MaxIter");
        if usize::BITS == 64 {
            assert!(boundary.is_err());
        } else {
            assert_eq!(boundary.unwrap(), usize::MAX);
        }
        assert!(positive_integer(&Value::Num((usize::MAX as f64) + 1.0), "MaxIter").is_err());
    }

    #[test]
    fn mnrfit_multiclass_nominal_shape() {
        let out = block_on(mnrfit_builtin(
            tensor_value(vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0], 6, 1),
            tensor_value(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 6, 1),
            vec![
                Value::String("IterationLimit".into()),
                Value::Num(200.0),
                Value::String("Tolerance".into()),
                Value::Num(1.0e-8),
            ],
        ))
        .unwrap();
        let Value::Tensor(b) = out else {
            panic!("expected B tensor");
        };
        assert_eq!(b.shape, vec![2, 2]);
        assert!(b.materialize_f64().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn mnrfit_binomial_counts_and_weights() {
        let out = block_on(mnrfit_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            tensor_value(vec![0.0, 1.0, 3.0, 5.0, 5.0, 4.0, 2.0, 1.0], 4, 2),
            vec![
                Value::String("Weights".into()),
                tensor_value(vec![1.0, 1.0, 2.0, 2.0], 4, 1),
                Value::String("Constant".into()),
                Value::String("on".into()),
            ],
        ))
        .unwrap();
        let Value::Tensor(b) = out else {
            panic!("expected B tensor");
        };
        assert_eq!(b.shape, vec![2, 1]);
        assert!(b.materialize_f64().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn mnrfit_grouped_multinomial_counts() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let out = block_on(mnrfit_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            tensor_value(
                vec![8.0, 5.0, 2.0, 1.0, 1.0, 3.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0],
                4,
                3,
            ),
            vec![
                Value::String("IterationLimit".into()),
                Value::Num(200.0),
                Value::String("Tolerance".into()),
                Value::Num(1.0e-8),
            ],
        ))
        .unwrap();
        let values = output_list(out);
        let Value::Tensor(b) = &values[0] else {
            panic!("expected B tensor");
        };
        assert_eq!(b.shape, vec![2, 2]);
        assert!(matches!(&values[1], Value::Num(value) if value.is_finite() && *value >= 0.0));
        let Value::Struct(stats) = &values[2] else {
            panic!("expected stats struct");
        };
        let Value::Tensor(class_names) = stats.fields.get("classNames").unwrap() else {
            panic!("expected numeric class names");
        };
        assert_eq!(class_names.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn mnrfit_accepts_text_labels_and_omits_missing_rows() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let out = block_on(mnrfit_builtin(
            tensor_value(vec![0.0, 1.0, f64::NAN, 2.0, 3.0, 4.0], 6, 1),
            Value::StringArray(
                StringArray::new(
                    vec![
                        "low".into(),
                        "high".into(),
                        "low".into(),
                        "low".into(),
                        "high".into(),
                        "high".into(),
                    ],
                    vec![6, 1],
                )
                .unwrap(),
            ),
            vec![
                Value::String("Options".into()),
                Value::Struct({
                    let mut st = StructValue::new();
                    st.insert("IterationLimit", Value::Num(200.0));
                    st.insert("Tolerance", Value::Num(1.0e-8));
                    st
                }),
            ],
        ))
        .unwrap();
        let values = output_list(out);
        let Value::Struct(stats) = &values[2] else {
            panic!("expected stats struct");
        };
        assert!(
            matches!(stats.fields.get("dfe"), Some(Value::Num(value)) if (*value - 3.0).abs() < 1.0e-8)
        );
        let Value::StringArray(class_names) = stats.fields.get("classNames").unwrap() else {
            panic!("expected text class names");
        };
        assert_eq!(class_names.data, vec!["high", "low"]);
    }

    #[test]
    fn mnrfit_rejects_unsupported_model() {
        let err = block_on(mnrfit_builtin(
            tensor_value(vec![0.0, 1.0, 2.0, 3.0], 4, 1),
            tensor_value(vec![1.0, 1.0, 2.0, 2.0], 4, 1),
            vec![
                Value::String("Model".into()),
                Value::String("ordinal".into()),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:mnrfit:InvalidArgument"));
    }

    #[test]
    fn mnrfit_reads_typed_integer_class_vectors_exactly() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let out = block_on(mnrfit_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3, 4]), 5, 1),
            poisoned_int_tensor(IntegerStorage::U8(vec![1, 1, 2, 2, 2]), 5, 1),
            vec![
                Value::String("Weights".into()),
                poisoned_int_tensor(IntegerStorage::U16(vec![1, 1, 2, 2, 1]), 5, 1),
                Value::String("IterationLimit".into()),
                poisoned_int_tensor(IntegerStorage::U16(vec![200]), 1, 1),
                Value::String("Options".into()),
                Value::Struct({
                    let mut st = StructValue::new();
                    st.insert(
                        "Tolerance",
                        poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
                    );
                    st
                }),
            ],
        ))
        .unwrap();
        let values = output_list(out);
        let Value::Tensor(b) = &values[0] else {
            panic!("expected B tensor");
        };
        assert_eq!(b.shape, vec![2, 1]);
        assert!(b.materialize_f64().iter().all(|value| value.is_finite()));
        assert!(matches!(&values[1], Value::Num(value) if value.is_finite()));
    }

    #[test]
    fn mnrfit_reads_typed_integer_grouped_counts_exactly() {
        let out = block_on(mnrfit_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
            poisoned_int_tensor(IntegerStorage::U16(vec![0, 1, 3, 5, 5, 4, 2, 1]), 4, 2),
            vec![
                Value::String("MaxIter".into()),
                poisoned_int_tensor(IntegerStorage::U16(vec![200]), 1, 1),
            ],
        ))
        .unwrap();
        let Value::Tensor(b) = out else {
            panic!("expected B tensor");
        };
        assert_eq!(b.shape, vec![2, 1]);
        assert!(b.materialize_f64().iter().all(|value| value.is_finite()));
    }
}
