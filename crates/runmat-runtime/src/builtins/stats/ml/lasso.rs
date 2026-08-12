//! Lasso and elastic-net regularized linear regression.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, LogicalArray, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "lasso";
const EPS: f64 = 1.0e-12;

const OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Coefficient matrix with one column per Lambda value.",
}];

const OUTPUT_B_FITINFO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Coefficient matrix with one column per Lambda value.",
    },
    BuiltinParamDescriptor {
        name: "FitInfo",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Fit information structure containing Lambda, Alpha, Intercept, DF, MSE, and related fields.",
    },
];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predictor matrix with observations in rows and predictors in columns.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response vector with one value per row of X.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Lambda, Alpha, Standardize, Intercept, Weights, CV, NumLambda, LambdaRatio, MaxIter, and RelTol.",
};

const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];

const LASSO_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "B = lasso(X, y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = lasso(X, y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, FitInfo] = lasso(X, y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_B_FITINFO,
    },
    BuiltinSignatureDescriptor {
        label: "[B, FitInfo] = lasso(X, y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_B_FITINFO,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSO.INVALID_ARGUMENT",
    identifier: Some("RunMat:lasso:InvalidArgument"),
    when: "Inputs, option names, option values, dimensions, or tolerances are malformed.",
    message: "lasso: invalid argument",
};

const ERROR_CONVERGENCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSO.CONVERGENCE",
    identifier: Some("RunMat:lasso:Convergence"),
    when: "Coordinate descent cannot make numerical progress for the supplied data.",
    message: "lasso: convergence failure",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSO.INTERNAL",
    identifier: Some("RunMat:lasso:Internal"),
    when: "RunMat cannot construct lasso outputs.",
    message: "lasso: internal error",
};

const LASSO_ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_CONVERGENCE, ERROR_INTERNAL];

pub const LASSO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LASSO_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LASSO_ERRORS,
};

fn lasso_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn lasso_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    lasso_error(message, &ERROR_INVALID_ARGUMENT)
}

fn convergence_error(message: impl Into<String>) -> RuntimeError {
    lasso_error(message, &ERROR_CONVERGENCE)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    lasso_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Debug)]
struct LassoOptions {
    alpha: f64,
    lambda: Option<Vec<f64>>,
    lambda_ratio: f64,
    num_lambda: usize,
    standardize: bool,
    intercept: bool,
    weights: Option<Vec<f64>>,
    rel_tol: f64,
    max_iter: usize,
    cv: CvSpec,
    predictor_names: Option<Vec<String>>,
    df_max: Option<usize>,
    use_covariance: bool,
}

impl Default for LassoOptions {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            lambda: None,
            lambda_ratio: 1.0e-4,
            num_lambda: 100,
            standardize: true,
            intercept: true,
            weights: None,
            rel_tol: 1.0e-4,
            max_iter: 100_000,
            cv: CvSpec::Resubstitution,
            predictor_names: None,
            df_max: None,
            use_covariance: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CvSpec {
    Resubstitution,
    KFold(usize),
}

#[derive(Clone, Debug)]
struct PreparedData {
    x_original: Vec<f64>,
    x_work: Vec<f64>,
    y_original: Vec<f64>,
    y_work: Vec<f64>,
    weights: Vec<f64>,
    x_means: Vec<f64>,
    x_scales: Vec<f64>,
    y_mean: f64,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug)]
struct FitPath {
    lambdas: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    df: Vec<f64>,
    mse: Vec<f64>,
    iterations: Vec<f64>,
}

#[derive(Clone, Debug)]
struct FitResult {
    b: Value,
    fit_info: Value,
}

#[runtime_builtin(
    name = "lasso",
    category = "stats/ml",
    summary = "Fit lasso or elastic-net regularized linear regression models.",
    keywords = "lasso,elastic net,regularization,regression,statistics,machine learning",
    type_resolver(lasso_type),
    descriptor(crate::builtins::stats::ml::lasso::LASSO_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::lasso"
)]
pub(crate) async fn lasso_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let x = gathered(x).await?;
    let y = gathered(y).await?;
    let rest = gather_values(rest).await?;
    let options = parse_options(rest)?;
    let result = lasso_compute(x, y, options)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![result.b])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![result.b, result.fit_info],
        )),
        None => Ok(result.b),
    }
}

async fn gathered(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("lasso: {err}")))
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gathered(value).await?);
    }
    Ok(out)
}

fn lasso_compute(x: Value, y: Value, mut options: LassoOptions) -> BuiltinResult<FitResult> {
    let x = value_to_real_tensor("X", x)?;
    let y_values = value_to_real_vector("y", y)?;
    let rows = x.rows();
    let cols = x.cols();
    if rows == 0 || cols == 0 {
        return Err(invalid_argument(
            "lasso: X must be a nonempty numeric matrix",
        ));
    }
    if y_values.len() != rows {
        return Err(invalid_argument(format!(
            "lasso: y length {} must match the number of rows in X ({rows})",
            y_values.len()
        )));
    }
    let x_values = tensor::tensor_values_f64_cow(&x);
    if y_values.iter().any(|value| !value.is_finite())
        || x_values.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_argument(
            "lasso: X and y must contain finite real values",
        ));
    }
    if options
        .predictor_names
        .as_ref()
        .is_some_and(|names| names.len() != cols)
    {
        return Err(invalid_argument(
            "lasso: PredictorNames must contain one name per predictor",
        ));
    }
    if options
        .weights
        .as_ref()
        .is_some_and(|weights| weights.len() != rows)
    {
        return Err(invalid_argument(
            "lasso: Weights vector length must match the number of rows in X",
        ));
    }
    if !options.intercept {
        options.standardize = false;
    }

    let prepared = prepare_data(&x, &y_values, &options)?;
    let lambdas = match &options.lambda {
        Some(lambda) => sorted_unique_lambdas(lambda)?,
        None => default_lambda_sequence(&prepared, &options)?,
    };
    let fit_path = fit_path(&prepared, &lambdas, &options)?;
    let fit_path = apply_df_limit(fit_path, options.df_max)?;
    let cv = match options.cv {
        CvSpec::Resubstitution => None,
        CvSpec::KFold(k) => Some(cross_validate(
            &x,
            &y_values,
            &fit_path.lambdas,
            &options,
            k,
        )?),
    };
    let b = coefficients_value(&fit_path)?;
    let fit_info = fit_info_value(&fit_path, &options, cv)?;
    Ok(FitResult { b, fit_info })
}

fn parse_options(rest: Vec<Value>) -> BuiltinResult<LassoOptions> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "lasso: name-value options must be supplied in pairs",
        ));
    }
    let mut options = LassoOptions::default();
    let mut idx = 0usize;
    while idx < rest.len() {
        let name = scalar_text(&rest[idx], "option name")?;
        let value = &rest[idx + 1];
        match canonical_option(&name).as_str() {
            "alpha" => {
                options.alpha = scalar_f64(value, "Alpha")?;
                if !(options.alpha > 0.0 && options.alpha <= 1.0) {
                    return Err(invalid_argument(
                        "lasso: Alpha must be a scalar in the interval (0, 1]",
                    ));
                }
            }
            "lambda" => options.lambda = Some(nonnegative_vector(value, "Lambda")?),
            "lambdaratio" => {
                options.lambda_ratio = scalar_f64(value, "LambdaRatio")?;
                if !(options.lambda_ratio >= 0.0 && options.lambda_ratio <= 1.0) {
                    return Err(invalid_argument(
                        "lasso: LambdaRatio must be a scalar in the interval [0, 1]",
                    ));
                }
            }
            "numlambda" => options.num_lambda = positive_usize(value, "NumLambda")?,
            "standardize" => options.standardize = scalar_bool(value, "Standardize")?,
            "intercept" => options.intercept = scalar_bool(value, "Intercept")?,
            "weights" => options.weights = Some(positive_weight_vector(value, "Weights")?),
            "reltol" => {
                options.rel_tol = scalar_f64(value, "RelTol")?;
                if !(options.rel_tol > 0.0 && options.rel_tol.is_finite()) {
                    return Err(invalid_argument(
                        "lasso: RelTol must be positive and finite",
                    ));
                }
            }
            "maxiter" => options.max_iter = positive_usize(value, "MaxIter")?,
            "cv" => options.cv = parse_cv(value)?,
            "predictornames" => {
                options.predictor_names = Some(string_list(value, "PredictorNames")?)
            }
            "dfmax" => options.df_max = Some(positive_usize(value, "DFmax")?),
            "usecovariance" => {
                options.use_covariance = match value {
                    Value::String(text) => text.eq_ignore_ascii_case("true"),
                    Value::CharArray(chars) => chars
                        .data
                        .iter()
                        .collect::<String>()
                        .eq_ignore_ascii_case("true"),
                    _ => scalar_bool(value, "UseCovariance")?,
                };
            }
            "cacheSize" | "cachesize" | "abstol" | "b0" | "u0" | "rho" | "options" => {}
            "mcreps" => {
                if positive_usize(value, "MCReps")? != 1 {
                    return Err(invalid_argument(
                        "lasso: MCReps values greater than 1 are not supported",
                    ));
                }
            }
            other => {
                return Err(invalid_argument(format!(
                    "lasso: unsupported option '{other}'"
                )))
            }
        }
        idx += 2;
    }
    Ok(options)
}

fn canonical_option(name: &str) -> String {
    name.chars()
        .filter(|ch| *ch != '_' && *ch != '-')
        .collect::<String>()
        .to_ascii_lowercase()
}

fn value_to_real_tensor(label: &str, value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::LogicalArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data.len());
            Tensor::new(
                array
                    .data
                    .into_iter()
                    .map(|flag| if flag == 0 { 0.0 } else { 1.0 })
                    .collect(),
                shape,
            )
            .map_err(|err| invalid_argument(format!("lasso: {label}: {err}")))
        }
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|err| invalid_argument(format!("lasso: {label}: {err}"))),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1])
            .map_err(|err| invalid_argument(format!("lasso: {label}: {err}"))),
        other => Err(invalid_argument(format!(
            "lasso: {label} must be real numeric, got {other:?}"
        ))),
    }
}

fn value_to_real_vector(label: &str, value: Value) -> BuiltinResult<Vec<f64>> {
    let tensor = value_to_real_tensor(label, value)?;
    if tensor.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(format!("lasso: {label} must be a vector")));
    }
    Ok(tensor::tensor_into_values_f64(tensor))
}

fn scalar_text(value: &Value, label: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(invalid_argument(format!(
            "lasso: {label} must be a string scalar, got {other:?}"
        ))),
    }
}

fn scalar_f64(value: &Value, label: &str) -> BuiltinResult<f64> {
    let number = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            return Err(invalid_argument(format!(
                "lasso: {label} must be a numeric scalar, got {other:?}"
            )))
        }
    };
    if !number.is_finite() {
        return Err(invalid_argument(format!("lasso: {label} must be finite")));
    }
    Ok(number)
}

fn scalar_bool(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) if i.to_f64() == 0.0 || i.to_f64() == 1.0 => Ok(i.to_f64() != 0.0),
        Value::String(text)
            if text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("false") =>
        {
            Ok(text.eq_ignore_ascii_case("true"))
        }
        Value::CharArray(chars) => {
            let text: String = chars.data.iter().collect();
            if text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("false") {
                Ok(text.eq_ignore_ascii_case("true"))
            } else {
                Err(invalid_argument(format!(
                    "lasso: {label} must be logical scalar"
                )))
            }
        }
        _ => Err(invalid_argument(format!(
            "lasso: {label} must be logical scalar"
        ))),
    }
}

fn positive_usize(value: &Value, label: &str) -> BuiltinResult<usize> {
    let raw = scalar_f64(value, label)?;
    if raw < 1.0
        || raw.fract() != 0.0
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
        return Err(invalid_argument(format!(
            "lasso: {label} must be a positive integer scalar"
        )));
    }
    Ok(raw as usize)
}

fn numeric_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = match value {
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) {
                return Err(invalid_argument(format!(
                    "lasso: {label} must be a numeric vector"
                )));
            }
            tensor::tensor_values_f64(tensor)
        }
        Value::LogicalArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data.len());
            if !is_vector_shape(&shape) {
                return Err(invalid_argument(format!(
                    "lasso: {label} must be a numeric vector"
                )));
            }
            array
                .data
                .iter()
                .map(|flag| if *flag == 0 { 0.0 } else { 1.0 })
                .collect()
        }
        Value::Num(n) => vec![*n],
        Value::Int(i) => vec![i.to_f64()],
        other => {
            return Err(invalid_argument(format!(
                "lasso: {label} must be a numeric vector, got {other:?}"
            )))
        }
    };
    if data.is_empty() || data.iter().any(|value| !value.is_finite()) {
        return Err(invalid_argument(format!(
            "lasso: {label} must contain finite values"
        )));
    }
    Ok(data)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|dim| **dim > 1).count() <= 1
}

fn nonnegative_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = numeric_vector(value, label)?;
    if data.iter().any(|value| *value < 0.0) {
        return Err(invalid_argument(format!(
            "lasso: {label} values must be nonnegative"
        )));
    }
    Ok(data)
}

fn positive_weight_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = numeric_vector(value, label)?;
    if data.iter().any(|value| *value < 0.0) || data.iter().all(|value| *value == 0.0) {
        return Err(invalid_argument(format!(
            "lasso: {label} must contain nonnegative values with positive total weight"
        )));
    }
    Ok(data)
}

fn parse_cv(value: &Value) -> BuiltinResult<CvSpec> {
    match value {
        Value::String(text) if text.eq_ignore_ascii_case("resubstitution") => {
            Ok(CvSpec::Resubstitution)
        }
        Value::CharArray(chars) => {
            let text: String = chars.data.iter().collect();
            if text.eq_ignore_ascii_case("resubstitution") {
                Ok(CvSpec::Resubstitution)
            } else {
                Err(invalid_argument(
                    "lasso: CV supports 'resubstitution' or a positive integer",
                ))
            }
        }
        _ => {
            let k = positive_usize(value, "CV")?;
            if k < 2 {
                return Err(invalid_argument("lasso: CV fold count must be at least 2"));
            }
            Ok(CvSpec::KFold(k))
        }
    }
}

fn string_list(value: &Value, label: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(chars) => Ok(vec![chars.data.iter().collect()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| scalar_text(item, label))
            .collect(),
        other => Err(invalid_argument(format!(
            "lasso: {label} must be a string array or cell array of character vectors, got {other:?}"
        ))),
    }
}

fn prepare_data(x: &Tensor, y: &[f64], options: &LassoOptions) -> BuiltinResult<PreparedData> {
    let rows = x.rows();
    let cols = x.cols();
    let x_values = tensor::tensor_values_f64_cow(x);
    let mut weights = options.weights.clone().unwrap_or_else(|| vec![1.0; rows]);
    let weight_sum: f64 = weights.iter().sum();
    if !(weight_sum > 0.0 && weight_sum.is_finite()) {
        return Err(invalid_argument(
            "lasso: Weights must have a positive finite sum",
        ));
    }
    for weight in &mut weights {
        *weight /= weight_sum;
    }

    let y_mean = if options.intercept {
        weighted_mean(y, &weights)
    } else {
        0.0
    };
    let y_work = y.iter().map(|value| *value - y_mean).collect::<Vec<_>>();

    let mut x_means = vec![0.0; cols];
    let mut x_scales = vec![1.0; cols];
    let mut x_work = vec![0.0; x_values.len()];
    for col in 0..cols {
        let mean = if options.intercept {
            weighted_column_mean(&x_values, rows, &weights, col)
        } else {
            0.0
        };
        let mut scale = 1.0;
        if options.intercept && options.standardize {
            let variance = (0..rows)
                .map(|row| {
                    let centered = x_values[row + col * rows] - mean;
                    weights[row] * centered * centered
                })
                .sum::<f64>();
            scale = variance.sqrt();
            if scale <= EPS || !scale.is_finite() {
                scale = 1.0;
            }
        }
        x_means[col] = mean;
        x_scales[col] = scale;
        for row in 0..rows {
            let idx = row + col * rows;
            x_work[idx] = (x_values[idx] - mean) / scale;
        }
    }

    Ok(PreparedData {
        x_original: x_values.to_vec(),
        x_work,
        y_original: y.to_vec(),
        y_work,
        weights,
        x_means,
        x_scales,
        y_mean,
        rows,
        cols,
    })
}

fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| value * weight)
        .sum()
}

fn weighted_column_mean(x_values: &[f64], rows: usize, weights: &[f64], col: usize) -> f64 {
    (0..rows)
        .map(|row| x_values[row + col * rows] * weights[row])
        .sum()
}

fn default_lambda_sequence(data: &PreparedData, options: &LassoOptions) -> BuiltinResult<Vec<f64>> {
    let lambda_max = (0..data.cols)
        .map(|col| {
            (0..data.rows)
                .map(|row| data.weights[row] * x_at(data, row, col) * data.y_work[row])
                .sum::<f64>()
                .abs()
        })
        .fold(0.0, f64::max)
        / options.alpha.max(EPS);
    if options.num_lambda == 1 {
        return Ok(vec![lambda_max]);
    }
    if lambda_max <= EPS {
        return Ok(vec![0.0]);
    }
    let ratio = options.lambda_ratio;
    if ratio <= EPS {
        let mut values = geometric_sequence(lambda_max, 1.0e-4, options.num_lambda);
        if let Some(last) = values.last_mut() {
            *last = 0.0;
        }
        values.sort_by(|a, b| a.total_cmp(b));
        return Ok(values);
    }
    let mut values = geometric_sequence(lambda_max, ratio, options.num_lambda);
    values.sort_by(|a, b| a.total_cmp(b));
    Ok(values)
}

fn geometric_sequence(max_value: f64, ratio: f64, count: usize) -> Vec<f64> {
    let min_value = max_value * ratio;
    (0..count)
        .map(|idx| {
            let t = idx as f64 / (count - 1) as f64;
            max_value * (min_value / max_value).powf(t)
        })
        .collect()
}

fn sorted_unique_lambdas(lambda: &[f64]) -> BuiltinResult<Vec<f64>> {
    if lambda.is_empty() {
        return Err(invalid_argument("lasso: Lambda must not be empty"));
    }
    let mut values = lambda.to_vec();
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup_by(|a, b| (*a - *b).abs() <= EPS);
    Ok(values)
}

fn fit_path(
    data: &PreparedData,
    lambdas_ascending: &[f64],
    options: &LassoOptions,
) -> BuiltinResult<FitPath> {
    let mut descending = lambdas_ascending.to_vec();
    descending.sort_by(|a, b| b.total_cmp(a));
    let mut beta_scaled = vec![0.0; data.cols];
    let mut entries = Vec::with_capacity(descending.len());
    for lambda in descending {
        let iterations = fit_single_lambda(data, &mut beta_scaled, lambda, options)?;
        let beta_original = unstandardize_coefficients(data, &beta_scaled);
        let intercept = if options.intercept {
            data.y_mean
                - beta_original
                    .iter()
                    .zip(data.x_means.iter())
                    .map(|(beta, mean)| beta * mean)
                    .sum::<f64>()
        } else {
            0.0
        };
        let mse = weighted_mse(data, &beta_original, intercept);
        let df = beta_original
            .iter()
            .filter(|value| value.abs() > 1.0e-8)
            .count() as f64;
        entries.push((lambda, beta_original, intercept, df, mse, iterations as f64));
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(FitPath {
        lambdas: entries.iter().map(|entry| entry.0).collect(),
        coefficients: entries.iter().map(|entry| entry.1.clone()).collect(),
        intercepts: entries.iter().map(|entry| entry.2).collect(),
        df: entries.iter().map(|entry| entry.3).collect(),
        mse: entries.iter().map(|entry| entry.4).collect(),
        iterations: entries.iter().map(|entry| entry.5).collect(),
    })
}

fn fit_single_lambda(
    data: &PreparedData,
    beta: &mut [f64],
    lambda: f64,
    options: &LassoOptions,
) -> BuiltinResult<usize> {
    let mut residual = data
        .y_work
        .iter()
        .enumerate()
        .map(|(row, y)| {
            let predicted = (0..data.cols)
                .map(|col| x_at(data, row, col) * beta[col])
                .sum::<f64>();
            y - predicted
        })
        .collect::<Vec<_>>();
    let mut column_norms = vec![0.0; data.cols];
    for (col, norm) in column_norms.iter_mut().enumerate() {
        *norm = (0..data.rows)
            .map(|row| data.weights[row] * x_at(data, row, col) * x_at(data, row, col))
            .sum::<f64>();
    }
    for iter in 1..=options.max_iter {
        let mut max_delta = 0.0_f64;
        let mut max_beta = 0.0_f64;
        for col in 0..data.cols {
            if column_norms[col] <= EPS {
                beta[col] = 0.0;
                continue;
            }
            let old = beta[col];
            let rho = (0..data.rows)
                .map(|row| {
                    data.weights[row]
                        * x_at(data, row, col)
                        * (residual[row] + x_at(data, row, col) * old)
                })
                .sum::<f64>();
            let denom = column_norms[col] + lambda * (1.0 - options.alpha);
            if denom <= EPS || !denom.is_finite() {
                return Err(convergence_error(
                    "lasso: coordinate descent encountered a singular predictor column",
                ));
            }
            let new_beta = soft_threshold(rho, lambda * options.alpha) / denom;
            let delta = new_beta - old;
            if delta != 0.0 {
                for (row, value) in residual.iter_mut().enumerate() {
                    *value -= x_at(data, row, col) * delta;
                }
                beta[col] = new_beta;
            }
            max_delta = max_delta.max(delta.abs());
            max_beta = max_beta.max(new_beta.abs());
        }
        if max_delta <= options.rel_tol * max_beta.max(1.0) {
            return Ok(iter);
        }
    }
    Ok(options.max_iter)
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

fn x_at(data: &PreparedData, row: usize, col: usize) -> f64 {
    data.x_work[row + col * data.rows]
}

fn unstandardize_coefficients(data: &PreparedData, beta_scaled: &[f64]) -> Vec<f64> {
    beta_scaled
        .iter()
        .zip(data.x_scales.iter())
        .map(|(beta, scale)| beta / scale)
        .collect()
}

fn weighted_mse(data: &PreparedData, beta: &[f64], intercept: f64) -> f64 {
    (0..data.rows)
        .map(|row| {
            let predicted = intercept
                + (0..data.cols)
                    .map(|col| data.x_original[row + col * data.rows] * beta[col])
                    .sum::<f64>();
            let err = data.y_original[row] - predicted;
            data.weights[row] * err * err
        })
        .sum()
}

fn apply_df_limit(mut path: FitPath, df_max: Option<usize>) -> BuiltinResult<FitPath> {
    let Some(max_df) = df_max else {
        return Ok(path);
    };
    let keep = path
        .df
        .iter()
        .enumerate()
        .filter_map(|(idx, df)| (*df as usize <= max_df).then_some(idx))
        .collect::<Vec<_>>();
    if keep.is_empty() {
        return Err(invalid_argument(
            "lasso: DFmax excludes every fitted Lambda value",
        ));
    }
    path.lambdas = keep.iter().map(|idx| path.lambdas[*idx]).collect();
    path.coefficients = keep
        .iter()
        .map(|idx| path.coefficients[*idx].clone())
        .collect();
    path.intercepts = keep.iter().map(|idx| path.intercepts[*idx]).collect();
    path.df = keep.iter().map(|idx| path.df[*idx]).collect();
    path.mse = keep.iter().map(|idx| path.mse[*idx]).collect();
    path.iterations = keep.iter().map(|idx| path.iterations[*idx]).collect();
    Ok(path)
}

#[derive(Clone, Debug)]
struct CvInfo {
    mse: Vec<f64>,
    se: Vec<f64>,
    lambda_min_mse: f64,
    lambda_1se: f64,
    index_min_mse: usize,
    index_1se: usize,
}

fn cross_validate(
    x: &Tensor,
    y: &[f64],
    lambdas: &[f64],
    options: &LassoOptions,
    k: usize,
) -> BuiltinResult<CvInfo> {
    if k > y.len() {
        return Err(invalid_argument(
            "lasso: CV fold count cannot exceed the number of observations",
        ));
    }
    let mut fold_errors = vec![vec![0.0; lambdas.len()]; k];
    for fold in 0..k {
        let train_rows = (0..x.rows())
            .filter(|row| row % k != fold)
            .collect::<Vec<_>>();
        let test_rows = (0..x.rows())
            .filter(|row| row % k == fold)
            .collect::<Vec<_>>();
        let train_x = subset_rows_tensor(x, &train_rows)?;
        let train_y = train_rows.iter().map(|row| y[*row]).collect::<Vec<_>>();
        let mut fold_options = options.clone();
        fold_options.cv = CvSpec::Resubstitution;
        fold_options.lambda = Some(lambdas.to_vec());
        if let Some(weights) = &options.weights {
            fold_options.weights = Some(train_rows.iter().map(|row| weights[*row]).collect());
        }
        let prepared = prepare_data(&train_x, &train_y, &fold_options)?;
        let path = fit_path(&prepared, lambdas, &fold_options)?;
        for (lambda_idx, beta) in path.coefficients.iter().enumerate() {
            let intercept = path.intercepts[lambda_idx];
            fold_errors[fold][lambda_idx] = validation_mse(
                x,
                y,
                beta,
                intercept,
                &test_rows,
                options.weights.as_deref(),
            );
        }
    }
    let mut mse = vec![0.0; lambdas.len()];
    let mut se = vec![0.0; lambdas.len()];
    for lambda_idx in 0..lambdas.len() {
        let values = (0..k)
            .map(|fold| fold_errors[fold][lambda_idx])
            .collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / k as f64;
        let variance = if k > 1 {
            values
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (k - 1) as f64
        } else {
            0.0
        };
        mse[lambda_idx] = mean;
        se[lambda_idx] = variance.sqrt() / (k as f64).sqrt();
    }
    let index_min_mse = mse
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let threshold = mse[index_min_mse] + se[index_min_mse];
    let index_1se = mse
        .iter()
        .enumerate()
        .filter(|(_, value)| **value <= threshold)
        .max_by(|(left, _), (right, _)| lambdas[*left].total_cmp(&lambdas[*right]))
        .map(|(idx, _)| idx)
        .unwrap_or(index_min_mse);
    Ok(CvInfo {
        mse,
        se,
        lambda_min_mse: lambdas[index_min_mse],
        lambda_1se: lambdas[index_1se],
        index_min_mse: index_min_mse + 1,
        index_1se: index_1se + 1,
    })
}

fn subset_rows_tensor(x: &Tensor, rows: &[usize]) -> BuiltinResult<Tensor> {
    let mut data = Vec::with_capacity(rows.len() * x.cols());
    for col in 0..x.cols() {
        for row in rows {
            data.push(
                x.get2(*row, col)
                    .map_err(|err| invalid_argument(format!("lasso: {err}")))?,
            );
        }
    }
    Tensor::new(data, vec![rows.len(), x.cols()])
        .map_err(|err| invalid_argument(format!("lasso: {err}")))
}

fn validation_mse(
    x: &Tensor,
    y: &[f64],
    beta: &[f64],
    intercept: f64,
    rows: &[usize],
    weights: Option<&[f64]>,
) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for row in rows {
        let predicted = intercept
            + (0..x.cols())
                .map(|col| x.get2(*row, col).unwrap_or(0.0) * beta[col])
                .sum::<f64>();
        let err = y[*row] - predicted;
        let weight = weights.map(|values| values[*row]).unwrap_or(1.0);
        sum += weight * err * err;
        weight_sum += weight;
    }
    if weight_sum <= EPS {
        return f64::NAN;
    }
    sum / weight_sum
}

fn coefficients_value(path: &FitPath) -> BuiltinResult<Value> {
    let rows = path
        .coefficients
        .first()
        .map(|coeffs| coeffs.len())
        .unwrap_or(0);
    let cols = path.coefficients.len();
    let mut data = Vec::with_capacity(rows * cols);
    for lambda_idx in 0..cols {
        for row in 0..rows {
            data.push(path.coefficients[lambda_idx][row]);
        }
    }
    Tensor::new(data, vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| internal_error(format!("lasso: {err}")))
}

fn row_tensor(values: &[f64]) -> BuiltinResult<Value> {
    Tensor::new(values.to_vec(), vec![1, values.len()])
        .map(Value::Tensor)
        .map_err(|err| internal_error(format!("lasso: {err}")))
}

fn scalar_tensor(value: f64) -> BuiltinResult<Value> {
    Tensor::new(vec![value], vec![1, 1])
        .map(Value::Tensor)
        .map_err(|err| internal_error(format!("lasso: {err}")))
}

fn fit_info_value(
    path: &FitPath,
    options: &LassoOptions,
    cv: Option<CvInfo>,
) -> BuiltinResult<Value> {
    let mut st = StructValue::new();
    st.insert("Intercept", row_tensor(&path.intercepts)?);
    st.insert("Lambda", row_tensor(&path.lambdas)?);
    st.insert("Alpha", Value::Num(options.alpha));
    st.insert("DF", row_tensor(&path.df)?);
    st.insert(
        "MSE",
        row_tensor(
            cv.as_ref()
                .map(|info| info.mse.as_slice())
                .unwrap_or(&path.mse),
        )?,
    );
    st.insert("PredictorNames", predictor_names_value(options)?);
    st.insert("UseCovariance", logical_scalar(false)?);
    st.insert("Iterations", row_tensor(&path.iterations)?);
    if let Some(cv) = cv {
        st.insert("SE", row_tensor(&cv.se)?);
        st.insert("LambdaMinMSE", scalar_tensor(cv.lambda_min_mse)?);
        st.insert("Lambda1SE", scalar_tensor(cv.lambda_1se)?);
        st.insert("IndexMinMSE", Value::Num(cv.index_min_mse as f64));
        st.insert("Index1SE", Value::Num(cv.index_1se as f64));
    }
    Ok(Value::Struct(st))
}

fn predictor_names_value(options: &LassoOptions) -> BuiltinResult<Value> {
    let names = options.predictor_names.as_deref().unwrap_or(&[]);
    let data = names
        .iter()
        .map(|name| Value::CharArray(CharArray::new_row(name)))
        .collect::<Vec<_>>();
    CellArray::new(data, 1, names.len())
        .map(Value::Cell)
        .map_err(|err| internal_error(format!("lasso: {err}")))
}

fn logical_scalar(value: bool) -> BuiltinResult<Value> {
    LogicalArray::new(vec![if value { 1 } else { 0 }], vec![1, 1])
        .map(Value::LogicalArray)
        .map_err(|err| internal_error(format!("lasso: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::StringArray;
    use runmat_value::{IntValue, IntegerStorage};

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    fn cleared_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).unwrap();
        Value::Tensor(tensor)
    }

    fn output_pair(value: Value) -> (Value, Value) {
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        (values[0].clone(), values[1].clone())
    }

    fn row_field<'a>(info: &'a StructValue, name: &str) -> &'a Tensor {
        let Some(Value::Tensor(tensor)) = info.fields.get(name) else {
            panic!("expected tensor field {name}");
        };
        tensor
    }

    #[test]
    fn explicit_lambda_recovers_sparse_coefficients_with_intercept() {
        let x = tensor(
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, //
                1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            vec![5, 2],
        );
        let y = tensor(vec![1.0, 3.0, 5.0, 7.0, 9.0], vec![5, 1]);
        let out = block_on(lasso_builtin(
            x,
            y,
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.0], vec![1, 1]),
                Value::CharArray(CharArray::new_row("Standardize")),
                Value::Bool(false),
            ],
        ))
        .unwrap();
        let Value::Tensor(coeffs) = out else {
            panic!("expected coefficient tensor");
        };
        assert_eq!(coeffs.shape, vec![2, 1]);
        assert!((coeffs.materialize_f64()[0] - 2.0).abs() < 1.0e-8);
        assert!(coeffs.materialize_f64()[1].abs() < 1.0e-8);
    }

    #[test]
    fn alpha_weights_and_fitinfo_output_are_supported() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let x = tensor(
            vec![
                0.0, 1.0, 2.0, 3.0, //
                0.0, 0.0, 1.0, 1.0,
            ],
            vec![4, 2],
        );
        let y = tensor(vec![0.0, 1.0, 3.0, 4.0], vec![4, 1]);
        let out = block_on(lasso_builtin(
            x,
            y,
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.0, 0.2], vec![1, 2]),
                Value::CharArray(CharArray::new_row("Alpha")),
                Value::Num(0.5),
                Value::CharArray(CharArray::new_row("Weights")),
                tensor(vec![1.0, 2.0, 1.0, 2.0], vec![4, 1]),
                Value::CharArray(CharArray::new_row("PredictorNames")),
                Value::StringArray(
                    StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap(),
                ),
            ],
        ))
        .unwrap();
        let (b, fit_info) = output_pair(out);
        let Value::Tensor(coeffs) = b else {
            panic!("expected coefficients");
        };
        assert_eq!(coeffs.shape, vec![2, 2]);
        let Value::Struct(info) = fit_info else {
            panic!("expected FitInfo struct");
        };
        assert!(info.fields.contains_key("Intercept"));
        assert!(info.fields.contains_key("Lambda"));
        assert!(info.fields.contains_key("DF"));
        assert!(info.fields.contains_key("MSE"));
        assert!(info.fields.contains_key("PredictorNames"));
    }

    #[test]
    fn lasso_reads_typed_integer_storage_exactly() {
        let x = poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), vec![4, 1]);
        let y = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7]), vec![4, 1]);
        let out = block_on(lasso_builtin(
            x,
            y,
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
                Value::CharArray(CharArray::new_row("Alpha")),
                cleared_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]),
                Value::CharArray(CharArray::new_row("Weights")),
                poisoned_int_tensor(IntegerStorage::U8(vec![1, 1, 1, 1]), vec![4, 1]),
                Value::CharArray(CharArray::new_row("MaxIter")),
                cleared_int_tensor(IntegerStorage::U16(vec![50]), vec![1, 1]),
                Value::CharArray(CharArray::new_row("Standardize")),
                Value::Bool(false),
            ],
        ))
        .unwrap();
        let Value::Tensor(coeffs) = out else {
            panic!("expected coefficient tensor");
        };
        assert_eq!(coeffs.shape, vec![1, 1]);
        assert!((coeffs.materialize_f64()[0] - 2.0).abs() < 1.0e-8);
    }

    #[test]
    fn lasso_positive_usize_parses_integer_bounds_exactly() {
        assert_eq!(
            positive_usize(&Value::Int(IntValue::U16(3)), "NumLambda").unwrap(),
            3
        );
        assert_eq!(
            positive_usize(
                &cleared_int_tensor(IntegerStorage::U64(vec![3]), vec![1, 1]),
                "NumLambda"
            )
            .unwrap(),
            3
        );

        for value in [
            Value::Int(IntValue::I8(-1)),
            Value::Num(1.5),
            Value::Num(usize::MAX as f64),
            Value::Num(usize::MAX as f64 + 1.0),
            cleared_int_tensor(IntegerStorage::U64(vec![usize::MAX as u64]), vec![1, 1]),
        ] {
            assert!(positive_usize(&value, "NumLambda").is_err());
        }
    }

    #[test]
    fn cross_validation_adds_minimum_error_fields() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let x = tensor(
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, //
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            ],
            vec![6, 2],
        );
        let y = tensor(vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0], vec![6, 1]);
        let out = block_on(lasso_builtin(
            x,
            y,
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.0, 0.1, 0.3], vec![1, 3]),
                Value::CharArray(CharArray::new_row("CV")),
                Value::Num(3.0),
            ],
        ))
        .unwrap();
        let (_, fit_info) = output_pair(out);
        let Value::Struct(info) = fit_info else {
            panic!("expected FitInfo");
        };
        assert!(info.fields.contains_key("SE"));
        assert!(info.fields.contains_key("LambdaMinMSE"));
        assert!(info.fields.contains_key("Lambda1SE"));
        assert!(info.fields.contains_key("IndexMinMSE"));
        assert!(info.fields.contains_key("Index1SE"));
    }

    #[test]
    fn dfmax_filters_cv_fields_consistently() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let x = tensor(
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, //
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            ],
            vec![6, 2],
        );
        let y = tensor(vec![0.0, 2.0, 4.0, 7.0, 8.0, 11.0], vec![6, 1]);
        let out = block_on(lasso_builtin(
            x,
            y,
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.0, 10.0], vec![1, 2]),
                Value::CharArray(CharArray::new_row("DFmax")),
                Value::Num(1.0),
                Value::CharArray(CharArray::new_row("CV")),
                Value::Num(3.0),
            ],
        ))
        .unwrap();
        let (b, fit_info) = output_pair(out);
        let Value::Tensor(coeffs) = b else {
            panic!("expected coefficients");
        };
        let Value::Struct(info) = fit_info else {
            panic!("expected FitInfo");
        };
        let lambda = row_field(&info, "Lambda");
        let mse = row_field(&info, "MSE");
        let se = row_field(&info, "SE");
        assert_eq!(coeffs.cols, lambda.materialize_f64().len());
        assert_eq!(mse.materialize_f64().len(), lambda.materialize_f64().len());
        assert_eq!(se.materialize_f64().len(), lambda.materialize_f64().len());
        let Some(Value::Num(index_min_mse)) = info.fields.get("IndexMinMSE") else {
            panic!("expected IndexMinMSE");
        };
        let Some(Value::Num(index_1se)) = info.fields.get("Index1SE") else {
            panic!("expected Index1SE");
        };
        assert!(*index_min_mse >= 1.0 && *index_min_mse <= lambda.materialize_f64().len() as f64);
        assert!(*index_1se >= 1.0 && *index_1se <= lambda.materialize_f64().len() as f64);
    }

    #[test]
    fn omitted_predictor_names_and_use_covariance_report_actual_state() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let x = tensor(
            vec![
                0.0, 1.0, 2.0, 3.0, //
                1.0, 1.0, 1.0, 1.0,
            ],
            vec![4, 2],
        );
        let y = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let out = block_on(lasso_builtin(
            x,
            y,
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.1], vec![1, 1]),
                Value::CharArray(CharArray::new_row("UseCovariance")),
                Value::Bool(true),
            ],
        ))
        .unwrap();
        let (_, fit_info) = output_pair(out);
        let Value::Struct(info) = fit_info else {
            panic!("expected FitInfo");
        };
        let Some(Value::Cell(names)) = info.fields.get("PredictorNames") else {
            panic!("expected PredictorNames cell");
        };
        assert_eq!(names.data.len(), 0);
        assert_eq!(names.shape, vec![1, 0]);
        let Some(Value::LogicalArray(use_covariance)) = info.fields.get("UseCovariance") else {
            panic!("expected UseCovariance logical");
        };
        assert_eq!(use_covariance.data, vec![0]);
    }

    #[test]
    fn matrix_shaped_lambda_and_weights_are_rejected() {
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let y = tensor(vec![1.0, 2.0], vec![2, 1]);
        let err = block_on(lasso_builtin(
            x.clone(),
            y.clone(),
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.0, 0.1, 0.2, 0.3], vec![2, 2]),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lasso:InvalidArgument"));

        let err = block_on(lasso_builtin(
            x,
            y,
            vec![
                Value::CharArray(CharArray::new_row("Weights")),
                tensor(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lasso:InvalidArgument"));
    }

    #[test]
    fn validation_mse_uses_heldout_weights() {
        let x = Tensor::new(vec![0.0, 0.0], vec![2, 1]).unwrap();
        let weighted = validation_mse(&x, &[1.0, 3.0], &[0.0], 0.0, &[0, 1], Some(&[9.0, 1.0]));
        assert!((weighted - 1.8).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_dimensions_error() {
        let err = block_on(lasso_builtin(
            tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            tensor(vec![1.0, 2.0, 3.0], vec![3, 1]),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lasso:InvalidArgument"));
    }
}
