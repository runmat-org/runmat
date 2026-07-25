//! Lasso and elastic-net regularized generalized linear models.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, LogicalArray, ResolveContext, StructValue, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "lassoglm";
const EPS: f64 = 1.0e-12;
const MAX_NUM_LAMBDA: usize = 10_000;
const MAX_ITERATIONS: usize = 1_000_000;

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
        description:
            "Fit information structure containing Lambda, Intercept, Deviance, DF, and diagnostics.",
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
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Response vector, or two-column binomial successes/trials matrix.",
};

const PARAM_DISTR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "distr",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution name: normal, binomial, or poisson.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Lambda, Alpha, Standardize, Intercept, Weights, Offset, CV, NumLambda, LambdaRatio, MaxIter, RelTol, and Options.",
};

const INPUTS_REQUIRED: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_DISTR];
const INPUTS_FULL: [BuiltinParamDescriptor; 4] = [PARAM_X, PARAM_Y, PARAM_DISTR, PARAM_OPTIONS];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "B = lassoglm(X, Y, distr)",
        inputs: &INPUTS_REQUIRED,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = lassoglm(X, Y, distr, Name, Value)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, FitInfo] = lassoglm(X, Y, distr)",
        inputs: &INPUTS_REQUIRED,
        outputs: &OUTPUT_B_FITINFO,
    },
    BuiltinSignatureDescriptor {
        label: "[B, FitInfo] = lassoglm(X, Y, distr, Name, Value)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_B_FITINFO,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSOGLM.INVALID_ARGUMENT",
    identifier: Some("RunMat:lassoglm:InvalidArgument"),
    when: "Inputs, distribution name, dimensions, option names, or option values are malformed.",
    message: "lassoglm: invalid argument",
};

const ERROR_CONVERGENCE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSOGLM.CONVERGENCE",
    identifier: Some("RunMat:lassoglm:Convergence"),
    when: "The regularized GLM solver cannot make numerical progress.",
    message: "lassoglm: convergence failure",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LASSOGLM.INTERNAL",
    identifier: Some("RunMat:lassoglm:Internal"),
    when: "RunMat cannot construct lassoglm outputs.",
    message: "lassoglm: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_CONVERGENCE, ERROR_INTERNAL];

pub const LASSOGLM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn lassoglm_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Distribution {
    Normal,
    Binomial,
    Poisson,
}

impl Distribution {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "distribution")?;
        match canonical_name(&text).as_str() {
            "normal" | "gaussian" => Ok(Self::Normal),
            "binomial" => Ok(Self::Binomial),
            "poisson" => Ok(Self::Poisson),
            other => Err(invalid(format!(
                "lassoglm: unsupported distribution '{other}'"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Binomial => "binomial",
            Self::Poisson => "poisson",
        }
    }
}

#[derive(Clone, Debug)]
struct Options {
    alpha: f64,
    lambda: Option<Vec<f64>>,
    lambda_ratio: f64,
    num_lambda: usize,
    standardize: bool,
    intercept: bool,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    binomial_size: Option<Vec<f64>>,
    link: Option<String>,
    est_disp: Option<String>,
    rel_tol: f64,
    max_iter: usize,
    cv: CvSpec,
    predictor_names: Option<Vec<String>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            lambda: None,
            lambda_ratio: 1.0e-4,
            num_lambda: 100,
            standardize: true,
            intercept: true,
            weights: None,
            offset: None,
            binomial_size: None,
            link: None,
            est_disp: None,
            rel_tol: 1.0e-4,
            max_iter: 1000,
            cv: CvSpec::Resubstitution,
            predictor_names: None,
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
    y: Vec<f64>,
    trials: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    x_means: Vec<f64>,
    x_scales: Vec<f64>,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug)]
struct FitPath {
    lambdas: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    deviance: Vec<f64>,
    df: Vec<f64>,
    iterations: Vec<f64>,
}

#[derive(Clone, Debug)]
struct CvInfo {
    deviance: Vec<f64>,
    se: Vec<f64>,
    lambda_min_deviance: f64,
    lambda_1se: f64,
    index_min_deviance: usize,
    index_1se: usize,
}

#[derive(Clone, Debug)]
struct FitResult {
    b: Value,
    fit_info: Value,
}

#[runtime_builtin(
    name = "lassoglm",
    category = "stats/ml",
    summary = "Fit lasso or elastic-net regularized generalized linear models.",
    keywords = "lassoglm,lasso,elastic net,generalized linear model,glm,binomial,poisson,statistics,machine learning",
    type_resolver(lassoglm_type),
    descriptor(crate::builtins::stats::ml::lassoglm::LASSOGLM_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::lassoglm"
)]
pub(crate) async fn lassoglm_builtin(
    x: Value,
    y: Value,
    distr: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let x = gathered(x).await?;
    let y = gathered(y).await?;
    let distr = gathered(distr).await?;
    let rest = gather_values(rest).await?;
    let distribution = Distribution::parse(&distr)?;
    let options = parse_options(rest)?;
    let result = lassoglm_compute(x, y, distribution, options)?;
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
        .map_err(|err| invalid(format!("lassoglm: {err}")))
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gathered(value).await?);
    }
    Ok(out)
}

fn lassoglm_compute(
    x: Value,
    y: Value,
    distribution: Distribution,
    options: Options,
) -> BuiltinResult<FitResult> {
    let x = value_to_real_tensor("X", x)?;
    let rows = x.rows();
    let cols = x.cols();
    if rows == 0 || cols == 0 {
        return Err(invalid("lassoglm: X must be a nonempty numeric matrix"));
    }
    validate_distribution_options(distribution, &options)?;
    let response = response_values(y, distribution, rows, &options)?;
    if response.0.len() != rows {
        return Err(invalid(format!(
            "lassoglm: Y length {} must match the number of rows in X ({rows})",
            response.0.len()
        )));
    }
    if x.data.iter().any(|value| !value.is_finite()) {
        return Err(invalid("lassoglm: X must contain finite real values"));
    }
    if options
        .predictor_names
        .as_ref()
        .is_some_and(|names| names.len() != cols)
    {
        return Err(invalid(
            "lassoglm: PredictorNames must contain one name per predictor",
        ));
    }
    if options
        .weights
        .as_ref()
        .is_some_and(|weights| weights.len() != rows)
    {
        return Err(invalid(
            "lassoglm: Weights vector length must match the number of rows in X",
        ));
    }
    if options
        .offset
        .as_ref()
        .is_some_and(|offset| offset.len() != rows)
    {
        return Err(invalid(
            "lassoglm: Offset vector length must match the number of rows in X",
        ));
    }
    let prepared = prepare_data(&x, response, &options)?;
    let lambdas = match &options.lambda {
        Some(values) => sorted_unique_lambdas(values)?,
        None => default_lambda_sequence(&prepared, distribution, &options)?,
    };
    let fit_path = fit_path(&prepared, distribution, &lambdas, &options)?;
    let cv = match options.cv {
        CvSpec::Resubstitution => None,
        CvSpec::KFold(k) => Some(cross_validate(
            &x,
            &prepared,
            distribution,
            &fit_path.lambdas,
            &options,
            k,
        )?),
    };
    let b = coefficients_value(&fit_path)?;
    let fit_info = fit_info_value(&fit_path, distribution, &options, cv)?;
    Ok(FitResult { b, fit_info })
}

fn parse_options(rest: Vec<Value>) -> BuiltinResult<Options> {
    if !rest.len().is_multiple_of(2) {
        return Err(invalid(
            "lassoglm: name-value options must be supplied in pairs",
        ));
    }
    let mut options = Options::default();
    let mut idx = 0usize;
    while idx < rest.len() {
        let name = scalar_text(&rest[idx], "option name")?;
        let value = &rest[idx + 1];
        match canonical_name(&name).as_str() {
            "alpha" => {
                options.alpha = scalar_f64(value, "Alpha")?;
                if !(options.alpha > 0.0 && options.alpha <= 1.0) {
                    return Err(invalid(
                        "lassoglm: Alpha must be a scalar in the interval (0, 1]",
                    ));
                }
            }
            "lambda" => options.lambda = Some(nonnegative_vector(value, "Lambda")?),
            "lambdaratio" => {
                options.lambda_ratio = scalar_f64(value, "LambdaRatio")?;
                if !(options.lambda_ratio >= 0.0 && options.lambda_ratio <= 1.0) {
                    return Err(invalid(
                        "lassoglm: LambdaRatio must be a scalar in the interval [0, 1]",
                    ));
                }
            }
            "numlambda" => options.num_lambda = bounded_usize(value, "NumLambda", MAX_NUM_LAMBDA)?,
            "standardize" => options.standardize = scalar_bool(value, "Standardize")?,
            "intercept" | "constant" => options.intercept = scalar_bool(value, "Intercept")?,
            "weights" => options.weights = Some(nonnegative_weight_vector(value, "Weights")?),
            "offset" => options.offset = Some(numeric_vector(value, "Offset")?),
            "binomialsize" => options.binomial_size = Some(positive_vector(value, "BinomialSize")?),
            "reltol" | "tolx" | "tolfun" => {
                options.rel_tol = scalar_f64(value, &name)?;
                if !(options.rel_tol > 0.0 && options.rel_tol.is_finite()) {
                    return Err(invalid("lassoglm: RelTol must be positive and finite"));
                }
            }
            "maxiter" => options.max_iter = bounded_usize(value, "MaxIter", MAX_ITERATIONS)?,
            "cv" => options.cv = parse_cv(value)?,
            "predictornames" => {
                options.predictor_names = Some(string_list(value, "PredictorNames")?)
            }
            "options" => apply_stat_options(value, &mut options)?,
            "link" => options.link = Some(scalar_text(value, "Link")?),
            "estdisp" => options.est_disp = Some(scalar_text(value, "EstDisp")?),
            "mcreps" => {
                if positive_usize(value, "MCReps")? != 1 {
                    return Err(invalid(
                        "lassoglm: MCReps values greater than 1 are not supported",
                    ));
                }
            }
            other => return Err(invalid(format!("lassoglm: unsupported option '{other}'"))),
        }
        idx += 2;
    }
    Ok(options)
}

fn apply_stat_options(value: &Value, options: &mut Options) -> BuiltinResult<()> {
    let Value::Struct(st) = value else {
        if is_empty_numeric(value) {
            return Ok(());
        }
        return Err(invalid("lassoglm: Options must be [] or a statset struct"));
    };
    if let Some(value) = nonempty_field(st, "MaxIter") {
        options.max_iter = bounded_usize(value, "Options.MaxIter", MAX_ITERATIONS)?;
    }
    if let Some(value) = nonempty_field(st, "TolX").or_else(|| nonempty_field(st, "TolFun")) {
        options.rel_tol = scalar_f64(value, "Options.TolX")?;
        if !(options.rel_tol > 0.0 && options.rel_tol.is_finite()) {
            return Err(invalid(
                "lassoglm: Options tolerance must be positive and finite",
            ));
        }
    }
    if let Some(value) = nonempty_field(st, "UseParallel") {
        if scalar_bool(value, "Options.UseParallel")? {
            return Err(invalid(
                "lassoglm: parallel statset Options are not supported by RunMat's CPU solver",
            ));
        }
    }
    Ok(())
}

fn validate_distribution_options(
    distribution: Distribution,
    options: &Options,
) -> BuiltinResult<()> {
    if let Some(link) = &options.link {
        let canonical = canonical_name(link);
        let supported = match distribution {
            Distribution::Normal => canonical == "identity",
            Distribution::Binomial => canonical == "logit",
            Distribution::Poisson => canonical == "log",
        };
        if !supported {
            return Err(invalid(format!(
                "lassoglm: Link '{link}' is not supported for {} distribution",
                distribution.as_str()
            )));
        }
    }
    if let Some(est_disp) = &options.est_disp {
        if canonical_name(est_disp) != "off" {
            return Err(invalid(
                "lassoglm: EstDisp values other than 'off' are not supported",
            ));
        }
    }
    if options.binomial_size.is_some() && distribution != Distribution::Binomial {
        return Err(invalid(
            "lassoglm: BinomialSize is only valid for binomial distribution",
        ));
    }
    Ok(())
}

fn nonempty_field<'a>(st: &'a StructValue, name: &str) -> Option<&'a Value> {
    st.fields.get(name).filter(|value| !is_empty_numeric(value))
}

fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.data.is_empty())
}

fn canonical_name(name: &str) -> String {
    name.chars()
        .filter(|ch| *ch != '_' && *ch != '-')
        .collect::<String>()
        .to_ascii_lowercase()
}

fn value_to_real_tensor(label: &str, value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => tensor::integer_tensor_to_f64(tensor)
            .map_err(|err| invalid(format!("lassoglm: {label}: {err}"))),
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
            .map_err(|err| invalid(format!("lassoglm: {label}: {err}")))
        }
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|err| invalid(format!("lassoglm: {label}: {err}"))),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1])
            .map_err(|err| invalid(format!("lassoglm: {label}: {err}"))),
        other => Err(invalid(format!(
            "lassoglm: {label} must be real numeric, got {other:?}"
        ))),
    }
}

fn response_values(
    value: Value,
    distribution: Distribution,
    expected_rows: usize,
    options: &Options,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let tensor = value_to_real_tensor("Y", value)?;
    if tensor.data.iter().any(|value| !value.is_finite()) {
        return Err(invalid("lassoglm: Y must contain finite real values"));
    }
    match distribution {
        Distribution::Normal => {
            if !is_vector_shape(&tensor.shape) {
                return Err(invalid("lassoglm: normal Y must be a vector"));
            }
            Ok((tensor.data, Vec::new()))
        }
        Distribution::Poisson => {
            if !is_vector_shape(&tensor.shape) {
                return Err(invalid("lassoglm: poisson Y must be a vector"));
            }
            if tensor.data.iter().any(|value| *value < 0.0) {
                return Err(invalid("lassoglm: poisson Y must be nonnegative"));
            }
            Ok((tensor.data, Vec::new()))
        }
        Distribution::Binomial => {
            if tensor.cols() == 2 && tensor.rows() == expected_rows {
                if options.binomial_size.is_some() {
                    return Err(invalid(
                        "lassoglm: BinomialSize cannot be combined with two-column binomial counts",
                    ));
                }
                let mut successes = Vec::with_capacity(tensor.rows());
                let mut trials = Vec::with_capacity(tensor.rows());
                for row in 0..tensor.rows() {
                    let y = tensor.get2(row, 0).unwrap_or(0.0);
                    let n = tensor.get2(row, 1).unwrap_or(0.0);
                    if y < 0.0 || n <= 0.0 || y > n {
                        return Err(invalid(
                            "lassoglm: binomial count response must satisfy 0 <= successes <= trials",
                        ));
                    }
                    successes.push(y / n);
                    trials.push(n);
                }
                Ok((successes, trials))
            } else {
                if !is_vector_shape(&tensor.shape) {
                    return Err(invalid(
                        "lassoglm: binomial Y must be a vector or two-column count matrix",
                    ));
                }
                if tensor.data.iter().any(|value| *value < 0.0 || *value > 1.0) {
                    return Err(invalid(
                        "lassoglm: binomial vector Y must contain probabilities in [0,1]",
                    ));
                }
                let trials = match &options.binomial_size {
                    Some(values) => expand_binomial_size(values, expected_rows)?,
                    None => Vec::new(),
                };
                Ok((tensor.data, trials))
            }
        }
    }
}

fn expand_binomial_size(values: &[f64], rows: usize) -> BuiltinResult<Vec<f64>> {
    if values.len() == 1 {
        return Ok(vec![values[0]; rows]);
    }
    if values.len() != rows {
        return Err(invalid(
            "lassoglm: BinomialSize must be scalar or contain one value per observation",
        ));
    }
    Ok(values.to_vec())
}

fn scalar_text(value: &Value, label: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(invalid(format!(
            "lassoglm: {label} must be a string scalar, got {other:?}"
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
            tensor::tensor_values_f64(tensor)[0]
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            return Err(invalid(format!(
                "lassoglm: {label} must be a numeric scalar, got {other:?}"
            )))
        }
    };
    if !number.is_finite() {
        return Err(invalid(format!("lassoglm: {label} must be finite")));
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
            if text.eq_ignore_ascii_case("true")
                || text.eq_ignore_ascii_case("false")
                || text.eq_ignore_ascii_case("on")
                || text.eq_ignore_ascii_case("off") =>
        {
            Ok(text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("on"))
        }
        Value::CharArray(chars) => {
            let text: String = chars.data.iter().collect();
            if text.eq_ignore_ascii_case("true")
                || text.eq_ignore_ascii_case("false")
                || text.eq_ignore_ascii_case("on")
                || text.eq_ignore_ascii_case("off")
            {
                Ok(text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("on"))
            } else {
                Err(invalid(format!("lassoglm: {label} must be logical scalar")))
            }
        }
        _ => Err(invalid(format!("lassoglm: {label} must be logical scalar"))),
    }
}

fn positive_usize(value: &Value, label: &str) -> BuiltinResult<usize> {
    let raw = scalar_f64(value, label)?;
    if raw < 1.0 || raw.fract() != 0.0 || raw > usize::MAX as f64 {
        return Err(invalid(format!(
            "lassoglm: {label} must be a positive integer scalar"
        )));
    }
    Ok(raw as usize)
}

fn bounded_usize(value: &Value, label: &str, max: usize) -> BuiltinResult<usize> {
    let parsed = positive_usize(value, label)?;
    if parsed > max {
        return Err(invalid(format!(
            "lassoglm: {label} must be no greater than {max}"
        )));
    }
    Ok(parsed)
}

fn numeric_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = match value {
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) {
                return Err(invalid(format!(
                    "lassoglm: {label} must be a numeric vector"
                )));
            }
            tensor::tensor_values_f64(tensor)
        }
        Value::LogicalArray(array) => {
            let shape = tensor::default_shape_for(&array.shape, array.data.len());
            if !is_vector_shape(&shape) {
                return Err(invalid(format!(
                    "lassoglm: {label} must be a numeric vector"
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
            return Err(invalid(format!(
                "lassoglm: {label} must be a numeric vector, got {other:?}"
            )))
        }
    };
    if data.is_empty() || data.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "lassoglm: {label} must contain finite values"
        )));
    }
    Ok(data)
}

fn nonnegative_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = numeric_vector(value, label)?;
    if data.iter().any(|value| *value < 0.0) {
        return Err(invalid(format!(
            "lassoglm: {label} values must be nonnegative"
        )));
    }
    Ok(data)
}

fn positive_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = numeric_vector(value, label)?;
    if data.iter().any(|value| *value <= 0.0) {
        return Err(invalid(format!(
            "lassoglm: {label} values must be positive"
        )));
    }
    Ok(data)
}

fn nonnegative_weight_vector(value: &Value, label: &str) -> BuiltinResult<Vec<f64>> {
    let data = numeric_vector(value, label)?;
    if data.iter().any(|value| *value < 0.0) || data.iter().all(|value| *value == 0.0) {
        return Err(invalid(format!(
            "lassoglm: {label} must contain nonnegative values with positive total weight"
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
                Err(invalid(
                    "lassoglm: CV supports 'resubstitution' or a positive integer",
                ))
            }
        }
        _ => {
            let k = positive_usize(value, "CV")?;
            if k < 2 {
                return Err(invalid("lassoglm: CV fold count must be at least 2"));
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
        other => Err(invalid(format!(
            "lassoglm: {label} must be a string array or cell array of character vectors, got {other:?}"
        ))),
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|dim| **dim > 1).count() <= 1
}

fn prepare_data(
    x: &Tensor,
    response: (Vec<f64>, Vec<f64>),
    options: &Options,
) -> BuiltinResult<PreparedData> {
    let rows = x.rows();
    let cols = x.cols();
    let mut weights = options.weights.clone().unwrap_or_else(|| vec![1.0; rows]);
    let trials = if response.1.is_empty() {
        vec![1.0; rows]
    } else {
        response.1
    };
    let offset = options.offset.clone().unwrap_or_else(|| vec![0.0; rows]);
    if offset.iter().any(|value| !value.is_finite()) {
        return Err(invalid("lassoglm: Offset must contain finite values"));
    }
    for (weight, trial) in weights.iter_mut().zip(trials.iter()) {
        *weight *= *trial;
    }
    let weight_sum: f64 = weights.iter().sum();
    if !(weight_sum > 0.0 && weight_sum.is_finite()) {
        return Err(invalid("lassoglm: Weights must have a positive finite sum"));
    }
    for weight in &mut weights {
        *weight /= weight_sum;
    }

    let mut x_means = vec![0.0; cols];
    let mut x_scales = vec![1.0; cols];
    let mut x_work = vec![0.0; x.data.len()];
    for col in 0..cols {
        let mean = if options.intercept {
            weighted_column_mean(x, &weights, col)
        } else {
            0.0
        };
        let mut scale = 1.0;
        if options.intercept && options.standardize {
            let variance = (0..rows)
                .map(|row| {
                    let centered = x.get2(row, col).unwrap_or(0.0) - mean;
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
            x_work[idx] = (x.data[idx] - mean) / scale;
        }
    }

    Ok(PreparedData {
        x_original: x.data.clone(),
        x_work,
        y: response.0,
        trials,
        weights,
        offset,
        x_means,
        x_scales,
        rows,
        cols,
    })
}

fn weighted_column_mean(x: &Tensor, weights: &[f64], col: usize) -> f64 {
    (0..x.rows())
        .map(|row| x.get2(row, col).unwrap_or(0.0) * weights[row])
        .sum()
}

fn default_lambda_sequence(
    data: &PreparedData,
    distribution: Distribution,
    options: &Options,
) -> BuiltinResult<Vec<f64>> {
    let intercept = initial_intercept(data, distribution, options.intercept);
    let beta = vec![0.0; data.cols];
    let gradient = gradient_at(data, distribution, &beta, intercept)?;
    let lambda_max =
        gradient.iter().map(|value| value.abs()).fold(0.0, f64::max) / options.alpha.max(EPS);
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
        return Err(invalid("lassoglm: Lambda must not be empty"));
    }
    let mut values = lambda.to_vec();
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup_by(|a, b| (*a - *b).abs() <= EPS);
    Ok(values)
}

fn fit_path(
    data: &PreparedData,
    distribution: Distribution,
    lambdas_ascending: &[f64],
    options: &Options,
) -> BuiltinResult<FitPath> {
    let mut descending = lambdas_ascending.to_vec();
    descending.sort_by(|a, b| b.total_cmp(a));
    let mut beta_scaled = vec![0.0; data.cols];
    let mut intercept_scaled = initial_intercept(data, distribution, options.intercept);
    if !options.intercept {
        intercept_scaled = 0.0;
    }
    let mut entries = Vec::with_capacity(descending.len());
    for lambda in descending {
        let iterations = fit_single_lambda(
            data,
            distribution,
            &mut beta_scaled,
            &mut intercept_scaled,
            lambda,
            options,
        )?;
        let beta_original = unstandardize_coefficients(data, &beta_scaled);
        let intercept = if options.intercept {
            intercept_scaled
                - beta_original
                    .iter()
                    .zip(data.x_means.iter())
                    .map(|(beta, mean)| beta * mean)
                    .sum::<f64>()
        } else {
            0.0
        };
        let deviance = model_deviance(data, distribution, &beta_original, intercept)?;
        let df = beta_original
            .iter()
            .filter(|value| value.abs() > 1.0e-8)
            .count() as f64;
        entries.push((
            lambda,
            beta_original,
            intercept,
            deviance,
            df,
            iterations as f64,
        ));
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(FitPath {
        lambdas: entries.iter().map(|entry| entry.0).collect(),
        coefficients: entries.iter().map(|entry| entry.1.clone()).collect(),
        intercepts: entries.iter().map(|entry| entry.2).collect(),
        deviance: entries.iter().map(|entry| entry.3).collect(),
        df: entries.iter().map(|entry| entry.4).collect(),
        iterations: entries.iter().map(|entry| entry.5).collect(),
    })
}

fn fit_single_lambda(
    data: &PreparedData,
    distribution: Distribution,
    beta: &mut [f64],
    intercept: &mut f64,
    lambda: f64,
    options: &Options,
) -> BuiltinResult<usize> {
    let mut objective = objective_value(data, distribution, beta, *intercept, lambda, options)?;
    for iter in 1..=options.max_iter {
        let (grad_beta, grad_intercept, lip) =
            gradient_and_lipschitz(data, distribution, beta, *intercept)?;
        let mut step = 1.0 / lip.max(EPS);
        let old_beta = beta.to_vec();
        let old_intercept = *intercept;
        let mut accepted = false;
        let mut new_objective = objective;
        for _ in 0..30 {
            let candidate_intercept = if options.intercept {
                old_intercept - step * grad_intercept
            } else {
                0.0
            };
            let mut candidate_beta = vec![0.0; beta.len()];
            let ridge = lambda * (1.0 - options.alpha);
            for col in 0..beta.len() {
                let raw = old_beta[col] - step * grad_beta[col];
                candidate_beta[col] =
                    soft_threshold(raw, step * lambda * options.alpha) / (1.0 + step * ridge);
            }
            new_objective = objective_value(
                data,
                distribution,
                &candidate_beta,
                candidate_intercept,
                lambda,
                options,
            )?;
            if new_objective.is_finite() && new_objective <= objective + 1.0e-10 {
                beta.copy_from_slice(&candidate_beta);
                *intercept = candidate_intercept;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            return Err(convergence(
                "lassoglm: proximal-gradient step could not decrease the objective",
            ));
        }
        let max_delta = beta
            .iter()
            .zip(old_beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold((*intercept - old_intercept).abs(), f64::max);
        let max_param = beta
            .iter()
            .map(|value| value.abs())
            .fold(intercept.abs(), f64::max)
            .max(1.0);
        objective = new_objective;
        if max_delta <= options.rel_tol * max_param {
            return Ok(iter);
        }
    }
    Ok(options.max_iter)
}

fn gradient_at(
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
) -> BuiltinResult<Vec<f64>> {
    let (grad, _, _) = gradient_and_lipschitz(data, distribution, beta, intercept)?;
    Ok(grad)
}

fn gradient_and_lipschitz(
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
) -> BuiltinResult<(Vec<f64>, f64, f64)> {
    let mut grad = vec![0.0; data.cols];
    let mut grad_intercept = 0.0;
    let mut max_weight = 0.0_f64;
    for row in 0..data.rows {
        let eta = linear_predictor(data, beta, intercept, row, true);
        let (_, residual, curvature) = mean_residual_curvature(distribution, data.y[row], eta)?;
        let weight = data.weights[row];
        for (col, grad_value) in grad.iter_mut().enumerate() {
            *grad_value += weight * residual * x_at(data, row, col);
        }
        grad_intercept += weight * residual;
        max_weight = max_weight.max(weight * curvature);
    }
    let norm_bound = (0..data.rows)
        .map(|row| {
            (0..data.cols)
                .map(|col| x_at(data, row, col).abs())
                .sum::<f64>()
                + 1.0
        })
        .map(|row_norm| row_norm * row_norm)
        .fold(1.0, f64::max);
    let lip = max_weight.max(EPS) * norm_bound;
    Ok((grad, grad_intercept, lip))
}

fn mean_residual_curvature(
    distribution: Distribution,
    y: f64,
    eta: f64,
) -> BuiltinResult<(f64, f64, f64)> {
    match distribution {
        Distribution::Normal => Ok((eta, eta - y, 1.0)),
        Distribution::Binomial => {
            let p = logistic(eta);
            Ok((p, p - y, (p * (1.0 - p)).max(1.0e-6)))
        }
        Distribution::Poisson => {
            let mu = safe_exp(eta)?;
            Ok((mu, mu - y, mu.max(1.0e-6)))
        }
    }
}

fn objective_value(
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
    lambda: f64,
    options: &Options,
) -> BuiltinResult<f64> {
    let mut loss = 0.0;
    for row in 0..data.rows {
        let eta = linear_predictor(data, beta, intercept, row, true);
        loss += data.weights[row] * unit_negative_log_likelihood(distribution, data.y[row], eta)?;
    }
    let l1 = beta.iter().map(|value| value.abs()).sum::<f64>();
    let l2 = beta.iter().map(|value| value * value).sum::<f64>();
    Ok(loss + lambda * (options.alpha * l1 + 0.5 * (1.0 - options.alpha) * l2))
}

fn unit_negative_log_likelihood(
    distribution: Distribution,
    y: f64,
    eta: f64,
) -> BuiltinResult<f64> {
    match distribution {
        Distribution::Normal => {
            let err = y - eta;
            Ok(0.5 * err * err)
        }
        Distribution::Binomial => Ok(log1pexp(eta) - y * eta),
        Distribution::Poisson => Ok(safe_exp(eta)? - y * eta),
    }
}

fn initial_intercept(data: &PreparedData, distribution: Distribution, intercept: bool) -> f64 {
    if !intercept {
        return 0.0;
    }
    let mean = data
        .y
        .iter()
        .zip(data.weights.iter())
        .map(|(y, w)| y * w)
        .sum::<f64>();
    match distribution {
        Distribution::Normal => mean,
        Distribution::Binomial => {
            let p = mean.clamp(1.0e-6, 1.0 - 1.0e-6);
            (p / (1.0 - p)).ln()
        }
        Distribution::Poisson => mean.max(1.0e-6).ln(),
    }
}

fn model_deviance(
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
) -> BuiltinResult<f64> {
    let mut dev = 0.0;
    for row in 0..data.rows {
        let eta = data.offset[row]
            + intercept
            + (0..data.cols)
                .map(|col| data.x_original[row + col * data.rows] * beta[col])
                .sum::<f64>();
        let contribution = deviance_contribution(distribution, data.y[row], eta)?;
        dev += data.weights[row] * contribution;
    }
    Ok(dev)
}

fn deviance_contribution(distribution: Distribution, y: f64, eta: f64) -> BuiltinResult<f64> {
    match distribution {
        Distribution::Normal => {
            let err = y - eta;
            Ok(err * err)
        }
        Distribution::Binomial => {
            let p = logistic(eta).clamp(EPS, 1.0 - EPS);
            let y = y.clamp(0.0, 1.0);
            Ok(binomial_deviance(y, p))
        }
        Distribution::Poisson => {
            let mu = safe_exp(eta)?.max(EPS);
            Ok(2.0 * (poisson_term(y, y.max(EPS)) - poisson_term(y, mu)))
        }
    }
}

fn binomial_deviance(y: f64, p: f64) -> f64 {
    if y <= EPS {
        2.0 * (1.0 / (1.0 - p).max(EPS)).ln()
    } else if y >= 1.0 - EPS {
        2.0 * (1.0 / p.max(EPS)).ln()
    } else {
        2.0 * (y * (y / p.max(EPS)).ln() + (1.0 - y) * ((1.0 - y) / (1.0 - p).max(EPS)).ln())
    }
}

fn poisson_term(y: f64, mu: f64) -> f64 {
    if y <= EPS {
        -y + mu
    } else {
        y * (y / mu).ln() - y + mu
    }
}

fn linear_predictor(
    data: &PreparedData,
    beta: &[f64],
    intercept: f64,
    row: usize,
    standardized: bool,
) -> f64 {
    let base = data.offset[row] + intercept;
    if standardized {
        base + (0..data.cols)
            .map(|col| x_at(data, row, col) * beta[col])
            .sum::<f64>()
    } else {
        base + (0..data.cols)
            .map(|col| data.x_original[row + col * data.rows] * beta[col])
            .sum::<f64>()
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

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

fn safe_exp(value: f64) -> BuiltinResult<f64> {
    if value > 700.0 {
        return Err(convergence("lassoglm: exponential mean overflowed"));
    }
    Ok(value.exp())
}

fn log1pexp(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

fn cross_validate(
    x: &Tensor,
    data: &PreparedData,
    distribution: Distribution,
    lambdas: &[f64],
    options: &Options,
    k: usize,
) -> BuiltinResult<CvInfo> {
    if k > data.rows {
        return Err(invalid(
            "lassoglm: CV fold count cannot exceed the number of observations",
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
        let train_response = (
            train_rows
                .iter()
                .map(|row| data.y[*row])
                .collect::<Vec<_>>(),
            train_rows
                .iter()
                .map(|row| data.trials[*row])
                .collect::<Vec<_>>(),
        );
        let mut fold_options = options.clone();
        fold_options.cv = CvSpec::Resubstitution;
        fold_options.lambda = Some(lambdas.to_vec());
        if let Some(weights) = &options.weights {
            fold_options.weights = Some(train_rows.iter().map(|row| weights[*row]).collect());
        }
        if let Some(offset) = &options.offset {
            fold_options.offset = Some(train_rows.iter().map(|row| offset[*row]).collect());
        }
        if let Some(binomial_size) = &options.binomial_size {
            fold_options.binomial_size =
                Some(train_rows.iter().map(|row| binomial_size[*row]).collect());
        }
        let prepared = prepare_data(&train_x, train_response, &fold_options)?;
        let path = fit_path(&prepared, distribution, lambdas, &fold_options)?;
        for (lambda_idx, beta) in path.coefficients.iter().enumerate() {
            let intercept = path.intercepts[lambda_idx];
            fold_errors[fold][lambda_idx] =
                validation_deviance(x, data, distribution, beta, intercept, &test_rows)?;
        }
    }
    let mut deviance = vec![0.0; lambdas.len()];
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
        deviance[lambda_idx] = mean;
        se[lambda_idx] = variance.sqrt() / (k as f64).sqrt();
    }
    let index_min_deviance = deviance
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let threshold = deviance[index_min_deviance] + se[index_min_deviance];
    let index_1se = deviance
        .iter()
        .enumerate()
        .filter(|(_, value)| **value <= threshold)
        .max_by(|(left, _), (right, _)| lambdas[*left].total_cmp(&lambdas[*right]))
        .map(|(idx, _)| idx)
        .unwrap_or(index_min_deviance);
    Ok(CvInfo {
        deviance,
        se,
        lambda_min_deviance: lambdas[index_min_deviance],
        lambda_1se: lambdas[index_1se],
        index_min_deviance: index_min_deviance + 1,
        index_1se: index_1se + 1,
    })
}

fn subset_rows_tensor(x: &Tensor, rows: &[usize]) -> BuiltinResult<Tensor> {
    let mut data = Vec::with_capacity(rows.len() * x.cols());
    for col in 0..x.cols() {
        for row in rows {
            data.push(
                x.get2(*row, col)
                    .map_err(|err| invalid(format!("lassoglm: {err}")))?,
            );
        }
    }
    Tensor::new(data, vec![rows.len(), x.cols()]).map_err(|err| invalid(format!("lassoglm: {err}")))
}

fn validation_deviance(
    x: &Tensor,
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
    rows: &[usize],
) -> BuiltinResult<f64> {
    if rows.is_empty() {
        return Ok(f64::NAN);
    }
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for row in rows {
        let eta = data.offset[*row]
            + intercept
            + (0..x.cols())
                .map(|col| x.get2(*row, col).unwrap_or(0.0) * beta[col])
                .sum::<f64>();
        sum += data.weights[*row] * deviance_contribution(distribution, data.y[*row], eta)?;
        weight_sum += data.weights[*row];
    }
    if weight_sum <= EPS {
        return Ok(f64::NAN);
    }
    Ok(sum / weight_sum)
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
        .map_err(|err| internal(format!("lassoglm: {err}")))
}

fn row_tensor(values: &[f64]) -> BuiltinResult<Value> {
    Tensor::new(values.to_vec(), vec![1, values.len()])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("lassoglm: {err}")))
}

fn scalar_tensor(value: f64) -> BuiltinResult<Value> {
    Tensor::new(vec![value], vec![1, 1])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("lassoglm: {err}")))
}

fn logical_scalar(value: bool) -> BuiltinResult<Value> {
    LogicalArray::new(vec![if value { 1 } else { 0 }], vec![1, 1])
        .map(Value::LogicalArray)
        .map_err(|err| internal(format!("lassoglm: {err}")))
}

fn fit_info_value(
    path: &FitPath,
    distribution: Distribution,
    options: &Options,
    cv: Option<CvInfo>,
) -> BuiltinResult<Value> {
    let mut st = StructValue::new();
    st.insert("Intercept", row_tensor(&path.intercepts)?);
    st.insert("Lambda", row_tensor(&path.lambdas)?);
    st.insert("Alpha", Value::Num(options.alpha));
    st.insert(
        "Deviance",
        row_tensor(
            cv.as_ref()
                .map(|info| info.deviance.as_slice())
                .unwrap_or(&path.deviance),
        )?,
    );
    st.insert("DF", row_tensor(&path.df)?);
    st.insert("Iterations", row_tensor(&path.iterations)?);
    st.insert(
        "Distribution",
        Value::String(distribution.as_str().to_string()),
    );
    st.insert("PredictorNames", predictor_names_value(options)?);
    st.insert("UseCovariance", logical_scalar(false)?);
    if let Some(cv) = cv {
        st.insert("SE", row_tensor(&cv.se)?);
        st.insert("LambdaMinDeviance", scalar_tensor(cv.lambda_min_deviance)?);
        st.insert("Lambda1SE", scalar_tensor(cv.lambda_1se)?);
        st.insert("IndexMinDeviance", Value::Num(cv.index_min_deviance as f64));
        st.insert("Index1SE", Value::Num(cv.index_1se as f64));
    }
    Ok(Value::Struct(st))
}

fn predictor_names_value(options: &Options) -> BuiltinResult<Value> {
    let names = options.predictor_names.as_deref().unwrap_or(&[]);
    let data = names
        .iter()
        .map(|name| Value::CharArray(CharArray::new_row(name)))
        .collect::<Vec<_>>();
    CellArray::new(data, 1, names.len())
        .map(Value::Cell)
        .map_err(|err| internal(format!("lassoglm: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, StringArray};

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let mut tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
        tensor.data.fill(f64::NAN);
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
    fn normal_explicit_lambda_matches_linear_trend() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(lassoglm_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5, 1]),
            tensor(vec![1.0, 3.0, 5.0, 7.0, 9.0], vec![5, 1]),
            Value::CharArray(CharArray::new_row("normal")),
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.0], vec![1, 1]),
                Value::CharArray(CharArray::new_row("Standardize")),
                Value::Bool(false),
            ],
        ))
        .unwrap();
        let (b, fit_info) = output_pair(out);
        let Value::Tensor(coeffs) = b else {
            panic!("expected B tensor");
        };
        assert_eq!(coeffs.shape, vec![1, 1]);
        assert!((coeffs.data[0] - 2.0).abs() < 1.0e-3);
        let Value::Struct(info) = fit_info else {
            panic!("expected FitInfo");
        };
        let intercept = row_field(&info, "Intercept");
        assert!((intercept.data[0] - 1.0).abs() < 1.0e-3);
        assert!(info.fields.contains_key("Deviance"));
        assert!(info.fields.contains_key("Distribution"));
    }

    #[test]
    fn binomial_and_poisson_paths_return_fitinfo() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(lassoglm_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![6, 1]),
            tensor(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6, 1]),
            Value::String("binomial".to_string()),
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.0, 0.1], vec![1, 2]),
                Value::CharArray(CharArray::new_row("CV")),
                Value::Num(3.0),
                Value::CharArray(CharArray::new_row("MaxIter")),
                Value::Num(500.0),
            ],
        ))
        .unwrap();
        let (b, fit_info) = output_pair(out);
        let Value::Tensor(coeffs) = b else {
            panic!("expected B tensor");
        };
        assert_eq!(coeffs.shape, vec![1, 2]);
        let Value::Struct(info) = fit_info else {
            panic!("expected FitInfo");
        };
        assert!(info.fields.contains_key("SE"));
        assert!(info.fields.contains_key("LambdaMinDeviance"));

        let out = block_on(lassoglm_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
            tensor(vec![1.0, 2.0, 4.0, 7.0], vec![4, 1]),
            Value::String("poisson".to_string()),
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.0], vec![1, 1]),
                Value::CharArray(CharArray::new_row("Options")),
                {
                    let mut st = StructValue::new();
                    st.insert("MaxIter", Value::Num(400.0));
                    st.insert("TolX", Value::Num(1.0e-6));
                    Value::Struct(st)
                },
            ],
        ))
        .unwrap();
        let (b, _) = output_pair(out);
        let Value::Tensor(coeffs) = b else {
            panic!("expected B tensor");
        };
        assert_eq!(coeffs.shape, vec![1, 1]);
        assert!(coeffs.data[0].is_finite());
    }

    #[test]
    fn binomial_size_and_single_row_count_response_are_supported() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(lassoglm_builtin(
            tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
            tensor(vec![0.2, 0.5, 0.8], vec![3, 1]),
            Value::String("binomial".to_string()),
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.0, 0.2], vec![1, 2]),
                Value::CharArray(CharArray::new_row("BinomialSize")),
                tensor(vec![10.0, 12.0, 14.0], vec![3, 1]),
                Value::CharArray(CharArray::new_row("Weights")),
                tensor(vec![1.0, 2.0, 1.0], vec![3, 1]),
                Value::CharArray(CharArray::new_row("CV")),
                Value::Num(3.0),
            ],
        ))
        .unwrap();
        let (b, fit_info) = output_pair(out);
        let Value::Tensor(coeffs) = b else {
            panic!("expected B tensor");
        };
        assert_eq!(coeffs.shape, vec![1, 2]);
        let Value::Struct(info) = fit_info else {
            panic!("expected FitInfo");
        };
        let deviance = row_field(&info, "Deviance");
        assert_eq!(deviance.data.len(), 2);
        assert!(deviance.data.iter().all(|value| value.is_finite()));

        let out = block_on(lassoglm_builtin(
            tensor(vec![1.0], vec![1, 1]),
            tensor(vec![8.0, 10.0], vec![1, 2]),
            Value::String("binomial".to_string()),
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                tensor(vec![0.1], vec![1, 1]),
            ],
        ))
        .unwrap();
        let (b, _) = output_pair(out);
        let Value::Tensor(coeffs) = b else {
            panic!("expected B tensor");
        };
        assert_eq!(coeffs.shape, vec![1, 1]);
    }

    #[test]
    fn rejects_unsupported_link_and_excessive_resource_options() {
        let err = block_on(lassoglm_builtin(
            tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
            tensor(vec![0.0, 1.0, 1.0], vec![3, 1]),
            Value::String("binomial".to_string()),
            vec![
                Value::CharArray(CharArray::new_row("Link")),
                Value::CharArray(CharArray::new_row("probit")),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lassoglm:InvalidArgument"));

        let err = block_on(lassoglm_builtin(
            tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
            tensor(vec![0.0, 1.0, 1.0], vec![3, 1]),
            Value::String("binomial".to_string()),
            vec![
                Value::CharArray(CharArray::new_row("NumLambda")),
                Value::Num((MAX_NUM_LAMBDA + 1) as f64),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lassoglm:InvalidArgument"));
    }

    #[test]
    fn validates_binomial_counts_and_predictor_names() {
        let err = block_on(lassoglm_builtin(
            tensor(vec![0.0, 1.0], vec![2, 1]),
            tensor(vec![2.0, 1.0, 1.0, 1.0], vec![2, 2]),
            Value::String("binomial".to_string()),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lassoglm:InvalidArgument"));

        let err = block_on(lassoglm_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
            tensor(vec![1.0, 2.0], vec![2, 1]),
            Value::String("normal".to_string()),
            vec![
                Value::CharArray(CharArray::new_row("PredictorNames")),
                Value::StringArray(StringArray::new(vec!["x".into()], vec![1, 1]).unwrap()),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:lassoglm:InvalidArgument"));
    }

    #[test]
    fn lassoglm_reads_typed_integer_storage_exactly() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(lassoglm_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3, 4]), 5, 1),
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7, 9]), 5, 1),
            Value::String("normal".to_string()),
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                poisoned_int_tensor(IntegerStorage::U8(vec![0]), 1, 1),
                Value::CharArray(CharArray::new_row("Weights")),
                poisoned_int_tensor(IntegerStorage::U16(vec![1, 1, 2, 2, 1]), 5, 1),
                Value::CharArray(CharArray::new_row("MaxIter")),
                poisoned_int_tensor(IntegerStorage::U16(vec![300]), 1, 1),
                Value::CharArray(CharArray::new_row("Standardize")),
                Value::Bool(false),
            ],
        ))
        .unwrap();
        let (b, fit_info) = output_pair(out);
        let Value::Tensor(coeffs) = b else {
            panic!("expected B tensor");
        };
        assert_eq!(coeffs.shape, vec![1, 1]);
        assert!((coeffs.data[0] - 2.0).abs() < 1.0e-3);
        let Value::Struct(info) = fit_info else {
            panic!("expected FitInfo");
        };
        assert!(info.fields.contains_key("Lambda"));
    }

    #[test]
    fn lassoglm_reads_typed_integer_binomial_counts_exactly() {
        let out = block_on(lassoglm_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1),
            poisoned_int_tensor(IntegerStorage::U16(vec![0, 1, 3, 5, 5, 4, 3, 6]), 4, 2),
            Value::String("binomial".to_string()),
            vec![
                Value::CharArray(CharArray::new_row("Lambda")),
                poisoned_int_tensor(IntegerStorage::U8(vec![0]), 1, 1),
                Value::CharArray(CharArray::new_row("CV")),
                poisoned_int_tensor(IntegerStorage::U8(vec![2]), 1, 1),
            ],
        ))
        .unwrap();
        let Value::Tensor(coeffs) = out else {
            panic!("expected B tensor");
        };
        assert_eq!(coeffs.shape, vec![1, 1]);
        assert!(coeffs.data[0].is_finite());
    }
}
