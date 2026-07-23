//! Linear classification model compatibility surface.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, LogicalArray, ObjectInstance, ResolveContext, StringArray, StructValue, Tensor,
    Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::builtins::table::{
    is_tabular_object, table_variable_names_from_object, table_variables,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

pub(crate) const CLASSIFICATION_LINEAR_CLASS: &str = "ClassificationLinear";

const FITCLINEAR_NAME: &str = "fitclinear";
const PREDICT_NAME: &str = "predict";
const DEFAULT_ITERATIONS: usize = 500;
const EPS: f64 = 1.0e-12;

const OUTPUT_MDL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Mdl",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "ClassificationLinear object containing coefficients, class names, and fit metadata.",
}];

const OUTPUT_MDL_FITINFO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Mdl",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "ClassificationLinear object.",
    },
    BuiltinParamDescriptor {
        name: "FitInfo",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Training diagnostics for each Lambda value.",
    },
];

const PARAM_TBL_OR_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tblOrX",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input table or predictor matrix.",
};

const PARAM_Y_OR_RESPONSE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "yOrResponse",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Response vector, response variable name, or table formula.",
};

const PARAM_REST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Learner, Lambda, Regularization, Solver, ObservationsIn, FitBias, ClassNames, PredictorNames, ResponseName, Weights, Prior, and ScoreTransform.",
};

const FITCLINEAR_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_TBL_OR_X, PARAM_Y_OR_RESPONSE];
const FITCLINEAR_INPUTS_FULL: [BuiltinParamDescriptor; 3] =
    [PARAM_TBL_OR_X, PARAM_Y_OR_RESPONSE, PARAM_REST];

const FITCLINEAR_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(X, Y)",
        inputs: &FITCLINEAR_INPUTS_X_Y,
        outputs: &OUTPUT_MDL,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(Tbl, ResponseVarName)",
        inputs: &FITCLINEAR_INPUTS_X_Y,
        outputs: &OUTPUT_MDL,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(Tbl, formula)",
        inputs: &FITCLINEAR_INPUTS_X_Y,
        outputs: &OUTPUT_MDL,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(Tbl, Y)",
        inputs: &FITCLINEAR_INPUTS_X_Y,
        outputs: &OUTPUT_MDL,
    },
    BuiltinSignatureDescriptor {
        label: "Mdl = fitclinear(___, Name, Value)",
        inputs: &FITCLINEAR_INPUTS_FULL,
        outputs: &OUTPUT_MDL,
    },
    BuiltinSignatureDescriptor {
        label: "[Mdl, FitInfo] = fitclinear(___)",
        inputs: &FITCLINEAR_INPUTS_FULL,
        outputs: &OUTPUT_MDL_FITINFO,
    },
];

const ERROR_FITCLINEAR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITCLINEAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:fitclinear:InvalidArgument"),
    when: "Inputs, binary response labels, dimensions, or name-value options are malformed or unsupported.",
    message: "fitclinear: invalid argument",
};

const ERROR_FITCLINEAR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FITCLINEAR.INTERNAL",
    identifier: Some("RunMat:fitclinear:Internal"),
    when: "RunMat cannot construct the ClassificationLinear result.",
    message: "fitclinear: internal error",
};

const ERROR_PREDICT_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PREDICT.INVALID_ARGUMENT",
    identifier: Some("RunMat:predict:InvalidArgument"),
    when: "Linear-classifier prediction inputs are malformed or incompatible with the model.",
    message: "predict: invalid argument",
};

const ERROR_PREDICT_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PREDICT.INTERNAL",
    identifier: Some("RunMat:predict:Internal"),
    when: "Linear-classifier prediction cannot construct its output.",
    message: "predict: internal error",
};

const FITCLINEAR_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ERROR_FITCLINEAR_INVALID_ARGUMENT, ERROR_FITCLINEAR_INTERNAL];

pub const FITCLINEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FITCLINEAR_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FITCLINEAR_ERRORS,
};

fn fitclinear_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn fitclinear_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(FITCLINEAR_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fitclinear_invalid(message: impl Into<String>) -> RuntimeError {
    fitclinear_error(message, &ERROR_FITCLINEAR_INVALID_ARGUMENT)
}

fn fitclinear_internal(message: impl Into<String>) -> RuntimeError {
    fitclinear_error(message, &ERROR_FITCLINEAR_INTERNAL)
}

fn predict_invalid(message: impl Into<String>) -> RuntimeError {
    predict_error(message, &ERROR_PREDICT_INVALID_ARGUMENT)
}

fn predict_internal(message: impl Into<String>) -> RuntimeError {
    predict_error(message, &ERROR_PREDICT_INTERNAL)
}

fn predict_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(PREDICT_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Learner {
    Svm,
    Logistic,
}

impl Learner {
    fn as_str(self) -> &'static str {
        match self {
            Learner::Svm => "svm",
            Learner::Logistic => "logistic",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Regularization {
    Ridge,
    Lasso,
}

impl Regularization {
    fn as_str(self) -> &'static str {
        match self {
            Regularization::Ridge => "ridge",
            Regularization::Lasso => "lasso",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObservationsIn {
    Rows,
    Columns,
}

impl ObservationsIn {
    fn as_str(self) -> &'static str {
        match self {
            ObservationsIn::Rows => "rows",
            ObservationsIn::Columns => "columns",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScoreTransform {
    None,
    Logit,
}

impl ScoreTransform {
    fn as_str(self) -> &'static str {
        match self {
            ScoreTransform::None => "none",
            ScoreTransform::Logit => "logit",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ClassLabel {
    Numeric(f64),
    Text(String),
    Logical(bool),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelKind {
    Numeric,
    Text,
    Logical,
}

#[derive(Clone, Debug)]
struct LabelVector {
    labels: Vec<ClassLabel>,
    kind: LabelKind,
}

#[derive(Clone, Debug)]
enum WeightSpec {
    Values(Vec<f64>),
    Variable(String),
}

#[derive(Clone, Debug)]
struct FitOptions {
    learner: Learner,
    lambdas: Option<Vec<f64>>,
    regularization: Regularization,
    solver: String,
    observations_in: ObservationsIn,
    fit_bias: bool,
    class_names: Option<Vec<ClassLabel>>,
    predictor_names: Option<Vec<String>>,
    response_name: Option<String>,
    weights: Option<WeightSpec>,
    prior: Option<Vec<f64>>,
    score_transform: Option<ScoreTransform>,
    iteration_limit: usize,
    beta0: Option<Vec<f64>>,
    bias0: Option<f64>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            learner: Learner::Svm,
            lambdas: None,
            regularization: Regularization::Ridge,
            solver: "auto".to_string(),
            observations_in: ObservationsIn::Rows,
            fit_bias: true,
            class_names: None,
            predictor_names: None,
            response_name: None,
            weights: None,
            prior: None,
            score_transform: None,
            iteration_limit: DEFAULT_ITERATIONS,
            beta0: None,
            bias0: None,
        }
    }
}

#[derive(Clone, Debug)]
struct FitSpec {
    x: Tensor,
    class_indices: Vec<usize>,
    class_names: Vec<ClassLabel>,
    class_kind: LabelKind,
    predictor_names: Vec<String>,
    response_name: String,
    weights: Vec<f64>,
    priors: Vec<f64>,
    options: FitOptions,
}

#[derive(Clone, Debug)]
struct TrainedModel {
    beta: Vec<f64>,
    bias: f64,
    objective: f64,
    iterations: usize,
    gradient_norm: f64,
}

#[derive(Clone, Debug)]
struct FitResult {
    object: ObjectInstance,
    fit_info: Value,
}

#[runtime_builtin(
    name = "fitclinear",
    category = "stats/ml",
    summary = "Fit a binary linear classification model.",
    keywords = "fitclinear,classification linear,svm,logistic regression,machine learning,statistics",
    type_resolver(fitclinear_type),
    descriptor(crate::builtins::stats::ml::classification_linear::FITCLINEAR_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::classification_linear"
)]
pub(crate) async fn fitclinear_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let first = gather(first).await?;
    let rest = gather_all(rest).await?;
    let spec = build_fit_spec(first, rest)?;
    let result = fit_models(&spec)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![Value::Object(result.object)])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![Value::Object(result.object), result.fit_info],
        )),
        None => Ok(Value::Object(result.object)),
    }
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| fitclinear_invalid(format!("fitclinear: {err}")))
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}

fn build_fit_spec(first: Value, mut rest: Vec<Value>) -> BuiltinResult<FitSpec> {
    if is_table_value(&first) {
        build_table_fit_spec(first, rest)
    } else {
        if rest.is_empty() {
            return Err(fitclinear_invalid(
                "fitclinear: matrix input requires response vector Y",
            ));
        }
        let y = rest.remove(0);
        build_matrix_fit_spec(first, y, rest)
    }
}

fn is_table_value(value: &Value) -> bool {
    matches!(value, Value::Object(object) if is_tabular_object(object))
}

fn build_matrix_fit_spec(
    x_value: Value,
    y_value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<FitSpec> {
    let mut options = FitOptions::default();
    parse_fit_options(&rest, &mut options)?;
    if let Some(WeightSpec::Variable(_)) = options.weights {
        return Err(fitclinear_invalid(
            "fitclinear: weights variable names require table input",
        ));
    }
    let x = normalize_observation_matrix(numeric_matrix_for_fit(x_value)?, &options)?;
    let labels = label_vector(y_value, "Y")?;
    if labels.labels.len() != x.rows {
        return Err(fitclinear_invalid(
            "fitclinear: length(Y) must match the number of observations in X",
        ));
    }
    let predictor_names = if let Some(names) = options.predictor_names.clone() {
        if names.len() != x.cols {
            return Err(fitclinear_invalid(format!(
                "fitclinear: PredictorNames must contain {} names",
                x.cols
            )));
        }
        names
    } else {
        (1..=x.cols).map(|idx| format!("x{idx}")).collect()
    };
    let response_name = options
        .response_name
        .clone()
        .unwrap_or_else(|| "Y".to_string());
    let weights = match &options.weights {
        Some(WeightSpec::Values(values)) => values.clone(),
        Some(WeightSpec::Variable(_)) => unreachable!("checked above"),
        None => vec![1.0; x.rows],
    };
    finalize_fit_spec(x, labels, predictor_names, response_name, weights, options)
}

fn build_table_fit_spec(first: Value, rest: Vec<Value>) -> BuiltinResult<FitSpec> {
    let object = match first {
        Value::Object(object) => object,
        _ => unreachable!("table checked"),
    };
    let variable_names = table_variable_names_from_object(&object)
        .map_err(|err| fitclinear_invalid(format!("fitclinear: {err}")))?;
    if variable_names.len() < 2 {
        return Err(fitclinear_invalid(
            "fitclinear: table input must contain at least one predictor and one response",
        ));
    }
    let variables =
        table_variables(&object).map_err(|err| fitclinear_invalid(format!("fitclinear: {err}")))?;
    let mut options = FitOptions::default();
    let mut rest_start = 0usize;
    let mut response_name = variable_names
        .last()
        .cloned()
        .ok_or_else(|| fitclinear_invalid("fitclinear: response variable missing"))?;
    let mut external_y = None;
    let mut formula_predictors = None;
    if let Some(first_arg) = rest.first() {
        if !is_name_value_option(first_arg) {
            if let Some(text) = scalar_text(first_arg) {
                if text.contains('~') {
                    let parsed = parse_formula(&text, &variable_names)?;
                    response_name = parsed.0;
                    formula_predictors = Some(parsed.1);
                } else {
                    response_name = text;
                }
            } else {
                external_y = Some(label_vector(first_arg.clone(), "Y")?);
                response_name = options
                    .response_name
                    .clone()
                    .unwrap_or_else(|| "Y".to_string());
            }
            rest_start = 1;
        }
    }
    parse_fit_options(&rest[rest_start..], &mut options)?;
    if options.observations_in == ObservationsIn::Columns {
        return Err(fitclinear_invalid(
            "fitclinear: ObservationsIn='columns' is not supported for table input",
        ));
    }
    response_name = options.response_name.clone().unwrap_or(response_name);
    let has_external_y = external_y.is_some();
    let labels = if let Some(labels) = external_y {
        labels
    } else {
        let value = variables.fields.get(&response_name).ok_or_else(|| {
            fitclinear_invalid(format!(
                "fitclinear: response variable '{response_name}' is not in the table"
            ))
        })?;
        label_vector(value.clone(), &response_name)?
    };
    let predictor_names = if let Some(names) = options.predictor_names.clone() {
        names
    } else if let Some(names) = formula_predictors {
        names
    } else {
        let weight_variable = match &options.weights {
            Some(WeightSpec::Variable(name)) => Some(name.as_str()),
            _ => None,
        };
        variable_names
            .iter()
            .filter(|name| {
                (has_external_y || **name != response_name)
                    && weight_variable != Some(name.as_str())
            })
            .cloned()
            .collect()
    };
    if predictor_names.is_empty() {
        return Err(fitclinear_invalid(
            "fitclinear: at least one predictor variable is required",
        ));
    }
    let mut columns = Vec::with_capacity(predictor_names.len());
    for name in &predictor_names {
        let value = variables.fields.get(name).ok_or_else(|| {
            fitclinear_invalid(format!(
                "fitclinear: predictor variable '{name}' is not in the table"
            ))
        })?;
        let tensor = numeric_matrix_for_fit(value.clone()).map_err(|_| {
            fitclinear_invalid(format!(
                "fitclinear: predictor variable '{name}' must be numeric"
            ))
        })?;
        columns.push(vector_values(&tensor, name)?);
    }
    let x = columns_to_tensor(columns)?;
    let weights = match &options.weights {
        Some(WeightSpec::Values(values)) => values.clone(),
        Some(WeightSpec::Variable(name)) => {
            let value = variables.fields.get(name).ok_or_else(|| {
                fitclinear_invalid(format!(
                    "fitclinear: weights variable '{name}' is not in the table"
                ))
            })?;
            numeric_vector(value, "Weights")?
        }
        None => vec![1.0; x.rows],
    };
    finalize_fit_spec(x, labels, predictor_names, response_name, weights, options)
}

fn finalize_fit_spec(
    x: Tensor,
    labels: LabelVector,
    predictor_names: Vec<String>,
    response_name: String,
    weights: Vec<f64>,
    mut options: FitOptions,
) -> BuiltinResult<FitSpec> {
    if labels.labels.len() != x.rows {
        return Err(fitclinear_invalid(
            "fitclinear: response length must match the number of observations",
        ));
    }
    if weights.len() != x.rows {
        return Err(fitclinear_invalid(
            "fitclinear: Weights length must match the number of observations",
        ));
    }
    if predictor_names.len() != x.cols {
        return Err(fitclinear_invalid(format!(
            "fitclinear: PredictorNames must contain {} names",
            x.cols
        )));
    }
    for weight in &weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(fitclinear_invalid(
                "fitclinear: weights must be nonnegative finite values",
            ));
        }
    }
    let class_names = if let Some(names) = options.class_names.clone() {
        validate_class_names(names, labels.kind)?
    } else {
        unique_labels(&labels.labels, labels.kind)
    };
    if class_names.len() != 2 {
        return Err(fitclinear_invalid(
            "fitclinear: exactly two classes are required",
        ));
    }
    let mut clean_rows = Vec::new();
    let mut clean_classes = Vec::new();
    let mut clean_weights = Vec::new();
    for row in 0..x.rows {
        let weight = weights[row];
        if weight == 0.0 || is_missing_label(&labels.labels[row]) {
            continue;
        }
        let class_index = class_names
            .iter()
            .position(|label| same_label(label, &labels.labels[row]))
            .ok_or_else(|| {
                fitclinear_invalid("fitclinear: response contains a class outside ClassNames")
            })?;
        let mut row_values = Vec::with_capacity(x.cols);
        let mut complete = true;
        for col in 0..x.cols {
            let value = x_value(&x, row, col);
            if value.is_infinite() {
                return Err(fitclinear_invalid(
                    "fitclinear: predictor values must not contain Inf",
                ));
            }
            if value.is_nan() {
                complete = false;
                break;
            }
            row_values.push(value);
        }
        if complete {
            clean_rows.push(row_values);
            clean_classes.push(class_index);
            clean_weights.push(weight);
        }
    }
    if clean_classes.is_empty() {
        return Err(fitclinear_invalid(
            "fitclinear: at least one complete weighted observation is required",
        ));
    }
    if !(clean_classes.contains(&0) && clean_classes.contains(&1)) {
        return Err(fitclinear_invalid(
            "fitclinear: both classes must be observed after removing missing rows",
        ));
    }
    let clean_x = rows_to_tensor(clean_rows)?;
    let lambdas = options
        .lambdas
        .clone()
        .unwrap_or_else(|| vec![1.0 / clean_x.rows.max(1) as f64]);
    if lambdas.is_empty() {
        return Err(fitclinear_invalid(
            "fitclinear: Lambda must contain at least one value",
        ));
    }
    for lambda in &lambdas {
        if !lambda.is_finite() || *lambda < 0.0 {
            return Err(fitclinear_invalid(
                "fitclinear: Lambda values must be nonnegative finite scalars",
            ));
        }
    }
    options.lambdas = Some(lambdas);
    if let Some(beta0) = &options.beta0 {
        if beta0.len() != clean_x.cols {
            return Err(fitclinear_invalid(format!(
                "fitclinear: Beta must contain {} coefficients",
                clean_x.cols
            )));
        }
    }
    let priors = class_priors(&clean_classes, &clean_weights, options.prior.clone())?;
    Ok(FitSpec {
        x: clean_x,
        class_indices: clean_classes,
        class_names,
        class_kind: labels.kind,
        predictor_names,
        response_name,
        weights: clean_weights,
        priors,
        options,
    })
}

fn class_priors(
    class_indices: &[usize],
    weights: &[f64],
    explicit: Option<Vec<f64>>,
) -> BuiltinResult<Vec<f64>> {
    if let Some(prior) = explicit {
        if prior.len() != 2 {
            return Err(fitclinear_invalid(
                "fitclinear: Prior must contain two class probabilities",
            ));
        }
        if prior.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(fitclinear_invalid(
                "fitclinear: Prior values must be nonnegative finite values",
            ));
        }
        let sum = prior.iter().sum::<f64>();
        if sum <= 0.0 {
            return Err(fitclinear_invalid(
                "fitclinear: Prior values must have positive total",
            ));
        }
        return Ok(prior.into_iter().map(|value| value / sum).collect());
    }
    let mut counts = [0.0, 0.0];
    for (idx, weight) in class_indices.iter().zip(weights.iter()) {
        counts[*idx] += *weight;
    }
    let total = counts[0] + counts[1];
    Ok(vec![counts[0] / total, counts[1] / total])
}

fn parse_fit_options(values: &[Value], options: &mut FitOptions) -> BuiltinResult<()> {
    if !values.len().is_multiple_of(2) {
        return Err(fitclinear_invalid(
            "fitclinear: name-value options must be paired",
        ));
    }
    let mut index = 0usize;
    while index < values.len() {
        let name = scalar_text(&values[index])
            .ok_or_else(|| fitclinear_invalid("fitclinear: option name must be text"))?;
        let value = &values[index + 1];
        match name.to_ascii_lowercase().as_str() {
            "learner" => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: Learner must be text"))?;
                options.learner = match text.to_ascii_lowercase().as_str() {
                    "svm" => Learner::Svm,
                    "logistic" => Learner::Logistic,
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported Learner '{other}'"
                        )));
                    }
                };
            }
            "lambda" => options.lambdas = Some(numeric_vector(value, "Lambda")?),
            "regularization" => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: Regularization must be text"))?;
                options.regularization = match text.to_ascii_lowercase().as_str() {
                    "ridge" => Regularization::Ridge,
                    "lasso" => Regularization::Lasso,
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported Regularization '{other}'"
                        )));
                    }
                };
            }
            "solver" => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: Solver must be text"))?;
                match text.to_ascii_lowercase().as_str() {
                    "auto" | "sgd" | "asgd" | "bfgs" | "lbfgs" | "sparsa" | "dual" => {
                        options.solver = text.to_ascii_lowercase();
                    }
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported Solver '{other}'"
                        )));
                    }
                }
            }
            "observationsin" => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: ObservationsIn must be text"))?;
                options.observations_in = match text.to_ascii_lowercase().as_str() {
                    "rows" => ObservationsIn::Rows,
                    "columns" => ObservationsIn::Columns,
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported ObservationsIn '{other}'"
                        )));
                    }
                };
            }
            "fitbias" => options.fit_bias = logical_scalar(value, "FitBias")?,
            "classnames" => {
                let labels = label_vector(value.clone(), "ClassNames")?;
                options.class_names = Some(labels.labels);
            }
            "predictornames" => {
                options.predictor_names = Some(text_list(value, "PredictorNames")?);
            }
            "responsename" => {
                options.response_name =
                    Some(scalar_text(value).ok_or_else(|| {
                        fitclinear_invalid("fitclinear: ResponseName must be text")
                    })?);
            }
            "weights" => {
                options.weights = Some(if let Some(name) = scalar_text(value) {
                    WeightSpec::Variable(name)
                } else {
                    WeightSpec::Values(numeric_vector(value, "Weights")?)
                });
            }
            "prior" => {
                if let Some(text) = scalar_text(value) {
                    match text.to_ascii_lowercase().as_str() {
                        "empirical" => options.prior = None,
                        "uniform" => options.prior = Some(vec![0.5, 0.5]),
                        other => {
                            return Err(fitclinear_invalid(format!(
                                "fitclinear: unsupported Prior '{other}'"
                            )));
                        }
                    }
                } else {
                    options.prior = Some(numeric_vector(value, "Prior")?);
                }
            }
            "scoretransform" => {
                let text = scalar_text(value)
                    .ok_or_else(|| fitclinear_invalid("fitclinear: ScoreTransform must be text"))?;
                options.score_transform = Some(match text.to_ascii_lowercase().as_str() {
                    "none" => ScoreTransform::None,
                    "logit" => ScoreTransform::Logit,
                    other => {
                        return Err(fitclinear_invalid(format!(
                            "fitclinear: unsupported ScoreTransform '{other}'"
                        )));
                    }
                });
            }
            "iterationlimit" | "passlimit" => {
                options.iteration_limit = positive_integer(value, &name)?;
            }
            "beta" => options.beta0 = Some(numeric_vector(value, "Beta")?),
            "bias" => options.bias0 = Some(numeric_scalar(value, "Bias")?),
            "batchlimit"
            | "betatolerance"
            | "deltagradienttolerance"
            | "gradienttolerance"
            | "numcheckconvergence"
            | "postfitbias"
            | "verbose" => {
                // Accepted for script compatibility. The CPU solver below is deterministic
                // batch gradient descent and records convergence diagnostics in FitInfo.
            }
            "cost"
            | "crossval"
            | "cvpartition"
            | "holdout"
            | "kfold"
            | "leaveout"
            | "optimizehyperparameters"
            | "hyperparameteroptimizationoptions" => {
                return Err(fitclinear_invalid(format!(
                    "fitclinear: option '{name}' is not supported yet"
                )));
            }
            other => {
                return Err(fitclinear_invalid(format!(
                    "fitclinear: unknown option '{other}'"
                )))
            }
        }
        index += 2;
    }
    Ok(())
}

fn parse_formula(text: &str, variables: &[String]) -> BuiltinResult<(String, Vec<String>)> {
    let Some((lhs, rhs)) = text.split_once('~') else {
        return Err(fitclinear_invalid("fitclinear: formula must contain '~'"));
    };
    let response = lhs.trim().to_string();
    if response.is_empty() {
        return Err(fitclinear_invalid("fitclinear: formula response is empty"));
    }
    let mut predictors = Vec::new();
    for term in rhs.split('+') {
        let term = term.trim();
        if term.is_empty() || term == "1" {
            continue;
        }
        if term.contains('*') || term.contains(':') || term.contains('^') {
            return Err(fitclinear_invalid(
                "fitclinear: formula interactions are not supported yet",
            ));
        }
        if !variables.iter().any(|name| name == term) {
            return Err(fitclinear_invalid(format!(
                "fitclinear: formula predictor '{term}' is not in the table"
            )));
        }
        predictors.push(term.to_string());
    }
    if predictors.is_empty() {
        return Err(fitclinear_invalid(
            "fitclinear: formula must contain at least one predictor",
        ));
    }
    Ok((response, predictors))
}

fn fit_models(spec: &FitSpec) -> BuiltinResult<FitResult> {
    let lambdas = spec
        .options
        .lambdas
        .as_ref()
        .expect("lambdas populated during finalize");
    let mut models = Vec::with_capacity(lambdas.len());
    for lambda in lambdas {
        models.push(train_for_lambda(spec, *lambda)?);
    }
    let beta_value = beta_value(&models, spec.x.cols)?;
    let bias_value = row_tensor(models.iter().map(|model| model.bias).collect())?;
    let mut object = ObjectInstance::new(CLASSIFICATION_LINEAR_CLASS.to_string());
    object
        .properties
        .insert("Beta".to_string(), beta_value.clone());
    object
        .properties
        .insert("Bias".to_string(), bias_value.clone());
    object
        .properties
        .insert("Lambda".to_string(), row_tensor(lambdas.clone())?);
    object.properties.insert(
        "Learner".to_string(),
        Value::String(spec.options.learner.as_str().to_string()),
    );
    object.properties.insert(
        "Regularization".to_string(),
        Value::String(spec.options.regularization.as_str().to_string()),
    );
    object.properties.insert(
        "Solver".to_string(),
        Value::String(spec.options.solver.clone()),
    );
    object.properties.insert(
        "ScoreTransform".to_string(),
        Value::String(score_transform(spec).as_str().to_string()),
    );
    object
        .properties
        .insert("FitBias".to_string(), Value::Bool(spec.options.fit_bias));
    object.properties.insert(
        "ResponseName".to_string(),
        Value::String(spec.response_name.clone()),
    );
    object.properties.insert(
        "PredictorNames".to_string(),
        Value::StringArray(StringArray {
            data: spec.predictor_names.clone(),
            shape: vec![1, spec.predictor_names.len()],
            rows: 1,
            cols: spec.predictor_names.len(),
        }),
    );
    object.properties.insert(
        "ClassNames".to_string(),
        labels_value(&spec.class_names, spec.class_kind)?,
    );
    object
        .properties
        .insert("Prior".to_string(), row_tensor(spec.priors.clone())?);
    object.properties.insert(
        "NumObservations".to_string(),
        Value::Num(spec.x.rows as f64),
    );
    object
        .properties
        .insert("NumPredictors".to_string(), Value::Num(spec.x.cols as f64));
    object.properties.insert(
        "__RunMatLinearClassKind".to_string(),
        Value::String(label_kind_name(spec.class_kind).to_string()),
    );
    let mut parameters = StructValue::new();
    parameters.insert(
        "Learner",
        Value::String(spec.options.learner.as_str().to_string()),
    );
    parameters.insert(
        "Regularization",
        Value::String(spec.options.regularization.as_str().to_string()),
    );
    parameters.insert("Solver", Value::String(spec.options.solver.clone()));
    parameters.insert(
        "ObservationsIn",
        Value::String(spec.options.observations_in.as_str().to_string()),
    );
    parameters.insert("FitBias", Value::Bool(spec.options.fit_bias));
    parameters.insert(
        "IterationLimit",
        Value::Num(spec.options.iteration_limit as f64),
    );
    object
        .properties
        .insert("ModelParameters".to_string(), Value::Struct(parameters));
    let fit_info = fit_info_value(spec, lambdas, &models)?;
    Ok(FitResult { object, fit_info })
}

fn train_for_lambda(spec: &FitSpec, lambda: f64) -> BuiltinResult<TrainedModel> {
    let predictors = spec.x.cols;
    let mut beta = spec
        .options
        .beta0
        .clone()
        .unwrap_or_else(|| vec![0.0; predictors]);
    let mut bias = spec.options.bias0.unwrap_or_else(|| {
        if spec.options.fit_bias {
            (spec.priors[1] / spec.priors[0]).ln() / 2.0
        } else {
            0.0
        }
    });
    let total_weight = spec.weights.iter().sum::<f64>().max(EPS);
    let base_step = if spec.options.learner == Learner::Logistic {
        0.25
    } else {
        0.15
    };
    let mut last_grad_norm = f64::INFINITY;
    for iter in 0..spec.options.iteration_limit.max(1) {
        let mut grad_beta = vec![0.0; predictors];
        let mut grad_bias = 0.0;
        for row in 0..spec.x.rows {
            let y = if spec.class_indices[row] == 1 {
                1.0
            } else {
                -1.0
            };
            let margin = dot_row(&spec.x, row, &beta) + bias;
            let coeff = match spec.options.learner {
                Learner::Logistic => {
                    let ym = (y * margin).clamp(-40.0, 40.0);
                    -y / (1.0 + ym.exp())
                }
                Learner::Svm => {
                    if y * margin < 1.0 {
                        -y
                    } else {
                        0.0
                    }
                }
            } * spec.weights[row]
                / total_weight;
            if coeff != 0.0 {
                for (col, grad) in grad_beta.iter_mut().enumerate() {
                    *grad += coeff * x_value(&spec.x, row, col);
                }
                if spec.options.fit_bias {
                    grad_bias += coeff;
                }
            }
        }
        match spec.options.regularization {
            Regularization::Ridge => {
                for (grad, coef) in grad_beta.iter_mut().zip(beta.iter()) {
                    *grad += lambda * *coef;
                }
            }
            Regularization::Lasso => {}
        }
        last_grad_norm = grad_beta
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            .hypot(grad_bias.abs());
        let step = base_step / (1.0 + lambda + 0.01 * iter as f64).sqrt();
        for (coef, grad) in beta.iter_mut().zip(grad_beta.iter()) {
            *coef -= step * *grad;
            if spec.options.regularization == Regularization::Lasso && lambda > 0.0 {
                *coef = soft_threshold(*coef, step * lambda);
            }
        }
        if spec.options.fit_bias {
            bias -= step * grad_bias;
        } else {
            bias = 0.0;
        }
        if last_grad_norm < 1.0e-7 {
            break;
        }
    }
    Ok(TrainedModel {
        objective: objective(spec, &beta, bias, lambda),
        beta,
        bias,
        iterations: spec.options.iteration_limit.max(1),
        gradient_norm: last_grad_norm,
    })
}

fn objective(spec: &FitSpec, beta: &[f64], bias: f64, lambda: f64) -> f64 {
    let total_weight = spec.weights.iter().sum::<f64>().max(EPS);
    let mut loss = 0.0;
    for row in 0..spec.x.rows {
        let y = if spec.class_indices[row] == 1 {
            1.0
        } else {
            -1.0
        };
        let margin = dot_row(&spec.x, row, beta) + bias;
        let row_loss = match spec.options.learner {
            Learner::Logistic => (1.0 + (-y * margin).clamp(-40.0, 40.0).exp()).ln(),
            Learner::Svm => (1.0 - y * margin).max(0.0),
        };
        loss += spec.weights[row] * row_loss / total_weight;
    }
    let penalty = match spec.options.regularization {
        Regularization::Ridge => 0.5 * lambda * beta.iter().map(|value| value * value).sum::<f64>(),
        Regularization::Lasso => lambda * beta.iter().map(|value| value.abs()).sum::<f64>(),
    };
    loss + penalty
}

fn fit_info_value(
    spec: &FitSpec,
    lambdas: &[f64],
    models: &[TrainedModel],
) -> BuiltinResult<Value> {
    let mut info = StructValue::new();
    info.insert("Lambda", row_tensor(lambdas.to_vec())?);
    info.insert(
        "Objective",
        row_tensor(models.iter().map(|model| model.objective).collect())?,
    );
    info.insert(
        "NumIterations",
        row_tensor(models.iter().map(|model| model.iterations as f64).collect())?,
    );
    info.insert(
        "NumPasses",
        row_tensor(models.iter().map(|model| model.iterations as f64).collect())?,
    );
    info.insert(
        "GradientNorm",
        row_tensor(models.iter().map(|model| model.gradient_norm).collect())?,
    );
    info.insert("PassLimit", Value::Num(spec.options.iteration_limit as f64));
    info.insert("Solver", Value::String(spec.options.solver.clone()));
    info.insert(
        "Learner",
        Value::String(spec.options.learner.as_str().to_string()),
    );
    info.insert(
        "Regularization",
        Value::String(spec.options.regularization.as_str().to_string()),
    );
    info.insert("TerminationCode", row_tensor(vec![1.0; models.len()])?);
    info.insert(
        "TerminationStatus",
        Value::String("Iteration limit reached or gradient tolerance satisfied".to_string()),
    );
    Ok(Value::Struct(info))
}

pub(crate) fn predict_classification_linear_object(
    object: ObjectInstance,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    let options = parse_predict_options(&rest)?;
    let predictor_names = string_array_property(&object, "PredictorNames")?;
    let x = predictors_for_prediction(xnew, &predictor_names, options.observations_in)?;
    let beta = tensor_property(&object, "Beta")?;
    if beta.shape.len() > 2 || beta.rows != predictor_names.len() {
        return Err(predict_invalid("predict: model Beta has invalid shape"));
    }
    let bias = numeric_vector_property(&object, "Bias")?;
    if bias.len() != beta.cols {
        return Err(predict_invalid(
            "predict: model Bias length does not match Beta",
        ));
    }
    let class_names = labels_from_value(
        object
            .properties
            .get("ClassNames")
            .cloned()
            .ok_or_else(|| predict_invalid("predict: model is missing ClassNames"))?,
        "ClassNames",
    )?;
    if class_names.labels.len() != 2 {
        return Err(predict_invalid(
            "predict: model must contain two class names",
        ));
    }
    let score_transform = score_transform_property(&object)?;
    let mut label_indices = Vec::with_capacity(x.rows * beta.cols);
    let mut score_data = Vec::with_capacity(x.rows * 2 * beta.cols);
    for model_idx in 0..beta.cols {
        let model_beta = (0..beta.rows)
            .map(|row| beta.data[row + model_idx * beta.rows])
            .collect::<Vec<_>>();
        for row in 0..x.rows {
            let margin = dot_row(&x, row, &model_beta) + bias[model_idx];
            label_indices.push(if margin >= 0.0 { 1 } else { 0 });
        }
        for class_col in 0..2 {
            for row in 0..x.rows {
                let margin = dot_row(&x, row, &model_beta) + bias[model_idx];
                let value = match score_transform {
                    ScoreTransform::Logit => {
                        let p = sigmoid(margin);
                        if class_col == 0 {
                            1.0 - p
                        } else {
                            p
                        }
                    }
                    ScoreTransform::None => {
                        if class_col == 0 {
                            -margin
                        } else {
                            margin
                        }
                    }
                };
                score_data.push(value);
            }
        }
    }
    let label = predicted_labels_value(
        &class_names.labels,
        class_names.kind,
        &label_indices,
        x.rows,
        beta.cols,
    )?;
    let score_shape = if beta.cols == 1 {
        vec![x.rows, 2]
    } else {
        vec![x.rows, 2, beta.cols]
    };
    let score = Value::Tensor(
        Tensor::new(score_data, score_shape)
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    Ok(vec![label, score])
}

#[derive(Clone, Copy, Debug)]
struct PredictOptions {
    observations_in: ObservationsIn,
}

fn parse_predict_options(values: &[Value]) -> BuiltinResult<PredictOptions> {
    if !values.len().is_multiple_of(2) {
        return Err(predict_invalid(
            "predict: name-value options must be paired",
        ));
    }
    let mut options = PredictOptions {
        observations_in: ObservationsIn::Rows,
    };
    let mut index = 0usize;
    while index < values.len() {
        let name = scalar_text(&values[index])
            .ok_or_else(|| predict_invalid("predict: option name must be text"))?;
        let value = &values[index + 1];
        match name.to_ascii_lowercase().as_str() {
            "observationsin" => {
                let text = scalar_text(value)
                    .ok_or_else(|| predict_invalid("predict: ObservationsIn must be text"))?;
                options.observations_in = match text.to_ascii_lowercase().as_str() {
                    "rows" => ObservationsIn::Rows,
                    "columns" => ObservationsIn::Columns,
                    other => {
                        return Err(predict_invalid(format!(
                            "predict: unsupported ObservationsIn '{other}'"
                        )));
                    }
                };
            }
            other => {
                return Err(predict_invalid(format!(
                    "predict: unknown option '{other}'"
                )));
            }
        }
        index += 2;
    }
    Ok(options)
}

fn predictors_for_prediction(
    value: Value,
    predictor_names: &[String],
    observations_in: ObservationsIn,
) -> BuiltinResult<Tensor> {
    if let Value::Object(object) = &value {
        if is_tabular_object(object) {
            if observations_in == ObservationsIn::Columns {
                return Err(predict_invalid(
                    "predict: ObservationsIn='columns' is not supported for table input",
                ));
            }
            let variables = table_variables(object)
                .map_err(|err| predict_invalid(format!("predict: {err}")))?;
            let mut columns = Vec::with_capacity(predictor_names.len());
            for name in predictor_names {
                let raw = variables.fields.get(name).ok_or_else(|| {
                    predict_invalid(format!("predict: missing predictor '{name}'"))
                })?;
                let tensor = numeric_matrix_for_predict(raw.clone()).map_err(|_| {
                    predict_invalid(format!("predict: predictor '{name}' must be numeric"))
                })?;
                columns.push(vector_values_predict(&tensor, name)?);
            }
            return columns_to_tensor_predict(columns);
        }
    }
    let tensor = normalize_prediction_matrix(numeric_matrix_for_predict(value)?, observations_in)?;
    if tensor.shape.len() > 2 || tensor.cols != predictor_names.len() {
        return Err(predict_invalid(format!(
            "predict: X must have {} predictor columns",
            predictor_names.len()
        )));
    }
    Ok(tensor)
}

fn numeric_matrix_for_fit(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::SparseTensor(sparse) => sparse
            .to_dense()
            .map_err(|err| fitclinear_invalid(format!("fitclinear: {err}"))),
        other => tensor::value_into_tensor_for(FITCLINEAR_NAME, other)
            .map_err(|err| fitclinear_invalid(format!("fitclinear: {err}"))),
    }
}

fn numeric_matrix_for_predict(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::SparseTensor(sparse) => sparse
            .to_dense()
            .map_err(|err| predict_invalid(format!("predict: {err}"))),
        other => tensor::value_into_tensor_for(PREDICT_NAME, other)
            .map_err(|err| predict_invalid(format!("predict: {err}"))),
    }
}

fn normalize_observation_matrix(x: Tensor, options: &FitOptions) -> BuiltinResult<Tensor> {
    normalize_prediction_matrix(x, options.observations_in)
        .map_err(|err| fitclinear_invalid(err.message))
}

fn normalize_prediction_matrix(
    x: Tensor,
    observations_in: ObservationsIn,
) -> BuiltinResult<Tensor> {
    if x.shape.len() > 2 {
        return Err(predict_invalid("predict: X must be a 2-D numeric matrix"));
    }
    match observations_in {
        ObservationsIn::Rows => Ok(x),
        ObservationsIn::Columns => {
            transpose_tensor(&x).map_err(|err| predict_internal(format!("predict: {err}")))
        }
    }
}

fn transpose_tensor(x: &Tensor) -> Result<Tensor, String> {
    let mut data = vec![0.0; x.rows * x.cols];
    for row in 0..x.rows {
        for col in 0..x.cols {
            data[col + row * x.cols] = x_value(x, row, col);
        }
    }
    Tensor::new(data, vec![x.cols, x.rows])
}

fn label_vector(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    labels_from_value(value, name).map_err(|err| fitclinear_invalid(err.message))
}

fn labels_from_value(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    match value {
        Value::Tensor(tensor) => Ok(LabelVector {
            labels: vector_values_predict(&tensor, name)?
                .into_iter()
                .map(ClassLabel::Numeric)
                .collect(),
            kind: LabelKind::Numeric,
        }),
        Value::Num(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Numeric(value)],
            kind: LabelKind::Numeric,
        }),
        Value::String(text) => Ok(LabelVector {
            labels: vec![ClassLabel::Text(text)],
            kind: LabelKind::Text,
        }),
        Value::CharArray(array) => Ok(LabelVector {
            labels: char_rows(&array).into_iter().map(ClassLabel::Text).collect(),
            kind: LabelKind::Text,
        }),
        Value::StringArray(array) => Ok(LabelVector {
            labels: array.data.into_iter().map(ClassLabel::Text).collect(),
            kind: LabelKind::Text,
        }),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for item in cell.data {
                labels.push(ClassLabel::Text(scalar_text(&item).ok_or_else(|| {
                    predict_invalid(format!("predict: {name} cell entries must be text"))
                })?));
            }
            Ok(LabelVector {
                labels,
                kind: LabelKind::Text,
            })
        }
        Value::Bool(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Logical(value)],
            kind: LabelKind::Logical,
        }),
        Value::LogicalArray(array) => Ok(LabelVector {
            labels: array
                .data
                .into_iter()
                .map(|value| ClassLabel::Logical(value != 0))
                .collect(),
            kind: LabelKind::Logical,
        }),
        other => Err(predict_invalid(format!(
            "predict: {name} must be numeric, logical, string, character, or cellstr labels; got {other:?}"
        ))),
    }
}

fn validate_class_names(names: Vec<ClassLabel>, kind: LabelKind) -> BuiltinResult<Vec<ClassLabel>> {
    if names.len() != 2 {
        return Err(fitclinear_invalid(
            "fitclinear: ClassNames must contain exactly two values",
        ));
    }
    let mut out = Vec::new();
    for name in names {
        if is_missing_label(&name) {
            return Err(fitclinear_invalid(
                "fitclinear: ClassNames must not contain missing values",
            ));
        }
        if label_kind(&name) != kind {
            return Err(fitclinear_invalid(
                "fitclinear: ClassNames must have the same type as the response",
            ));
        }
        if out.iter().any(|existing| same_label(existing, &name)) {
            return Err(fitclinear_invalid(
                "fitclinear: ClassNames must contain unique values",
            ));
        }
        out.push(name);
    }
    Ok(out)
}

fn unique_labels(labels: &[ClassLabel], kind: LabelKind) -> Vec<ClassLabel> {
    let mut out = Vec::new();
    for label in labels {
        if is_missing_label(label) {
            continue;
        }
        if !out.iter().any(|existing| same_label(existing, label)) {
            out.push(label.clone());
        }
    }
    match kind {
        LabelKind::Numeric => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            _ => Ordering::Equal,
        }),
        LabelKind::Text => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Text(a), ClassLabel::Text(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }),
        LabelKind::Logical => out.sort_by_key(|label| match label {
            ClassLabel::Logical(value) => *value,
            _ => false,
        }),
    }
    out
}

fn label_kind(label: &ClassLabel) -> LabelKind {
    match label {
        ClassLabel::Numeric(_) => LabelKind::Numeric,
        ClassLabel::Text(_) => LabelKind::Text,
        ClassLabel::Logical(_) => LabelKind::Logical,
    }
}

fn label_kind_name(kind: LabelKind) -> &'static str {
    match kind {
        LabelKind::Numeric => "numeric",
        LabelKind::Text => "text",
        LabelKind::Logical => "logical",
    }
}

fn same_label(left: &ClassLabel, right: &ClassLabel) -> bool {
    match (left, right) {
        (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => (a == b) || (a.is_nan() && b.is_nan()),
        (ClassLabel::Text(a), ClassLabel::Text(b)) => a == b,
        (ClassLabel::Logical(a), ClassLabel::Logical(b)) => a == b,
        _ => false,
    }
}

fn is_missing_label(label: &ClassLabel) -> bool {
    matches!(label, ClassLabel::Numeric(value) if value.is_nan())
}

fn labels_value(labels: &[ClassLabel], kind: LabelKind) -> BuiltinResult<Value> {
    predicted_labels_value(
        labels,
        kind,
        &(0..labels.len()).collect::<Vec<_>>(),
        labels.len(),
        1,
    )
}

fn predicted_labels_value(
    labels: &[ClassLabel],
    kind: LabelKind,
    indices: &[usize],
    rows: usize,
    cols: usize,
) -> BuiltinResult<Value> {
    let shape = vec![rows, cols];
    match kind {
        LabelKind::Numeric => Ok(Value::Tensor(
            Tensor::new(
                indices
                    .iter()
                    .map(|idx| match labels.get(*idx) {
                        Some(ClassLabel::Numeric(value)) => *value,
                        _ => f64::NAN,
                    })
                    .collect(),
                shape,
            )
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
        )),
        LabelKind::Text => Ok(Value::StringArray(StringArray {
            data: indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Text(value)) => value.clone(),
                    _ => String::new(),
                })
                .collect(),
            shape,
            rows,
            cols,
        })),
        LabelKind::Logical => Ok(Value::LogicalArray(
            LogicalArray::new(
                indices
                    .iter()
                    .map(|idx| match labels.get(*idx) {
                        Some(ClassLabel::Logical(value)) if *value => 1,
                        _ => 0,
                    })
                    .collect(),
                shape,
            )
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
        )),
    }
}

fn score_transform(spec: &FitSpec) -> ScoreTransform {
    spec.options
        .score_transform
        .unwrap_or(match spec.options.learner {
            Learner::Logistic => ScoreTransform::Logit,
            Learner::Svm => ScoreTransform::None,
        })
}

fn score_transform_property(object: &ObjectInstance) -> BuiltinResult<ScoreTransform> {
    match object.properties.get("ScoreTransform") {
        Some(Value::String(text)) if text.eq_ignore_ascii_case("none") => Ok(ScoreTransform::None),
        Some(Value::String(text)) if text.eq_ignore_ascii_case("logit") => {
            Ok(ScoreTransform::Logit)
        }
        _ => Err(predict_invalid(
            "predict: model has unsupported ScoreTransform",
        )),
    }
}

fn string_array_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<String>> {
    match object.properties.get(name) {
        Some(Value::StringArray(array)) => Ok(array.data.clone()),
        Some(Value::String(text)) => Ok(vec![text.clone()]),
        _ => Err(predict_invalid(format!("predict: model is missing {name}"))),
    }
}

fn tensor_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Tensor> {
    match object.properties.get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        Some(Value::Num(value)) => Tensor::new(vec![*value], vec![1, 1])
            .map_err(|err| predict_internal(format!("predict: {err}"))),
        _ => Err(predict_invalid(format!("predict: model is missing {name}"))),
    }
}

fn numeric_vector_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    match object.properties.get(name) {
        Some(Value::Tensor(tensor)) => vector_values_predict(tensor, name),
        Some(Value::Num(value)) => Ok(vec![*value]),
        _ => Err(predict_invalid(format!("predict: model is missing {name}"))),
    }
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

fn text_list(value: &Value, name: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(array) => Ok(char_rows(array)),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| {
                scalar_text(value).ok_or_else(|| {
                    fitclinear_invalid(format!("fitclinear: {name} entries must be text"))
                })
            })
            .collect(),
        _ => Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be text or a text collection"
        ))),
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

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Tensor(tensor) => vector_values(tensor, name),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|value| if *value == 0 { 0.0 } else { 1.0 })
            .collect()),
        Value::Bool(value) => Ok(vec![if *value { 1.0 } else { 0.0 }]),
        _ => Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be numeric"
        ))),
    }
}

fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    vector_values_predict(tensor, name).map_err(|err| fitclinear_invalid(err.message))
}

fn vector_values_predict(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    if tensor.shape.len() > 2 || (tensor.rows != 1 && tensor.cols != 1) {
        return Err(predict_invalid(format!("predict: {name} must be a vector")));
    }
    Ok(tensor.data.clone())
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        _ => Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be a numeric scalar"
        ))),
    }
}

fn logical_scalar(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor)
            if tensor.data.len() == 1 && (tensor.data[0] == 0.0 || tensor.data[0] == 1.0) =>
        {
            Ok(tensor.data[0] != 0.0)
        }
        _ => Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be logical scalar"
        ))),
    }
}

fn positive_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    let value = numeric_scalar(value, name)?;
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(fitclinear_invalid(format!(
            "fitclinear: {name} must be a positive integer"
        )));
    }
    Ok(value as usize)
}

fn is_name_value_option(value: &Value) -> bool {
    scalar_text(value).is_some_and(|text| {
        matches!(
            text.to_ascii_lowercase().as_str(),
            "batchlimit"
                | "beta"
                | "betatolerance"
                | "bias"
                | "classnames"
                | "cost"
                | "crossval"
                | "cvpartition"
                | "deltagradienttolerance"
                | "fitbias"
                | "gradienttolerance"
                | "holdout"
                | "hyperparameteroptimizationoptions"
                | "iterationlimit"
                | "kfold"
                | "lambda"
                | "learner"
                | "leaveout"
                | "numcheckconvergence"
                | "observationsin"
                | "optimizehyperparameters"
                | "passlimit"
                | "postfitbias"
                | "predictornames"
                | "prior"
                | "regularization"
                | "responsename"
                | "scoretransform"
                | "solver"
                | "verbose"
                | "weights"
        )
    })
}

fn columns_to_tensor(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Err(fitclinear_invalid("fitclinear: no predictor columns"));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        if column.len() != rows {
            return Err(fitclinear_invalid(
                "fitclinear: predictor variables must have the same height",
            ));
        }
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols])
        .map_err(|err| fitclinear_internal(format!("fitclinear: {err}")))
}

fn columns_to_tensor_predict(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    columns_to_tensor(columns).map_err(|err| predict_invalid(err.message))
}

fn rows_to_tensor(rows: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if rows.is_empty() {
        return Err(fitclinear_invalid("fitclinear: no complete predictor rows"));
    }
    let row_count = rows.len();
    let col_count = rows[0].len();
    let mut data = Vec::with_capacity(row_count * col_count);
    for col in 0..col_count {
        for row in &rows {
            data.push(row[col]);
        }
    }
    Tensor::new(data, vec![row_count, col_count])
        .map_err(|err| fitclinear_internal(format!("fitclinear: {err}")))
}

fn beta_value(models: &[TrainedModel], predictors: usize) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(predictors * models.len());
    for model in models {
        data.extend(model.beta.iter().copied());
    }
    Ok(Value::Tensor(
        Tensor::new(data, vec![predictors, models.len()])
            .map_err(|err| fitclinear_internal(format!("fitclinear: {err}")))?,
    ))
}

fn row_tensor(values: Vec<f64>) -> BuiltinResult<Value> {
    let cols = values.len();
    Ok(Value::Tensor(Tensor::new(values, vec![1, cols]).map_err(
        |err| fitclinear_internal(format!("fitclinear: {err}")),
    )?))
}

fn x_value(x: &Tensor, row: usize, col: usize) -> f64 {
    x.data[row + col * x.rows]
}

fn dot_row(x: &Tensor, row: usize, beta: &[f64]) -> f64 {
    beta.iter()
        .enumerate()
        .map(|(col, coef)| x_value(x, row, col) * coef)
        .sum()
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

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::table::table_from_columns;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    #[tokio::test]
    async fn fitclinear_svm_predicts_binary_numeric_labels() {
        let model = fitclinear_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
            vec![
                tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
                Value::String("Lambda".into()),
                Value::Num(0.0),
            ],
        )
        .await
        .unwrap();
        let outputs = predict_classification_linear_object(
            match model {
                Value::Object(object) => object,
                _ => panic!("expected object"),
            },
            tensor(vec![0.2, 2.8], vec![2, 1]),
            Vec::new(),
        )
        .unwrap();
        match &outputs[0] {
            Value::Tensor(labels) => assert_eq!(labels.data, vec![0.0, 1.0]),
            other => panic!("expected labels, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(score) => assert_eq!(score.shape, vec![2, 2]),
            other => panic!("expected score, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fitclinear_logistic_returns_fitinfo_and_probability_scores() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let value = fitclinear_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
            vec![
                tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
                Value::String("Learner".into()),
                Value::String("logistic".into()),
                Value::String("Lambda".into()),
                Value::Num(0.0),
            ],
        )
        .await
        .unwrap();
        let (object, info) = match value {
            Value::OutputList(mut outputs) => {
                let info = outputs.pop().unwrap();
                let object = match outputs.pop().unwrap() {
                    Value::Object(object) => object,
                    other => panic!("expected object, got {other:?}"),
                };
                (object, info)
            }
            other => panic!("expected outputs, got {other:?}"),
        };
        assert!(matches!(info, Value::Struct(_)));
        let outputs = predict_classification_linear_object(
            object,
            tensor(vec![0.2, 2.8], vec![2, 1]),
            Vec::new(),
        )
        .unwrap();
        match &outputs[1] {
            Value::Tensor(score) => {
                assert_eq!(score.shape, vec![2, 2]);
                assert!((score.data[0] + score.data[2] - 1.0).abs() < 1.0e-8);
            }
            other => panic!("expected score, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fitclinear_observations_in_columns_and_text_classnames() {
        let model = fitclinear_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]),
            vec![
                Value::StringArray(StringArray {
                    data: vec!["low".into(), "low".into(), "high".into(), "high".into()],
                    shape: vec![4, 1],
                    rows: 4,
                    cols: 1,
                }),
                Value::String("ObservationsIn".into()),
                Value::String("columns".into()),
                Value::String("ClassNames".into()),
                Value::StringArray(StringArray {
                    data: vec!["low".into(), "high".into()],
                    shape: vec![2, 1],
                    rows: 2,
                    cols: 1,
                }),
            ],
        )
        .await
        .unwrap();
        let outputs = predict_classification_linear_object(
            match model {
                Value::Object(object) => object,
                _ => panic!("expected object"),
            },
            tensor(vec![0.2, 2.8], vec![1, 2]),
            vec![
                Value::String("ObservationsIn".into()),
                Value::String("columns".into()),
            ],
        )
        .unwrap();
        match &outputs[0] {
            Value::StringArray(labels) => assert_eq!(labels.data, vec!["low", "high"]),
            other => panic!("expected string labels, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fitclinear_rejects_multiclass_response() {
        let err = fitclinear_builtin(
            tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
            vec![tensor(vec![0.0, 1.0, 2.0], vec![3, 1])],
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("exactly two classes"));
    }

    #[tokio::test]
    async fn fitclinear_table_external_y_honors_response_name() {
        let table = table_from_columns(
            vec!["A".into(), "B".into()],
            vec![
                tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
                tensor(vec![1.0, 1.0, 1.0, 1.0], vec![4, 1]),
            ],
        )
        .unwrap();
        let model = fitclinear_builtin(
            table,
            vec![
                tensor(vec![0.0, 0.0, 1.0, 1.0], vec![4, 1]),
                Value::String("ResponseName".into()),
                Value::String("Outcome".into()),
            ],
        )
        .await
        .unwrap();
        match model {
            Value::Object(object) => {
                assert_eq!(
                    object.properties.get("ResponseName"),
                    Some(&Value::String("Outcome".into()))
                );
            }
            other => panic!("expected object, got {other:?}"),
        }
    }
}
