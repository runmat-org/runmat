//! Linear regression model compatibility surface.

use nalgebra::{DMatrix, DVector};
use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, FitlmOption,
    FITLM_ERROR_INTERNAL as ERROR_FITLM_INTERNAL,
    FITLM_ERROR_INVALID_ARGUMENT as ERROR_FITLM_INVALID_ARGUMENT,
    FITLM_ERROR_NUMERICAL as ERROR_FITLM_NUMERICAL, FITLM_INTEGER_CONTROL_EXTENSION,
    FITLM_INTEGER_PREDICTOR_EXTENSION, FITLM_INTEGER_RESPONSE_EXTENSION,
    FITLM_INTEGER_SELECTOR_EXTENSION, FITLM_INTEGER_WEIGHTS_EXTENSION,
    FITLM_RESIDENT_FALLBACK_EXTENSION, PREDICT_ERROR_INTERNAL as ERROR_PREDICT_INTERNAL,
    PREDICT_ERROR_INVALID_ARGUMENT as ERROR_PREDICT_INVALID_ARGUMENT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, NumericScalar, ObjectInstance, StringArray, StructValue, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::builtins::stats::summary::distribution_math::{student_t_cdf_upper, student_t_inv};
use crate::builtins::table::{
    is_tabular_object, table_from_columns, table_height, table_variable_names_from_object,
    table_variables,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const FITLM_NAME: &str = "fitlm";
const PREDICT_NAME: &str = "predict";
const EPS: f64 = 1.0e-12;

fn fitlm_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(FITLM_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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

fn fitlm_invalid(message: impl Into<String>) -> RuntimeError {
    fitlm_error(message, &ERROR_FITLM_INVALID_ARGUMENT)
}

fn fitlm_numerical(message: impl Into<String>) -> RuntimeError {
    fitlm_error(message, &ERROR_FITLM_NUMERICAL)
}

fn fitlm_internal(message: impl Into<String>) -> RuntimeError {
    fitlm_error(message, &ERROR_FITLM_INTERNAL)
}

pub(crate) fn predict_invalid(message: impl Into<String>) -> RuntimeError {
    predict_error(message, &ERROR_PREDICT_INVALID_ARGUMENT)
}

fn predict_internal(message: impl Into<String>) -> RuntimeError {
    predict_error(message, &ERROR_PREDICT_INTERNAL)
}

#[derive(Clone, Debug)]
struct FitOptions {
    intercept: bool,
    predictor_vars: Option<Vec<String>>,
    response_var: Option<String>,
    var_names: Option<Vec<String>>,
    exclude: Option<ExcludeSpec>,
    weights: Option<WeightSpec>,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            intercept: true,
            predictor_vars: None,
            response_var: None,
            var_names: None,
            exclude: None,
            weights: None,
        }
    }
}

#[derive(Clone, Debug)]
enum ExcludeSpec {
    Mask(Vec<bool>),
    Indices(Vec<usize>),
}

#[derive(Clone, Debug)]
enum WeightSpec {
    Values(Vec<f64>),
    Variable(String),
}

#[derive(Clone, Debug)]
struct PredictOptions {
    alpha: f64,
    prediction: PredictionKind,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            prediction: PredictionKind::Curve,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PredictionKind {
    Curve,
    Observation,
}

#[derive(Clone, Debug)]
struct FitSpec {
    x: Tensor,
    y: Vec<f64>,
    predictor_names: Vec<String>,
    response_name: String,
    terms: Vec<Vec<u32>>,
    coefficient_names: Vec<String>,
    model_formula: String,
    exclude: Option<ExcludeSpec>,
    weights: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct CleanData {
    design: DMatrix<f64>,
    weighted_design: DMatrix<f64>,
    y: DVector<f64>,
    weighted_y: DVector<f64>,
    weights: Vec<f64>,
}

#[derive(Clone, Debug)]
struct OlsFit {
    beta: Vec<f64>,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
    covariance: Vec<f64>,
    se: Vec<f64>,
    tstat: Vec<f64>,
    pvalue: Vec<f64>,
    sse: f64,
    mse: f64,
    rmse: f64,
    r_squared: f64,
    adjusted_r_squared: f64,
    dfe: f64,
    rank: usize,
    observations: usize,
}

#[runtime_builtin(
    name = "fitlm",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::linear_model"
)]
async fn fitlm_builtin(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_fitlm_integer_extensions(&first, &rest)?;
    ensure_fitlm_resident_fallback_extension(&first, &rest)?;
    let first = gather(first)
        .await
        .map_err(|err| fitlm_invalid(err.message))?;
    let rest = gather_all(rest)
        .await
        .map_err(|err| fitlm_invalid(err.message))?;
    let spec = build_fit_spec(first, rest)?;
    let fit = fit_ols(&spec)?;
    model_object(spec, fit).map(Value::Object)
}

fn ensure_fitlm_integer_extensions(first: &Value, rest: &[Value]) -> BuiltinResult<()> {
    let option_start = if is_table_value(first) {
        usize::from(
            rest.first()
                .is_some_and(|value| !is_name_value_option(value)),
        )
    } else {
        if is_typed_integer_value(first) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FITLM_INTEGER_PREDICTOR_EXTENSION,
                FITLM_NAME,
            )?;
        }
        if rest.first().is_some_and(is_typed_integer_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FITLM_INTEGER_RESPONSE_EXTENSION,
                FITLM_NAME,
            )?;
        }
        1 + usize::from(
            rest.get(1)
                .is_some_and(|value| !is_name_value_option(value)),
        )
    };

    let option_values = rest.get(option_start..).unwrap_or(&[]);
    let mut response_name: Option<String> = None;
    let mut predictor_names: Option<Vec<String>> = None;
    let mut weights_name: Option<String> = None;
    for pair in option_values.chunks_exact(2) {
        let Some(name) = scalar_text(&pair[0]) else {
            continue;
        };
        match FitlmOption::from_public_name(&name) {
            Some(FitlmOption::Weights) => {
                if is_typed_integer_value(&pair[1]) {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &FITLM_INTEGER_WEIGHTS_EXTENSION,
                        FITLM_NAME,
                    )?;
                }
                weights_name = scalar_text(&pair[1]);
            }
            Some(FitlmOption::Exclude) if is_typed_integer_value(&pair[1]) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FITLM_INTEGER_SELECTOR_EXTENSION,
                    FITLM_NAME,
                )?;
            }
            Some(FitlmOption::ResponseVar) => response_name = scalar_text(&pair[1]),
            Some(FitlmOption::PredictorVars) => {
                predictor_names = text_list(&pair[1], "PredictorVars").ok();
            }
            Some(FitlmOption::Intercept) if is_typed_integer_value(&pair[1]) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FITLM_INTEGER_CONTROL_EXTENSION,
                    FITLM_NAME,
                )?;
            }
            _ => {}
        }
    }

    if is_table_value(first) {
        let Value::Object(object) = first else {
            unreachable!("table values are objects")
        };
        let names = table_variable_names_from_object(object)
            .map_err(|error| fitlm_invalid(format!("fitlm: {error}")))?;
        let positional = rest.first().and_then(scalar_text);
        if response_name.is_none() {
            response_name = positional.as_ref().and_then(|text| {
                if let Some((lhs, _)) = text.split_once('~') {
                    Some(lhs.trim().to_string())
                } else if is_known_model_spec(text) {
                    None
                } else {
                    Some(text.clone())
                }
            });
        }
        let response_name = response_name.or_else(|| names.last().cloned());
        let mut active_predictors = predictor_names.unwrap_or_else(|| {
            if let Some(formula) = positional.as_ref().filter(|text| text.contains('~')) {
                let rhs = formula
                    .split_once('~')
                    .map(|(_, rhs)| rhs)
                    .unwrap_or_default();
                formula_rhs_tokens(rhs)
                    .into_iter()
                    .filter_map(|term| {
                        let term = term.trim();
                        names
                            .iter()
                            .any(|name| name == term)
                            .then(|| term.to_string())
                    })
                    .collect()
            } else {
                names
                    .iter()
                    .filter(|name| Some(*name) != response_name.as_ref())
                    .cloned()
                    .collect()
            }
        });
        if let Some(weights_name) = &weights_name {
            active_predictors.retain(|name| name != weights_name);
        }
        let variables =
            table_variables(object).map_err(|error| fitlm_invalid(format!("fitlm: {error}")))?;
        for (name, value) in &variables.fields {
            if !is_typed_integer_value(value) {
                continue;
            }
            let extension = if response_name.as_ref() == Some(name) {
                &FITLM_INTEGER_RESPONSE_EXTENSION
            } else if weights_name.as_ref() == Some(name) {
                &FITLM_INTEGER_WEIGHTS_EXTENSION
            } else if active_predictors.iter().any(|predictor| predictor == name) {
                &FITLM_INTEGER_PREDICTOR_EXTENSION
            } else {
                continue;
            };
            crate::compatibility::ensure_builtin_extension_enabled(extension, FITLM_NAME)?;
        }
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn ensure_fitlm_resident_fallback_extension(first: &Value, rest: &[Value]) -> BuiltinResult<()> {
    let mut resident = matches!(first, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)));
    if is_table_value(first) {
        let Value::Object(object) = first else {
            unreachable!("table values are objects")
        };
        resident |= table_variables(object)
            .map_err(|error| fitlm_invalid(format!("fitlm: {error}")))?
            .fields
            .values()
            .any(|value| matches!(value, Value::GpuTensor(_)));
    }
    if resident {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FITLM_RESIDENT_FALLBACK_EXTENSION,
            FITLM_NAME,
        )?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) async fn predict_builtin(
    model: Value,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    super::predict::predict_builtin(model, xnew, rest).await
}

pub(crate) fn predict_linear_model_dispatch(
    model: Value,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    let options = parse_predict_options(rest)?;
    let output = predict_linear_model(model, xnew, options)?;
    Ok(vec![output.0, output.1])
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| fitlm_invalid(format!("fitlm: {err}")))
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
            return Err(fitlm_invalid(
                "fitlm: matrix input requires response vector y",
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
    let x = tensor::value_into_tensor_for(FITLM_NAME, x_value)
        .map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    ensure_exact_fitlm_integer_tensor_boundary(&x, "X")?;
    if x.shape.len() > 2 {
        return Err(fitlm_invalid("fitlm: X must be a 2-D numeric matrix"));
    }
    let y_tensor = tensor::value_into_tensor_for(FITLM_NAME, y_value)
        .map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    ensure_exact_fitlm_integer_tensor_boundary(&y_tensor, "y")?;
    let y = vector_values(&y_tensor, "y")?;
    if y.len() != x.rows {
        return Err(fitlm_invalid(
            "fitlm: length(y) must match the number of rows in X",
        ));
    }
    let cols = x.cols;
    let mut options = FitOptions::default();
    let mut model_spec = ModelSpec::Linear;
    let mut index = 0;
    if index < rest.len() && !is_name_value_option(&rest[index]) {
        model_spec = parse_model_spec(&rest[index], cols)?;
        index += 1;
    }
    parse_fit_name_values(&rest[index..], &mut options)?;
    if options.response_var.is_some() {
        return Err(fitlm_invalid(
            "fitlm: ResponseVar is only valid for table input",
        ));
    }
    if options.predictor_vars.is_some() {
        return Err(fitlm_invalid(
            "fitlm: PredictorVars is only valid for table input",
        ));
    }
    let (predictor_names, response_name) = if let Some(names) = options.var_names.clone() {
        if names.len() != cols + 1 {
            return Err(fitlm_invalid(format!(
                "fitlm: VarNames must contain {} predictor names and one response name",
                cols
            )));
        }
        (names[..cols].to_vec(), names[cols].clone())
    } else {
        (
            (1..=cols).map(|idx| format!("x{idx}")).collect(),
            "y".to_string(),
        )
    };
    let terms = model_terms(model_spec, cols, options.intercept);
    let coefficient_names = coefficient_names(&terms, &predictor_names);
    let model_formula = formula_text(&response_name, &coefficient_names);
    let weights = match options.weights {
        Some(WeightSpec::Values(values)) => Some(values),
        Some(WeightSpec::Variable(_)) => {
            return Err(fitlm_invalid(
                "fitlm: weights variable names require table input",
            ));
        }
        None => None,
    };
    Ok(FitSpec {
        x,
        y,
        predictor_names,
        response_name,
        terms,
        coefficient_names,
        model_formula,
        exclude: options.exclude,
        weights,
    })
}

fn build_table_fit_spec(first: Value, rest: Vec<Value>) -> BuiltinResult<FitSpec> {
    let object = match first {
        Value::Object(object) => object,
        _ => unreachable!("table checked"),
    };
    let variable_names = table_variable_names_from_object(&object)
        .map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    if variable_names.len() < 2 {
        return Err(fitlm_invalid(
            "fitlm: table input must contain at least one predictor and one response",
        ));
    }
    let variables =
        table_variables(&object).map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    let mut options = FitOptions::default();
    let mut model_spec = ModelSpec::Linear;
    let mut formula: Option<String> = None;
    let mut response_arg: Option<String> = None;
    let mut index = 0;
    if index < rest.len() && !is_name_value_option(&rest[index]) {
        if let Some(text) = scalar_text(&rest[index]) {
            if text.contains('~') {
                formula = Some(text);
            } else if is_known_model_spec(&text) {
                model_spec = parse_model_spec_text(&text, variable_names.len() - 1)?;
            } else {
                response_arg = Some(text);
            }
        } else {
            model_spec = parse_model_spec(&rest[index], variable_names.len() - 1)?;
        }
        index += 1;
    }
    parse_fit_name_values(&rest[index..], &mut options)?;
    let response_name = formula
        .as_ref()
        .and_then(|text| text.split_once('~').map(|(lhs, _)| lhs.trim().to_string()))
        .or(options.response_var.clone())
        .or(response_arg)
        .unwrap_or_else(|| variable_names.last().cloned().unwrap());
    if !variable_names.iter().any(|name| name == &response_name) {
        return Err(fitlm_invalid(format!(
            "fitlm: response variable '{response_name}' is not in the table"
        )));
    }
    let mut predictor_names = if let Some(text) = &formula {
        parse_formula_predictors(text, &variable_names, &response_name, &mut options)?
    } else if let Some(names) = options.predictor_vars.clone() {
        names
    } else {
        variable_names
            .iter()
            .filter(|name| *name != &response_name)
            .cloned()
            .collect()
    };
    let weights = match options.weights.clone() {
        Some(WeightSpec::Variable(name)) => {
            let value = variables.fields.get(&name).ok_or_else(|| {
                fitlm_invalid(format!(
                    "fitlm: weights variable '{name}' is not in the table"
                ))
            })?;
            predictor_names.retain(|predictor| predictor != &name);
            Some(numeric_vector(value, "Weights")?)
        }
        Some(WeightSpec::Values(values)) => Some(values),
        None => None,
    };
    if predictor_names.is_empty() && !options.intercept {
        return Err(fitlm_invalid("fitlm: model must contain at least one term"));
    }
    for name in &predictor_names {
        if !variable_names.iter().any(|candidate| candidate == name) {
            return Err(fitlm_invalid(format!(
                "fitlm: predictor variable '{name}' is not in the table"
            )));
        }
    }
    let y_value = variables
        .fields
        .get(&response_name)
        .ok_or_else(|| fitlm_invalid("fitlm: response variable missing from table"))?;
    let y_tensor = tensor::value_into_tensor_for(FITLM_NAME, y_value.clone())
        .map_err(|err| fitlm_invalid(format!("fitlm: {err}")))?;
    ensure_exact_fitlm_integer_tensor_boundary(&y_tensor, &response_name)?;
    let y = vector_values(&y_tensor, &response_name)?;
    let x = if predictor_names.is_empty() {
        Tensor::new(Vec::new(), vec![y.len(), 0])
            .map_err(|err| fitlm_internal(format!("fitlm: {err}")))?
    } else {
        let mut columns = Vec::with_capacity(predictor_names.len());
        for name in &predictor_names {
            let value = variables
                .fields
                .get(name)
                .ok_or_else(|| fitlm_invalid(format!("fitlm: predictor '{name}' missing")))?;
            let tensor = tensor::value_into_tensor_for(FITLM_NAME, value.clone())
                .map_err(|_| fitlm_invalid(format!("fitlm: predictor '{name}' must be numeric")))?;
            ensure_exact_fitlm_integer_tensor_boundary(&tensor, name)?;
            let values = vector_values(&tensor, name)?;
            if values.len() != y.len() {
                return Err(fitlm_invalid(format!(
                    "fitlm: predictor '{name}' height does not match response"
                )));
            }
            columns.push(values);
        }
        columns_to_tensor(columns)?
    };
    let terms = if let Some(text) = &formula {
        parse_formula_terms(text, &predictor_names, options.intercept)?
    } else {
        model_terms(model_spec, predictor_names.len(), options.intercept)
    };
    let coefficient_names = coefficient_names(&terms, &predictor_names);
    let model_formula = formula
        .as_ref()
        .cloned()
        .unwrap_or_else(|| formula_text(&response_name, &coefficient_names));
    Ok(FitSpec {
        x,
        y,
        predictor_names,
        response_name,
        terms,
        coefficient_names,
        model_formula,
        exclude: options.exclude,
        weights,
    })
}

fn parse_fit_name_values(values: &[Value], options: &mut FitOptions) -> BuiltinResult<()> {
    if !values.len().is_multiple_of(2) {
        return Err(fitlm_invalid("fitlm: name-value options must be paired"));
    }
    let mut index = 0;
    while index < values.len() {
        let name = scalar_text(&values[index])
            .ok_or_else(|| fitlm_invalid("fitlm: option name must be text"))?;
        let value = &values[index + 1];
        match FitlmOption::from_public_name(&name) {
            Some(FitlmOption::Intercept) => options.intercept = bool_scalar(value, "Intercept")?,
            Some(FitlmOption::ResponseVar) => {
                options.response_var = Some(
                    scalar_text(value)
                        .ok_or_else(|| fitlm_invalid("fitlm: ResponseVar must be text"))?,
                );
            }
            Some(FitlmOption::PredictorVars) => {
                options.predictor_vars = Some(text_list(value, "PredictorVars")?);
            }
            Some(FitlmOption::VarNames) => options.var_names = Some(text_list(value, "VarNames")?),
            Some(FitlmOption::Exclude) => options.exclude = Some(exclude_spec(value)?),
            Some(FitlmOption::Weights) => {
                options.weights = Some(if let Some(name) = scalar_text(value) {
                    WeightSpec::Variable(name)
                } else {
                    WeightSpec::Values(numeric_vector(value, "Weights")?)
                });
            }
            Some(FitlmOption::RobustOpts | FitlmOption::CategoricalVars) => {
                return Err(fitlm_invalid(format!(
                    "fitlm: option '{name}' is not supported yet"
                )));
            }
            None => return Err(fitlm_invalid(format!("fitlm: unknown option '{name}'"))),
        }
        index += 2;
    }
    Ok(())
}

fn parse_predict_options(values: Vec<Value>) -> BuiltinResult<PredictOptions> {
    if !values.len().is_multiple_of(2) {
        return Err(predict_invalid(
            "predict: name-value options must be paired",
        ));
    }
    let mut options = PredictOptions::default();
    let mut index = 0;
    while index < values.len() {
        let name = scalar_text(&values[index])
            .ok_or_else(|| predict_invalid("predict: option name must be text"))?;
        let value = &values[index + 1];
        if name.eq_ignore_ascii_case("Alpha") {
            let alpha =
                numeric_scalar(value, "Alpha").map_err(|err| predict_invalid(err.message))?;
            if !(0.0..1.0).contains(&alpha) {
                return Err(predict_invalid("predict: Alpha must be between 0 and 1"));
            }
            options.alpha = alpha;
        } else if name.eq_ignore_ascii_case("Prediction") {
            let text = scalar_text(value)
                .ok_or_else(|| predict_invalid("predict: Prediction must be text"))?;
            if text.eq_ignore_ascii_case("curve") {
                options.prediction = PredictionKind::Curve;
            } else if text.eq_ignore_ascii_case("observation") {
                options.prediction = PredictionKind::Observation;
            } else {
                return Err(predict_invalid(
                    "predict: Prediction must be 'curve' or 'observation'",
                ));
            }
        } else if name.eq_ignore_ascii_case("Simultaneous") {
            let simultaneous =
                bool_scalar(value, "Simultaneous").map_err(|err| predict_invalid(err.message))?;
            if simultaneous {
                return Err(predict_invalid(
                    "predict: Simultaneous=true intervals are not supported yet",
                ));
            }
        } else {
            return Err(predict_invalid(format!("predict: unknown option '{name}'")));
        }
        index += 2;
    }
    Ok(options)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelSpec {
    Constant,
    Linear,
    Interactions,
    PureQuadratic,
    Quadratic,
}

fn parse_model_spec(value: &Value, predictors: usize) -> BuiltinResult<ModelSpec> {
    if let Some(text) = scalar_text(value) {
        parse_model_spec_text(&text, predictors)
    } else {
        Err(fitlm_invalid(
            "fitlm: numeric terms matrices are not supported yet",
        ))
    }
}

fn parse_model_spec_text(text: &str, _predictors: usize) -> BuiltinResult<ModelSpec> {
    if text.eq_ignore_ascii_case("constant") {
        Ok(ModelSpec::Constant)
    } else if text.eq_ignore_ascii_case("linear") {
        Ok(ModelSpec::Linear)
    } else if text.eq_ignore_ascii_case("interactions") {
        Ok(ModelSpec::Interactions)
    } else if text.eq_ignore_ascii_case("purequadratic") {
        Ok(ModelSpec::PureQuadratic)
    } else if text.eq_ignore_ascii_case("quadratic") {
        Ok(ModelSpec::Quadratic)
    } else {
        Err(fitlm_invalid(format!(
            "fitlm: unsupported model specification '{text}'"
        )))
    }
}

fn is_known_model_spec(text: &str) -> bool {
    matches!(
        text.to_ascii_lowercase().as_str(),
        "constant" | "linear" | "interactions" | "purequadratic" | "quadratic"
    )
}

fn model_terms(spec: ModelSpec, predictors: usize, intercept: bool) -> Vec<Vec<u32>> {
    let mut terms = Vec::new();
    if intercept {
        terms.push(vec![0; predictors]);
    }
    if matches!(
        spec,
        ModelSpec::Linear
            | ModelSpec::Interactions
            | ModelSpec::PureQuadratic
            | ModelSpec::Quadratic
    ) {
        for col in 0..predictors {
            let mut term = vec![0; predictors];
            term[col] = 1;
            terms.push(term);
        }
    }
    if matches!(spec, ModelSpec::Interactions | ModelSpec::Quadratic) {
        for left in 0..predictors {
            for right in left + 1..predictors {
                let mut term = vec![0; predictors];
                term[left] = 1;
                term[right] = 1;
                terms.push(term);
            }
        }
    }
    if matches!(spec, ModelSpec::PureQuadratic | ModelSpec::Quadratic) {
        for col in 0..predictors {
            let mut term = vec![0; predictors];
            term[col] = 2;
            terms.push(term);
        }
    }
    terms
}

fn parse_formula_predictors(
    formula: &str,
    table_names: &[String],
    response_name: &str,
    options: &mut FitOptions,
) -> BuiltinResult<Vec<String>> {
    let (_, rhs) = formula
        .split_once('~')
        .ok_or_else(|| fitlm_invalid("fitlm: formula must contain '~'"))?;
    let mut out = Vec::new();
    for raw in formula_rhs_tokens(rhs) {
        let term = raw.trim();
        if term.is_empty() || term == "1" {
            continue;
        }
        if term == "0" || term == "-1" {
            options.intercept = false;
            continue;
        }
        if term.contains(':') || term.contains('*') || term.contains('^') {
            return Err(fitlm_invalid(
                "fitlm: formula interactions and powers are not supported yet",
            ));
        }
        if term == response_name {
            return Err(fitlm_invalid("fitlm: response cannot also be a predictor"));
        }
        if !table_names.iter().any(|name| name == term) {
            return Err(fitlm_invalid(format!(
                "fitlm: formula predictor '{term}' is not in the table"
            )));
        }
        if !out.iter().any(|name| name == term) {
            out.push(term.to_string());
        }
    }
    Ok(out)
}

fn parse_formula_terms(
    formula: &str,
    predictor_names: &[String],
    intercept: bool,
) -> BuiltinResult<Vec<Vec<u32>>> {
    let (_, rhs) = formula
        .split_once('~')
        .ok_or_else(|| fitlm_invalid("fitlm: formula must contain '~'"))?;
    let mut terms = Vec::new();
    if intercept {
        terms.push(vec![0; predictor_names.len()]);
    }
    for raw in formula_rhs_tokens(rhs) {
        let term = raw.trim();
        if term.is_empty() || term == "1" || term == "0" || term == "-1" {
            continue;
        }
        let position = predictor_names
            .iter()
            .position(|name| name == term)
            .ok_or_else(|| fitlm_invalid(format!("fitlm: unknown formula term '{term}'")))?;
        let mut exponents = vec![0; predictor_names.len()];
        exponents[position] = 1;
        terms.push(exponents);
    }
    Ok(terms)
}

fn formula_rhs_tokens(rhs: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut sign = 1_i8;
    let mut saw_token = false;
    for ch in rhs.chars() {
        match ch {
            '+' | '-' => {
                if saw_token || !current.trim().is_empty() {
                    let token = current.trim();
                    if !token.is_empty() {
                        out.push(if sign < 0 {
                            format!("-{token}")
                        } else {
                            token.to_string()
                        });
                    }
                    current.clear();
                    saw_token = false;
                }
                sign = if ch == '-' { -1 } else { 1 };
            }
            _ => {
                if !ch.is_whitespace() {
                    saw_token = true;
                }
                current.push(ch);
            }
        }
    }
    let token = current.trim();
    if !token.is_empty() {
        out.push(if sign < 0 {
            format!("-{token}")
        } else {
            token.to_string()
        });
    }
    out
}

fn coefficient_names(terms: &[Vec<u32>], predictor_names: &[String]) -> Vec<String> {
    terms
        .iter()
        .map(|term| {
            if term.iter().all(|power| *power == 0) {
                "(Intercept)".to_string()
            } else {
                let mut factors = Vec::new();
                for (name, power) in predictor_names.iter().zip(term) {
                    match *power {
                        0 => {}
                        1 => factors.push(name.clone()),
                        n => factors.push(format!("{name}^{n}")),
                    }
                }
                factors.join(":")
            }
        })
        .collect()
}

fn formula_text(response_name: &str, coefficient_names: &[String]) -> String {
    let rhs = if coefficient_names.is_empty() {
        "0".to_string()
    } else {
        coefficient_names
            .iter()
            .map(|name| {
                if name == "(Intercept)" {
                    "1".to_string()
                } else {
                    name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" + ")
    };
    format!("{response_name} ~ {rhs}")
}

fn fit_ols(spec: &FitSpec) -> BuiltinResult<OlsFit> {
    let clean = clean_data(spec)?;
    let observations = clean.y.len();
    if observations == 0 {
        return Err(fitlm_invalid(
            "fitlm: at least one complete observation is required",
        ));
    }
    let svd = clean.weighted_design.clone().svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| fitlm_numerical("fitlm: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| fitlm_numerical("fitlm: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.iter().copied().collect::<Vec<_>>();
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = (observations.max(spec.terms.len()) as f64) * f64::EPSILON * largest.max(1.0);
    let rank = singular_values
        .iter()
        .filter(|value| value.abs() > tolerance)
        .count();
    let mut beta = DVector::zeros(spec.terms.len());
    let mut inv_xtx = DMatrix::<f64>::zeros(spec.terms.len(), spec.terms.len());
    for (index, singular_value) in singular_values.iter().copied().enumerate() {
        if singular_value.abs() <= tolerance {
            continue;
        }
        let uty = u.column(index).dot(&clean.weighted_y);
        let contribution = uty / singular_value;
        for row in 0..spec.terms.len() {
            beta[row] += v_t[(index, row)] * contribution;
        }
        let inv_s2 = 1.0 / (singular_value * singular_value);
        for row in 0..spec.terms.len() {
            for col in 0..spec.terms.len() {
                inv_xtx[(row, col)] += v_t[(index, row)] * v_t[(index, col)] * inv_s2;
            }
        }
    }
    let fitted_vec = &clean.design * &beta;
    let residual_vec = &clean.y - &fitted_vec;
    let sse = residual_vec
        .iter()
        .zip(clean.weights.iter())
        .map(|(residual, weight)| residual * residual * weight)
        .sum::<f64>();
    let weight_sum = clean.weights.iter().sum::<f64>();
    let y_mean = if weight_sum > 0.0 {
        clean
            .y
            .iter()
            .zip(clean.weights.iter())
            .map(|(y, weight)| y * weight)
            .sum::<f64>()
            / weight_sum
    } else {
        f64::NAN
    };
    let sst = clean
        .y
        .iter()
        .zip(clean.weights.iter())
        .map(|(y, weight)| {
            let diff = y - y_mean;
            diff * diff * weight
        })
        .sum::<f64>();
    let dfe = observations as f64 - rank as f64;
    let mse = if dfe > 0.0 { sse / dfe } else { f64::NAN };
    let rmse = mse.sqrt();
    let r_squared = if sst > EPS {
        1.0 - sse / sst
    } else if sse <= EPS {
        1.0
    } else {
        f64::NAN
    };
    let adjusted_r_squared = if dfe > 0.0 && observations > 1 && r_squared.is_finite() {
        1.0 - (1.0 - r_squared) * ((observations - 1) as f64) / dfe
    } else {
        f64::NAN
    };
    let covariance = inv_xtx.iter().map(|value| value * mse).collect::<Vec<_>>();
    let mut se = Vec::with_capacity(spec.terms.len());
    let mut tstat = Vec::with_capacity(spec.terms.len());
    let mut pvalue = Vec::with_capacity(spec.terms.len());
    for i in 0..spec.terms.len() {
        let variance = covariance[i + i * spec.terms.len()];
        let std_error = if variance >= 0.0 {
            variance.sqrt()
        } else {
            f64::NAN
        };
        let t = beta[i] / std_error;
        let p = if dfe > 0.0 {
            (2.0 * student_t_cdf_upper(t.abs(), dfe)).min(1.0)
        } else {
            f64::NAN
        };
        se.push(std_error);
        tstat.push(t);
        pvalue.push(p);
    }
    Ok(OlsFit {
        beta: beta.iter().copied().collect(),
        fitted: fitted_vec.iter().copied().collect(),
        residuals: residual_vec.iter().copied().collect(),
        covariance,
        se,
        tstat,
        pvalue,
        sse,
        mse,
        rmse,
        r_squared,
        adjusted_r_squared,
        dfe,
        rank,
        observations,
    })
}

fn clean_data(spec: &FitSpec) -> BuiltinResult<CleanData> {
    let rows = spec.x.rows;
    let cols = spec.x.cols;
    let exclude_mask = exclude_mask(spec.exclude.as_ref(), rows)?;
    let weights = spec.weights.as_ref();
    if let Some(weights) = weights {
        if weights.len() != rows {
            return Err(fitlm_invalid(
                "fitlm: Weights length must match the number of observations",
            ));
        }
    }
    let mut design_rows = Vec::new();
    let mut y_values = Vec::new();
    let mut clean_weights = Vec::new();
    for row in 0..rows {
        if exclude_mask.as_ref().is_some_and(|mask| mask[row]) {
            continue;
        }
        let y = spec.y[row];
        let weight = weights.map(|w| w[row]).unwrap_or(1.0);
        if weight < 0.0 || !weight.is_finite() {
            return Err(fitlm_invalid(
                "fitlm: weights must be nonnegative finite values",
            ));
        }
        let mut raw = Vec::with_capacity(cols);
        let mut has_nan = y.is_nan();
        let mut has_inf = y.is_infinite();
        for col in 0..cols {
            let value = x_value(&spec.x, row, col);
            has_nan |= value.is_nan();
            has_inf |= value.is_infinite();
            raw.push(value);
        }
        if has_inf {
            return Err(fitlm_invalid("fitlm: X and y must not contain Inf values"));
        }
        if has_nan || weight == 0.0 {
            continue;
        }
        for term in &spec.terms {
            design_rows.push(evaluate_term(&raw, term));
        }
        y_values.push(y);
        clean_weights.push(weight);
    }
    let observations = y_values.len();
    let predictors = spec.terms.len();
    let design = DMatrix::from_row_slice(observations, predictors, &design_rows);
    let mut weighted_rows = design_rows;
    let mut weighted_y = y_values.clone();
    for row in 0..observations {
        let scale = clean_weights[row].sqrt();
        weighted_y[row] *= scale;
        for col in 0..predictors {
            weighted_rows[row * predictors + col] *= scale;
        }
    }
    Ok(CleanData {
        design,
        weighted_design: DMatrix::from_row_slice(observations, predictors, &weighted_rows),
        y: DVector::from_column_slice(&y_values),
        weighted_y: DVector::from_column_slice(&weighted_y),
        weights: clean_weights,
    })
}

fn evaluate_term(row: &[f64], term: &[u32]) -> f64 {
    let mut value = 1.0;
    for (x, power) in row.iter().zip(term) {
        match *power {
            0 => {}
            1 => value *= *x,
            2 => value *= x * x,
            n => value *= x.powi(n as i32),
        }
    }
    value
}

fn model_object(spec: FitSpec, fit: OlsFit) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(runmat_types::standard::LINEAR_MODEL.to_string());
    object.properties.insert(
        "Coefficients".to_string(),
        coefficients_table(&spec.coefficient_names, &fit)?,
    );
    object.properties.insert(
        "CoefficientNames".to_string(),
        string_row(spec.coefficient_names.clone())?,
    );
    object
        .properties
        .insert("Formula".to_string(), Value::String(spec.model_formula));
    object.properties.insert(
        "NumObservations".to_string(),
        Value::Num(fit.observations as f64),
    );
    object
        .properties
        .insert("DFE".to_string(), Value::Num(fit.dfe));
    object
        .properties
        .insert("RMSE".to_string(), Value::Num(fit.rmse));
    let mut rsquared = StructValue::new();
    rsquared.insert("Ordinary", Value::Num(fit.r_squared));
    rsquared.insert("Adjusted", Value::Num(fit.adjusted_r_squared));
    object
        .properties
        .insert("Rsquared".to_string(), Value::Struct(rsquared));
    let mut model_criterion = StructValue::new();
    model_criterion.insert("MSE", Value::Num(fit.mse));
    model_criterion.insert("SSE", Value::Num(fit.sse));
    object
        .properties
        .insert("ModelCriterion".to_string(), Value::Struct(model_criterion));
    object.properties.insert(
        "Fitted".to_string(),
        Value::Tensor(
            Tensor::new(fit.fitted.clone(), vec![fit.fitted.len(), 1]).map_err(|err| {
                fitlm_internal(format!("fitlm: failed to construct fitted values: {err}"))
            })?,
        ),
    );
    let mut residuals = StructValue::new();
    residuals.insert(
        "Raw",
        Value::Tensor(
            Tensor::new(fit.residuals.clone(), vec![fit.residuals.len(), 1]).map_err(|err| {
                fitlm_internal(format!("fitlm: failed to construct residuals: {err}"))
            })?,
        ),
    );
    object
        .properties
        .insert("Residuals".to_string(), Value::Struct(residuals));
    object.properties.insert(
        "PredictorNames".to_string(),
        string_row(spec.predictor_names.clone())?,
    );
    object.properties.insert(
        "ResponseName".to_string(),
        Value::String(spec.response_name),
    );
    object.properties.insert(
        "NumPredictors".to_string(),
        Value::Num(spec.predictor_names.len() as f64),
    );
    object.properties.insert(
        "HasIntercept".to_string(),
        Value::Bool(has_intercept(&spec.terms)),
    );
    object
        .properties
        .insert("DesignRank".to_string(), Value::Num(fit.rank as f64));
    object
        .properties
        .insert("__RunMatTerms".to_string(), terms_tensor(&spec.terms)?);
    object
        .properties
        .insert("__RunMatBeta".to_string(), numeric_column(&fit.beta)?);
    object.properties.insert(
        "__RunMatCovariance".to_string(),
        Value::Tensor(
            Tensor::new(
                fit.covariance.clone(),
                vec![spec.terms.len(), spec.terms.len()],
            )
            .map_err(|err| fitlm_internal(format!("fitlm: covariance shape error: {err}")))?,
        ),
    );
    object
        .properties
        .insert("__RunMatMSE".to_string(), Value::Num(fit.mse));
    object
        .properties
        .insert("__RunMatDFE".to_string(), Value::Num(fit.dfe));
    Ok(object)
}

fn coefficients_table(names: &[String], fit: &OlsFit) -> BuiltinResult<Value> {
    let columns = vec![
        numeric_column(&fit.beta)?,
        numeric_column(&fit.se)?,
        numeric_column(&fit.tstat)?,
        numeric_column(&fit.pvalue)?,
    ];
    let table = table_from_columns(
        vec![
            "Estimate".to_string(),
            "SE".to_string(),
            "tStat".to_string(),
            "pValue".to_string(),
        ],
        columns,
    )
    .map_err(|err| fitlm_internal(format!("fitlm: {err}")))?;
    let mut object = match table {
        Value::Object(object) => object,
        _ => unreachable!("table_from_columns returns object"),
    };
    let row_names = string_column(names.to_vec())?;
    if let Some(Value::Struct(props)) = object.properties.get_mut("Properties") {
        props.insert("RowNames", row_names.clone());
    }
    if let Some(Value::Struct(props)) = object.properties.get_mut("__table_properties") {
        props.insert("RowNames", row_names.clone());
    }
    object
        .properties
        .insert("__CoefficientNames".to_string(), row_names);
    Ok(Value::Object(object))
}

fn predict_linear_model(
    model: Value,
    xnew: Value,
    options: PredictOptions,
) -> BuiltinResult<(Value, Value)> {
    let object = match model {
        Value::Object(object) if object.class_name.is(runmat_types::standard::LINEAR_MODEL) => {
            object
        }
        other => {
            return Err(predict_invalid(format!(
                "predict: expected LinearModel object, got {other:?}"
            )));
        }
    };
    let terms = terms_from_object(&object)?;
    let beta = numeric_property(&object, "__RunMatBeta")?;
    let covariance = matrix_property(&object, "__RunMatCovariance")?;
    let predictor_names = string_property(&object, "PredictorNames")?;
    let x = predictors_for_prediction(xnew, &predictor_names)?;
    let design = design_matrix(&x, &terms);
    let rows = x.rows;
    let cols = terms.len();
    if beta.len() != cols {
        return Err(predict_invalid(
            "predict: model coefficient length is invalid",
        ));
    }
    let mut ypred = Vec::with_capacity(rows);
    let mut ci_low = Vec::with_capacity(rows);
    let mut ci_high = Vec::with_capacity(rows);
    let mse = numeric_scalar_property(&object, "__RunMatMSE")?;
    let dfe = numeric_scalar_property(&object, "__RunMatDFE")?;
    let crit = if dfe > 0.0 {
        student_t_inv(1.0 - options.alpha / 2.0, dfe)
    } else {
        f64::NAN
    };
    for row in 0..rows {
        let mut y = 0.0;
        for col in 0..cols {
            y += design[row * cols + col] * beta[col];
        }
        ypred.push(y);
        let leverage = leverage_for_row(&design[row * cols..row * cols + cols], &covariance, cols);
        let variance = match options.prediction {
            PredictionKind::Curve => leverage,
            PredictionKind::Observation => leverage + mse,
        };
        let delta = if variance >= 0.0 {
            crit * variance.sqrt()
        } else {
            f64::NAN
        };
        ci_low.push(y - delta);
        ci_high.push(y + delta);
    }
    let mut intervals = ci_low;
    intervals.extend(ci_high);
    let y_value = Value::Tensor(
        Tensor::new(ypred, vec![rows, 1])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    let ci_value = Value::Tensor(
        Tensor::new(intervals, vec![rows, 2])
            .map_err(|err| predict_internal(format!("predict: {err}")))?,
    );
    Ok((y_value, ci_value))
}

fn predictors_for_prediction(value: Value, predictor_names: &[String]) -> BuiltinResult<Tensor> {
    if let Value::Object(object) = &value {
        if is_tabular_object(object) {
            if predictor_names.is_empty() {
                let rows = table_height(object)
                    .map_err(|err| predict_invalid(format!("predict: {err}")))?;
                return Tensor::new(Vec::new(), vec![rows, 0])
                    .map_err(|err| predict_internal(format!("predict: {err}")));
            }
            let variables = table_variables(object)
                .map_err(|err| predict_invalid(format!("predict: {err}")))?;
            let mut columns = Vec::with_capacity(predictor_names.len());
            let mut height = None;
            for name in predictor_names {
                let raw = variables.fields.get(name).ok_or_else(|| {
                    predict_invalid(format!("predict: missing predictor '{name}'"))
                })?;
                let tensor =
                    tensor::value_into_tensor_for(PREDICT_NAME, raw.clone()).map_err(|_| {
                        predict_invalid(format!("predict: predictor '{name}' must be numeric"))
                    })?;
                let values = vector_values_predict(&tensor, name)?;
                if let Some(expected) = height {
                    if values.len() != expected {
                        return Err(predict_invalid(
                            "predict: predictor table variables must have the same height",
                        ));
                    }
                } else {
                    height = Some(values.len());
                }
                columns.push(values);
            }
            return columns_to_tensor_predict(columns);
        }
    }
    let tensor = tensor::value_into_tensor_for(PREDICT_NAME, value)
        .map_err(|err| predict_invalid(format!("predict: {err}")))?;
    if tensor.shape.len() > 2 || tensor.cols != predictor_names.len() {
        return Err(predict_invalid(format!(
            "predict: Xnew must have {} predictor columns",
            predictor_names.len()
        )));
    }
    Ok(tensor)
}

fn design_matrix(x: &Tensor, terms: &[Vec<u32>]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.rows * terms.len());
    for row in 0..x.rows {
        let raw = (0..x.cols)
            .map(|col| x_value(x, row, col))
            .collect::<Vec<_>>();
        for term in terms {
            out.push(evaluate_term(&raw, term));
        }
    }
    out
}

fn leverage_for_row(row: &[f64], covariance: &[f64], cols: usize) -> f64 {
    let mut value = 0.0;
    for i in 0..cols {
        for j in 0..cols {
            value += row[i] * covariance[i + j * cols] * row[j];
        }
    }
    value
}

fn terms_from_object(object: &ObjectInstance) -> BuiltinResult<Vec<Vec<u32>>> {
    let tensor = match object.properties.get("__RunMatTerms") {
        Some(Value::Tensor(tensor)) => tensor,
        _ => return Err(predict_invalid("predict: model is missing term metadata")),
    };
    let mut terms = Vec::with_capacity(tensor.rows);
    for row in 0..tensor.rows {
        let mut term = Vec::with_capacity(tensor.cols);
        for col in 0..tensor.cols {
            let value = x_value(tensor, row, col);
            if value < 0.0 || value.fract().abs() > EPS {
                return Err(predict_invalid("predict: model term metadata is invalid"));
            }
            term.push(value as u32);
        }
        terms.push(term);
    }
    Ok(terms)
}

fn numeric_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    match object.properties.get(name) {
        Some(Value::Tensor(tensor)) => Ok(tensor::tensor_values_f64(tensor)),
        Some(Value::Num(value)) => Ok(vec![*value]),
        _ => Err(predict_invalid(format!(
            "predict: model is missing numeric property '{name}'"
        ))),
    }
}

fn matrix_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<f64>> {
    match object.properties.get(name) {
        Some(Value::Tensor(tensor)) if tensor.rows == tensor.cols => {
            Ok(tensor::tensor_values_f64(tensor))
        }
        _ => Err(predict_invalid(format!(
            "predict: model is missing matrix property '{name}'"
        ))),
    }
}

fn numeric_scalar_property(object: &ObjectInstance, name: &str) -> BuiltinResult<f64> {
    match object.properties.get(name) {
        Some(Value::Num(value)) => Ok(*value),
        Some(Value::Tensor(tensor)) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        _ => Err(predict_invalid(format!(
            "predict: model is missing scalar property '{name}'"
        ))),
    }
}

fn string_property(object: &ObjectInstance, name: &str) -> BuiltinResult<Vec<String>> {
    match object.properties.get(name) {
        Some(Value::StringArray(array)) => Ok(array.data.clone()),
        Some(Value::Cell(cell)) => cell
            .data
            .iter()
            .map(|value| {
                scalar_text(value).ok_or_else(|| {
                    predict_invalid(format!("predict: property '{name}' is invalid"))
                })
            })
            .collect(),
        _ => Err(predict_invalid(format!(
            "predict: model is missing string property '{name}'"
        ))),
    }
}

fn vector_values(tensor: &Tensor, label: &str) -> BuiltinResult<Vec<f64>> {
    if tensor.shape.len() > 2 || !(tensor.rows == 1 || tensor.cols == 1) {
        return Err(fitlm_invalid(format!("fitlm: {label} must be a vector")));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

fn ensure_exact_fitlm_integer_tensor_boundary(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for integer in storage.exact_values() {
        if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
            return Err(fitlm_invalid(format!(
                "fitlm: integer {role} values must be exactly representable as double"
            )));
        }
    }
    Ok(())
}

fn ensure_exact_fitlm_integer_scalar_boundary(
    value: NumericScalar,
    role: &str,
) -> BuiltinResult<()> {
    let Some(integer) = value.into_int_value() else {
        return Ok(());
    };
    if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
        return Err(fitlm_invalid(format!(
            "fitlm: integer {role} values must be exactly representable as double"
        )));
    }
    Ok(())
}

fn vector_values_predict(tensor: &Tensor, label: &str) -> BuiltinResult<Vec<f64>> {
    if tensor.shape.len() > 2 || !(tensor.rows == 1 || tensor.cols == 1) {
        return Err(predict_invalid(format!(
            "predict: {label} must be a vector"
        )));
    }
    Ok(tensor::tensor_values_f64(tensor))
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => {
            ensure_exact_fitlm_integer_scalar_boundary(NumericScalar::from(value.clone()), name)?;
            Ok(vec![value.to_f64()])
        }
        Value::Tensor(tensor) => {
            ensure_exact_fitlm_integer_tensor_boundary(tensor, name)?;
            vector_values(tensor, name)
        }
        _ => Err(fitlm_invalid(format!("fitlm: {name} must be numeric"))),
    }
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return Ok(integer.to_f64());
    }
    match value {
        Value::Num(value) => Ok(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        _ => Err(fitlm_invalid(format!("fitlm: {name} must be a scalar"))),
    }
}

fn exclude_spec(value: &Value) -> BuiltinResult<ExcludeSpec> {
    match value {
        Value::Bool(value) => Ok(ExcludeSpec::Mask(vec![*value])),
        Value::LogicalArray(array) => Ok(ExcludeSpec::Mask(
            array.data.iter().map(|value| *value != 0).collect(),
        )),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let indices = tensor
                .integer_storage()
                .expect("integer storage checked above")
                .exact_values()
                .iter()
                .map(|value| value.try_to_usize().filter(|value| *value > 0))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| fitlm_invalid("fitlm: Exclude indices must be positive integers"))?;
            Ok(ExcludeSpec::Indices(indices))
        }
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64(tensor);
            let mut indices = Vec::with_capacity(values.len());
            for value in values {
                if value < 1.0 || value.fract().abs() > EPS {
                    return Err(fitlm_invalid(
                        "fitlm: Exclude indices must be positive integers",
                    ));
                }
                indices.push(value as usize);
            }
            Ok(ExcludeSpec::Indices(indices))
        }
        _ => Err(fitlm_invalid(
            "fitlm: Exclude must be logical mask or row indices",
        )),
    }
}

fn exclude_mask(spec: Option<&ExcludeSpec>, rows: usize) -> BuiltinResult<Option<Vec<bool>>> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let mut mask = vec![false; rows];
    match spec {
        ExcludeSpec::Mask(values) => {
            if values.len() == 1 {
                mask.fill(values[0]);
            } else if values.len() == rows {
                mask.clone_from(values);
            } else {
                return Err(fitlm_invalid(
                    "fitlm: Exclude mask length must match the number of observations",
                ));
            }
        }
        ExcludeSpec::Indices(indices) => {
            for index in indices {
                if *index == 0 || *index > rows {
                    return Err(fitlm_invalid("fitlm: Exclude index exceeds observations"));
                }
                mask[*index - 1] = true;
            }
        }
    }
    Ok(Some(mask))
}

fn bool_scalar(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return Ok(!integer.is_zero());
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0] != 0.0)
        }
        _ => Err(fitlm_invalid(format!(
            "fitlm: {name} must be logical scalar"
        ))),
    }
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) => Some(chars_to_string(chars)),
        Value::StringArray(array) if array.data.len() == 1 => array.data.first().cloned(),
        _ => None,
    }
}

fn text_list(value: &Value, name: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(chars) => Ok(vec![chars_to_string(chars)]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|value| {
                scalar_text(value)
                    .ok_or_else(|| fitlm_invalid(format!("fitlm: {name} entries must be text")))
            })
            .collect(),
        _ => Err(fitlm_invalid(format!(
            "fitlm: {name} must be text or a text collection"
        ))),
    }
}

fn chars_to_string(chars: &CharArray) -> String {
    chars.data.iter().collect()
}

fn is_name_value_option(value: &Value) -> bool {
    scalar_text(value).is_some_and(|text| FitlmOption::from_public_name(&text).is_some())
}

fn columns_to_tensor(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|err| fitlm_internal(format!("fitlm: {err}")));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        if column.len() != rows {
            return Err(fitlm_invalid(
                "fitlm: predictor variables must have the same height",
            ));
        }
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols]).map_err(|err| fitlm_internal(format!("fitlm: {err}")))
}

fn columns_to_tensor_predict(columns: Vec<Vec<f64>>) -> BuiltinResult<Tensor> {
    if columns.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|err| predict_internal(format!("predict: {err}")));
    }
    let rows = columns[0].len();
    let cols = columns.len();
    let mut data = Vec::with_capacity(rows * cols);
    for column in columns {
        data.extend(column);
    }
    Tensor::new(data, vec![rows, cols]).map_err(|err| predict_internal(format!("predict: {err}")))
}

fn x_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    tensor::tensor_value_f64(tensor, col * tensor.rows + row)
}

fn terms_tensor(terms: &[Vec<u32>]) -> BuiltinResult<Value> {
    let rows = terms.len();
    let cols = terms.first().map(|term| term.len()).unwrap_or(0);
    let mut data = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for term in terms {
            data.push(term[col] as f64);
        }
    }
    Ok(Value::Tensor(Tensor::new(data, vec![rows, cols]).map_err(
        |err| fitlm_internal(format!("fitlm: terms metadata error: {err}")),
    )?))
}

fn numeric_column(values: &[f64]) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        Tensor::new(values.to_vec(), vec![values.len(), 1])
            .map_err(|err| fitlm_internal(format!("fitlm: numeric column error: {err}")))?,
    ))
}

fn string_row(values: Vec<String>) -> BuiltinResult<Value> {
    let len = values.len();
    Ok(Value::StringArray(
        StringArray::new(values, vec![1, len])
            .map_err(|err| fitlm_internal(format!("fitlm: string row error: {err}")))?,
    ))
}

fn string_column(values: Vec<String>) -> BuiltinResult<Value> {
    let len = values.len();
    Ok(Value::StringArray(
        StringArray::new(values, vec![len, 1])
            .map_err(|err| fitlm_internal(format!("fitlm: string column error: {err}")))?,
    ))
}

fn has_intercept(terms: &[Vec<u32>]) -> bool {
    terms
        .iter()
        .any(|term| term.iter().all(|power| *power == 0))
}

#[cfg(test)]
mod tests;
