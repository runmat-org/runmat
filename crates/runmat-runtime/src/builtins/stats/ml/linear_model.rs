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

mod training;
use training::{evaluate_term, fit_ols};
mod admission;
mod options;
mod prediction;
mod projection;
use admission::{build_fit_spec, is_table_value};
use options::{
    coefficient_names, formula_text, model_terms, parse_fit_name_values, parse_formula_predictors,
    parse_formula_terms, parse_model_spec, parse_predict_options, ModelSpec,
};
use prediction::predict_linear_model;
use projection::model_object;
mod values;
use values::{
    bool_scalar, columns_to_tensor, columns_to_tensor_predict,
    ensure_exact_fitlm_integer_scalar_boundary, ensure_exact_fitlm_integer_tensor_boundary,
    exclude_mask, exclude_spec, has_intercept, is_name_value_option, matrix_property,
    numeric_column, numeric_property, numeric_scalar, numeric_scalar_property, numeric_vector,
    scalar_text, string_column, string_property, string_row, terms_from_object, terms_tensor,
    text_list, vector_values, vector_values_predict, x_value,
};

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
    let first = gather(first).await?;
    let rest = gather_all(rest).await?;
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
    gather_if_needed_async(&value).await
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
