//! Linear classification model compatibility surface.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, FitclinearOption, FitclinearSolver,
    FITCLINEAR_ERROR_INTERNAL as ERROR_FITCLINEAR_INTERNAL,
    FITCLINEAR_ERROR_INVALID_ARGUMENT as ERROR_FITCLINEAR_INVALID_ARGUMENT,
    FITCLINEAR_INTEGER_NUMERIC_ARGUMENT_EXTENSION as INTEGER_NUMERIC_ARGUMENT_EXTENSION,
    FITCLINEAR_INTEGER_PREDICTOR_EXTENSION as INTEGER_PREDICTOR_EXTENSION,
    FITCLINEAR_RESIDENT_INPUT_EXTENSION as RESIDENT_INPUT_EXTENSION,
    PREDICT_ERROR_INTERNAL as ERROR_PREDICT_INTERNAL,
    PREDICT_ERROR_INVALID_ARGUMENT as ERROR_PREDICT_INVALID_ARGUMENT,
};
use runmat_macros::runtime_builtin;
use runmat_types::IntegerClass;
use runmat_value::{
    CharArray, IntValue, IntegerStorage, LogicalArray, ObjectInstance, StringArray, StringElement,
    Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::builtins::table::{
    is_tabular_object, table_variable_names_from_object, table_variables,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const FITCLINEAR_NAME: &str = "fitclinear";
const PREDICT_NAME: &str = "predict";
const DEFAULT_ITERATIONS: usize = 500;
const EPS: f64 = 1.0e-12;

mod training;
use training::fit_models;
mod header;
use header::{
    preflight_matrix_rank, preflight_response_shape, FitclinearArguments, HeaderInspection,
    NamedOption, PreparedOption,
};
mod admission;
mod options;
mod prediction;
mod projection;
use admission::{build_fit_spec, classification_matrix_dimensions, is_table_value};
use options::{
    class_priors, decode_fit_option, parse_fit_options, parse_formula, FitclinearAction,
};
pub(crate) use prediction::predict_classification_linear_object;
use projection::fit_info_value;
mod labels;
mod values;
use labels::{
    is_missing_label, label_kind_name, label_vector, labels_from_value, labels_value,
    numeric_vector_property, predicted_labels_value, same_label, score_transform,
    score_transform_property, string_array_property, tensor_property, unique_labels,
    validate_class_names,
};
use values::{
    beta_value, char_rows, columns_to_tensor, columns_to_tensor_predict, dot_row,
    ensure_integer_tensor_exact, ensure_integer_tensor_exact_predict, logical_scalar,
    numeric_scalar, numeric_vector, positive_integer, predict_scalar_text, row_tensor,
    rows_to_tensor, scalar_text, sigmoid, soft_threshold, text_list, vector_values,
    vector_values_predict, x_value,
};

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
    Integer(IntValue),
    Text(StringElement),
    Logical(bool),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelKind {
    Numeric,
    Integer(IntegerClass),
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
    solver: FitclinearSolver,
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
            solver: FitclinearSolver::Auto,
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
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::classification_linear"
)]
pub(crate) async fn fitclinear_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let inspection = HeaderInspection::inspect(&first, rest.view())?;
    ensure_fitclinear_extensions(&first, rest.view(), &inspection)?;
    preflight_matrix_rank(&first)?;
    let arguments = inspection
        .validate(rest.len())?
        .preflight_values(rest.view())?;
    preflight_response_shape(&first, rest.view(), &arguments)?;
    let first = gather(first).await?;
    let rest = gather_all(rest, &arguments, runtime.as_ref()).await?;
    let spec = build_fit_spec(first, rest.view(), arguments)?;
    let result = fit_models(runtime.as_ref(), &spec)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![Value::Object(result.object)])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![Value::Object(result.object), result.fit_info],
        )),
        None => Ok(Value::Object(result.object)),
    }
}

fn enable_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, FITCLINEAR_NAME)
}

fn typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn ensure_fitclinear_extensions(
    first: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    inspection: &HeaderInspection,
) -> BuiltinResult<()> {
    if crate::dispatcher::value_contains_gpu(first)
        || rest.iter().any(crate::dispatcher::value_contains_gpu)
    {
        enable_extension(&RESIDENT_INPUT_EXTENSION)?;
    }
    if is_table_value(first) {
        ensure_table_integer_extensions(first, rest, inspection)?;
    } else if typed_integer_value(first) {
        enable_extension(&INTEGER_PREDICTOR_EXTENSION)?;
    }
    for (index, identity) in inspection.known_options() {
        if rest.get(index + 1).is_some_and(typed_integer_value)
            && identity != FitclinearOption::ClassNames
        {
            enable_extension(&INTEGER_NUMERIC_ARGUMENT_EXTENSION)?;
        }
    }
    Ok(())
}

fn ensure_table_integer_extensions(
    first: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    inspection: &HeaderInspection,
) -> BuiltinResult<()> {
    let Value::Object(object) = first else {
        return Ok(());
    };
    let names = table_variable_names_from_object(object)
        .map_err(|err| fitclinear_invalid(format!("fitclinear: {err}")))?;
    let variables =
        table_variables(object).map_err(|err| fitclinear_invalid(format!("fitclinear: {err}")))?;
    let mut response = names.last().cloned().unwrap_or_default();
    let mut predictors: Option<Vec<String>> = None;
    let mut external_y = false;
    if let Some(arg) = rest.first().filter(|_| inspection.option_start == 1) {
        if let Some(text) = scalar_text(arg)? {
            if let Some((lhs, rhs)) = text.split_once('~') {
                response = lhs.trim().to_string();
                predictors = Some(
                    rhs.split('+')
                        .map(str::trim)
                        .filter(|term| !term.is_empty() && *term != "1")
                        .map(str::to_string)
                        .collect(),
                );
            } else {
                response = text;
            }
        } else {
            external_y = true;
        }
    }
    let mut weight_variable = None;
    for (index, identity) in inspection.known_options() {
        let Some(value) = rest.get(index + 1) else {
            continue;
        };
        match identity {
            FitclinearOption::ResponseName if !external_y => {
                if let Some(value) = scalar_text(value)? {
                    response = value;
                }
            }
            FitclinearOption::PredictorNames => {
                predictors = Some(text_list(value, "PredictorNames")?)
            }
            FitclinearOption::Weights => weight_variable = scalar_text(value)?,
            _ => {}
        }
    }
    let predictors = predictors.unwrap_or_else(|| {
        names
            .iter()
            .filter(|name| {
                (external_y || **name != response)
                    && weight_variable.as_deref() != Some(name.as_str())
            })
            .cloned()
            .collect()
    });
    if predictors
        .iter()
        .any(|name| variables.fields.get(name).is_some_and(typed_integer_value))
    {
        enable_extension(&INTEGER_PREDICTOR_EXTENSION)?;
    }
    if weight_variable
        .as_ref()
        .is_some_and(|name| variables.fields.get(name).is_some_and(typed_integer_value))
    {
        enable_extension(&INTEGER_NUMERIC_ARGUMENT_EXTENSION)?;
    }
    Ok(())
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await
}

async fn gather_all(
    values: crate::call::BuiltinArguments,
    arguments: &FitclinearArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    if !values.iter().any(crate::dispatcher::value_contains_gpu) {
        return Ok(values);
    }
    values
        .try_map_preserving_roots_async(values.len(), runtime, async |index| {
            let is_option_name = index >= arguments.option_start
                && (index - arguments.option_start).is_multiple_of(2);
            let is_ignored_value = arguments
                .options
                .iter()
                .any(|option| option.name.index + 1 == index && option.name.identity.ignored());
            if is_option_name || is_ignored_value {
                Ok(values[index].clone())
            } else {
                gather(values[index].clone()).await
            }
        })
        .await
}

fn numeric_matrix_for_fit(value: Value) -> BuiltinResult<Tensor> {
    let tensor = match value {
        Value::SparseTensor(sparse) => sparse
            .to_dense()
            .map_err(|err| fitclinear_invalid(format!("fitclinear: {err}"))),
        other => tensor::value_into_tensor_for(FITCLINEAR_NAME, other)
            .map_err(|err| fitclinear_invalid(format!("fitclinear: {err}"))),
    }?;
    ensure_integer_tensor_exact(&tensor, "predictors")?;
    Ok(tensor)
}

fn numeric_matrix_for_predict(value: Value) -> BuiltinResult<Tensor> {
    let tensor = match value {
        Value::SparseTensor(sparse) => sparse
            .to_dense()
            .map_err(|err| predict_invalid(format!("predict: {err}"))),
        other => tensor::value_into_tensor_for(PREDICT_NAME, other)
            .map_err(|err| predict_invalid(format!("predict: {err}"))),
    }?;
    ensure_integer_tensor_exact_predict(&tensor, "predictors")?;
    Ok(tensor)
}

fn normalize_observation_matrix(x: Tensor, options: &FitOptions) -> BuiltinResult<Tensor> {
    normalize_prediction_matrix(x, options.observations_in)
        .map_err(|err| fitclinear_invalid(err.message))
}

fn normalize_prediction_matrix(
    x: Tensor,
    observations_in: ObservationsIn,
) -> BuiltinResult<Tensor> {
    if x.shape().len() > 2 {
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
    let (rows, columns) = classification_matrix_dimensions(x);
    let mut data = vec![0.0; rows * columns];
    for row in 0..rows {
        for col in 0..columns {
            data[col + row * columns] = x_value(x, row, col);
        }
    }
    Tensor::new(data, vec![columns, rows])
}

#[cfg(test)]
mod tests;
