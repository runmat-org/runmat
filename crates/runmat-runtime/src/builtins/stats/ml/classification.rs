//! Discriminant classification compatibility surface.

mod coefficients;
mod header;
mod labels;
mod model;
mod prediction;
mod prior;

use self::{coefficients::*, labels::*, model::*, prediction::*};

use nalgebra::DMatrix;
use runmat_builtins::{
    BuiltinErrorDescriptor, DiscriminantType, CLASSIFY_ERROR_INTERNAL,
    CLASSIFY_ERROR_INVALID_ARGUMENT, CLASSIFY_ERROR_NUMERICAL, CLASSIFY_INTEGER_GROUP_EXTENSION,
    CLASSIFY_INTEGER_PRIOR_EXTENSION, CLASSIFY_INTEGER_SAMPLE_EXTENSION,
    CLASSIFY_INTEGER_TRAINING_EXTENSION, CLASSIFY_LOGICAL_PREDICTOR_EXTENSION,
    CLASSIFY_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericScalar, Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const CLASSIFY_NAME: &str = "classify";
const EPS: f64 = 1.0e-10;
const REGULARIZATION: f64 = 1.0e-9;

fn classify_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(CLASSIFY_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    classify_error(message, &CLASSIFY_ERROR_INVALID_ARGUMENT)
}

fn numerical_error(message: impl Into<String>) -> RuntimeError {
    classify_error(message, &CLASSIFY_ERROR_NUMERICAL)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    classify_error(message, &CLASSIFY_ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelKind {
    Numeric(NumericDType),
    String,
    Char,
    Cell,
    Logical,
}

#[derive(Clone, Debug, PartialEq)]
enum ClassLabel {
    Numeric(NumericScalar),
    Text(runmat_value::StringElement),
    Logical(bool),
}

#[derive(Clone, Debug)]
struct LabelVector {
    labels: Vec<ClassLabel>,
    kind: LabelKind,
}

#[derive(Clone, Debug)]
struct FloatingMatrix {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug)]
struct ClassStats {
    label: ClassLabel,
    rows: Vec<usize>,
    mean: Vec<f64>,
    covariance: DMatrix<f64>,
    inverse: DMatrix<f64>,
    log_det: f64,
}

#[derive(Clone, Debug)]
struct PreparedModel {
    kind: LabelKind,
    discriminant: DiscriminantType,
    classes: Vec<ClassStats>,
    priors: Vec<f64>,
    shared_inverse: Option<DMatrix<f64>>,
    shared_log_det: Option<f64>,
    cols: usize,
}

#[derive(Clone, Debug)]
struct PredictionResult {
    labels: Vec<usize>,
    posterior: Option<Tensor>,
    logp: Option<Tensor>,
}

#[runtime_builtin(
    name = "classify",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::classification"
)]
async fn classify_builtin(
    sample: Value,
    training: Value,
    group: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if matches!(requested_outputs, Some(count) if count > 5) {
        return Err(invalid_argument("classify: too many output arguments"));
    }
    if rest.len() > 2 {
        return Err(invalid_argument(
            "classify: accepts at most type and prior arguments",
        ));
    }
    ensure_compatibility_extensions(&sample, &training, &group, rest.view())?;
    let admitted = header::admit(&sample, &training, group, rest.view())?;
    if requested_outputs == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    let sample = gathered(sample).await?;
    let training = gathered(training).await?;
    let group = admitted.group.materialize().await?;
    let prior = admitted.prior.materialize().await?;
    let output_limit = requested_outputs.unwrap_or(1).max(1);
    let result = classify_compute(
        sample,
        training,
        group,
        admitted.discriminant,
        prior,
        output_limit,
    )?;
    match requested_outputs {
        Some(1) => Ok(Value::output_list(vec![result[0].clone()])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count, result,
        )),
        None => Ok(result[0].clone()),
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn prior_struct_field<'a>(value: &'a Value, lower: &str, upper: &str) -> Option<&'a Value> {
    let Value::Struct(prior) = value else {
        return None;
    };
    prior.fields.get(lower).or_else(|| prior.fields.get(upper))
}

fn ensure_compatibility_extensions(
    sample: &Value,
    training: &Value,
    group: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for (value, extension) in [
        (sample, &CLASSIFY_INTEGER_SAMPLE_EXTENSION),
        (training, &CLASSIFY_INTEGER_TRAINING_EXTENSION),
        (group, &CLASSIFY_INTEGER_GROUP_EXTENSION),
    ] {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(extension, CLASSIFY_NAME)?;
        }
    }
    if is_logical_value(sample) || is_logical_value(training) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLASSIFY_LOGICAL_PREDICTOR_EXTENSION,
            CLASSIFY_NAME,
        )?;
    }
    if let Some(prior) = rest.get(1) {
        if is_typed_integer_value(prior)
            || prior_struct_field(prior, "prob", "Prob").is_some_and(is_typed_integer_value)
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CLASSIFY_INTEGER_PRIOR_EXTENSION,
                CLASSIFY_NAME,
            )?;
        }
        if prior_struct_field(prior, "group", "Group").is_some_and(is_typed_integer_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CLASSIFY_INTEGER_GROUP_EXTENSION,
                CLASSIFY_NAME,
            )?;
        }
    }
    if [sample, training, group]
        .into_iter()
        .chain(rest.iter())
        .any(crate::dispatcher::value_contains_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CLASSIFY_RESIDENT_INPUT_EXTENSION,
            CLASSIFY_NAME,
        )?;
    }
    Ok(())
}

async fn gathered(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await.map_err(|cause| {
        let mut error = invalid_argument(format!("classify: {cause}"));
        error.gpu_gather_retry = cause.gpu_gather_retry;
        error.span = cause.span;
        error.source = Some(Box::new(cause));
        error
    })
}

fn classify_compute(
    sample: Value,
    training: Value,
    group: LabelVector,
    discriminant: DiscriminantType,
    prior: prior::PriorSpec,
    output_limit: usize,
) -> BuiltinResult<Vec<Value>> {
    let sample = numeric_matrix(sample, "sample")?;
    let training = numeric_matrix(training, "training")?;
    if sample.cols != training.cols {
        return Err(invalid_argument(
            "classify: sample and training must have the same number of columns",
        ));
    }
    if sample.cols == 0 || training.rows == 0 {
        return Err(invalid_argument(
            "classify: sample and training must be nonempty numeric matrices",
        ));
    }
    if sample.values.iter().any(|value| !value.is_finite())
        || training.values.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_argument(
            "classify: sample and training must contain finite values",
        ));
    }
    if group.labels.len() != training.rows {
        return Err(invalid_argument(
            "classify: group length must match the number of rows in training",
        ));
    }
    let prepared = prepare_model(&training, group, discriminant, prior)?;
    let predictions = predict_rows(&prepared, &sample, output_limit >= 3, output_limit >= 4)?;
    let class = labels_value(&prepared.classes, prepared.kind, &predictions.labels)?;
    if output_limit == 1 {
        return Ok(vec![class]);
    }
    let training_predictions = predict_rows(&prepared, &training, false, false)?;
    let err = apparent_error(&prepared, &training_predictions.labels);
    let mut outputs = vec![class, Value::Num(err)];
    if output_limit >= 3 {
        outputs.push(Value::Tensor(predictions.posterior.ok_or_else(|| {
            internal_error("classify: posterior output was not materialized")
        })?));
    }
    if output_limit >= 4 {
        outputs.push(Value::Tensor(predictions.logp.ok_or_else(|| {
            internal_error("classify: log-density output was not materialized")
        })?));
    }
    if output_limit >= 5 {
        outputs.push(coeff_value(&prepared)?);
    }
    Ok(outputs)
}

fn numeric_matrix(value: Value, name: &str) -> BuiltinResult<FloatingMatrix> {
    let tensor = tensor::value_into_tensor_for(CLASSIFY_NAME, value)
        .map_err(|err| invalid_argument(format!("classify: {name}: {err}")))?;
    if tensor.shape().len() > 2 {
        return Err(invalid_argument(format!(
            "classify: {name} must be a 2-D numeric matrix"
        )));
    }
    ensure_exact_integer_tensor_boundary(&tensor, name)?;
    let (rows, cols) = classification_matrix_dimensions(&tensor);
    Ok(FloatingMatrix {
        values: tensor.materialize_f64(),
        rows,
        cols,
    })
}

/// Matrix geometry used by the classifier's column-major numerical core.
///
/// A rank-one dense value is one row with `n` predictors, while rank zero has
/// no matrix geometry. The predictor header applies its own separate shape
/// projection before this numerical matrix is materialized.
fn classification_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}

fn parse_type(value: &Value) -> BuiltinResult<DiscriminantType> {
    let text = scalar_text(value, "type")?;
    DiscriminantType::from_public(&text).ok_or_else(|| {
        invalid_argument(format!(
            "classify: unsupported discriminant type '{}'",
            canonical(&text)
        ))
    })
}

#[cfg(test)]
#[path = "classification/tests.rs"]
mod tests;
