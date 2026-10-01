//! Discriminant classification compatibility surface.

mod header;

use std::cmp::Ordering;

use nalgebra::{DMatrix, DVector};
use runmat_builtins::{
    BuiltinErrorDescriptor, DiscriminantType, CLASSIFY_ERROR_INTERNAL,
    CLASSIFY_ERROR_INVALID_ARGUMENT, CLASSIFY_ERROR_NUMERICAL, CLASSIFY_INTEGER_GROUP_EXTENSION,
    CLASSIFY_INTEGER_PRIOR_EXTENSION, CLASSIFY_INTEGER_SAMPLE_EXTENSION,
    CLASSIFY_INTEGER_TRAINING_EXTENSION, CLASSIFY_LOGICAL_PREDICTOR_EXTENSION,
    CLASSIFY_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, LogicalArray, NumericDType, NumericScalar, NumericStorage, StringArray,
    StructValue, Tensor, Value,
};

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
    Text(String),
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
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if let Some(0) = requested_outputs {
        return Ok(Value::OutputList(Vec::new()));
    }
    if matches!(requested_outputs, Some(count) if count > 5) {
        return Err(invalid_argument("classify: too many output arguments"));
    }
    if rest.len() > 2 {
        return Err(invalid_argument(
            "classify: accepts at most type and prior arguments",
        ));
    }
    ensure_compatibility_extensions(&sample, &training, &group, &rest)?;
    let discriminant = header::admit(&sample, &training, &group, &rest)?;
    let sample = gathered(sample).await?;
    let training = gathered(training).await?;
    let group = gathered(group).await?;
    let prior = match rest.into_iter().nth(1) {
        Some(value) => Some(gathered(value).await?),
        None => None,
    };
    let output_limit = requested_outputs.unwrap_or(1).max(1);
    let result = classify_compute(
        sample,
        training,
        group,
        discriminant,
        prior.as_ref(),
        output_limit,
    )?;
    match requested_outputs {
        Some(1) => Ok(Value::OutputList(vec![result[0].clone()])),
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
    rest: &[Value],
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
    group: Value,
    discriminant: DiscriminantType,
    prior: Option<&Value>,
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
    let labels = labels_from_value(group, "group")?;
    if labels.labels.len() != training.rows {
        return Err(invalid_argument(
            "classify: group length must match the number of rows in training",
        ));
    }
    let prepared = prepare_model(&training, labels, discriminant, prior)?;
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
    if tensor.shape.len() > 2 {
        return Err(invalid_argument(format!(
            "classify: {name} must be a 2-D numeric matrix"
        )));
    }
    ensure_exact_integer_tensor_boundary(&tensor, name)?;
    Ok(FloatingMatrix {
        values: tensor.materialize_f64(),
        rows: tensor.rows,
        cols: tensor.cols,
    })
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

fn prepare_model(
    training: &FloatingMatrix,
    labels: LabelVector,
    discriminant: DiscriminantType,
    prior_value: Option<&Value>,
) -> BuiltinResult<PreparedModel> {
    let unique = unique_labels(&labels.labels, labels.kind);
    if unique.len() < 2 {
        return Err(invalid_argument(
            "classify: group must contain at least two nonmissing classes",
        ));
    }
    let mut rows_by_class = vec![Vec::<usize>::new(); unique.len()];
    for (row, label) in labels.labels.iter().enumerate() {
        if is_missing_label(label) {
            continue;
        }
        let Some(class_index) = unique.iter().position(|class| same_label(class, label)) else {
            continue;
        };
        rows_by_class[class_index].push(row);
    }
    if rows_by_class.iter().any(Vec::is_empty) {
        return Err(invalid_argument(
            "classify: every class must contain at least one training row",
        ));
    }
    let priors = parse_prior(prior_value, &unique, labels.kind, &rows_by_class)?;
    let cols = training.cols;
    let mut classes = Vec::with_capacity(unique.len());
    for (class_index, rows) in rows_by_class.iter().enumerate() {
        let mean = class_mean(training, rows);
        let mut covariance = class_covariance(training, rows, &mean)?;
        if discriminant.uses_diagonal_covariance() {
            covariance = diagonalized(&covariance);
        }
        let (inverse, log_det) = invert_covariance(&covariance, "class covariance")?;
        classes.push(ClassStats {
            label: unique[class_index].clone(),
            rows: rows.clone(),
            mean,
            covariance,
            inverse,
            log_det,
        });
    }
    let (shared_inverse, shared_log_det) = if discriminant.uses_shared_covariance() {
        let mut pooled = pooled_covariance(&classes, cols)?;
        if discriminant.uses_diagonal_covariance() {
            pooled = diagonalized(&pooled);
        }
        let (inverse, log_det) = invert_covariance(&pooled, "pooled covariance")?;
        (Some(inverse), Some(log_det))
    } else {
        (None, None)
    };
    Ok(PreparedModel {
        kind: labels.kind,
        discriminant,
        classes,
        priors,
        shared_inverse,
        shared_log_det,
        cols,
    })
}

fn class_mean(training: &FloatingMatrix, rows: &[usize]) -> Vec<f64> {
    let mut mean = vec![0.0; training.cols];
    for row in rows {
        for col in 0..training.cols {
            mean[col] += training.values[row + col * training.rows];
        }
    }
    for value in &mut mean {
        *value /= rows.len() as f64;
    }
    mean
}

fn class_covariance(
    training: &FloatingMatrix,
    rows: &[usize],
    mean: &[f64],
) -> BuiltinResult<DMatrix<f64>> {
    let cols = training.cols;
    let mut cov = DMatrix::<f64>::zeros(cols, cols);
    if rows.len() < 2 {
        for idx in 0..cols {
            cov[(idx, idx)] = REGULARIZATION;
        }
        return Ok(cov);
    }
    for row in rows {
        for a in 0..cols {
            let da = training.values[row + a * training.rows] - mean[a];
            for b in 0..cols {
                let db = training.values[row + b * training.rows] - mean[b];
                cov[(a, b)] += da * db;
            }
        }
    }
    let denom = (rows.len() - 1) as f64;
    for value in cov.iter_mut() {
        *value /= denom;
    }
    Ok(cov)
}

fn pooled_covariance(classes: &[ClassStats], cols: usize) -> BuiltinResult<DMatrix<f64>> {
    let total_rows = classes.iter().map(|class| class.rows.len()).sum::<usize>();
    if total_rows <= classes.len() {
        let mut identity = DMatrix::<f64>::zeros(cols, cols);
        for idx in 0..cols {
            identity[(idx, idx)] = REGULARIZATION;
        }
        return Ok(identity);
    }
    let mut pooled = DMatrix::<f64>::zeros(cols, cols);
    for class in classes {
        let weight = class.rows.len().saturating_sub(1) as f64;
        pooled += class.covariance.clone() * weight;
    }
    pooled /= (total_rows - classes.len()) as f64;
    Ok(pooled)
}

fn diagonalized(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let mut out = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());
    for idx in 0..matrix.nrows().min(matrix.ncols()) {
        out[(idx, idx)] = matrix[(idx, idx)];
    }
    out
}

fn invert_covariance(
    matrix: &DMatrix<f64>,
    label: &'static str,
) -> BuiltinResult<(DMatrix<f64>, f64)> {
    let mut adjusted = matrix.clone();
    for idx in 0..adjusted.nrows().min(adjusted.ncols()) {
        adjusted[(idx, idx)] += REGULARIZATION;
    }
    let lu = adjusted.lu();
    let det = lu.determinant();
    if !det.is_finite() || det.abs() <= EPS {
        return Err(numerical_error(format!("classify: singular {label}")));
    }
    let Some(inverse) = lu.try_inverse() else {
        return Err(numerical_error(format!("classify: singular {label}")));
    };
    Ok((inverse, det.abs().ln()))
}

fn predict_rows(
    model: &PreparedModel,
    data: &FloatingMatrix,
    need_posterior: bool,
    need_logp: bool,
) -> BuiltinResult<PredictionResult> {
    let rows = data.rows;
    let class_count = model.classes.len();
    let mut label_indices = Vec::new();
    label_indices
        .try_reserve_exact(rows)
        .map_err(|err| internal_error(format!("classify: cannot allocate labels: {err}")))?;
    let mut posterior_data = if need_posterior {
        let count = rows
            .checked_mul(class_count)
            .ok_or_else(|| internal_error("classify: posterior shape overflows"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|err| internal_error(format!("classify: cannot allocate posterior: {err}")))?;
        values.resize(count, 0.0);
        Some(values)
    } else {
        None
    };
    let mut logp_data = if need_logp {
        let mut values = Vec::new();
        values.try_reserve_exact(rows).map_err(|err| {
            internal_error(format!("classify: cannot allocate log density: {err}"))
        })?;
        values.resize(rows, 0.0);
        Some(values)
    } else {
        None
    };
    for row in 0..rows {
        let x = row_vector(data, row);
        let mut scores = Vec::new();
        scores.try_reserve_exact(class_count).map_err(|err| {
            internal_error(format!("classify: cannot allocate class scores: {err}"))
        })?;
        for class_index in 0..class_count {
            scores.push(discriminant_score(model, class_index, &x)?);
        }
        let best = scores
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| internal_error("classify: no classes"))?;
        label_indices.push(best);
        if model.discriminant == DiscriminantType::Mahalanobis {
            if let Some(posterior) = &mut posterior_data {
                for class_index in 0..class_count {
                    posterior[row + class_index * rows] = f64::NAN;
                }
            }
            if let Some(logp) = &mut logp_data {
                logp[row] = f64::NAN;
            }
        } else if need_posterior || need_logp {
            let logp = log_sum_exp(&scores);
            if let Some(values) = &mut logp_data {
                values[row] = logp;
            }
            if let Some(posterior) = &mut posterior_data {
                for class_index in 0..class_count {
                    posterior[row + class_index * rows] = (scores[class_index] - logp).exp();
                }
            }
        }
    }
    Ok(PredictionResult {
        labels: label_indices,
        posterior: posterior_data
            .map(|values| {
                Tensor::new(values, vec![rows, class_count])
                    .map_err(|err| internal_error(format!("classify: {err}")))
            })
            .transpose()?,
        logp: logp_data
            .map(|values| {
                Tensor::new(values, vec![rows, 1])
                    .map_err(|err| internal_error(format!("classify: {err}")))
            })
            .transpose()?,
    })
}

fn row_vector(data: &FloatingMatrix, row: usize) -> DVector<f64> {
    DVector::from_iterator(
        data.cols,
        (0..data.cols).map(|col| data.values[row + col * data.rows]),
    )
}

fn mean_vector(mean: &[f64]) -> DVector<f64> {
    DVector::from_iterator(mean.len(), mean.iter().copied())
}

fn discriminant_score(
    model: &PreparedModel,
    class_index: usize,
    x: &DVector<f64>,
) -> BuiltinResult<f64> {
    let class = &model.classes[class_index];
    let mean = mean_vector(&class.mean);
    let diff = x - &mean;
    let (inverse, log_det) = if let Some(inverse) = &model.shared_inverse {
        (inverse, model.shared_log_det.unwrap_or(0.0))
    } else {
        (&class.inverse, class.log_det)
    };
    let distance = (diff.transpose() * inverse * diff)[(0, 0)];
    if model.discriminant == DiscriminantType::Mahalanobis {
        Ok(-distance)
    } else {
        let gaussian_norm = -0.5 * (model.cols as f64) * (2.0 * std::f64::consts::PI).ln();
        Ok(model.priors[class_index].ln() + gaussian_norm - 0.5 * log_det - 0.5 * distance)
    }
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));
    max + values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f64>()
        .ln()
}

fn apparent_error(model: &PreparedModel, predicted: &[usize]) -> f64 {
    let mut total = 0.0;
    for (class_index, class) in model.classes.iter().enumerate() {
        let mut wrong = 0usize;
        for row in &class.rows {
            if predicted.get(*row).copied() != Some(class_index) {
                wrong += 1;
            }
        }
        let class_error = if class.rows.is_empty() {
            0.0
        } else {
            wrong as f64 / class.rows.len() as f64
        };
        total += model.priors[class_index] * class_error;
    }
    total
}

fn parse_prior(
    prior: Option<&Value>,
    labels: &[ClassLabel],
    kind: LabelKind,
    rows_by_class: &[Vec<usize>],
) -> BuiltinResult<Vec<f64>> {
    let raw = match prior {
        None => vec![1.0; labels.len()],
        Some(value) if string_matches(value, "empirical") => rows_by_class
            .iter()
            .map(|rows| rows.len() as f64)
            .collect::<Vec<_>>(),
        Some(Value::Struct(st)) => prior_from_struct(st, labels, kind)?,
        Some(value) => numeric_vector(value, "prior")?,
    };
    if raw.len() != labels.len() {
        return Err(invalid_argument(
            "classify: prior must contain one probability per class",
        ));
    }
    if raw.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(invalid_argument(
            "classify: prior probabilities must be finite and nonnegative",
        ));
    }
    let total = raw.iter().sum::<f64>();
    if total <= 0.0 {
        return Err(invalid_argument(
            "classify: prior probabilities must have positive total",
        ));
    }
    Ok(raw.into_iter().map(|value| value / total).collect())
}

fn prior_from_struct(
    st: &StructValue,
    labels: &[ClassLabel],
    kind: LabelKind,
) -> BuiltinResult<Vec<f64>> {
    let group_value = st
        .fields
        .get("group")
        .or_else(|| st.fields.get("Group"))
        .ok_or_else(|| invalid_argument("classify: prior struct must contain group field"))?
        .clone();
    let prob_value = st
        .fields
        .get("prob")
        .or_else(|| st.fields.get("Prob"))
        .ok_or_else(|| invalid_argument("classify: prior struct must contain prob field"))?;
    let groups = labels_from_value(group_value, "prior.group")?;
    if !label_kinds_compatible(groups.kind, kind) {
        return Err(invalid_argument(
            "classify: prior group labels must match group label type",
        ));
    }
    let probs = numeric_vector(prob_value, "prior.prob")?;
    if groups.labels.len() != probs.len() {
        return Err(invalid_argument(
            "classify: prior group and prob lengths must match",
        ));
    }
    let mut out = vec![0.0; labels.len()];
    for (group, prob) in groups.labels.iter().zip(probs.iter()) {
        let Some(index) = labels.iter().position(|label| same_label(label, group)) else {
            return Err(invalid_argument(
                "classify: prior struct contains unknown group label",
            ));
        };
        out[index] = *prob;
    }
    Ok(out)
}

fn coeff_value(model: &PreparedModel) -> BuiltinResult<Value> {
    let k = model.classes.len();
    let entry_count = k
        .checked_mul(k)
        .ok_or_else(|| internal_error("classify: coefficient cell shape overflows"))?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entry_count)
        .map_err(|err| internal_error(format!("classify: cannot allocate coefficients: {err}")))?;
    for row in 0..k {
        for col in 0..k {
            let coeff = pairwise_coeff(model, row, col)?;
            let mut st = StructValue::new();
            st.insert(
                "type",
                Value::String(model.discriminant.property_name().to_string()),
            );
            st.insert("name1", labels_value(&model.classes, model.kind, &[row])?);
            st.insert("name2", labels_value(&model.classes, model.kind, &[col])?);
            st.insert("const", Value::Num(coeff.0));
            st.insert(
                "linear",
                Value::Tensor(
                    Tensor::new(coeff.1, vec![model.cols, 1])
                        .map_err(|err| internal_error(format!("classify: {err}")))?,
                ),
            );
            if !matches!(
                model.discriminant,
                DiscriminantType::Linear | DiscriminantType::DiagLinear
            ) {
                st.insert(
                    "quadratic",
                    Value::Tensor(
                        Tensor::new(coeff.2, vec![model.cols, model.cols])
                            .map_err(|err| internal_error(format!("classify: {err}")))?,
                    ),
                );
            }
            entries.push(Value::Struct(st));
        }
    }
    Ok(Value::Cell(CellArray::new(entries, k, k).map_err(
        |err| internal_error(format!("classify: {err}")),
    )?))
}

fn pairwise_coeff(
    model: &PreparedModel,
    i: usize,
    j: usize,
) -> BuiltinResult<(f64, Vec<f64>, Vec<f64>)> {
    let quadratic_len = model
        .cols
        .checked_mul(model.cols)
        .ok_or_else(|| internal_error("classify: quadratic coefficient shape overflows"))?;
    if i == j {
        let mut quadratic = Vec::new();
        quadratic.try_reserve_exact(quadratic_len).map_err(|err| {
            internal_error(format!(
                "classify: cannot allocate quadratic coefficients: {err}"
            ))
        })?;
        quadratic.resize(quadratic_len, 0.0);
        return Ok((0.0, vec![0.0; model.cols], quadratic));
    }
    let class_i = &model.classes[i];
    let class_j = &model.classes[j];
    let mean_i = mean_vector(&class_i.mean);
    let mean_j = mean_vector(&class_j.mean);
    let (inv_i, logdet_i) = if let Some(inv) = &model.shared_inverse {
        (inv, model.shared_log_det.unwrap_or(0.0))
    } else {
        (&class_i.inverse, class_i.log_det)
    };
    let (inv_j, logdet_j) = if let Some(inv) = &model.shared_inverse {
        (inv, model.shared_log_det.unwrap_or(0.0))
    } else {
        (&class_j.inverse, class_j.log_det)
    };
    let linear_vec = mean_i.transpose() * inv_i - mean_j.transpose() * inv_j;
    let const_term = if model.discriminant == DiscriminantType::Mahalanobis {
        -0.5 * ((mean_i.transpose() * inv_i * &mean_i)[(0, 0)]
            - (mean_j.transpose() * inv_j * &mean_j)[(0, 0)])
    } else {
        log_prior_ratio(model.priors[i], model.priors[j])
            - 0.5 * (logdet_i - logdet_j)
            - 0.5
                * ((mean_i.transpose() * inv_i * &mean_i)[(0, 0)]
                    - (mean_j.transpose() * inv_j * &mean_j)[(0, 0)])
    };
    let quadratic = if model.discriminant.uses_shared_covariance() {
        let mut values = Vec::new();
        values.try_reserve_exact(quadratic_len).map_err(|err| {
            internal_error(format!(
                "classify: cannot allocate quadratic coefficients: {err}"
            ))
        })?;
        values.resize(quadratic_len, 0.0);
        values
    } else {
        ((inv_j - inv_i) * 0.5).iter().copied().collect()
    };
    Ok((const_term, linear_vec.iter().copied().collect(), quadratic))
}

fn log_prior_ratio(left: f64, right: f64) -> f64 {
    match (left > 0.0, right > 0.0) {
        (true, true) => (left / right).ln(),
        (true, false) => f64::INFINITY,
        (false, true) => f64::NEG_INFINITY,
        (false, false) => 0.0,
    }
}

fn labels_from_value(value: Value, name: &str) -> BuiltinResult<LabelVector> {
    match value {
        Value::Tensor(tensor) => {
            let dtype = tensor.numeric_dtype();
            Ok(LabelVector {
                labels: vector_values(&tensor, name)?
                    .into_iter()
                    .map(ClassLabel::Numeric)
                    .collect(),
                kind: LabelKind::Numeric(dtype),
            })
        }
        Value::Num(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Numeric(NumericScalar::F64(value))],
            kind: LabelKind::Numeric(NumericDType::F64),
        }),
        Value::Int(value) => {
            let value = NumericScalar::from(value);
            Ok(LabelVector {
                labels: vec![ClassLabel::Numeric(value)],
                kind: LabelKind::Numeric(value.numeric_dtype()),
            })
        }
        Value::String(text) => Ok(LabelVector {
            labels: vec![ClassLabel::Text(text)],
            kind: LabelKind::String,
        }),
        Value::CharArray(array) => Ok(LabelVector {
            labels: char_rows(&array).into_iter().map(ClassLabel::Text).collect(),
            kind: LabelKind::Char,
        }),
        Value::StringArray(array) => Ok(LabelVector {
            labels: array.data.into_iter().map(ClassLabel::Text).collect(),
            kind: LabelKind::String,
        }),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for item in cell.data {
                labels.push(ClassLabel::Text(scalar_text(&item, name)?));
            }
            Ok(LabelVector {
                labels,
                kind: LabelKind::Cell,
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
        other => Err(invalid_argument(format!(
            "classify: {name} must be numeric, logical, string, character, or cellstr labels; got {other:?}"
        ))),
    }
}

fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<NumericScalar>> {
    if tensor.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(format!(
            "classify: {name} must be a vector"
        )));
    }
    (0..tensor.len())
        .map(|index| {
            tensor.numeric_value_at(index).ok_or_else(|| {
                invalid_argument(format!("classify: {name} numeric storage is invalid"))
            })
        })
        .collect()
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.shape.iter().filter(|dim| **dim > 1).count() > 1 {
                return Err(invalid_argument(format!(
                    "classify: {name} must be a vector"
                )));
            }
            ensure_exact_integer_tensor_boundary(tensor, name)?;
            Ok(tensor::tensor_values_f64(tensor))
        }
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => {
            ensure_exact_integer_scalar_boundary(NumericScalar::from(value.clone()), name)?;
            Ok(vec![value.to_f64()])
        }
        other => Err(invalid_argument(format!(
            "classify: {name} must be a numeric vector, got {other:?}"
        ))),
    }
}

fn ensure_exact_integer_tensor_boundary(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    for index in 0..tensor.len() {
        if let Some(value) = tensor.numeric_value_at(index) {
            ensure_exact_integer_scalar_boundary(value, name)?;
        }
    }
    Ok(())
}

fn ensure_exact_integer_scalar_boundary(value: NumericScalar, name: &str) -> BuiltinResult<()> {
    if value
        .into_int_value()
        .is_some_and(|integer| !crate::builtins::common::validation::integer_is_exact_f64(&integer))
    {
        return Err(invalid_argument(format!(
            "classify: {name} integer values must be exactly representable as double"
        )));
    }
    Ok(())
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
        LabelKind::Numeric(_) => out.sort_by(|left, right| match (left, right) {
            (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => {
                numeric_label_cmp(*a, *b).unwrap_or(Ordering::Equal)
            }
            _ => Ordering::Equal,
        }),
        LabelKind::String | LabelKind::Char | LabelKind::Cell => {
            out.sort_by(|left, right| match (left, right) {
                (ClassLabel::Text(a), ClassLabel::Text(b)) => a.cmp(b),
                _ => Ordering::Equal,
            })
        }
        LabelKind::Logical => out.sort_by_key(|label| match label {
            ClassLabel::Logical(value) => *value,
            _ => false,
        }),
    }
    out
}

fn numeric_label_cmp(left: NumericScalar, right: NumericScalar) -> Option<Ordering> {
    match (left, right) {
        (NumericScalar::F64(left), NumericScalar::F64(right)) => left.partial_cmp(&right),
        (NumericScalar::F32(left), NumericScalar::F32(right)) => left.partial_cmp(&right),
        (NumericScalar::I8(left), NumericScalar::I8(right)) => Some(left.cmp(&right)),
        (NumericScalar::I16(left), NumericScalar::I16(right)) => Some(left.cmp(&right)),
        (NumericScalar::I32(left), NumericScalar::I32(right)) => Some(left.cmp(&right)),
        (NumericScalar::I64(left), NumericScalar::I64(right)) => Some(left.cmp(&right)),
        (NumericScalar::U8(left), NumericScalar::U8(right)) => Some(left.cmp(&right)),
        (NumericScalar::U16(left), NumericScalar::U16(right)) => Some(left.cmp(&right)),
        (NumericScalar::U32(left), NumericScalar::U32(right)) => Some(left.cmp(&right)),
        (NumericScalar::U64(left), NumericScalar::U64(right)) => Some(left.cmp(&right)),
        _ => None,
    }
}

fn numeric_label_is_nan(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.is_nan(),
        NumericScalar::F32(value) => value.is_nan(),
        NumericScalar::I8(_)
        | NumericScalar::I16(_)
        | NumericScalar::I32(_)
        | NumericScalar::I64(_)
        | NumericScalar::U8(_)
        | NumericScalar::U16(_)
        | NumericScalar::U32(_)
        | NumericScalar::U64(_) => false,
    }
}

fn same_label(left: &ClassLabel, right: &ClassLabel) -> bool {
    match (left, right) {
        (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => {
            (a == b) || (numeric_label_is_nan(*a) && numeric_label_is_nan(*b))
        }
        (ClassLabel::Text(a), ClassLabel::Text(b)) => a == b,
        (ClassLabel::Logical(a), ClassLabel::Logical(b)) => a == b,
        _ => false,
    }
}

fn is_missing_label(label: &ClassLabel) -> bool {
    match label {
        ClassLabel::Numeric(value) => numeric_label_is_nan(*value),
        ClassLabel::Text(value) => value.is_empty() || value == "<missing>",
        ClassLabel::Logical(_) => false,
    }
}

fn label_kinds_compatible(left: LabelKind, right: LabelKind) -> bool {
    left == right || (is_text_kind(left) && is_text_kind(right))
}

fn is_text_kind(kind: LabelKind) -> bool {
    matches!(kind, LabelKind::String | LabelKind::Char | LabelKind::Cell)
}

fn labels_value(
    classes: &[ClassStats],
    kind: LabelKind,
    indices: &[usize],
) -> BuiltinResult<Value> {
    let labels = classes
        .iter()
        .map(|class| class.label.clone())
        .collect::<Vec<_>>();
    labels_from_indices(&labels, kind, indices)
}

fn labels_from_indices(
    labels: &[ClassLabel],
    kind: LabelKind,
    indices: &[usize],
) -> BuiltinResult<Value> {
    match kind {
        LabelKind::Numeric(dtype) => {
            let values = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Numeric(value)) => Ok(*value),
                    _ => Err(internal_error("classify: invalid numeric label index")),
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            let mut storage = NumericStorage::zeros(dtype, values.len());
            for (index, value) in values.into_iter().enumerate() {
                storage
                    .set_value(index, value)
                    .map_err(|err| internal_error(format!("classify: {err}")))?;
            }
            Ok(Value::Tensor(
                Tensor::from_numeric_storage(storage, vec![indices.len(), 1])
                    .map_err(|err| internal_error(format!("classify: {err}")))?,
            ))
        }
        LabelKind::String => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Text(value)) => value.clone(),
                    _ => String::new(),
                })
                .collect::<Vec<_>>();
            Ok(Value::StringArray(
                StringArray::new(data, vec![indices.len(), 1])
                    .map_err(|err| internal_error(format!("classify: {err}")))?,
            ))
        }
        LabelKind::Char => {
            let rows = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Text(value)) => value.clone(),
                    _ => String::new(),
                })
                .collect::<Vec<_>>();
            Ok(Value::CharArray(char_array_from_rows(&rows)?))
        }
        LabelKind::Cell => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Text(value)) => Value::CharArray(CharArray::new_row(value)),
                    _ => Value::CharArray(CharArray::new_row("")),
                })
                .collect::<Vec<_>>();
            Ok(Value::Cell(
                CellArray::new(data, indices.len(), 1)
                    .map_err(|err| internal_error(format!("classify: {err}")))?,
            ))
        }
        LabelKind::Logical => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Logical(value)) if *value => 1,
                    _ => 0,
                })
                .collect::<Vec<_>>();
            Ok(Value::LogicalArray(
                LogicalArray::new(data, vec![indices.len(), 1])
                    .map_err(|err| internal_error(format!("classify: {err}")))?,
            ))
        }
    }
}

fn char_rows(array: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        rows.push(array.data[start..start + array.cols].iter().collect());
    }
    rows
}

fn char_array_from_rows(rows: &[String]) -> BuiltinResult<CharArray> {
    let width = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(rows.len() * width);
    for row in rows {
        let mut chars = row.chars().collect::<Vec<_>>();
        chars.resize(width, ' ');
        data.extend(chars);
    }
    CharArray::new(data, rows.len(), width)
        .map_err(|err| internal_error(format!("classify: {err}")))
}

fn scalar_text(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(invalid_argument(format!(
            "classify: {name} must be a string scalar, got {other:?}"
        ))),
    }
}

fn string_matches(value: &Value, expected: &str) -> bool {
    scalar_text(value, "text")
        .map(|text| text.eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}

#[cfg(test)]
#[path = "classification/tests.rs"]
mod tests;
