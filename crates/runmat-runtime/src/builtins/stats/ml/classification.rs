//! Discriminant classification compatibility surface.

use std::cmp::Ordering;

use nalgebra::{DMatrix, DVector};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, LogicalArray, ResolveContext, StringArray, StructValue, Tensor, Type,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const CLASSIFY_NAME: &str = "classify";
const EPS: f64 = 1.0e-10;
const REGULARIZATION: f64 = 1.0e-9;

const OUTPUT_CLASS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "class",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predicted class labels for sample observations.",
};

const OUTPUT_ERR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "err",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Apparent training-set error rate.",
};

const OUTPUT_POSTERIOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "posterior",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Posterior class probabilities for sample observations.",
};

const OUTPUT_LOGP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "logp",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Log unconditional density for sample observations.",
};

const OUTPUT_COEFF: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "coeff",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Pairwise discriminant boundary coefficients.",
};

const PARAM_SAMPLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sample",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample observations in rows.",
};

const PARAM_TRAINING: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "training",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Training observations in rows.",
};

const PARAM_GROUP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "group",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Training group labels.",
};

const PARAM_TYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "type",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("linear"),
    description: "Discriminant type: linear, quadratic, diagLinear, diagQuadratic, or mahalanobis.",
};

const PARAM_PRIOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "prior",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Prior probabilities as a numeric vector, 'empirical', or a struct with group and prob fields.",
};

const INPUTS_BASIC: [BuiltinParamDescriptor; 3] = [PARAM_SAMPLE, PARAM_TRAINING, PARAM_GROUP];
const INPUTS_FULL: [BuiltinParamDescriptor; 5] = [
    PARAM_SAMPLE,
    PARAM_TRAINING,
    PARAM_GROUP,
    PARAM_TYPE,
    PARAM_PRIOR,
];
const OUTPUT_CLASS_ONLY: [BuiltinParamDescriptor; 1] = [OUTPUT_CLASS];
const OUTPUT_ALL: [BuiltinParamDescriptor; 5] = [
    OUTPUT_CLASS,
    OUTPUT_ERR,
    OUTPUT_POSTERIOR,
    OUTPUT_LOGP,
    OUTPUT_COEFF,
];

const CLASSIFY_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "class = classify(sample, training, group)",
        inputs: &INPUTS_BASIC,
        outputs: &OUTPUT_CLASS_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "class = classify(sample, training, group, type, prior)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_CLASS_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[class,err,posterior,logp,coeff] = classify(___)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_ALL,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLASSIFY.INVALID_ARGUMENT",
    identifier: Some("RunMat:classify:InvalidArgument"),
    when: "Inputs, labels, dimensions, discriminant type, or priors are malformed.",
    message: "classify: invalid argument",
};

const ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLASSIFY.NUMERICAL",
    identifier: Some("RunMat:classify:Numerical"),
    when: "Covariance matrices cannot be solved numerically.",
    message: "classify: numerical failure",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLASSIFY.INTERNAL",
    identifier: Some("RunMat:classify:Internal"),
    when: "RunMat cannot construct classification outputs.",
    message: "classify: internal error",
};

const CLASSIFY_ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_NUMERICAL, ERROR_INTERNAL];

pub const CLASSIFY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLASSIFY_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLASSIFY_ERRORS,
};

fn classify_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

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
    classify_error(message, &ERROR_INVALID_ARGUMENT)
}

fn numerical_error(message: impl Into<String>) -> RuntimeError {
    classify_error(message, &ERROR_NUMERICAL)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    classify_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DiscriminantType {
    Linear,
    Quadratic,
    DiagLinear,
    DiagQuadratic,
    Mahalanobis,
}

impl DiscriminantType {
    fn property_name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
            Self::DiagLinear => "diagLinear",
            Self::DiagQuadratic => "diagQuadratic",
            Self::Mahalanobis => "mahalanobis",
        }
    }

    fn uses_shared_covariance(self) -> bool {
        matches!(self, Self::Linear | Self::DiagLinear)
    }

    fn uses_diagonal_covariance(self) -> bool {
        matches!(self, Self::DiagLinear | Self::DiagQuadratic)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelKind {
    Numeric,
    String,
    Char,
    Cell,
    Logical,
}

#[derive(Clone, Debug, PartialEq)]
enum ClassLabel {
    Numeric(f64),
    Text(String),
    Logical(bool),
}

#[derive(Clone, Debug)]
struct LabelVector {
    labels: Vec<ClassLabel>,
    kind: LabelKind,
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
    posterior: Tensor,
    logp: Tensor,
}

#[runtime_builtin(
    name = "classify",
    category = "stats/ml",
    summary = "Classify observations using discriminant analysis.",
    keywords = "classify,discriminant analysis,lda,qda,classification,statistics,machine learning",
    type_resolver(classify_type),
    descriptor(crate::builtins::stats::ml::classification::CLASSIFY_DESCRIPTOR),
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
    let sample = gathered(sample).await?;
    let training = gathered(training).await?;
    let group = gathered(group).await?;
    let rest = gather_values(rest).await?;
    let output_limit = requested_outputs.unwrap_or(1).max(1);
    let result = classify_compute(sample, training, group, rest, output_limit)?;
    match requested_outputs {
        Some(1) => Ok(Value::OutputList(vec![result[0].clone()])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count, result,
        )),
        None => Ok(result[0].clone()),
    }
}

async fn gathered(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("classify: {err}")))
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gathered(value).await?);
    }
    Ok(out)
}

fn classify_compute(
    sample: Value,
    training: Value,
    group: Value,
    rest: Vec<Value>,
    output_limit: usize,
) -> BuiltinResult<Vec<Value>> {
    if rest.len() > 2 {
        return Err(invalid_argument(
            "classify: accepts at most type and prior arguments",
        ));
    }
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
    if sample.data.iter().any(|value| !value.is_finite())
        || training.data.iter().any(|value| !value.is_finite())
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
    let discriminant = if let Some(value) = rest.first() {
        parse_type(value)?
    } else {
        DiscriminantType::Linear
    };
    let prepared = prepare_model(&training, labels, discriminant, rest.get(1))?;
    let predictions = predict_rows(&prepared, &sample)?;
    let class = labels_value(&prepared.classes, prepared.kind, &predictions.labels)?;
    if output_limit == 1 {
        return Ok(vec![class]);
    }
    let training_predictions = predict_rows(&prepared, &training)?;
    let err = apparent_error(&prepared, &training_predictions.labels);
    let mut outputs = vec![class, Value::Num(err)];
    if output_limit >= 3 {
        outputs.push(Value::Tensor(predictions.posterior));
    }
    if output_limit >= 4 {
        outputs.push(Value::Tensor(predictions.logp));
    }
    if output_limit >= 5 {
        outputs.push(coeff_value(&prepared)?);
    }
    Ok(outputs)
}

fn numeric_matrix(value: Value, name: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor::value_into_tensor_for(CLASSIFY_NAME, value)
        .map_err(|err| invalid_argument(format!("classify: {name}: {err}")))?;
    if tensor.shape.len() > 2 {
        return Err(invalid_argument(format!(
            "classify: {name} must be a 2-D numeric matrix"
        )));
    }
    Ok(tensor)
}

fn parse_type(value: &Value) -> BuiltinResult<DiscriminantType> {
    let text = scalar_text(value, "type")?;
    match canonical(&text).as_str() {
        "linear" => Ok(DiscriminantType::Linear),
        "quadratic" => Ok(DiscriminantType::Quadratic),
        "diaglinear" => Ok(DiscriminantType::DiagLinear),
        "diagquadratic" => Ok(DiscriminantType::DiagQuadratic),
        "mahalanobis" => Ok(DiscriminantType::Mahalanobis),
        other => Err(invalid_argument(format!(
            "classify: unsupported discriminant type '{other}'"
        ))),
    }
}

fn prepare_model(
    training: &Tensor,
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

fn class_mean(training: &Tensor, rows: &[usize]) -> Vec<f64> {
    let mut mean = vec![0.0; training.cols];
    for row in rows {
        for col in 0..training.cols {
            mean[col] += training.data[row + col * training.rows];
        }
    }
    for value in &mut mean {
        *value /= rows.len() as f64;
    }
    mean
}

fn class_covariance(
    training: &Tensor,
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
            let da = training.data[row + a * training.rows] - mean[a];
            for b in 0..cols {
                let db = training.data[row + b * training.rows] - mean[b];
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

fn predict_rows(model: &PreparedModel, data: &Tensor) -> BuiltinResult<PredictionResult> {
    let rows = data.rows;
    let class_count = model.classes.len();
    let mut label_indices = Vec::with_capacity(rows);
    let mut posterior_data = vec![0.0; rows * class_count];
    let mut logp_data = vec![0.0; rows];
    for row in 0..rows {
        let x = row_vector(data, row);
        let mut scores = Vec::with_capacity(class_count);
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
            for class_index in 0..class_count {
                posterior_data[row + class_index * rows] = f64::NAN;
            }
            logp_data[row] = f64::NAN;
        } else {
            let logp = log_sum_exp(&scores);
            logp_data[row] = logp;
            for class_index in 0..class_count {
                posterior_data[row + class_index * rows] = (scores[class_index] - logp).exp();
            }
        }
    }
    Ok(PredictionResult {
        labels: label_indices,
        posterior: Tensor::new(posterior_data, vec![rows, class_count])
            .map_err(|err| internal_error(format!("classify: {err}")))?,
        logp: Tensor::new(logp_data, vec![rows, 1])
            .map_err(|err| internal_error(format!("classify: {err}")))?,
    })
}

fn row_vector(data: &Tensor, row: usize) -> DVector<f64> {
    DVector::from_iterator(
        data.cols,
        (0..data.cols).map(|col| data.data[row + col * data.rows]),
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
    let mut entries = Vec::with_capacity(k * k);
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
    if i == j {
        return Ok((
            0.0,
            vec![0.0; model.cols],
            vec![0.0; model.cols * model.cols],
        ));
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
        DMatrix::<f64>::zeros(model.cols, model.cols)
    } else {
        (inv_j - inv_i) * 0.5
    };
    Ok((
        const_term,
        linear_vec.iter().copied().collect(),
        quadratic.iter().copied().collect(),
    ))
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
        Value::Tensor(tensor) => Ok(LabelVector {
            labels: vector_values(&tensor, name)?
                .into_iter()
                .map(ClassLabel::Numeric)
                .collect(),
            kind: LabelKind::Numeric,
        }),
        Value::Num(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Numeric(value)],
            kind: LabelKind::Numeric,
        }),
        Value::Int(value) => Ok(LabelVector {
            labels: vec![ClassLabel::Numeric(value.to_f64())],
            kind: LabelKind::Numeric,
        }),
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

fn vector_values(tensor: &Tensor, name: &str) -> BuiltinResult<Vec<f64>> {
    if tensor.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(format!(
            "classify: {name} must be a vector"
        )));
    }
    Ok(tensor.data.clone())
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) => vector_values(tensor, name),
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![value.to_f64()]),
        other => Err(invalid_argument(format!(
            "classify: {name} must be a numeric vector, got {other:?}"
        ))),
    }
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

fn same_label(left: &ClassLabel, right: &ClassLabel) -> bool {
    match (left, right) {
        (ClassLabel::Numeric(a), ClassLabel::Numeric(b)) => (a == b) || (a.is_nan() && b.is_nan()),
        (ClassLabel::Text(a), ClassLabel::Text(b)) => a == b,
        (ClassLabel::Logical(a), ClassLabel::Logical(b)) => a == b,
        _ => false,
    }
}

fn is_missing_label(label: &ClassLabel) -> bool {
    match label {
        ClassLabel::Numeric(value) => value.is_nan(),
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
        LabelKind::Numeric => {
            let data = indices
                .iter()
                .map(|idx| match labels.get(*idx) {
                    Some(ClassLabel::Numeric(value)) => *value,
                    _ => f64::NAN,
                })
                .collect::<Vec<_>>();
            Ok(Value::Tensor(
                Tensor::new(data, vec![indices.len(), 1])
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
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn classify(sample: Value, training: Value, group: Value, rest: Vec<Value>) -> Vec<Value> {
        let Value::OutputList(values) =
            block_on(classify_builtin(sample, training, group, rest)).expect("classify")
        else {
            panic!("expected output list");
        };
        values
    }

    fn with_outputs<T>(count: usize, f: impl FnOnce() -> T) -> T {
        let _guard = crate::output_count::push_output_count(Some(count));
        f()
    }

    #[test]
    fn classify_linear_predicts_numeric_groups() {
        with_outputs(5, || {
            let values = classify(
                tensor(vec![0.2, 2.2], vec![2, 1]),
                tensor(vec![0.0, 0.5, 2.0, 2.5], vec![4, 1]),
                tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]),
                Vec::new(),
            );
            let Value::Tensor(labels) = &values[0] else {
                panic!("labels");
            };
            assert_eq!(labels.data, vec![1.0, 2.0]);
            assert!(matches!(values[1], Value::Num(err) if err <= EPS));
            let Value::Tensor(posterior) = &values[2] else {
                panic!("posterior");
            };
            assert_eq!(posterior.shape, vec![2, 2]);
            assert!(posterior.data[0] > posterior.data[2]);
            assert!(posterior.data[3] > posterior.data[1]);
            let Value::Cell(coeff) = &values[4] else {
                panic!("coeff");
            };
            assert_eq!(coeff.rows, 2);
            assert_eq!(coeff.cols, 2);
            assert!(
                matches!(&coeff.data[1], Value::Struct(st) if st.fields.contains_key("linear"))
            );
        });
    }

    #[test]
    fn classify_quadratic_preserves_text_labels() {
        with_outputs(5, || {
            let groups = Value::StringArray(
                StringArray::new(
                    vec!["low".into(), "low".into(), "high".into(), "high".into()],
                    vec![4, 1],
                )
                .unwrap(),
            );
            let values = classify(
                tensor(vec![0.1, 3.2], vec![2, 1]),
                tensor(vec![0.0, 0.4, 3.0, 3.4], vec![4, 1]),
                groups,
                vec![Value::String("quadratic".into())],
            );
            let Value::StringArray(labels) = &values[0] else {
                panic!("labels");
            };
            assert_eq!(labels.data, vec!["low", "high"]);
            let Value::Cell(coeff) = &values[4] else {
                panic!("coeff");
            };
            assert!(
                matches!(&coeff.data[1], Value::Struct(st) if st.fields.contains_key("quadratic"))
            );
        });
    }

    #[test]
    fn classify_empirical_prior_and_missing_labels() {
        with_outputs(3, || {
            let values = classify(
                tensor(vec![0.1, 10.0], vec![2, 1]),
                tensor(vec![0.0, 0.2, 10.0, 10.2, 99.0], vec![5, 1]),
                tensor(vec![1.0, 1.0, 2.0, 2.0, f64::NAN], vec![5, 1]),
                vec![
                    Value::String("diagLinear".into()),
                    Value::String("empirical".into()),
                ],
            );
            let Value::Tensor(labels) = &values[0] else {
                panic!("labels");
            };
            assert_eq!(labels.data, vec![1.0, 2.0]);
            let Value::Tensor(posterior) = &values[2] else {
                panic!("posterior");
            };
            assert_eq!(posterior.shape, vec![2, 2]);
        });
    }

    #[test]
    fn classify_preserves_char_matrix_labels() {
        with_outputs(1, || {
            let groups = Value::CharArray(
                CharArray::new(vec!['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], 4, 2).unwrap(),
            );
            let values = classify(
                tensor(vec![0.1, 3.2], vec![2, 1]),
                tensor(vec![0.0, 0.4, 3.0, 3.4], vec![4, 1]),
                groups,
                Vec::new(),
            );
            let Value::CharArray(labels) = &values[0] else {
                panic!("labels");
            };
            assert_eq!(labels.rows, 2);
            assert_eq!(labels.cols, 2);
            assert_eq!(labels.data, vec!['a', 'a', 'b', 'b']);
        });
    }

    #[test]
    fn classify_accepts_cellstr_labels_and_struct_priors() {
        with_outputs(2, || {
            let groups = Value::Cell(
                CellArray::new(
                    vec![
                        Value::CharArray(CharArray::new_row("left")),
                        Value::CharArray(CharArray::new_row("left")),
                        Value::CharArray(CharArray::new_row("right")),
                        Value::CharArray(CharArray::new_row("right")),
                    ],
                    4,
                    1,
                )
                .unwrap(),
            );
            let mut prior = StructValue::new();
            prior.insert(
                "group",
                Value::StringArray(
                    StringArray::new(vec!["left".into(), "right".into()], vec![2, 1]).unwrap(),
                ),
            );
            prior.insert(
                "prob",
                Value::Tensor(Tensor::new(vec![0.25, 0.75], vec![2, 1]).unwrap()),
            );
            let values = classify(
                tensor(vec![0.1, 3.2], vec![2, 1]),
                tensor(vec![0.0, 0.4, 3.0, 3.4], vec![4, 1]),
                groups,
                vec![Value::String("linear".into()), Value::Struct(prior)],
            );
            let Value::Cell(labels) = &values[0] else {
                panic!("labels");
            };
            assert_eq!(labels.rows, 2);
            assert!(matches!(values[1], Value::Num(err) if err <= EPS));
        });
    }

    #[test]
    fn classify_rejects_bad_dimensions() {
        let err = block_on(classify_builtin(
            tensor(vec![1.0, 2.0], vec![1, 2]),
            tensor(vec![1.0, 2.0], vec![2, 1]),
            tensor(vec![1.0, 2.0], vec![2, 1]),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message.contains("same number of columns"));
    }
}
