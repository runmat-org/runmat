//! t-SNE dimensionality reduction compatibility surface.

use std::cmp::Ordering;

use nalgebra::{DMatrix, SymmetricEigen};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{random, random_args::keyword_of, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "tsne";
const DEFAULT_MAX_ITER: usize = 1000;
const MAX_WORK_CELLS: usize = 4_000_000;
const MAX_EMBEDDING_DIMS: usize = 128;
const MAX_ITERATIONS: usize = 100_000;
const MAX_NUM_PRINT: usize = 1_000_000;
const MIN_PROB: f64 = 1.0e-12;

const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Low-dimensional embedding matrix.",
}];

const OUTPUT_Y_LOSS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Low-dimensional embedding matrix.",
    },
    BuiltinParamDescriptor {
        name: "loss",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Kullback-Leibler divergence for the final embedding.",
    },
];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation matrix with observations in rows.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as Algorithm, Distance, NumDimensions, NumPCAComponents, Perplexity, Standardize, InitialY, LearnRate, Options, Theta, and Verbose.",
};

const INPUTS_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUTS_X_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_OPTIONS];

const TSNE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = tsne(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tsne(X, Name, Value)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[Y, loss] = tsne(___)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_Y_LOSS,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TSNE.INVALID_ARGUMENT",
    identifier: Some("RunMat:tsne:InvalidArgument"),
    when: "Inputs, dimensions, distances, initialization, or name-value options are malformed or unsupported.",
    message: "tsne: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TSNE.INTERNAL",
    identifier: Some("RunMat:tsne:Internal"),
    when: "RunMat cannot allocate or construct the t-SNE result.",
    message: "tsne: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const TSNE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TSNE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn tsne_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn tsne_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    tsne_error(message, &ERROR_INVALID_ARGUMENT)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    tsne_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Algorithm {
    Exact,
    BarnesHut,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DistanceMetric {
    Euclidean,
    Seuclidean,
    Cityblock,
    Chebychev,
    Minkowski,
    Mahalanobis,
    Cosine,
    Correlation,
    Spearman,
    Hamming,
    Jaccard,
}

#[derive(Clone, Debug)]
struct Options {
    algorithm: Algorithm,
    distance: DistanceMetric,
    exaggeration: f64,
    num_dimensions: usize,
    num_pca_components: usize,
    perplexity: f64,
    standardize: bool,
    initial_y: Option<Tensor>,
    learn_rate: f64,
    num_print: usize,
    max_iter: usize,
    tol_fun: f64,
    theta: f64,
    verbose: usize,
    cache_size: Option<CacheSize>,
}

#[derive(Clone, Debug)]
enum CacheSize {
    Maximal,
    Megabytes(f64),
}

impl Default for Options {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::BarnesHut,
            distance: DistanceMetric::Euclidean,
            exaggeration: 4.0,
            num_dimensions: 2,
            num_pca_components: 0,
            perplexity: 30.0,
            standardize: false,
            initial_y: None,
            learn_rate: 500.0,
            num_print: 20,
            max_iter: DEFAULT_MAX_ITER,
            tol_fun: 1.0e-10,
            theta: 0.5,
            verbose: 0,
            cache_size: None,
        }
    }
}

#[derive(Clone, Debug)]
struct PreparedData {
    rows: Vec<Vec<f64>>,
    good_rows: Vec<usize>,
    original_rows: usize,
}

#[derive(Clone, Debug)]
struct EmbeddingResult {
    embedding: Tensor,
    loss: f64,
}

#[runtime_builtin(
    name = "tsne",
    category = "stats/ml",
    summary = "Embed high-dimensional observations using t-distributed stochastic neighbor embedding.",
    keywords = "tsne,t-SNE,dimensionality reduction,embedding,statistics,machine learning",
    type_resolver(tsne_type),
    descriptor(crate::builtins::stats::ml::tsne::TSNE_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::tsne"
)]
pub(crate) async fn tsne_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let x = gather(x).await?;
    let rest = gather_all(rest).await?;
    let options = parse_options(rest)?;
    let tensor =
        tensor::value_into_tensor_for(NAME, x).map_err(|err| invalid(format!("tsne: {err}")))?;
    let tensor =
        tensor::integer_tensor_to_f64(tensor).map_err(|err| invalid(format!("tsne: {err}")))?;
    let result = compute_tsne(tensor, options)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![Value::Tensor(result.embedding)])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![Value::Tensor(result.embedding), Value::Num(result.loss)],
        )),
        None => Ok(Value::Tensor(result.embedding)),
    }
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("tsne: {err}")))
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}

fn parse_options(values: Vec<Value>) -> BuiltinResult<Options> {
    if !values.len().is_multiple_of(2) {
        return Err(invalid("tsne: name-value options must be paired"));
    }
    let mut options = Options::default();
    let mut index = 0usize;
    while index < values.len() {
        let name = keyword_of(&values[index])
            .ok_or_else(|| invalid("tsne: option name must be text"))?
            .to_ascii_lowercase();
        let value = &values[index + 1];
        match name.as_str() {
            "algorithm" => {
                let text = text_scalar(value, "Algorithm")?;
                options.algorithm = match text.to_ascii_lowercase().as_str() {
                    "exact" => Algorithm::Exact,
                    "barneshut" => Algorithm::BarnesHut,
                    other => return Err(invalid(format!("tsne: unsupported Algorithm '{other}'"))),
                };
            }
            "cachesize" => {
                options.cache_size = Some(if let Some(text) = keyword_of(value) {
                    if text.eq_ignore_ascii_case("maximal") {
                        CacheSize::Maximal
                    } else {
                        return Err(invalid("tsne: CacheSize must be positive or 'maximal'"));
                    }
                } else {
                    CacheSize::Megabytes(positive_scalar(value, "CacheSize")?)
                });
            }
            "distance" => {
                if matches!(
                    value,
                    Value::FunctionHandle(_)
                        | Value::ExternalFunctionHandle(_)
                        | Value::MethodFunctionHandle(_)
                        | Value::BoundFunctionHandle { .. }
                ) {
                    return Err(invalid(
                        "tsne: custom distance function handles are not supported yet",
                    ));
                }
                let text = text_scalar(value, "Distance")?;
                let lower = text.to_ascii_lowercase();
                options.distance = match lower.as_str() {
                    "euclidean" => DistanceMetric::Euclidean,
                    "fasteuclidean" => DistanceMetric::Euclidean,
                    "seuclidean" => DistanceMetric::Seuclidean,
                    "fastseuclidean" => DistanceMetric::Seuclidean,
                    "cityblock" => DistanceMetric::Cityblock,
                    "chebychev" => DistanceMetric::Chebychev,
                    "minkowski" => DistanceMetric::Minkowski,
                    "mahalanobis" => DistanceMetric::Mahalanobis,
                    "cosine" => DistanceMetric::Cosine,
                    "correlation" => DistanceMetric::Correlation,
                    "spearman" => DistanceMetric::Spearman,
                    "hamming" => DistanceMetric::Hamming,
                    "jaccard" => DistanceMetric::Jaccard,
                    other => return Err(invalid(format!("tsne: unsupported Distance '{other}'"))),
                };
            }
            "exaggeration" => {
                options.exaggeration = bounded_scalar(value, "Exaggeration", 1.0, f64::INFINITY)?;
            }
            "numdimensions" => {
                options.num_dimensions =
                    bounded_positive_integer(value, "NumDimensions", MAX_EMBEDDING_DIMS)?;
            }
            "numpcacomponents" => {
                options.num_pca_components = nonnegative_integer(value, "NumPCAComponents")?;
            }
            "perplexity" => options.perplexity = positive_scalar(value, "Perplexity")?,
            "standardize" => options.standardize = bool_scalar(value, "Standardize")?,
            "initialy" => {
                let tensor = tensor::value_into_tensor_for(NAME, value.clone())
                    .map_err(|err| invalid(format!("tsne: InitialY {err}")))?;
                let tensor = tensor::integer_tensor_to_f64(tensor)
                    .map_err(|err| invalid(format!("tsne: InitialY {err}")))?;
                options.initial_y = Some(tensor);
            }
            "learnrate" => options.learn_rate = positive_scalar(value, "LearnRate")?,
            "numprint" => {
                options.num_print = bounded_positive_integer(value, "NumPrint", MAX_NUM_PRINT)?
            }
            "options" => apply_options_struct(&mut options, value)?,
            "theta" => options.theta = bounded_scalar(value, "Theta", 0.0, 1.0)?,
            "verbose" => {
                let value = nonnegative_integer(value, "Verbose")?;
                if value > 2 {
                    return Err(invalid("tsne: Verbose must be 0, 1, or 2"));
                }
                options.verbose = value;
            }
            other => return Err(invalid(format!("tsne: unknown option '{other}'"))),
        }
        index += 2;
    }
    Ok(options)
}

fn apply_options_struct(options: &mut Options, value: &Value) -> BuiltinResult<()> {
    let Value::Struct(fields) = value else {
        if is_empty_numeric(value) {
            return Ok(());
        }
        return Err(invalid("tsne: Options must be a struct"));
    };
    for (name, field) in &fields.fields {
        if is_empty_option_value(field) {
            continue;
        }
        match name.to_ascii_lowercase().as_str() {
            "maxiter" => {
                options.max_iter =
                    bounded_positive_integer(field, "Options.MaxIter", MAX_ITERATIONS)?
            }
            "tolfun" => options.tol_fun = positive_scalar(field, "Options.TolFun")?,
            "display" => {
                let display = text_scalar(field, "Options.Display")?;
                match display.to_ascii_lowercase().as_str() {
                    "off" | "final" | "iter" => {}
                    other => {
                        return Err(invalid(format!(
                            "tsne: unsupported Options.Display '{other}'"
                        )))
                    }
                }
            }
            "outputfcn" => {
                if !is_empty_numeric(field) {
                    return Err(invalid(
                        "tsne: Options.OutputFcn is not supported by the CPU runtime",
                    ));
                }
            }
            other => {
                return Err(invalid(format!(
                    "tsne: unsupported Options field '{other}'"
                )));
            }
        }
    }
    Ok(())
}

fn is_empty_option_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.data.is_empty(),
        Value::LogicalArray(array) => array.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::StringArray(array) => array.data.is_empty(),
        Value::CharArray(array) => array.data.is_empty(),
        _ => false,
    }
}

fn compute_tsne(x: Tensor, options: Options) -> BuiltinResult<EmbeddingResult> {
    if x.shape.len() > 2 {
        return Err(invalid("tsne: X must be a 2-D numeric matrix"));
    }
    let prepared = prepare_data(&x)?;
    if prepared.rows.is_empty() {
        return Err(invalid(
            "tsne: at least one complete observation is required",
        ));
    }
    if prepared.rows.len() < 2 {
        return single_row_embedding(prepared.rows.len(), options.num_dimensions);
    }
    if prepared.rows.len().saturating_mul(prepared.rows.len()) > MAX_WORK_CELLS {
        return Err(invalid(format!(
            "tsne: exact CPU implementation supports at most {MAX_WORK_CELLS} pairwise cells"
        )));
    }
    let feature_count = prepared.rows[0].len();
    if matches!(options.distance, DistanceMetric::Mahalanobis) || needs_pca(&options, feature_count)
    {
        guard_dense_feature_work(feature_count)?;
    }
    let mut rows = prepared.rows.clone();
    let _reporting_options = (options.num_print, options.verbose);
    let _barnes_hut_theta = options.theta;
    match options.algorithm {
        Algorithm::Exact | Algorithm::BarnesHut => {}
    }
    match &options.cache_size {
        Some(CacheSize::Megabytes(value)) if !value.is_finite() || *value <= 0.0 => {
            return Err(invalid("tsne: CacheSize must be positive"));
        }
        Some(CacheSize::Megabytes(_)) | Some(CacheSize::Maximal) | None => {}
    }
    if options.standardize {
        standardize_rows(&mut rows);
    }
    rows = apply_pca_if_requested(rows, options.num_pca_components)?;
    let distances = pairwise_squared_distances(&rows, options.distance)?;
    let probabilities = joint_probabilities(&distances, options.perplexity)?;
    let mut embedding = initial_embedding(&options, &prepared)?;
    center_embedding(&mut embedding, options.num_dimensions);
    let mut velocity = vec![0.0; embedding.len()];
    let mut gains = vec![1.0_f64; embedding.len()];
    let iterations = options.max_iter.max(1);
    for iter in 0..iterations {
        let exaggeration = if iter < 99 { options.exaggeration } else { 1.0 };
        let (gradient, _, grad_norm) = gradient_and_loss(
            &embedding,
            options.num_dimensions,
            &probabilities,
            exaggeration,
        );
        let momentum = if iter < 250 { 0.5 } else { 0.8 };
        for idx in 0..embedding.len() {
            gains[idx] = if (gradient[idx] > 0.0) != (velocity[idx] > 0.0) {
                gains[idx] + 0.2
            } else {
                (gains[idx] * 0.8).max(0.01)
            };
            velocity[idx] =
                momentum * velocity[idx] - options.learn_rate * gains[idx] * gradient[idx];
            embedding[idx] += velocity[idx];
        }
        center_embedding(&mut embedding, options.num_dimensions);
        if grad_norm < options.tol_fun {
            break;
        }
    }
    let (_, loss, _) = gradient_and_loss(&embedding, options.num_dimensions, &probabilities, 1.0);
    let output = Tensor::new(embedding, vec![prepared.rows.len(), options.num_dimensions])
        .map_err(|err| internal(format!("tsne: {err}")))?;
    Ok(EmbeddingResult {
        embedding: output,
        loss,
    })
}

fn single_row_embedding(rows: usize, dims: usize) -> BuiltinResult<EmbeddingResult> {
    let tensor = Tensor::new(vec![0.0; rows * dims], vec![rows, dims])
        .map_err(|err| internal(format!("tsne: {err}")))?;
    Ok(EmbeddingResult {
        embedding: tensor,
        loss: 0.0,
    })
}

fn prepare_data(x: &Tensor) -> BuiltinResult<PreparedData> {
    let mut rows = Vec::new();
    let mut good_rows = Vec::new();
    for row in 0..x.rows {
        let mut out = Vec::with_capacity(x.cols);
        let mut complete = true;
        for col in 0..x.cols {
            let value = x_value(x, row, col);
            if value.is_infinite() {
                return Err(invalid("tsne: X must not contain Inf values"));
            }
            if value.is_nan() {
                complete = false;
                continue;
            }
            out.push(value);
        }
        if complete {
            good_rows.push(row);
            rows.push(out);
        }
    }
    Ok(PreparedData {
        rows,
        good_rows,
        original_rows: x.rows,
    })
}

fn needs_pca(options: &Options, feature_count: usize) -> bool {
    options.num_pca_components > 0 && options.num_pca_components < feature_count
}

fn guard_dense_feature_work(feature_count: usize) -> BuiltinResult<()> {
    if feature_count.saturating_mul(feature_count) > MAX_WORK_CELLS {
        return Err(invalid(format!(
            "tsne: PCA and Mahalanobis distance support at most {MAX_WORK_CELLS} dense feature cells"
        )));
    }
    Ok(())
}

fn standardize_rows(rows: &mut [Vec<f64>]) {
    if rows.is_empty() {
        return;
    }
    let cols = rows[0].len();
    for col in 0..cols {
        let mean = rows.iter().map(|row| row[col]).sum::<f64>() / rows.len() as f64;
        let var = rows
            .iter()
            .map(|row| {
                let diff = row[col] - mean;
                diff * diff
            })
            .sum::<f64>()
            / (rows.len().saturating_sub(1).max(1) as f64);
        let scale = var.sqrt();
        for row in rows.iter_mut() {
            row[col] = if scale > 0.0 {
                (row[col] - mean) / scale
            } else {
                0.0
            };
        }
    }
}

fn apply_pca_if_requested(rows: Vec<Vec<f64>>, components: usize) -> BuiltinResult<Vec<Vec<f64>>> {
    if components == 0 || rows.is_empty() || components >= rows[0].len() {
        return Ok(rows);
    }
    let n = rows.len();
    let p = rows[0].len();
    let k = components.min(p);
    let mut centered = rows;
    let means = (0..p)
        .map(|col| centered.iter().map(|row| row[col]).sum::<f64>() / n as f64)
        .collect::<Vec<_>>();
    for row in &mut centered {
        for col in 0..p {
            row[col] -= means[col];
        }
    }
    let mut cov: DMatrix<f64> = DMatrix::zeros(p, p);
    let denom = n.saturating_sub(1).max(1) as f64;
    for row in &centered {
        for a in 0..p {
            for b in a..p {
                cov[(a, b)] += row[a] * row[b] / denom;
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            cov[(a, b)] = cov[(b, a)];
        }
    }
    let eig = SymmetricEigen::new(cov);
    let mut order: Vec<(usize, f64)> = (0..p)
        .map(|idx| (idx, eig.eigenvalues[idx]))
        .collect::<Vec<_>>();
    order.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    let mut projected = vec![vec![0.0; k]; n];
    for (new_col, (eig_col, _)) in order.into_iter().take(k).enumerate() {
        for row in 0..n {
            let mut value = 0.0;
            for old_col in 0..p {
                value += centered[row][old_col] * eig.eigenvectors[(old_col, eig_col)];
            }
            projected[row][new_col] = value;
        }
    }
    Ok(projected)
}

fn pairwise_squared_distances(
    rows: &[Vec<f64>],
    metric: DistanceMetric,
) -> BuiltinResult<Vec<f64>> {
    let n = rows.len();
    let mut data = vec![0.0; n * n];
    let prepared = match metric {
        DistanceMetric::Seuclidean => Some(MetricPrep::Scale(column_std(rows))),
        DistanceMetric::Mahalanobis => Some(MetricPrep::InverseCov(inverse_covariance(rows)?)),
        DistanceMetric::Spearman => Some(MetricPrep::Ranks(row_ranks(rows))),
        _ => None,
    };
    for i in 0..n {
        for j in (i + 1)..n {
            let value = match metric {
                DistanceMetric::Euclidean => euclidean_sq(&rows[i], &rows[j]),
                DistanceMetric::Seuclidean => match &prepared {
                    Some(MetricPrep::Scale(scale)) => seuclidean_sq(&rows[i], &rows[j], scale),
                    _ => unreachable!(),
                },
                DistanceMetric::Cityblock => cityblock(&rows[i], &rows[j]).powi(2),
                DistanceMetric::Chebychev => chebychev(&rows[i], &rows[j]).powi(2),
                DistanceMetric::Minkowski => euclidean_sq(&rows[i], &rows[j]),
                DistanceMetric::Mahalanobis => match &prepared {
                    Some(MetricPrep::InverseCov(inv)) => mahalanobis_sq(&rows[i], &rows[j], inv),
                    _ => unreachable!(),
                },
                DistanceMetric::Cosine => cosine_distance(&rows[i], &rows[j]).powi(2),
                DistanceMetric::Correlation => correlation_distance(&rows[i], &rows[j]).powi(2),
                DistanceMetric::Spearman => match &prepared {
                    Some(MetricPrep::Ranks(ranks)) => {
                        correlation_distance(&ranks[i], &ranks[j]).powi(2)
                    }
                    _ => unreachable!(),
                },
                DistanceMetric::Hamming => hamming_distance(&rows[i], &rows[j]).powi(2),
                DistanceMetric::Jaccard => jaccard_distance(&rows[i], &rows[j]).powi(2),
            };
            data[i + j * n] = value;
            data[j + i * n] = value;
        }
    }
    Ok(data)
}

enum MetricPrep {
    Scale(Vec<f64>),
    InverseCov(DMatrix<f64>),
    Ranks(Vec<Vec<f64>>),
}

fn joint_probabilities(distances: &[f64], perplexity: f64) -> BuiltinResult<Vec<f64>> {
    let n = integer_sqrt(distances.len())
        .ok_or_else(|| internal("tsne: distance matrix has invalid shape"))?;
    let effective_perplexity = perplexity.min((n - 1) as f64).max(1.0);
    let target_entropy = effective_perplexity.ln();
    let mut conditional = vec![0.0; n * n];
    for i in 0..n {
        let mut beta = 1.0;
        let mut beta_min = f64::NEG_INFINITY;
        let mut beta_max = f64::INFINITY;
        let mut row_prob = vec![0.0; n];
        for _ in 0..80 {
            let mut sum = 0.0;
            for j in 0..n {
                if i == j {
                    row_prob[j] = 0.0;
                } else {
                    let value = (-distances[i + j * n] * beta).exp();
                    row_prob[j] = value;
                    sum += value;
                }
            }
            if sum <= 0.0 || !sum.is_finite() {
                for (j, prob) in row_prob.iter_mut().enumerate() {
                    *prob = if i == j { 0.0 } else { 1.0 / (n - 1) as f64 };
                }
                break;
            }
            let mut entropy = 0.0;
            for prob in &mut row_prob {
                *prob /= sum;
                if *prob > 0.0 {
                    entropy -= *prob * prob.ln();
                }
            }
            let diff = entropy - target_entropy;
            if diff.abs() < 1.0e-5 {
                break;
            }
            if diff > 0.0 {
                beta_min = beta;
                beta = if beta_max.is_finite() {
                    (beta + beta_max) / 2.0
                } else {
                    beta * 2.0
                };
            } else {
                beta_max = beta;
                beta = if beta_min.is_finite() {
                    (beta + beta_min) / 2.0
                } else {
                    beta / 2.0
                };
            }
        }
        for j in 0..n {
            conditional[i + j * n] = row_prob[j];
        }
    }
    let mut joint = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                joint[i + j * n] = ((conditional[i + j * n] + conditional[j + i * n])
                    / (2.0 * n as f64))
                    .max(MIN_PROB);
            }
        }
    }
    let sum = joint.iter().sum::<f64>();
    if sum > 0.0 {
        for value in &mut joint {
            *value /= sum;
        }
    }
    Ok(joint)
}

fn initial_embedding(options: &Options, prepared: &PreparedData) -> BuiltinResult<Vec<f64>> {
    let rows = prepared.rows.len();
    let dims = options.num_dimensions;
    if let Some(initial) = &options.initial_y {
        if initial.shape.len() > 2 || initial.cols != dims {
            return Err(invalid(format!(
                "tsne: InitialY must have {} columns",
                dims
            )));
        }
        let mut data = Vec::with_capacity(rows * dims);
        if initial.rows == prepared.original_rows {
            for col in 0..dims {
                for &row in &prepared.good_rows {
                    let value = x_value(initial, row, col);
                    if !value.is_finite() {
                        return Err(invalid("tsne: InitialY values must be finite"));
                    }
                    data.push(value);
                }
            }
        } else if initial.rows == rows {
            for col in 0..dims {
                for row in 0..rows {
                    let value = x_value(initial, row, col);
                    if !value.is_finite() {
                        return Err(invalid("tsne: InitialY values must be finite"));
                    }
                    data.push(value);
                }
            }
        } else {
            return Err(invalid(
                "tsne: InitialY row count must match X or complete X rows",
            ));
        }
        return Ok(data);
    }
    let normals = random::generate_normal(rows * dims, NAME)?;
    Ok(normals.into_iter().map(|value| value * 1.0e-4).collect())
}

fn gradient_and_loss(
    embedding: &[f64],
    dims: usize,
    probabilities: &[f64],
    exaggeration: f64,
) -> (Vec<f64>, f64, f64) {
    let n = integer_sqrt(probabilities.len()).unwrap_or(0);
    let mut q_num = vec![0.0; n * n];
    let mut sum_q = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist = 0.0;
            for dim in 0..dims {
                let diff = embedding[i + dim * n] - embedding[j + dim * n];
                dist += diff * diff;
            }
            let value = 1.0 / (1.0 + dist);
            q_num[i + j * n] = value;
            q_num[j + i * n] = value;
            sum_q += 2.0 * value;
        }
    }
    let sum_q = sum_q.max(MIN_PROB);
    let mut gradient = vec![0.0; embedding.len()];
    let mut loss = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let p = probabilities[i + j * n].max(MIN_PROB);
            let q = (q_num[i + j * n] / sum_q).max(MIN_PROB);
            loss += p * (p / q).ln();
            let coeff = 4.0 * (exaggeration * p - q) * q_num[i + j * n];
            for dim in 0..dims {
                gradient[i + dim * n] += coeff * (embedding[i + dim * n] - embedding[j + dim * n]);
            }
        }
    }
    let grad_norm = gradient
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    (gradient, loss, grad_norm)
}

fn center_embedding(embedding: &mut [f64], dims: usize) {
    let rows = embedding.len() / dims.max(1);
    for dim in 0..dims {
        let start = dim * rows;
        let mean = embedding[start..start + rows].iter().sum::<f64>() / rows.max(1) as f64;
        for row in 0..rows {
            embedding[start + row] -= mean;
        }
    }
}

fn euclidean_sq(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum()
}

fn seuclidean_sq(left: &[f64], right: &[f64], scale: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .zip(scale.iter())
        .map(|((a, b), s)| {
            let diff = if *s > 0.0 { (a - b) / s } else { 0.0 };
            diff * diff
        })
        .sum()
}

fn cityblock(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .sum()
}

fn chebychev(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn cosine_distance(left: &[f64], right: &[f64]) -> f64 {
    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let nl = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let nr = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    if nl == 0.0 || nr == 0.0 {
        1.0
    } else {
        (1.0 - dot / (nl * nr)).max(0.0)
    }
}

fn correlation_distance(left: &[f64], right: &[f64]) -> f64 {
    let ml = left.iter().sum::<f64>() / left.len().max(1) as f64;
    let mr = right.iter().sum::<f64>() / right.len().max(1) as f64;
    let centered_l = left.iter().map(|value| value - ml).collect::<Vec<_>>();
    let centered_r = right.iter().map(|value| value - mr).collect::<Vec<_>>();
    cosine_distance(&centered_l, &centered_r)
}

fn hamming_distance(left: &[f64], right: &[f64]) -> f64 {
    let mismatches = left
        .iter()
        .zip(right.iter())
        .filter(|(a, b)| (*a - *b).abs() > 0.0)
        .count();
    mismatches as f64 / left.len().max(1) as f64
}

fn jaccard_distance(left: &[f64], right: &[f64]) -> f64 {
    let mut union = 0usize;
    let mut intersection = 0usize;
    for (a, b) in left.iter().zip(right.iter()) {
        let an = *a != 0.0;
        let bn = *b != 0.0;
        if an || bn {
            union += 1;
            if an && bn {
                intersection += 1;
            }
        }
    }
    if union == 0 {
        0.0
    } else {
        1.0 - intersection as f64 / union as f64
    }
}

fn column_std(rows: &[Vec<f64>]) -> Vec<f64> {
    if rows.is_empty() {
        return Vec::new();
    }
    let cols = rows[0].len();
    (0..cols)
        .map(|col| {
            let mean = rows.iter().map(|row| row[col]).sum::<f64>() / rows.len() as f64;
            let var = rows
                .iter()
                .map(|row| {
                    let diff = row[col] - mean;
                    diff * diff
                })
                .sum::<f64>()
                / rows.len().saturating_sub(1).max(1) as f64;
            var.sqrt()
        })
        .collect()
}

fn inverse_covariance(rows: &[Vec<f64>]) -> BuiltinResult<DMatrix<f64>> {
    if rows.is_empty() {
        return Ok(DMatrix::identity(0, 0));
    }
    let p = rows[0].len();
    let mut cov: DMatrix<f64> = DMatrix::zeros(p, p);
    let means = (0..p)
        .map(|col| rows.iter().map(|row| row[col]).sum::<f64>() / rows.len() as f64)
        .collect::<Vec<_>>();
    let denom = rows.len().saturating_sub(1).max(1) as f64;
    for row in rows {
        for a in 0..p {
            for b in a..p {
                cov[(a, b)] += (row[a] - means[a]) * (row[b] - means[b]) / denom;
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            cov[(a, b)] = cov[(b, a)];
        }
    }
    for idx in 0..p {
        cov[(idx, idx)] += 1.0e-10;
    }
    cov.try_inverse()
        .ok_or_else(|| invalid("tsne: Mahalanobis covariance is singular"))
}

fn mahalanobis_sq(left: &[f64], right: &[f64], inv_cov: &DMatrix<f64>) -> f64 {
    let diff = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let mut value = 0.0;
    for i in 0..diff.len() {
        for j in 0..diff.len() {
            value += diff[i] * inv_cov[(i, j)] * diff[j];
        }
    }
    value.max(0.0)
}

fn row_ranks(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter().map(|row| tied_ranks(row)).collect()
}

fn tied_ranks(values: &[f64]) -> Vec<f64> {
    let mut order = values.iter().copied().enumerate().collect::<Vec<_>>();
    order.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && order[end].1 == order[start].1 {
            end += 1;
        }
        let rank = (start + 1 + end) as f64 / 2.0;
        for idx in start..end {
            ranks[order[idx].0] = rank;
        }
        start = end;
    }
    ranks
}

fn x_value(x: &Tensor, row: usize, col: usize) -> f64 {
    x.data[row + col * x.rows]
}

fn integer_sqrt(value: usize) -> Option<usize> {
    let root = (value as f64).sqrt() as usize;
    if root.saturating_mul(root) == value {
        Some(root)
    } else {
        None
    }
}

fn text_scalar(value: &Value, name: &str) -> BuiltinResult<String> {
    keyword_of(value).ok_or_else(|| invalid(format!("tsne: {name} must be text")))
}

fn positive_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    let value = numeric_scalar(value, name)?;
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!("tsne: {name} must be a positive scalar")));
    }
    Ok(value)
}

fn bounded_scalar(value: &Value, name: &str, min: f64, max: f64) -> BuiltinResult<f64> {
    let value = numeric_scalar(value, name)?;
    if !value.is_finite() || value < min || value > max {
        return Err(invalid(format!(
            "tsne: {name} must be in the supported numeric range"
        )));
    }
    Ok(value)
}

fn bounded_positive_integer(value: &Value, name: &str, max: usize) -> BuiltinResult<usize> {
    integer_in_range(value, name, 1, max)
}

fn bounded_nonnegative_integer(value: &Value, name: &str, max: usize) -> BuiltinResult<usize> {
    integer_in_range(value, name, 0, max)
}

fn integer_in_range(value: &Value, name: &str, min: usize, max: usize) -> BuiltinResult<usize> {
    let value = numeric_scalar(value, name)?;
    if !value.is_finite() || value.fract() != 0.0 || value < min as f64 || value > max as f64 {
        let kind = if min == 0 {
            "a nonnegative integer"
        } else {
            "a positive integer"
        };
        return Err(invalid(format!(
            "tsne: {name} must be {kind} no greater than {max}"
        )));
    }
    Ok(value as usize)
}

fn nonnegative_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    bounded_nonnegative_integer(value, name, usize::MAX)
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0])
        }
        _ => Err(invalid(format!("tsne: {name} must be a numeric scalar"))),
    }
}

fn bool_scalar(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let raw = tensor::tensor_values_f64(tensor)[0];
            if raw == 0.0 || raw == 1.0 {
                Ok(raw != 0.0)
            } else {
                Err(invalid(format!("tsne: {name} must be logical scalar")))
            }
        }
        _ => Err(invalid(format!("tsne: {name} must be logical scalar"))),
    }
}

fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.data.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;
    use runmat_builtins::{IntegerStorage, StructValue};

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let mut tensor = Tensor::new_integer(storage, shape).unwrap();
        tensor.data.fill(f64::NAN);
        Value::Tensor(tensor)
    }

    fn reset_rng() -> std::sync::MutexGuard<'static, ()> {
        let guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        guard
    }

    #[tokio::test]
    async fn tsne_embeds_rows_and_returns_loss() {
        let _guard = reset_rng();
        let _out_guard = crate::output_count::push_output_count(Some(2));
        let value = tsne_builtin(
            tensor(vec![0.0, 0.1, 5.0, 5.1, 0.0, 0.1, 5.0, 5.1], vec![4, 2]),
            vec![
                Value::String("Algorithm".into()),
                Value::String("exact".into()),
                Value::String("Perplexity".into()),
                Value::Num(2.0),
                Value::String("Options".into()),
                {
                    let mut options = StructValue::new();
                    options.insert("MaxIter", Value::Num(50.0));
                    Value::Struct(options)
                },
            ],
        )
        .await
        .unwrap();
        match value {
            Value::OutputList(outputs) => {
                match &outputs[0] {
                    Value::Tensor(y) => {
                        assert_eq!(y.shape, vec![4, 2]);
                        assert!(y.data.iter().all(|value| value.is_finite()));
                    }
                    other => panic!("expected tensor, got {other:?}"),
                }
                assert!(matches!(&outputs[1], Value::Num(loss) if loss.is_finite()));
            }
            other => panic!("expected outputs, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tsne_removes_nan_rows_and_honors_num_dimensions() {
        let _guard = reset_rng();
        let value = tsne_builtin(
            tensor(
                vec![
                    0.0,
                    1.0,
                    f64::NAN,
                    3.0, //
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
                vec![4, 2],
            ),
            vec![
                Value::String("NumDimensions".into()),
                Value::Num(3.0),
                Value::String("InitialY".into()),
                Value::Tensor(Tensor::new(vec![0.0; 12], vec![4, 3]).unwrap()),
                Value::String("Options".into()),
                {
                    let mut options = StructValue::new();
                    options.insert("MaxIter", Value::Num(2.0));
                    Value::Struct(options)
                },
            ],
        )
        .await
        .unwrap();
        match value {
            Value::Tensor(y) => assert_eq!(y.shape, vec![3, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tsne_supports_standardize_pca_and_cosine_distance() {
        let _guard = reset_rng();
        let value = tsne_builtin(
            tensor(
                vec![0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 2.0, 1.0],
                vec![4, 3],
            ),
            vec![
                Value::String("Standardize".into()),
                Value::Bool(true),
                Value::String("NumPCAComponents".into()),
                Value::Num(2.0),
                Value::String("Distance".into()),
                Value::String("cosine".into()),
                Value::String("Options".into()),
                {
                    let mut options = StructValue::new();
                    options.insert("MaxIter", Value::Num(5.0));
                    Value::Struct(options)
                },
            ],
        )
        .await
        .unwrap();
        match value {
            Value::Tensor(y) => assert_eq!(y.shape, vec![4, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tsne_reads_typed_integer_data_initial_y_and_options_exactly() {
        let _guard = reset_rng();
        let value = tsne_builtin(
            poisoned_int_tensor(
                IntegerStorage::I16(vec![0, 1, 5, 6, 0, 1, 5, 6]),
                vec![4, 2],
            ),
            vec![
                Value::String("NumDimensions".into()),
                poisoned_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
                Value::String("NumPCAComponents".into()),
                poisoned_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
                Value::String("Perplexity".into()),
                poisoned_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
                Value::String("Standardize".into()),
                poisoned_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
                Value::String("InitialY".into()),
                poisoned_int_tensor(IntegerStorage::I16(vec![0; 8]), vec![4, 2]),
                Value::String("Options".into()),
                {
                    let mut options = StructValue::new();
                    options.insert(
                        "MaxIter",
                        poisoned_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                    );
                    Value::Struct(options)
                },
            ],
        )
        .await
        .unwrap();
        match value {
            Value::Tensor(y) => assert_eq!(y.shape, vec![4, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tsne_rejects_bad_initial_y_shape() {
        let err = tsne_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
            vec![
                Value::String("InitialY".into()),
                Value::Tensor(Tensor::new(vec![0.0; 3], vec![3, 1]).unwrap()),
            ],
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("InitialY"));
    }

    #[tokio::test]
    async fn tsne_rejects_unbounded_resource_options() {
        let err = tsne_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
            vec![Value::String("NumDimensions".into()), Value::Num(1.0e20)],
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("NumDimensions"));

        let err = tsne_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
            vec![Value::String("Options".into()), {
                let mut options = StructValue::new();
                options.insert("MaxIter", Value::Num(1.0e20));
                Value::Struct(options)
            }],
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("MaxIter"));
    }

    #[tokio::test]
    async fn tsne_rejects_dense_feature_work_before_allocation() {
        let x = Tensor::new(vec![1.0; 2 * 2001], vec![2, 2001]).unwrap();
        let err = tsne_builtin(
            Value::Tensor(x),
            vec![
                Value::String("Distance".into()),
                Value::String("mahalanobis".into()),
            ],
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("dense feature cells"));
    }

    #[tokio::test]
    async fn tsne_rejects_inf_even_after_nan_in_row() {
        let err = tsne_builtin(
            tensor(vec![f64::NAN, 1.0, f64::INFINITY, 2.0], vec![2, 2]),
            Vec::new(),
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("Inf"));
    }

    #[tokio::test]
    async fn tsne_rejects_unsupported_fast_distance_names() {
        let err = tsne_builtin(
            tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
            vec![
                Value::String("Distance".into()),
                Value::String("fastcityblock".into()),
            ],
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("unsupported Distance"));
    }
}
