//! Pairwise distance and squareform helpers for Statistics and Machine Learning workflows.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const PDIST_NAME: &str = "pdist";
const PDIST2_NAME: &str = "pdist2";
const KNNSEARCH_NAME: &str = "knnsearch";
const SQUAREFORM_NAME: &str = "squareform";
const EPS: f64 = 1.0e-12;

const OUTPUT_D: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Pairwise distances.",
}];

const OUTPUT_D_I: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected pairwise distances.",
    },
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based row indices from X for selected distances.",
    },
];

const OUTPUT_Z: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square symmetric matrix or condensed distance vector.",
}];

const OUTPUT_IDX_D: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Idx",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based indices of nearest rows in X, or cell array of index vectors when IncludeTies is true.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Distances to nearest rows in X, or cell array of distance vectors when IncludeTies is true.",
    },
];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation matrix with observations in rows.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second observation matrix with observations in rows.",
};

const PARAM_DISTANCE_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Distance metric and optional metric parameter.",
};

const PARAM_PDIST2_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Distance metric, metric parameter, or Smallest/Largest selection options.",
};

const PARAM_KNNSEARCH_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as K, Distance, P, Cov, Scale, IncludeTies, NSMethod, BucketSize, CacheSize, and SortIndices.",
};

const PARAM_SQUAREFORM_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Condensed distance vector or square distance matrix.",
};

const PARAM_SQUAREFORM_FORCE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "force",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Conversion force option: 'tomatrix' or 'tovector'.",
};

const INPUTS_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUTS_X_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_DISTANCE_OPTIONS];
const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_PDIST2_OPTIONS];
const INPUTS_X_Y_KNN_OPTIONS: [BuiltinParamDescriptor; 3] =
    [PARAM_X, PARAM_Y, PARAM_KNNSEARCH_OPTIONS];
const INPUTS_Z: [BuiltinParamDescriptor; 1] = [PARAM_SQUAREFORM_INPUT];
const INPUTS_Z_OPTIONS: [BuiltinParamDescriptor; 2] =
    [PARAM_SQUAREFORM_INPUT, PARAM_SQUAREFORM_FORCE];

const PDIST_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "D = pdist(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "D = pdist(X, Distance, DistParameter)",
        inputs: &INPUTS_X_OPTIONS,
        outputs: &OUTPUT_D,
    },
];

const PDIST2_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "D = pdist2(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "D = pdist2(X, Y, Distance, DistParameter, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_D_I,
    },
];

const KNNSEARCH_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Idx = knnsearch(X, Y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_IDX_D,
    },
    BuiltinSignatureDescriptor {
        label: "[Idx, D] = knnsearch(X, Y, Name, Value)",
        inputs: &INPUTS_X_Y_KNN_OPTIONS,
        outputs: &OUTPUT_IDX_D,
    },
];

const SQUAREFORM_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = squareform(Z)",
        inputs: &INPUTS_Z,
        outputs: &OUTPUT_Z,
    },
    BuiltinSignatureDescriptor {
        label: "Y = squareform(Z, Force)",
        inputs: &INPUTS_Z_OPTIONS,
        outputs: &OUTPUT_Z,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISTANCE.INVALID_ARGUMENT",
    identifier: Some("RunMat:distance:InvalidArgument"),
    when: "Inputs, dimensions, metrics, metric parameters, or selection options are malformed.",
    message: "distance helper: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISTANCE.INTERNAL",
    identifier: Some("RunMat:distance:Internal"),
    when: "RunMat cannot allocate or construct a distance output.",
    message: "distance helper: internal error",
};

const DISTANCE_ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const PDIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PDIST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DISTANCE_ERRORS,
};

pub const PDIST2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PDIST2_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DISTANCE_ERRORS,
};

pub const KNNSEARCH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &KNNSEARCH_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DISTANCE_ERRORS,
};

pub const SQUAREFORM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SQUAREFORM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DISTANCE_ERRORS,
};

fn matrix_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![None, None]),
    }
}

fn vector_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![Some(1), None]),
    }
}

fn distance_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    distance_descriptor_error(name, message, &ERROR_INVALID_ARGUMENT)
}

fn internal_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    distance_descriptor_error(name, message, &ERROR_INTERNAL)
}

fn distance_descriptor_error(
    name: &'static str,
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(name);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone, Debug)]
enum DistanceMetric {
    Euclidean,
    SquareEuclidean,
    Cityblock,
    Chebychev,
    Minkowski(f64),
    Seuclidean(Vec<f64>),
    Mahalanobis(Vec<f64>),
    Cosine,
    Correlation,
    Hamming,
    Jaccard,
    Spearman,
}

#[derive(Clone, Copy, Debug)]
enum Selection {
    Smallest(usize),
    Largest(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SquareformForce {
    Auto,
    ToMatrix,
    ToVector,
}

#[derive(Clone, Copy, Debug)]
struct KnnOptions {
    k: usize,
    include_ties: bool,
}

#[runtime_builtin(
    name = "pdist",
    category = "stats/ml",
    summary = "Compute pairwise distances between rows of an observation matrix.",
    keywords = "pdist,pairwise,distance,statistics,machine learning,clustering",
    type_resolver(vector_type),
    descriptor(crate::builtins::stats::ml::distance::PDIST_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn pdist_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let x = value_to_matrix(PDIST_NAME, x).await?;
    let metric = parse_metric(PDIST_NAME, &x, None, rest).await?;
    let distances = condensed_distances(PDIST_NAME, &x, &metric)?;
    Ok(Value::Tensor(distances))
}

#[runtime_builtin(
    name = "pdist2",
    category = "stats/ml",
    summary = "Compute pairwise distances between two sets of observations.",
    keywords = "pdist2,pairwise,distance,nearest neighbor,statistics,machine learning",
    type_resolver(matrix_type),
    descriptor(crate::builtins::stats::ml::distance::PDIST2_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn pdist2_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let x = value_to_matrix(PDIST2_NAME, x).await?;
    let y = value_to_matrix(PDIST2_NAME, y).await?;
    if x.cols != y.cols {
        return Err(distance_error(
            PDIST2_NAME,
            "pdist2: X and Y must have the same number of columns",
        ));
    }
    let (metric_args, selection) = split_pdist2_options(rest)?;
    let metric = parse_metric(PDIST2_NAME, &x, Some(&y), metric_args).await?;
    let distances = distance_matrix(PDIST2_NAME, &x, &y, &metric)?;
    match selection {
        Some(selection) => selected_distance_outputs(&distances, selection),
        None => {
            if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
                return Err(distance_error(
                    PDIST2_NAME,
                    "pdist2: second output is only available with Smallest or Largest",
                ));
            }
            match crate::output_count::current_output_count() {
                Some(0) => Ok(Value::OutputList(Vec::new())),
                Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                    out_count,
                    vec![Value::Tensor(distances)],
                )),
                None => Ok(Value::Tensor(distances)),
            }
        }
    }
}

#[runtime_builtin(
    name = "knnsearch",
    category = "stats/ml",
    summary = "Find k-nearest neighbors in an observation matrix.",
    keywords = "knnsearch,nearest neighbor,k nearest,statistics,machine learning,distance",
    type_resolver(matrix_type),
    descriptor(crate::builtins::stats::ml::distance::KNNSEARCH_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn knnsearch_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let x = value_to_matrix(KNNSEARCH_NAME, x).await?;
    let y = value_to_matrix(KNNSEARCH_NAME, y).await?;
    if x.cols != y.cols {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: X and Y must have the same number of columns",
        ));
    }
    let (metric_args, options) = parse_knnsearch_options(rest).await?;
    if options.k > x.rows {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: K must be <= size(X,1)",
        ));
    }
    let metric = parse_metric(KNNSEARCH_NAME, &x, Some(&y), metric_args).await?;
    knnsearch_outputs(&x, &y, &metric, options)
}

#[runtime_builtin(
    name = "squareform",
    category = "stats/ml",
    summary = "Convert between condensed distance vectors and square distance matrices.",
    keywords = "squareform,distance,condensed,symmetric,statistics,machine learning",
    type_resolver(matrix_type),
    descriptor(crate::builtins::stats::ml::distance::SQUAREFORM_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn squareform_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let tensor = value_to_matrix(SQUAREFORM_NAME, value).await?;
    let force = parse_squareform_force(rest)?;
    let output = squareform_compute(tensor, force)?;
    Ok(Value::Tensor(output))
}

pub(super) async fn condensed_distances_from_metric_args(
    name: &'static str,
    x: &Tensor,
    args: Vec<Value>,
) -> BuiltinResult<Tensor> {
    let metric = parse_metric(name, x, None, args).await?;
    condensed_distances(name, x, &metric)
}

async fn value_to_matrix(name: &'static str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| distance_error(name, format!("{name}: {err}")))?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| distance_error(name, format!("{name}: {err}")))?;
    if tensor.shape.len() > 2 {
        return Err(distance_error(
            name,
            format!("{name}: input must be a numeric vector or 2-D matrix"),
        ));
    }
    Ok(tensor)
}

async fn gather_value(name: &'static str, value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| distance_error(name, format!("{name}: {err}")))
}

async fn parse_metric(
    name: &'static str,
    x: &Tensor,
    _y: Option<&Tensor>,
    args: Vec<Value>,
) -> BuiltinResult<DistanceMetric> {
    if args.is_empty() {
        return Ok(DistanceMetric::Euclidean);
    }
    let metric_name = keyword_of(&args[0])
        .ok_or_else(|| distance_error(name, format!("{name}: distance metric must be text")))?;
    let metric_name = metric_name.to_ascii_lowercase();
    let rest = args[1..].to_vec();
    match metric_name.as_str() {
        "euclidean" => {
            ensure_no_metric_parameter(name, &rest, "euclidean").map(|()| DistanceMetric::Euclidean)
        }
        "squaredeuclidean" | "sqeuclidean" | "squared euclidean" => {
            ensure_no_metric_parameter(name, &rest, "squaredeuclidean")
                .map(|()| DistanceMetric::SquareEuclidean)
        }
        "cityblock" | "city block" | "manhattan" => {
            ensure_no_metric_parameter(name, &rest, "cityblock").map(|()| DistanceMetric::Cityblock)
        }
        "chebychev" | "chebyshev" | "cheby" => {
            ensure_no_metric_parameter(name, &rest, "chebychev").map(|()| DistanceMetric::Chebychev)
        }
        "minkowski" => {
            let p = if rest.is_empty() {
                2.0
            } else {
                scalar_parameter(name, &rest[0], "minkowski exponent").await?
            };
            if rest.len() > 1 || !p.is_finite() || p <= 0.0 {
                return Err(distance_error(
                    name,
                    format!("{name}: Minkowski exponent must be a positive finite scalar"),
                ));
            }
            Ok(DistanceMetric::Minkowski(p))
        }
        "seuclidean" | "standardizedeuclidean" | "standardized euclidean" => {
            let scale = if rest.is_empty() {
                variance_scale(name, x)?
            } else {
                let stddev =
                    vector_parameter(name, rest[0].clone(), x.cols, "seuclidean scale").await?;
                stddev
                    .into_iter()
                    .map(|value| value * value)
                    .collect::<Vec<_>>()
            };
            if rest.len() > 1 || scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                return Err(distance_error(
                    name,
                    format!("{name}: seuclidean scale values must be positive and finite"),
                ));
            }
            Ok(DistanceMetric::Seuclidean(scale))
        }
        "mahalanobis" | "mahal" => {
            let covariance = if rest.is_empty() {
                covariance_matrix(name, x)?
            } else {
                matrix_parameter(name, rest[0].clone(), x.cols, "mahalanobis covariance").await?
            };
            if rest.len() > 1 {
                return Err(distance_error(
                    name,
                    format!("{name}: mahalanobis accepts at most one covariance matrix"),
                ));
            }
            Ok(DistanceMetric::Mahalanobis(invert_spd_matrix(
                name,
                &covariance,
                x.cols,
                "mahalanobis covariance",
            )?))
        }
        "cosine" => {
            ensure_no_metric_parameter(name, &rest, "cosine").map(|()| DistanceMetric::Cosine)
        }
        "correlation" => ensure_no_metric_parameter(name, &rest, "correlation")
            .map(|()| DistanceMetric::Correlation),
        "hamming" => {
            ensure_no_metric_parameter(name, &rest, "hamming").map(|()| DistanceMetric::Hamming)
        }
        "jaccard" => {
            ensure_no_metric_parameter(name, &rest, "jaccard").map(|()| DistanceMetric::Jaccard)
        }
        "spearman" => {
            ensure_no_metric_parameter(name, &rest, "spearman").map(|()| DistanceMetric::Spearman)
        }
        other => Err(distance_error(
            name,
            format!("{name}: unsupported distance metric '{other}'"),
        )),
    }
}

fn ensure_no_metric_parameter(
    name: &'static str,
    args: &[Value],
    metric: &str,
) -> BuiltinResult<()> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(distance_error(
            name,
            format!("{name}: {metric} distance does not accept a distance parameter"),
        ))
    }
}

async fn scalar_parameter(name: &'static str, value: &Value, label: &str) -> BuiltinResult<f64> {
    let gathered = gather_value(name, value.clone()).await?;
    match gathered {
        Value::Num(value) => Ok(value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        other => Err(distance_error(
            name,
            format!("{name}: {label} must be a numeric scalar, got {other:?}"),
        )),
    }
}

async fn vector_parameter(
    name: &'static str,
    value: Value,
    expected_len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    let tensor = value_to_matrix(name, value).await?;
    if tensor.data.len() != expected_len {
        return Err(distance_error(
            name,
            format!("{name}: {label} length must match the number of columns"),
        ));
    }
    Ok(tensor.data)
}

async fn matrix_parameter(
    name: &'static str,
    value: Value,
    expected_size: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    let tensor = value_to_matrix(name, value).await?;
    if tensor.rows != expected_size || tensor.cols != expected_size {
        return Err(distance_error(
            name,
            format!("{name}: {label} must be a square matrix matching the number of columns"),
        ));
    }
    Ok(tensor.data)
}

fn split_pdist2_options(args: Vec<Value>) -> BuiltinResult<(Vec<Value>, Option<Selection>)> {
    let mut metric_args = Vec::with_capacity(args.len());
    let mut selection = None;
    let mut idx = 0usize;
    while idx < args.len() {
        let selector = keyword_of(&args[idx]).and_then(|text| {
            let lower = text.to_ascii_lowercase();
            if lower == "smallest" || lower == "largest" {
                Some(lower)
            } else {
                None
            }
        });
        if let Some(selector) = selector {
            if selection.is_some() || idx + 1 >= args.len() {
                return Err(distance_error(
                    PDIST2_NAME,
                    "pdist2: Smallest or Largest must be a single name-value pair",
                ));
            }
            let k = parse_positive_integer(PDIST2_NAME, &args[idx + 1], "pdist2 selection count")?;
            selection = Some(if selector == "smallest" {
                Selection::Smallest(k)
            } else {
                Selection::Largest(k)
            });
            idx += 2;
        } else {
            metric_args.push(args[idx].clone());
            idx += 1;
        }
    }
    Ok((metric_args, selection))
}

fn parse_positive_integer(name: &'static str, value: &Value, label: &str) -> BuiltinResult<usize> {
    let raw = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        Value::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        Value::LogicalArray(array) if array.data.len() == 1 => f64::from(array.data[0] != 0),
        other => {
            return Err(distance_error(
                name,
                format!("{name}: {label} must be a positive integer scalar, got {other:?}"),
            ))
        }
    };
    if !raw.is_finite() || raw < 1.0 || raw.fract().abs() > EPS || raw > usize::MAX as f64 {
        return Err(distance_error(
            name,
            format!("{name}: {label} must be a positive integer scalar"),
        ));
    }
    Ok(raw as usize)
}

async fn parse_knnsearch_options(args: Vec<Value>) -> BuiltinResult<(Vec<Value>, KnnOptions)> {
    let mut k = 1usize;
    let mut include_ties = false;
    let mut sort_indices = true;
    let mut metric_name: Option<Value> = None;
    let mut metric_keyword = "euclidean".to_string();
    let mut metric_parameter: Option<Value> = None;
    let mut ns_method: Option<String> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        let name = keyword_of(&args[idx]).ok_or_else(|| {
            distance_error(
                KNNSEARCH_NAME,
                "knnsearch: options must be supplied as name-value pairs",
            )
        })?;
        if idx + 1 >= args.len() {
            return Err(distance_error(
                KNNSEARCH_NAME,
                format!("knnsearch: missing value for option '{name}'"),
            ));
        }
        let value = args[idx + 1].clone();
        match name.as_str() {
            "k" => k = parse_positive_integer(KNNSEARCH_NAME, &value, "K")?,
            "includeties" => include_ties = parse_logical_scalar(&value, "IncludeTies")?,
            "sortindices" => sort_indices = parse_logical_scalar(&value, "SortIndices")?,
            "distance" => {
                let distance = keyword_of(&value).ok_or_else(|| {
                    distance_error(KNNSEARCH_NAME, "knnsearch: Distance must be text")
                })?;
                let normalized = match distance.as_str() {
                    "fasteuclidean" => "euclidean",
                    "fastseuclidean" => "seuclidean",
                    other => other,
                };
                metric_keyword = normalized.to_string();
                metric_name = Some(Value::from(normalized));
            }
            "p" | "cov" | "scale" => {
                if metric_parameter.is_some() {
                    return Err(distance_error(
                        KNNSEARCH_NAME,
                        "knnsearch: only one distance parameter may be specified",
                    ));
                }
                metric_parameter = Some(value);
            }
            "nsmethod" => {
                let method = keyword_of(&value).ok_or_else(|| {
                    distance_error(KNNSEARCH_NAME, "knnsearch: NSMethod must be text")
                })?;
                match method.as_str() {
                    "exhaustive" | "kdtree" => ns_method = Some(method),
                    other => {
                        return Err(distance_error(
                            KNNSEARCH_NAME,
                            format!("knnsearch: unsupported NSMethod '{other}'"),
                        ))
                    }
                }
            }
            "cachesize" => {
                if keyword_of(&value).as_deref() != Some("maximal") {
                    let raw = scalar_parameter(KNNSEARCH_NAME, &value, "CacheSize").await?;
                    if !raw.is_finite() || raw <= 0.0 {
                        return Err(distance_error(
                            KNNSEARCH_NAME,
                            "knnsearch: CacheSize must be positive",
                        ));
                    }
                }
            }
            "bucketsize" => {
                parse_positive_integer(KNNSEARCH_NAME, &value, "BucketSize")?;
            }
            other => {
                return Err(distance_error(
                    KNNSEARCH_NAME,
                    format!("knnsearch: unsupported option '{other}'"),
                ))
            }
        }
        idx += 2;
    }
    if matches!(ns_method.as_deref(), Some("kdtree"))
        && !matches!(
            metric_keyword.as_str(),
            "euclidean" | "cityblock" | "chebychev" | "chebyshev" | "cheby" | "minkowski"
        )
    {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: NSMethod='kdtree' is only valid for euclidean, cityblock, chebychev, or minkowski distances",
        ));
    }
    let _ = sort_indices;
    let mut metric_args = Vec::new();
    if let Some(metric) = metric_name {
        metric_args.push(metric);
        if let Some(parameter) = metric_parameter {
            metric_args.push(parameter);
        }
    } else if metric_parameter.is_some() {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: P, Cov, or Scale requires a Distance option",
        ));
    }
    Ok((metric_args, KnnOptions { k, include_ties }))
}

fn parse_logical_scalar(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Int(value) if value.to_i64() == 0 || value.to_i64() == 1 => Ok(value.to_i64() != 0),
        Value::Tensor(tensor)
            if tensor.data.len() == 1 && (tensor.data[0] == 0.0 || tensor.data[0] == 1.0) =>
        {
            Ok(tensor.data[0] != 0.0)
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(distance_error(
            KNNSEARCH_NAME,
            format!("knnsearch: {label} must be logical scalar, got {other:?}"),
        )),
    }
}

fn condensed_distances(
    name: &'static str,
    x: &Tensor,
    metric: &DistanceMetric,
) -> BuiltinResult<Tensor> {
    let rows = x.rows;
    let len = rows
        .checked_mul(rows.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| internal_error(name, "pdist: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error(name, "pdist: failed to allocate output vector"))?;
    for col in 0..rows {
        for row in (col + 1)..rows {
            out.push(row_distance(name, x, row, x, col, metric)?);
        }
    }
    Tensor::new(out, vec![1, len]).map_err(|err| internal_error(name, format!("pdist: {err}")))
}

fn distance_matrix(
    name: &'static str,
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
) -> BuiltinResult<Tensor> {
    let len = x
        .rows
        .checked_mul(y.rows)
        .ok_or_else(|| internal_error(name, "pdist2: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error(name, "pdist2: failed to allocate output matrix"))?;
    for col in 0..y.rows {
        for row in 0..x.rows {
            out.push(row_distance(name, x, row, y, col, metric)?);
        }
    }
    Tensor::new(out, vec![x.rows, y.rows])
        .map_err(|err| internal_error(name, format!("pdist2: {err}")))
}

fn row_distance(
    name: &'static str,
    x: &Tensor,
    xi: usize,
    y: &Tensor,
    yi: usize,
    metric: &DistanceMetric,
) -> BuiltinResult<f64> {
    if row_has_nan(x, xi) || row_has_nan(y, yi) {
        return Ok(f64::NAN);
    }
    match metric {
        DistanceMetric::Euclidean => {
            let mut sum = 0.0;
            for col in 0..x.cols {
                let diff = row_value(x, xi, col) - row_value(y, yi, col);
                sum += diff * diff;
            }
            Ok(sum.sqrt())
        }
        DistanceMetric::SquareEuclidean => {
            let mut sum = 0.0;
            for col in 0..x.cols {
                let diff = row_value(x, xi, col) - row_value(y, yi, col);
                sum += diff * diff;
            }
            Ok(sum)
        }
        DistanceMetric::Cityblock => {
            let mut sum = 0.0;
            for col in 0..x.cols {
                sum += (row_value(x, xi, col) - row_value(y, yi, col)).abs();
            }
            Ok(sum)
        }
        DistanceMetric::Chebychev => {
            let mut max = 0.0;
            for col in 0..x.cols {
                let diff = (row_value(x, xi, col) - row_value(y, yi, col)).abs();
                if diff > max {
                    max = diff;
                }
            }
            Ok(max)
        }
        DistanceMetric::Minkowski(p) => {
            let mut sum = 0.0;
            for col in 0..x.cols {
                sum += (row_value(x, xi, col) - row_value(y, yi, col))
                    .abs()
                    .powf(*p);
            }
            Ok(sum.powf(1.0 / p))
        }
        DistanceMetric::Seuclidean(scale) => {
            let mut sum = 0.0;
            for (col, scale_value) in scale.iter().enumerate().take(x.cols) {
                let diff = row_value(x, xi, col) - row_value(y, yi, col);
                sum += diff * diff / *scale_value;
            }
            Ok(sum.sqrt())
        }
        DistanceMetric::Mahalanobis(inv_covariance) => {
            let diff: Vec<f64> = (0..x.cols)
                .map(|col| row_value(x, xi, col) - row_value(y, yi, col))
                .collect();
            let mut total = 0.0;
            for row in 0..x.cols {
                for col in 0..x.cols {
                    total += diff[row] * matrix_value(inv_covariance, x.cols, row, col) * diff[col];
                }
            }
            if total < -EPS {
                return Err(distance_error(
                    name,
                    format!("{name}: mahalanobis covariance produced a negative quadratic form"),
                ));
            }
            Ok(total.max(0.0).sqrt())
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0;
            let mut x_norm = 0.0;
            let mut y_norm = 0.0;
            for col in 0..x.cols {
                let xv = row_value(x, xi, col);
                let yv = row_value(y, yi, col);
                dot += xv * yv;
                x_norm += xv * xv;
                y_norm += yv * yv;
            }
            if x_norm <= EPS || y_norm <= EPS {
                Ok(f64::NAN)
            } else {
                Ok(1.0 - dot / (x_norm.sqrt() * y_norm.sqrt()))
            }
        }
        DistanceMetric::Correlation => correlation_distance(x, xi, y, yi),
        DistanceMetric::Hamming => {
            let mut unequal = 0usize;
            for col in 0..x.cols {
                if row_value(x, xi, col) != row_value(y, yi, col) {
                    unequal += 1;
                }
            }
            Ok(unequal as f64 / x.cols as f64)
        }
        DistanceMetric::Jaccard => {
            let mut unequal_nonzero = 0usize;
            let mut nonzero_union = 0usize;
            for col in 0..x.cols {
                let xv = row_value(x, xi, col);
                let yv = row_value(y, yi, col);
                if xv != 0.0 || yv != 0.0 {
                    nonzero_union += 1;
                    if xv != yv {
                        unequal_nonzero += 1;
                    }
                }
            }
            if nonzero_union == 0 {
                Ok(0.0)
            } else {
                Ok(unequal_nonzero as f64 / nonzero_union as f64)
            }
        }
        DistanceMetric::Spearman => spearman_distance(x, xi, y, yi),
    }
}

fn row_value(tensor: &Tensor, row: usize, col: usize) -> f64 {
    tensor.data[col * tensor.rows + row]
}

fn row_has_nan(tensor: &Tensor, row: usize) -> bool {
    (0..tensor.cols).any(|col| row_value(tensor, row, col).is_nan())
}

fn matrix_value(data: &[f64], rows: usize, row: usize, col: usize) -> f64 {
    data[col * rows + row]
}

fn correlation_distance(x: &Tensor, xi: usize, y: &Tensor, yi: usize) -> BuiltinResult<f64> {
    let cols = x.cols;
    if cols == 0 {
        return Ok(f64::NAN);
    }
    let x_mean = (0..cols).map(|col| row_value(x, xi, col)).sum::<f64>() / cols as f64;
    let y_mean = (0..cols).map(|col| row_value(y, yi, col)).sum::<f64>() / cols as f64;
    let mut dot = 0.0;
    let mut x_norm = 0.0;
    let mut y_norm = 0.0;
    for col in 0..cols {
        let xv = row_value(x, xi, col) - x_mean;
        let yv = row_value(y, yi, col) - y_mean;
        dot += xv * yv;
        x_norm += xv * xv;
        y_norm += yv * yv;
    }
    if x_norm <= EPS || y_norm <= EPS {
        Ok(f64::NAN)
    } else {
        Ok(1.0 - dot / (x_norm.sqrt() * y_norm.sqrt()))
    }
}

fn spearman_distance(x: &Tensor, xi: usize, y: &Tensor, yi: usize) -> BuiltinResult<f64> {
    let x_rank = row_ranks(x, xi);
    let y_rank = row_ranks(y, yi);
    let rank_x = Tensor::new(x_rank, vec![1, x.cols])
        .map_err(|err| internal_error(PDIST_NAME, format!("spearman: {err}")))?;
    let rank_y = Tensor::new(y_rank, vec![1, y.cols])
        .map_err(|err| internal_error(PDIST_NAME, format!("spearman: {err}")))?;
    correlation_distance(&rank_x, 0, &rank_y, 0)
}

fn row_ranks(tensor: &Tensor, row: usize) -> Vec<f64> {
    let mut values = (0..tensor.cols)
        .map(|col| (col, row_value(tensor, row, col)))
        .collect::<Vec<_>>();
    values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
    let mut ranks = vec![0.0; tensor.cols];
    let mut idx = 0usize;
    while idx < values.len() {
        let start = idx;
        let value = values[idx].1;
        while idx < values.len() && values[idx].1 == value {
            idx += 1;
        }
        let rank = (start + 1 + idx) as f64 / 2.0;
        for entry in &values[start..idx] {
            ranks[entry.0] = rank;
        }
    }
    ranks
}

fn variance_scale(name: &'static str, x: &Tensor) -> BuiltinResult<Vec<f64>> {
    let cols = x.cols;
    let mut counts = vec![0usize; cols];
    let mut means = vec![0.0; cols];
    for row in 0..x.rows {
        for col in 0..cols {
            let value = row_value(x, row, col);
            if !value.is_nan() {
                counts[col] += 1;
                means[col] += value;
            }
        }
    }
    for col in 0..cols {
        if counts[col] < 2 {
            return Err(distance_error(
                name,
                format!("{name}: at least two non-NaN observations are required per column for seuclidean scale"),
            ));
        }
        means[col] /= counts[col] as f64;
    }
    let mut vars = vec![0.0; cols];
    for row in 0..x.rows {
        for col in 0..cols {
            let value = row_value(x, row, col);
            if !value.is_nan() {
                let diff = value - means[col];
                vars[col] += diff * diff;
            }
        }
    }
    for col in 0..cols {
        vars[col] /= (counts[col] - 1) as f64;
        if vars[col] <= EPS {
            vars[col] = f64::NAN;
        }
    }
    Ok(vars)
}

fn covariance_matrix(name: &'static str, x: &Tensor) -> BuiltinResult<Vec<f64>> {
    let cols = x.cols;
    let rows = x.rows;
    if rows < 2 {
        return Err(distance_error(
            name,
            format!("{name}: at least two observations are required for mahalanobis covariance"),
        ));
    }
    let mut means = vec![0.0; cols];
    for row in 0..x.rows {
        for (col, mean) in means.iter_mut().enumerate().take(cols) {
            *mean += row_value(x, row, col);
        }
    }
    for mean in &mut means {
        *mean /= rows as f64;
    }
    let len = cols
        .checked_mul(cols)
        .ok_or_else(|| internal_error(name, "mahalanobis covariance size overflow"))?;
    let mut covariance = Vec::new();
    covariance
        .try_reserve(len)
        .map_err(|_| internal_error(name, "mahalanobis covariance allocation failed"))?;
    covariance.resize(len, 0.0);
    accumulate_covariance(&mut covariance, x, &means);
    for value in &mut covariance {
        *value /= (rows - 1) as f64;
    }
    Ok(covariance)
}

fn accumulate_covariance(out: &mut [f64], data: &Tensor, means: &[f64]) {
    let cols = data.cols;
    for row in 0..data.rows {
        for a in 0..cols {
            let av = row_value(data, row, a) - means[a];
            for b in 0..cols {
                let bv = row_value(data, row, b) - means[b];
                out[b * cols + a] += av * bv;
            }
        }
    }
}

fn invert_spd_matrix(
    name: &'static str,
    matrix: &[f64],
    size: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    validate_symmetric_positive_definite(name, matrix, size, label)?;
    let width = size
        .checked_mul(2)
        .ok_or_else(|| internal_error(name, format!("{label}: augmented width overflow")))?;
    let len = size
        .checked_mul(width)
        .ok_or_else(|| internal_error(name, format!("{label}: augmented matrix overflow")))?;
    let mut aug = Vec::new();
    aug.try_reserve(len).map_err(|_| {
        internal_error(name, format!("{label}: augmented matrix allocation failed"))
    })?;
    aug.resize(len, 0.0);
    for row in 0..size {
        for col in 0..size {
            aug[row * width + col] = matrix_value(matrix, size, row, col);
        }
        aug[row * width + size + row] = 1.0;
    }
    for pivot in 0..size {
        let mut best = pivot;
        let mut best_abs = aug[pivot * width + pivot].abs();
        for row in (pivot + 1)..size {
            let value = aug[row * width + pivot].abs();
            if value > best_abs {
                best = row;
                best_abs = value;
            }
        }
        if best_abs <= EPS {
            return Err(distance_error(
                name,
                format!("{name}: {label} must be nonsingular"),
            ));
        }
        if best != pivot {
            for col in 0..width {
                aug.swap(pivot * width + col, best * width + col);
            }
        }
        let pivot_value = aug[pivot * width + pivot];
        for col in 0..width {
            aug[pivot * width + col] /= pivot_value;
        }
        for row in 0..size {
            if row == pivot {
                continue;
            }
            let factor = aug[row * width + pivot];
            if factor == 0.0 {
                continue;
            }
            for col in 0..width {
                aug[row * width + col] -= factor * aug[pivot * width + col];
            }
        }
    }
    let inv_len = size
        .checked_mul(size)
        .ok_or_else(|| internal_error(name, format!("{label}: inverse size overflow")))?;
    let mut inv = Vec::new();
    inv.try_reserve(inv_len)
        .map_err(|_| internal_error(name, format!("{label}: inverse allocation failed")))?;
    inv.resize(inv_len, 0.0);
    for row in 0..size {
        for col in 0..size {
            inv[col * size + row] = aug[row * width + size + col];
        }
    }
    Ok(inv)
}

fn validate_symmetric_positive_definite(
    name: &'static str,
    matrix: &[f64],
    size: usize,
    label: &str,
) -> BuiltinResult<()> {
    for row in 0..size {
        for col in 0..size {
            let a = matrix_value(matrix, size, row, col);
            let b = matrix_value(matrix, size, col, row);
            if !a.is_finite() || (a - b).abs() > 1.0e-9 {
                return Err(distance_error(
                    name,
                    format!("{name}: {label} must be symmetric positive definite"),
                ));
            }
        }
    }
    let len = size
        .checked_mul(size)
        .ok_or_else(|| internal_error(name, format!("{label}: Cholesky size overflow")))?;
    let mut chol = Vec::new();
    chol.try_reserve(len)
        .map_err(|_| internal_error(name, format!("{label}: Cholesky allocation failed")))?;
    chol.resize(len, 0.0);
    for i in 0..size {
        for j in 0..=i {
            let mut sum = matrix_value(matrix, size, i, j);
            for k in 0..j {
                sum -= chol[i * size + k] * chol[j * size + k];
            }
            if i == j {
                if sum <= EPS {
                    return Err(distance_error(
                        name,
                        format!("{name}: {label} must be symmetric positive definite"),
                    ));
                }
                chol[i * size + j] = sum.sqrt();
            } else {
                chol[i * size + j] = sum / chol[j * size + j];
            }
        }
    }
    Ok(())
}

fn selected_distance_outputs(distances: &Tensor, selection: Selection) -> BuiltinResult<Value> {
    let (selected_distances, selected_indices) = select_distances(distances, selection)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![
                Value::Tensor(selected_distances),
                Value::Tensor(selected_indices),
            ],
        )),
        None => Ok(Value::Tensor(selected_distances)),
    }
}

fn select_distances(distances: &Tensor, selection: Selection) -> BuiltinResult<(Tensor, Tensor)> {
    let k = match selection {
        Selection::Smallest(k) | Selection::Largest(k) => k,
    };
    if k > distances.rows {
        return Err(distance_error(
            PDIST2_NAME,
            "pdist2: Smallest or Largest count must be <= size(X,1)",
        ));
    }
    let len = k
        .checked_mul(distances.cols)
        .ok_or_else(|| internal_error(PDIST2_NAME, "pdist2: selected output size overflow"))?;
    let mut out = Vec::new();
    let mut indices = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error(PDIST2_NAME, "pdist2: selected output allocation failed"))?;
    indices
        .try_reserve(len)
        .map_err(|_| internal_error(PDIST2_NAME, "pdist2: selected index allocation failed"))?;
    for col in 0..distances.cols {
        let mut values = (0..distances.rows)
            .map(|row| (row_value(distances, row, col), row + 1))
            .collect::<Vec<_>>();
        match selection {
            Selection::Smallest(_) => {
                values.sort_by(|a, b| distance_order_ascending(a.0, a.1, b.0, b.1));
            }
            Selection::Largest(_) => {
                values.sort_by(|a, b| distance_order_descending(a.0, a.1, b.0, b.1));
            }
        }
        for (distance, index) in values.iter().take(k) {
            out.push(*distance);
            indices.push(*index as f64);
        }
    }
    let distance_tensor = Tensor::new(out, vec![k, distances.cols])
        .map_err(|err| internal_error(PDIST2_NAME, format!("pdist2: {err}")))?;
    let index_tensor = Tensor::new(indices, vec![k, distances.cols])
        .map_err(|err| internal_error(PDIST2_NAME, format!("pdist2: {err}")))?;
    Ok((distance_tensor, index_tensor))
}

fn knnsearch_outputs(
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
    options: KnnOptions,
) -> BuiltinResult<Value> {
    if options.include_ties {
        let (indices, distances) = knnsearch_tie_cells(x, y, metric, options.k)?;
        return match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![Value::Cell(indices), Value::Cell(distances)],
            )),
            None => Ok(Value::Cell(indices)),
        };
    }

    let (indices, distances) = knnsearch_numeric_outputs(x, y, metric, options.k)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![Value::Tensor(indices), Value::Tensor(distances)],
        )),
        None => Ok(Value::Tensor(indices)),
    }
}

fn knnsearch_numeric_outputs(
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
    k: usize,
) -> BuiltinResult<(Tensor, Tensor)> {
    let len = y
        .rows
        .checked_mul(k)
        .ok_or_else(|| internal_error(KNNSEARCH_NAME, "knnsearch: output size overflow"))?;
    let mut indices = Vec::new();
    let mut distances = Vec::new();
    indices
        .try_reserve(len)
        .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: index allocation failed"))?;
    distances
        .try_reserve(len)
        .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: distance allocation failed"))?;
    indices.resize(len, 0.0);
    distances.resize(len, 0.0);
    for query in 0..y.rows {
        let values = sorted_neighbors(x, y, metric, query)?;
        for (rank, (distance, index)) in values.iter().take(k).enumerate() {
            let pos = rank * y.rows + query;
            indices[pos] = *index as f64;
            distances[pos] = *distance;
        }
    }
    let index_tensor = Tensor::new(indices, vec![y.rows, k])
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
    let distance_tensor = Tensor::new(distances, vec![y.rows, k])
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
    Ok((index_tensor, distance_tensor))
}

fn knnsearch_tie_cells(
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
    k: usize,
) -> BuiltinResult<(CellArray, CellArray)> {
    let mut index_cells = Vec::new();
    let mut distance_cells = Vec::new();
    index_cells
        .try_reserve(y.rows)
        .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: cell allocation failed"))?;
    distance_cells
        .try_reserve(y.rows)
        .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: cell allocation failed"))?;
    for query in 0..y.rows {
        let values = sorted_neighbors(x, y, metric, query)?;
        let cutoff = values[k - 1].0;
        let mut selected = Vec::new();
        for (distance, index) in values {
            let include = if cutoff.is_nan() {
                distance.is_nan()
            } else {
                !distance.is_nan() && (distance - cutoff).abs() <= EPS
            };
            if include || selected.len() < k {
                selected.push((distance, index));
            }
            if !include && selected.len() >= k {
                break;
            }
        }
        let len = selected.len();
        let idx_tensor = Tensor::new(
            selected.iter().map(|(_, index)| *index as f64).collect(),
            vec![1, len],
        )
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
        let dist_tensor = Tensor::new(
            selected.iter().map(|(distance, _)| *distance).collect(),
            vec![1, len],
        )
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
        index_cells.push(Value::Tensor(idx_tensor));
        distance_cells.push(Value::Tensor(dist_tensor));
    }
    let indices = CellArray::new(index_cells, y.rows, 1)
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
    let distances = CellArray::new(distance_cells, y.rows, 1)
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
    Ok((indices, distances))
}

fn sorted_neighbors(
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
    query: usize,
) -> BuiltinResult<Vec<(f64, usize)>> {
    let mut values = Vec::new();
    values
        .try_reserve(x.rows)
        .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: neighbor allocation failed"))?;
    for row in 0..x.rows {
        values.push((
            row_distance(KNNSEARCH_NAME, x, row, y, query, metric)?,
            row + 1,
        ));
    }
    values.sort_by(|a, b| distance_order_ascending(a.0, a.1, b.0, b.1));
    Ok(values)
}

fn distance_order_ascending(
    left_distance: f64,
    left_index: usize,
    right_distance: f64,
    right_index: usize,
) -> Ordering {
    match (left_distance.is_nan(), right_distance.is_nan()) {
        (true, true) => left_index.cmp(&right_index),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left_distance
            .partial_cmp(&right_distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_index.cmp(&right_index)),
    }
}

fn distance_order_descending(
    left_distance: f64,
    left_index: usize,
    right_distance: f64,
    right_index: usize,
) -> Ordering {
    match (left_distance.is_nan(), right_distance.is_nan()) {
        (true, true) => left_index.cmp(&right_index),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => right_distance
            .partial_cmp(&left_distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_index.cmp(&right_index)),
    }
}

fn parse_squareform_force(args: Vec<Value>) -> BuiltinResult<SquareformForce> {
    if args.is_empty() {
        return Ok(SquareformForce::Auto);
    }
    if args.len() != 1 {
        return Err(distance_error(
            SQUAREFORM_NAME,
            "squareform: accepts at most one force option",
        ));
    }
    let text = keyword_of(&args[0]).ok_or_else(|| {
        distance_error(
            SQUAREFORM_NAME,
            "squareform: force option must be 'tomatrix' or 'tovector'",
        )
    })?;
    match text.to_ascii_lowercase().as_str() {
        "tomatrix" => Ok(SquareformForce::ToMatrix),
        "tovector" => Ok(SquareformForce::ToVector),
        other => Err(distance_error(
            SQUAREFORM_NAME,
            format!("squareform: unsupported force option '{other}'"),
        )),
    }
}

fn squareform_compute(tensor: Tensor, force: SquareformForce) -> BuiltinResult<Tensor> {
    let is_vector = tensor.rows == 1 || tensor.cols == 1;
    match force {
        SquareformForce::ToMatrix => vector_to_square(tensor),
        SquareformForce::ToVector => square_to_vector(tensor),
        SquareformForce::Auto if is_vector => vector_to_square(tensor),
        SquareformForce::Auto => square_to_vector(tensor),
    }
}

fn vector_to_square(tensor: Tensor) -> BuiltinResult<Tensor> {
    if !(tensor.rows == 1 || tensor.cols == 1) {
        return Err(distance_error(
            SQUAREFORM_NAME,
            "squareform: vector input is required for 'tomatrix'",
        ));
    }
    let len = tensor.data.len();
    let size = condensed_matrix_size(len).ok_or_else(|| {
        distance_error(
            SQUAREFORM_NAME,
            "squareform: vector length must be n*(n-1)/2 for an integer n",
        )
    })?;
    let out_len = size
        .checked_mul(size)
        .ok_or_else(|| internal_error(SQUAREFORM_NAME, "squareform: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve(out_len)
        .map_err(|_| internal_error(SQUAREFORM_NAME, "squareform: output allocation failed"))?;
    out.resize(out_len, 0.0);
    let mut idx = 0usize;
    for col in 0..size {
        for row in (col + 1)..size {
            let value = tensor.data[idx];
            out[col * size + row] = value;
            out[row * size + col] = value;
            idx += 1;
        }
    }
    Tensor::new(out, vec![size, size])
        .map_err(|err| internal_error(SQUAREFORM_NAME, format!("squareform: {err}")))
}

fn square_to_vector(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.rows != tensor.cols {
        return Err(distance_error(
            SQUAREFORM_NAME,
            "squareform: matrix input must be square",
        ));
    }
    for idx in 0..tensor.rows {
        if row_value(&tensor, idx, idx).abs() > EPS {
            return Err(distance_error(
                SQUAREFORM_NAME,
                "squareform: matrix diagonal must be zero",
            ));
        }
    }
    let len = tensor
        .rows
        .checked_mul(tensor.rows.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| internal_error(SQUAREFORM_NAME, "squareform: output size overflow"))?;
    let mut out = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error(SQUAREFORM_NAME, "squareform: output allocation failed"))?;
    for col in 0..tensor.rows {
        for row in (col + 1)..tensor.rows {
            let lower = row_value(&tensor, row, col);
            let upper = row_value(&tensor, col, row);
            if (lower - upper).abs() > EPS {
                return Err(distance_error(
                    SQUAREFORM_NAME,
                    "squareform: matrix must be symmetric",
                ));
            }
            out.push(upper);
        }
    }
    Tensor::new(out, vec![1, len])
        .map_err(|err| internal_error(SQUAREFORM_NAME, format!("squareform: {err}")))
}

fn condensed_matrix_size(len: usize) -> Option<usize> {
    if len == 0 {
        return Some(1);
    }
    let discriminant = 1usize.checked_add(len.checked_mul(8)?)?;
    let root = (discriminant as f64).sqrt().round() as usize;
    if root.checked_mul(root)? != discriminant {
        return None;
    }
    let numerator = root.checked_add(1)?;
    if numerator % 2 != 0 {
        return None;
    }
    Some(numerator / 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn tensor_out(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn pdist_default_and_cityblock_use_condensed_order() {
        let x = tensor(vec![0.0, 3.0, 4.0, 0.0, 0.0, 4.0, 0.0, 2.0], 4, 2);
        let out = block_on(pdist_builtin(x.clone(), Vec::new())).unwrap();
        let data = tensor_out(out).data;
        assert!((data[0] - 5.0).abs() < 1.0e-10);
        assert!((data[1] - 4.0).abs() < 1.0e-10);
        assert!((data[2] - 2.0).abs() < 1.0e-10);
        assert!((data[3] - 17.0_f64.sqrt()).abs() < 1.0e-10);
        assert!((data[4] - 13.0_f64.sqrt()).abs() < 1.0e-10);
        assert!((data[5] - 20.0_f64.sqrt()).abs() < 1.0e-10);

        let out = block_on(pdist_builtin(x, vec![Value::from("cityblock")])).unwrap();
        assert_eq!(tensor_out(out).data, vec![7.0, 4.0, 2.0, 5.0, 5.0, 6.0]);
    }

    #[test]
    fn pdist_supports_parameterized_metrics() {
        let x = tensor(vec![0.0, 3.0, 4.0, 0.0, 4.0, 0.0], 3, 2);
        let out = block_on(pdist_builtin(
            x.clone(),
            vec![Value::from("minkowski"), Value::Num(1.0)],
        ))
        .unwrap();
        assert_eq!(tensor_out(out).data, vec![7.0, 4.0, 5.0]);

        let out = block_on(pdist_builtin(
            x,
            vec![Value::from("seuclidean"), tensor(vec![1.0, 4.0], 1, 2)],
        ))
        .unwrap();
        assert!((tensor_out(out).data[0] - 10.0_f64.sqrt()).abs() < 1.0e-10);

        let out = block_on(pdist2_builtin(
            tensor(vec![0.0, 3.0, 0.0, 4.0], 2, 2),
            tensor(vec![0.0, 6.0], 1, 2),
            vec![Value::from("seuclidean")],
        ))
        .unwrap();
        assert!((tensor_out(out).data[0] - 4.5_f64.sqrt()).abs() < 1.0e-10);
    }

    #[test]
    fn nan_rows_propagate_and_invalid_covariance_is_rejected() {
        let x = tensor(vec![0.0, f64::NAN, 0.0, 1.0], 2, 2);
        let out = block_on(pdist_builtin(x, vec![Value::from("hamming")])).unwrap();
        assert!(tensor_out(out).data[0].is_nan());

        let err = block_on(pdist_builtin(
            tensor(vec![0.0, 1.0, 0.0, 1.0], 2, 2),
            vec![
                Value::from("mahalanobis"),
                tensor(vec![1.0, 2.0, 0.0, 1.0], 2, 2),
            ],
        ))
        .expect_err("asymmetric covariance should fail");
        assert!(err.to_string().contains("symmetric positive definite"));
    }

    #[test]
    fn pdist2_matrix_and_selection_modes_work() {
        let x = tensor(vec![0.0, 2.0, 0.0, 0.0], 2, 2);
        let y = tensor(vec![1.0, 3.0, 0.0, 0.0], 2, 2);
        let out = block_on(pdist2_builtin(
            x.clone(),
            y.clone(),
            vec![Value::from("squaredeuclidean")],
        ))
        .unwrap();
        assert_eq!(tensor_out(out).data, vec![1.0, 1.0, 9.0, 1.0]);

        let out = block_on(pdist2_builtin(
            x,
            y,
            vec![
                Value::from("euclidean"),
                Value::from("Smallest"),
                Value::Num(1.0),
            ],
        ))
        .unwrap();
        let tensor = tensor_out(out);
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.data, vec![1.0, 1.0]);
    }

    #[test]
    fn knnsearch_returns_indices_then_distances() {
        let x = tensor(vec![0.0, 2.0, 5.0, 0.0, 0.0, 0.0], 3, 2);
        let y = tensor(vec![1.0, 4.0, 0.0, 0.0], 2, 2);
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(knnsearch_builtin(
            x,
            y,
            vec![Value::from("K"), Value::Num(2.0)],
        ))
        .unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        let idx = tensor_out(values[0].clone());
        let dist = tensor_out(values[1].clone());
        assert_eq!(idx.shape, vec![2, 2]);
        assert_eq!(dist.shape, vec![2, 2]);
        assert_eq!(idx.data, vec![1.0, 3.0, 2.0, 2.0]);
        assert_eq!(dist.data, vec![1.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn knnsearch_supports_metrics_and_ties() {
        let x = tensor(vec![0.0, 2.0, -2.0, 0.0, 0.0, 0.0], 3, 2);
        let y = tensor(vec![0.0, 0.0], 1, 2);
        let out = block_on(knnsearch_builtin(
            x.clone(),
            y.clone(),
            vec![
                Value::from("Distance"),
                Value::from("cityblock"),
                Value::from("K"),
                Value::Num(2.0),
            ],
        ))
        .unwrap();
        let idx = tensor_out(out);
        assert_eq!(idx.shape, vec![1, 2]);
        assert_eq!(idx.data, vec![1.0, 2.0]);

        let out = block_on(knnsearch_builtin(
            x,
            y,
            vec![
                Value::from("K"),
                Value::Num(2.0),
                Value::from("IncludeTies"),
                Value::Bool(true),
            ],
        ))
        .unwrap();
        let Value::Cell(cells) = out else {
            panic!("expected cell output");
        };
        assert_eq!(cells.shape, vec![1, 1]);
        let Value::Tensor(tied) = &cells.data[0] else {
            panic!("expected tensor in cell");
        };
        assert_eq!(tied.shape, vec![1, 3]);
        assert_eq!(tied.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn knnsearch_handles_nan_rows_empty_queries_and_option_validation() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(knnsearch_builtin(
            tensor(vec![0.0, f64::NAN, 2.0], 3, 1),
            tensor(vec![1.0], 1, 1),
            vec![Value::from("K"), Value::Num(2.0)],
        ))
        .unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        assert_eq!(tensor_out(values[0].clone()).data, vec![1.0, 3.0]);
        assert_eq!(tensor_out(values[1].clone()).data, vec![1.0, 1.0]);

        let out = block_on(knnsearch_builtin(
            tensor(vec![0.0, 2.0], 2, 1),
            tensor(Vec::new(), 0, 1),
            vec![Value::from("K"), Value::Num(1.0)],
        ))
        .unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        assert_eq!(tensor_out(values[0].clone()).shape, vec![0, 1]);
        assert_eq!(tensor_out(values[1].clone()).shape, vec![0, 1]);

        let out = block_on(knnsearch_builtin(
            tensor(vec![0.0, 2.0, -2.0], 3, 1),
            tensor(vec![0.0], 1, 1),
            vec![
                Value::from("K"),
                Value::Num(2.0),
                Value::from("IncludeTies"),
                Value::Bool(true),
                Value::from("SortIndices"),
                Value::Bool(false),
            ],
        ))
        .unwrap();
        assert!(matches!(out, Value::OutputList(values) if matches!(&values[0], Value::Cell(_))));

        let err = block_on(knnsearch_builtin(
            tensor(vec![1.0, 0.0], 2, 1),
            tensor(vec![1.0], 1, 1),
            vec![
                Value::from("Distance"),
                Value::from("cosine"),
                Value::from("NSMethod"),
                Value::from("kdtree"),
            ],
        ))
        .expect_err("kdtree should reject cosine distance");
        assert!(err.to_string().contains("NSMethod='kdtree'"));
    }

    #[test]
    fn squareform_converts_both_directions() {
        let vector = tensor(
            vec![
                5.0,
                4.0,
                2.0,
                17.0_f64.sqrt(),
                13.0_f64.sqrt(),
                20.0_f64.sqrt(),
            ],
            1,
            6,
        );
        let matrix = tensor_out(block_on(squareform_builtin(vector, Vec::new())).unwrap());
        assert_eq!(matrix.shape, vec![4, 4]);
        assert_eq!(
            matrix.data,
            vec![
                0.0,
                5.0,
                4.0,
                2.0,
                5.0,
                0.0,
                17.0_f64.sqrt(),
                13.0_f64.sqrt(),
                4.0,
                17.0_f64.sqrt(),
                0.0,
                20.0_f64.sqrt(),
                2.0,
                13.0_f64.sqrt(),
                20.0_f64.sqrt(),
                0.0,
            ]
        );

        let vector =
            tensor_out(block_on(squareform_builtin(Value::Tensor(matrix), Vec::new())).unwrap());
        assert_eq!(vector.shape, vec![1, 6]);
        assert_eq!(
            vector.data,
            vec![
                5.0,
                4.0,
                2.0,
                17.0_f64.sqrt(),
                13.0_f64.sqrt(),
                20.0_f64.sqrt(),
            ]
        );
    }
}
