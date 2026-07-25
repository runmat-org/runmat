//! k-means clustering compatibility surface.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, StructValue, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{random, random_args::keyword_of, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "kmeans";
const EPS: f64 = 1.0e-12;
const DEFAULT_MAX_ITER: usize = 100;
const MAX_WORK_CELLS: usize = 20_000_000;

const OUTPUT_IDX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "idx",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based cluster index for each input observation.",
};

const OUTPUT_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Final cluster centroid matrix.",
};

const OUTPUT_SUMD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sumd",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Within-cluster sums of point-to-centroid distances.",
};

const OUTPUT_D: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Distance from every observation to every centroid.",
};

const OUTPUTS_IDX: [BuiltinParamDescriptor; 1] = [OUTPUT_IDX];
const OUTPUTS_IDX_C: [BuiltinParamDescriptor; 2] = [OUTPUT_IDX, OUTPUT_C];
const OUTPUTS_IDX_C_SUMD: [BuiltinParamDescriptor; 3] = [OUTPUT_IDX, OUTPUT_C, OUTPUT_SUMD];
const OUTPUTS_FULL: [BuiltinParamDescriptor; 4] = [OUTPUT_IDX, OUTPUT_C, OUTPUT_SUMD, OUTPUT_D];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation matrix with observations in rows.",
};

const PARAM_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of clusters.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Name-value options including Distance, Start, Replicates, MaxIter, EmptyAction, Display, OnlinePhase, and Options.",
};

const INPUTS_X_K: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_K];
const INPUTS_FULL: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_K, PARAM_OPTIONS];

const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "idx = kmeans(X, k)",
        inputs: &INPUTS_X_K,
        outputs: &OUTPUTS_IDX,
    },
    BuiltinSignatureDescriptor {
        label: "idx = kmeans(X, k, Name, Value)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUTS_IDX,
    },
    BuiltinSignatureDescriptor {
        label: "[idx, C] = kmeans(___)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUTS_IDX_C,
    },
    BuiltinSignatureDescriptor {
        label: "[idx, C, sumd] = kmeans(___)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUTS_IDX_C_SUMD,
    },
    BuiltinSignatureDescriptor {
        label: "[idx, C, sumd, D] = kmeans(___)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUTS_FULL,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KMEANS.INVALID_ARGUMENT",
    identifier: Some("RunMat:kmeans:InvalidArgument"),
    when: "Inputs, cluster counts, starts, distance metrics, or name-value options are malformed.",
    message: "kmeans: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KMEANS.INTERNAL",
    identifier: Some("RunMat:kmeans:Internal"),
    when: "RunMat cannot allocate or construct kmeans outputs.",
    message: "kmeans: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const KMEANS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn kmeans_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn kmeans_error(
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
    kmeans_error(message, &ERROR_INVALID_ARGUMENT)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    kmeans_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Distance {
    SqEuclidean,
    Cityblock,
    Cosine,
    Correlation,
    Hamming,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EmptyAction {
    Singleton,
    Error,
    Drop,
}

#[derive(Clone, Debug)]
enum StartSpec {
    Plus,
    Sample,
    Uniform,
    Cluster,
    Numeric(Vec<Vec<f64>>),
    NumericPages(Vec<Vec<Vec<f64>>>),
}

#[derive(Clone, Debug)]
struct Options {
    distance: Distance,
    empty_action: EmptyAction,
    max_iter: usize,
    online_phase: bool,
    replicates: usize,
    start: StartSpec,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            distance: Distance::SqEuclidean,
            empty_action: EmptyAction::Singleton,
            max_iter: DEFAULT_MAX_ITER,
            online_phase: false,
            replicates: 1,
            start: StartSpec::Plus,
        }
    }
}

#[derive(Clone, Debug)]
struct PreparedData {
    rows: Vec<Vec<f64>>,
    original_row_count: usize,
    original_to_valid: Vec<Option<usize>>,
    cols: usize,
}

#[derive(Clone, Debug)]
struct ReplicateResult {
    idx_valid: Vec<usize>,
    centers: Vec<Vec<f64>>,
    sumd: Vec<f64>,
    distances: Vec<Vec<f64>>,
    objective: f64,
}

#[runtime_builtin(
    name = "kmeans",
    category = "stats/ml",
    summary = "Partition observations into k clusters with k-means clustering.",
    keywords = "kmeans,k-means,clustering,statistics,machine learning",
    type_resolver(kmeans_type),
    descriptor(crate::builtins::stats::ml::kmeans::KMEANS_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::kmeans"
)]
async fn kmeans_builtin(x: Value, k: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let x = gather(x).await?;
    let k = gather(k).await?;
    let rest = gather_all(rest).await?;
    let mut options = parse_options(rest)?;
    let data = prepare_data(x, options.distance)?;
    let k = parse_k(&k, &options.start)?;
    validate_options(&mut options, k, &data)?;
    let result = run_kmeans(&data, k, &options)?;
    outputs_for_count(result, &data)
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("kmeans: {err}")))
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}

fn parse_options(args: Vec<Value>) -> BuiltinResult<Options> {
    if !args.len().is_multiple_of(2) {
        return Err(invalid("kmeans: options must be name-value pairs"));
    }
    let mut options = Options::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let name = keyword_of(&args[idx])
            .ok_or_else(|| invalid("kmeans: option name must be text"))?
            .to_ascii_lowercase();
        let value = &args[idx + 1];
        match name.as_str() {
            "distance" => options.distance = parse_distance(value)?,
            "emptyaction" => options.empty_action = parse_empty_action(value)?,
            "maxiter" => options.max_iter = parse_positive_usize(value, "MaxIter")?,
            "replicates" => options.replicates = parse_positive_usize(value, "Replicates")?,
            "start" => options.start = parse_start(value)?,
            "display" => {
                let display = keyword_of(value)
                    .ok_or_else(|| invalid("kmeans: Display must be 'off', 'final', or 'iter'"))?;
                match display.to_ascii_lowercase().as_str() {
                    "off" | "final" | "iter" => {}
                    other => return Err(invalid(format!("kmeans: unsupported Display '{other}'"))),
                }
            }
            "onlinephase" => {
                let flag = keyword_of(value)
                    .ok_or_else(|| invalid("kmeans: OnlinePhase must be 'off' or 'on'"))?;
                match flag.to_ascii_lowercase().as_str() {
                    "off" => options.online_phase = false,
                    "on" => options.online_phase = true,
                    other => {
                        return Err(invalid(format!(
                            "kmeans: unsupported OnlinePhase '{other}'"
                        )))
                    }
                }
            }
            "options" => match value {
                Value::Struct(options_struct) => {
                    apply_options_struct(&mut options, options_struct)?
                }
                _ if is_empty_numeric(value) => {}
                _ => {
                    return Err(invalid(
                        "kmeans: Options must be [] or a statset options struct",
                    ))
                }
            },
            other => return Err(invalid(format!("kmeans: unsupported option '{other}'"))),
        }
        idx += 2;
    }
    Ok(options)
}

fn parse_distance(value: &Value) -> BuiltinResult<Distance> {
    let text = keyword_of(value).ok_or_else(|| invalid("kmeans: Distance must be text"))?;
    match text.to_ascii_lowercase().as_str() {
        "sqeuclidean" | "squaredeuclidean" | "squared euclidean" => Ok(Distance::SqEuclidean),
        "cityblock" | "city block" | "manhattan" => Ok(Distance::Cityblock),
        "cosine" => Ok(Distance::Cosine),
        "correlation" => Ok(Distance::Correlation),
        "hamming" => Ok(Distance::Hamming),
        other => Err(invalid(format!("kmeans: unsupported Distance '{other}'"))),
    }
}

fn parse_empty_action(value: &Value) -> BuiltinResult<EmptyAction> {
    let text = keyword_of(value).ok_or_else(|| invalid("kmeans: EmptyAction must be text"))?;
    match text.to_ascii_lowercase().as_str() {
        "singleton" => Ok(EmptyAction::Singleton),
        "error" => Ok(EmptyAction::Error),
        "drop" => Ok(EmptyAction::Drop),
        other => Err(invalid(format!(
            "kmeans: unsupported EmptyAction '{other}'"
        ))),
    }
}

fn parse_start(value: &Value) -> BuiltinResult<StartSpec> {
    if let Some(text) = keyword_of(value) {
        return match text.to_ascii_lowercase().as_str() {
            "plus" => Ok(StartSpec::Plus),
            "sample" => Ok(StartSpec::Sample),
            "uniform" => Ok(StartSpec::Uniform),
            "cluster" => Ok(StartSpec::Cluster),
            other => Err(invalid(format!("kmeans: unsupported Start '{other}'"))),
        };
    }
    let tensor = tensor::value_into_tensor_for(NAME, value.clone())
        .map_err(|err| invalid(format!("kmeans: {err}")))?;
    let tensor =
        tensor::integer_tensor_to_f64(tensor).map_err(|err| invalid(format!("kmeans: {err}")))?;
    if tensor.shape.len() > 3 {
        return Err(invalid(
            "kmeans: numeric Start must be a matrix or 3-D array",
        ));
    }
    if tensor.shape.len() == 3 {
        let k = tensor.shape[0];
        let cols = tensor.shape[1];
        let reps = tensor.shape[2];
        let mut pages = Vec::with_capacity(reps);
        for page in 0..reps {
            let mut centers = vec![vec![0.0; cols]; k];
            for col in 0..cols {
                for (row, center_row) in centers.iter_mut().enumerate().take(k) {
                    let offset = row + col * k + page * k * cols;
                    center_row[col] = tensor.data[offset];
                }
            }
            pages.push(centers);
        }
        Ok(StartSpec::NumericPages(pages))
    } else {
        let (rows, cols) = tensor_rows_cols(&tensor);
        let mut centers = vec![vec![0.0; cols]; rows];
        for col in 0..cols {
            for (row, center_row) in centers.iter_mut().enumerate().take(rows) {
                center_row[col] = tensor.data[col * rows + row];
            }
        }
        Ok(StartSpec::Numeric(centers))
    }
}

fn parse_positive_usize(value: &Value, label: &str) -> BuiltinResult<usize> {
    let raw = scalar_number(value)
        .ok_or_else(|| invalid(format!("kmeans: {label} must be a positive integer scalar")))?;
    if !raw.is_finite() || raw < 1.0 || raw.fract().abs() > EPS {
        return Err(invalid(format!(
            "kmeans: {label} must be a positive integer scalar"
        )));
    }
    Ok(raw as usize)
}

fn parse_k(value: &Value, start: &StartSpec) -> BuiltinResult<usize> {
    if is_empty_numeric(value) {
        return infer_k_from_start(start).ok_or_else(|| {
            invalid("kmeans: k can be [] only when Start is a numeric start matrix or array")
        });
    }
    parse_positive_usize(value, "k")
}

fn infer_k_from_start(start: &StartSpec) -> Option<usize> {
    match start {
        StartSpec::Numeric(centers) => Some(centers.len()),
        StartSpec::NumericPages(pages) => pages.first().map(Vec::len),
        _ => None,
    }
}

fn scalar_number(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_values_f64(tensor)[0])
        }
        _ => None,
    }
}

fn is_empty_numeric(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.data.is_empty())
}

fn apply_options_struct(options: &mut Options, value: &StructValue) -> BuiltinResult<()> {
    for (name, field) in &value.fields {
        if is_empty_option_value(field) {
            continue;
        }
        match name.to_ascii_lowercase().as_str() {
            "maxiter" => options.max_iter = parse_positive_usize(field, "Options.MaxIter")?,
            "display" => {
                let display = keyword_of(field).ok_or_else(|| {
                    invalid("kmeans: Options.Display must be 'off', 'final', or 'iter'")
                })?;
                match display.to_ascii_lowercase().as_str() {
                    "off" | "final" | "iter" => {}
                    other => {
                        return Err(invalid(format!(
                            "kmeans: unsupported Options.Display '{other}'"
                        )))
                    }
                }
            }
            "useparallel" | "usesubstreams" => {
                if bool_option(field)? {
                    return Err(invalid(format!(
                        "kmeans: Options.{name} is not supported by the CPU runtime"
                    )));
                }
            }
            "streams" => {
                if !is_empty_numeric(field) {
                    return Err(invalid(
                        "kmeans: Options.Streams is not supported by the CPU runtime",
                    ));
                }
            }
            "tolx" | "tolfun" => {
                let value = scalar_number(field).ok_or_else(|| {
                    invalid(format!("kmeans: Options.{name} must be a numeric scalar"))
                })?;
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid(format!(
                        "kmeans: Options.{name} must be a nonnegative finite scalar"
                    )));
                }
            }
            _ => {}
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

fn bool_option(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) => Ok(*value != 0.0),
        Value::Int(value) => Ok(value.to_f64() != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_values_f64(tensor)[0] != 0.0)
        }
        other => Err(invalid(format!(
            "kmeans: option value must be a logical scalar, got {other:?}"
        ))),
    }
}

fn prepare_data(value: Value, distance: Distance) -> BuiltinResult<PreparedData> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|err| invalid(format!("kmeans: {err}")))?;
    let tensor =
        tensor::integer_tensor_to_f64(tensor).map_err(|err| invalid(format!("kmeans: {err}")))?;
    if tensor.shape.len() > 2 {
        return Err(invalid("kmeans: X must be a numeric vector or 2-D matrix"));
    }
    let (raw_rows, raw_cols) = tensor_rows_cols(&tensor);
    let vector_as_column = raw_rows == 1 || raw_cols == 1;
    let original_row_count = if vector_as_column {
        tensor.data.len()
    } else {
        raw_rows
    };
    let cols = if vector_as_column { 1 } else { raw_cols };
    let mut rows = Vec::new();
    let mut original_to_valid = Vec::with_capacity(original_row_count);
    for row in 0..original_row_count {
        let mut values = Vec::with_capacity(cols);
        let mut has_nan = false;
        for col in 0..cols {
            let value = if vector_as_column {
                tensor.data[row]
            } else {
                tensor.data[col * raw_rows + row]
            };
            if value.is_nan() {
                has_nan = true;
            } else if !value.is_finite() {
                return Err(invalid("kmeans: X cannot contain Inf values"));
            }
            values.push(value);
        }
        if has_nan {
            original_to_valid.push(None);
        } else {
            let transformed = transform_row(&values, distance)?;
            if transformed.iter().any(|value| value.is_nan()) {
                original_to_valid.push(None);
            } else {
                original_to_valid.push(Some(rows.len()));
                rows.push(transformed);
            }
        }
    }
    if rows.is_empty() {
        return Err(invalid(
            "kmeans: X must contain at least one complete observation",
        ));
    }
    Ok(PreparedData {
        rows,
        original_row_count,
        original_to_valid,
        cols,
    })
}

fn tensor_rows_cols(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape.as_slice() {
        [] => (1, 1),
        [n] => (*n, 1),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn transform_row(row: &[f64], distance: Distance) -> BuiltinResult<Vec<f64>> {
    match distance {
        Distance::Cosine => {
            let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
            if norm <= EPS {
                Ok(vec![f64::NAN; row.len()])
            } else {
                Ok(row.iter().map(|value| value / norm).collect())
            }
        }
        Distance::Correlation => {
            if row.is_empty() {
                return Ok(Vec::new());
            }
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            let centered = row.iter().map(|value| value - mean).collect::<Vec<_>>();
            let norm = centered
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            if norm <= EPS {
                Ok(vec![f64::NAN; row.len()])
            } else {
                Ok(centered.into_iter().map(|value| value / norm).collect())
            }
        }
        Distance::Hamming => {
            if row.iter().any(|value| *value != 0.0 && *value != 1.0) {
                return Err(invalid("kmeans: Hamming distance requires binary X values"));
            }
            Ok(row.to_vec())
        }
        _ => Ok(row.to_vec()),
    }
}

fn validate_options(options: &mut Options, k: usize, data: &PreparedData) -> BuiltinResult<()> {
    match &mut options.start {
        StartSpec::Numeric(centers) => {
            validate_centers(centers, k, data.cols)?;
            transform_centers(centers, options.distance)?;
            options.replicates = 1;
        }
        StartSpec::NumericPages(pages) => {
            if pages.is_empty() {
                return Err(invalid(
                    "kmeans: numeric Start array must have at least one page",
                ));
            }
            for centers in &mut *pages {
                validate_centers(centers, k, data.cols)?;
                transform_centers(centers, options.distance)?;
            }
            options.replicates = pages.len();
        }
        StartSpec::Uniform if options.distance == Distance::Hamming => {
            return Err(invalid(
                "kmeans: Start 'uniform' is not valid with Hamming distance",
            ));
        }
        _ => {}
    }
    let center_work = k
        .checked_mul(data.cols)
        .and_then(|value| value.checked_mul(options.replicates))
        .ok_or_else(|| internal("kmeans: work size overflow"))?;
    let assignment_work = data
        .rows
        .len()
        .checked_mul(k)
        .ok_or_else(|| internal("kmeans: assignment work size overflow"))?;
    let d_work = data
        .original_row_count
        .checked_mul(k)
        .ok_or_else(|| internal("kmeans: distance output size overflow"))?;
    if center_work > MAX_WORK_CELLS || assignment_work > MAX_WORK_CELLS || d_work > MAX_WORK_CELLS {
        return Err(invalid("kmeans: requested clustering work is too large"));
    }
    Ok(())
}

fn transform_centers(centers: &mut [Vec<f64>], distance: Distance) -> BuiltinResult<()> {
    for center in centers {
        let transformed = transform_row(center, distance)?;
        if transformed.iter().any(|value| value.is_nan()) {
            return Err(invalid(
                "kmeans: numeric Start contains a degenerate centroid for the selected Distance",
            ));
        }
        *center = transformed;
    }
    Ok(())
}

fn validate_centers(centers: &[Vec<f64>], k: usize, cols: usize) -> BuiltinResult<()> {
    if centers.len() != k || centers.iter().any(|row| row.len() != cols) {
        return Err(invalid(
            "kmeans: numeric Start must be k-by-p or k-by-p-by-r",
        ));
    }
    if centers
        .iter()
        .flat_map(|row| row.iter())
        .any(|value| !value.is_finite())
    {
        return Err(invalid("kmeans: numeric Start values must be finite"));
    }
    Ok(())
}

fn run_kmeans(data: &PreparedData, k: usize, options: &Options) -> BuiltinResult<ReplicateResult> {
    let starts = initial_centers(data, k, options)?;
    let mut best = None;
    for centers in starts {
        let result = run_replicate(data, centers, options)?;
        if best
            .as_ref()
            .map(|current: &ReplicateResult| result.objective < current.objective)
            .unwrap_or(true)
        {
            best = Some(result);
        }
    }
    best.ok_or_else(|| internal("kmeans: no replicate produced a result"))
}

fn initial_centers(
    data: &PreparedData,
    k: usize,
    options: &Options,
) -> BuiltinResult<Vec<Vec<Vec<f64>>>> {
    match &options.start {
        StartSpec::Numeric(centers) => Ok(vec![centers.clone()]),
        StartSpec::NumericPages(pages) => Ok(pages.clone()),
        _ if k > data.rows.len() && options.empty_action != EmptyAction::Drop => Err(invalid(
            "kmeans: k cannot exceed the number of complete observations unless EmptyAction is 'drop'",
        )),
        StartSpec::Plus => (0..options.replicates)
            .map(|_| plus_start(data, k, options.distance))
            .collect(),
        StartSpec::Sample => (0..options.replicates)
            .map(|_| sample_start(data, k))
            .collect(),
        StartSpec::Uniform => (0..options.replicates)
            .map(|_| uniform_start(data, k))
            .collect(),
        StartSpec::Cluster => (0..options.replicates)
            .map(|_| cluster_start(data, k, options))
            .collect(),
    }
}

fn sample_start(data: &PreparedData, k: usize) -> BuiltinResult<Vec<Vec<f64>>> {
    if k > data.rows.len() {
        return Err(invalid(
            "kmeans: Start 'sample' requires k no larger than the number of complete observations",
        ));
    }
    let mut scored = random::generate_uniform(data.rows.len(), NAME)?
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    Ok(scored
        .into_iter()
        .take(k)
        .map(|(idx, _)| data.rows[idx].clone())
        .collect())
}

fn plus_start(data: &PreparedData, k: usize, distance: Distance) -> BuiltinResult<Vec<Vec<f64>>> {
    if k > data.rows.len() {
        return Err(invalid(
            "kmeans: Start 'plus' requires k no larger than the number of complete observations",
        ));
    }
    let uniforms = random::generate_uniform(k.max(1), NAME)?;
    let first = ((uniforms[0] * data.rows.len() as f64).floor() as usize).min(data.rows.len() - 1);
    let mut centers = vec![data.rows[first].clone()];
    while centers.len() < k {
        let mut weights = Vec::with_capacity(data.rows.len());
        let mut total = 0.0;
        for row in &data.rows {
            let best = centers
                .iter()
                .map(|center| distance_value(row, center, distance))
                .fold(f64::INFINITY, f64::min);
            let weight = if best.is_finite() { best.max(0.0) } else { 0.0 };
            weights.push(weight);
            total += weight;
        }
        let next = if total <= EPS {
            first_unused_row(data, &centers, distance).unwrap_or(0)
        } else {
            let mut threshold = uniforms[centers.len()] * total;
            let mut selected = data.rows.len() - 1;
            for (idx, weight) in weights.iter().enumerate() {
                threshold -= *weight;
                if threshold <= 0.0 {
                    selected = idx;
                    break;
                }
            }
            selected
        };
        centers.push(data.rows[next].clone());
    }
    Ok(centers)
}

fn first_unused_row(
    data: &PreparedData,
    centers: &[Vec<f64>],
    distance: Distance,
) -> Option<usize> {
    data.rows.iter().position(|row| {
        centers
            .iter()
            .all(|center| distance_value(row, center, distance) > EPS)
    })
}

fn uniform_start(data: &PreparedData, k: usize) -> BuiltinResult<Vec<Vec<f64>>> {
    let cols = data.cols;
    let mut mins = vec![f64::INFINITY; cols];
    let mut maxs = vec![f64::NEG_INFINITY; cols];
    for row in &data.rows {
        for col in 0..cols {
            mins[col] = mins[col].min(row[col]);
            maxs[col] = maxs[col].max(row[col]);
        }
    }
    let uniforms = random::generate_uniform(k * cols, NAME)?;
    let mut centers = vec![vec![0.0; cols]; k];
    for center in 0..k {
        for col in 0..cols {
            let lo = mins[col];
            let hi = maxs[col];
            centers[center][col] = if hi > lo {
                lo + (hi - lo) * uniforms[center + col * k]
            } else {
                lo
            };
        }
    }
    Ok(centers)
}

fn cluster_start(data: &PreparedData, k: usize, options: &Options) -> BuiltinResult<Vec<Vec<f64>>> {
    let sample_size = data.rows.len().div_ceil(10).max(k).min(data.rows.len());
    let rows = if sample_size == data.rows.len() {
        data.rows.clone()
    } else {
        sample_start(data, sample_size)?
    };
    let subset = PreparedData {
        rows,
        original_row_count: sample_size,
        original_to_valid: (0..sample_size).map(Some).collect(),
        cols: data.cols,
    };
    if subset.rows.len() == k {
        return Ok(subset.rows);
    }
    let centers = plus_start(&subset, k, options.distance)?;
    let preliminary = Options {
        distance: options.distance,
        empty_action: EmptyAction::Singleton,
        max_iter: options.max_iter.clamp(1, 10),
        online_phase: false,
        replicates: 1,
        start: StartSpec::Numeric(centers.clone()),
    };
    run_replicate(&subset, centers, &preliminary).map(|result| result.centers)
}

fn run_replicate(
    data: &PreparedData,
    mut centers: Vec<Vec<f64>>,
    options: &Options,
) -> BuiltinResult<ReplicateResult> {
    let k = centers.len();
    let mut idx = vec![0usize; data.rows.len()];
    let iterations = if options.online_phase {
        options.max_iter.saturating_mul(2)
    } else {
        options.max_iter
    };
    for _ in 0..iterations {
        let assignment = assign_rows(&data.rows, &centers, options.distance);
        let changed = assignment.changed_from(&idx);
        idx = assignment.idx;
        let next_centers = update_centers(data, &idx, &assignment.distances, &centers, options)?;
        centers = next_centers;
        if !changed {
            break;
        }
    }
    let assignment = assign_rows(&data.rows, &centers, options.distance);
    idx = assignment.idx;
    let distances = assignment.distances;
    let sumd = sum_distances(k, &idx, &distances);
    let objective = sumd.iter().filter(|value| value.is_finite()).sum::<f64>();
    Ok(ReplicateResult {
        idx_valid: idx,
        centers,
        sumd,
        distances,
        objective,
    })
}

struct Assignment {
    idx: Vec<usize>,
    distances: Vec<Vec<f64>>,
}

impl Assignment {
    fn changed_from(&self, previous: &[usize]) -> bool {
        self.idx != previous
    }
}

fn assign_rows(rows: &[Vec<f64>], centers: &[Vec<f64>], distance: Distance) -> Assignment {
    let mut idx = Vec::with_capacity(rows.len());
    let mut all_distances = Vec::with_capacity(rows.len());
    for row in rows {
        let mut best_idx = 0usize;
        let mut best_distance = f64::INFINITY;
        let mut row_distances = Vec::with_capacity(centers.len());
        for (center_idx, center) in centers.iter().enumerate() {
            let d = distance_value(row, center, distance);
            row_distances.push(d);
            if d < best_distance {
                best_distance = d;
                best_idx = center_idx;
            }
        }
        idx.push(best_idx);
        all_distances.push(row_distances);
    }
    Assignment {
        idx,
        distances: all_distances,
    }
}

fn update_centers(
    data: &PreparedData,
    idx: &[usize],
    distances: &[Vec<f64>],
    previous: &[Vec<f64>],
    options: &Options,
) -> BuiltinResult<Vec<Vec<f64>>> {
    let k = previous.len();
    let mut members = vec![Vec::new(); k];
    for (row, cluster) in idx.iter().enumerate() {
        members[*cluster].push(row);
    }
    let mut centers = vec![vec![f64::NAN; data.cols]; k];
    let mut singleton_rows = vec![false; data.rows.len()];
    for cluster in 0..k {
        if members[cluster].is_empty() {
            match options.empty_action {
                EmptyAction::Error => {
                    return Err(invalid(
                        "kmeans: empty cluster encountered and EmptyAction is 'error'",
                    ))
                }
                EmptyAction::Drop => continue,
                EmptyAction::Singleton => {
                    let row = furthest_assigned_row(distances, idx, &singleton_rows);
                    singleton_rows[row] = true;
                    centers[cluster] = data.rows[row].clone();
                }
            }
        } else {
            centers[cluster] = centroid_for(&data.rows, &members[cluster], options.distance);
        }
    }
    Ok(centers)
}

fn furthest_assigned_row(distances: &[Vec<f64>], idx: &[usize], excluded: &[bool]) -> usize {
    let mut best_row = 0usize;
    let mut best_distance = f64::NEG_INFINITY;
    for (row, cluster) in idx.iter().enumerate() {
        if excluded.get(row).copied().unwrap_or(false) {
            continue;
        }
        let distance = distances[row][*cluster];
        if distance > best_distance {
            best_distance = distance;
            best_row = row;
        }
    }
    best_row
}

fn centroid_for(rows: &[Vec<f64>], members: &[usize], distance: Distance) -> Vec<f64> {
    let cols = rows[0].len();
    match distance {
        Distance::Cityblock | Distance::Hamming => {
            let mut center = vec![0.0; cols];
            for col in 0..cols {
                let mut values = members
                    .iter()
                    .map(|row| rows[*row][col])
                    .collect::<Vec<_>>();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                let mid = values.len() / 2;
                center[col] = if values.len() % 2 == 0 {
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[mid]
                };
                if distance == Distance::Hamming {
                    center[col] = if center[col] >= 0.5 { 1.0 } else { 0.0 };
                }
            }
            center
        }
        _ => {
            let mut center = vec![0.0; cols];
            for row in members {
                for (col, value) in center.iter_mut().enumerate().take(cols) {
                    *value += rows[*row][col];
                }
            }
            for value in &mut center {
                *value /= members.len() as f64;
            }
            match distance {
                Distance::Cosine => normalize_or_nan(center),
                Distance::Correlation => {
                    let mean = center.iter().sum::<f64>() / center.len() as f64;
                    normalize_or_nan(center.into_iter().map(|value| value - mean).collect())
                }
                _ => center,
            }
        }
    }
}

fn normalize_or_nan(mut row: Vec<f64>) -> Vec<f64> {
    let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= EPS {
        row.fill(f64::NAN);
    } else {
        for value in &mut row {
            *value /= norm;
        }
    }
    row
}

fn sum_distances(k: usize, idx: &[usize], distances: &[Vec<f64>]) -> Vec<f64> {
    let mut sumd = vec![0.0; k];
    for (row, cluster) in idx.iter().enumerate() {
        let value = distances[row][*cluster];
        if value.is_finite() {
            sumd[*cluster] += value;
        }
    }
    sumd
}

fn distance_value(row: &[f64], center: &[f64], distance: Distance) -> f64 {
    if row.iter().any(|v| v.is_nan()) || center.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    match distance {
        Distance::SqEuclidean => row
            .iter()
            .zip(center.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum(),
        Distance::Cityblock => row
            .iter()
            .zip(center.iter())
            .map(|(a, b)| (a - b).abs())
            .sum(),
        Distance::Cosine | Distance::Correlation => {
            1.0 - row
                .iter()
                .zip(center.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        }
        Distance::Hamming => {
            let mismatches = row
                .iter()
                .zip(center.iter())
                .filter(|(a, b)| (*a - *b).abs() > EPS)
                .count();
            mismatches as f64 / row.len() as f64
        }
    }
}

fn outputs_for_count(result: ReplicateResult, data: &PreparedData) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![idx_output(&result, data)?])),
        Some(2) => Ok(Value::OutputList(vec![
            idx_output(&result, data)?,
            Value::Tensor(matrix_output(&result.centers)?),
        ])),
        Some(3) => Ok(Value::OutputList(vec![
            idx_output(&result, data)?,
            Value::Tensor(matrix_output(&result.centers)?),
            Value::Tensor(sumd_output(&result)?),
        ])),
        Some(4) => Ok(Value::OutputList(vec![
            idx_output(&result, data)?,
            Value::Tensor(matrix_output(&result.centers)?),
            Value::Tensor(sumd_output(&result)?),
            Value::Tensor(distance_output(&result, data)?),
        ])),
        Some(_) => Err(invalid("kmeans: too many output arguments; maximum is 4")),
        None => idx_output(&result, data),
    }
}

fn idx_output(result: &ReplicateResult, data: &PreparedData) -> BuiltinResult<Value> {
    let mut values = vec![f64::NAN; data.original_row_count];
    for (original, valid) in data.original_to_valid.iter().enumerate() {
        if let Some(valid_idx) = valid {
            values[original] = result.idx_valid[*valid_idx] as f64 + 1.0;
        }
    }
    Tensor::new(values, vec![data.original_row_count, 1])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("kmeans: {err}")))
}

fn matrix_output(rows: &[Vec<f64>]) -> BuiltinResult<Tensor> {
    let row_count = rows.len();
    let cols = rows.first().map(|row| row.len()).unwrap_or(0);
    let len = row_count
        .checked_mul(cols)
        .ok_or_else(|| internal("kmeans: matrix output size overflow"))?;
    let mut data = vec![0.0; len];
    for row in 0..row_count {
        for col in 0..cols {
            data[col * row_count + row] = rows[row][col];
        }
    }
    Tensor::new(data, vec![row_count, cols]).map_err(|err| internal(format!("kmeans: {err}")))
}

fn sumd_output(result: &ReplicateResult) -> BuiltinResult<Tensor> {
    Tensor::new(result.sumd.clone(), vec![result.sumd.len(), 1])
        .map_err(|err| internal(format!("kmeans: {err}")))
}

fn distance_output(result: &ReplicateResult, data: &PreparedData) -> BuiltinResult<Tensor> {
    let k = result.centers.len();
    let len = data
        .original_row_count
        .checked_mul(k)
        .ok_or_else(|| internal("kmeans: distance output size overflow"))?;
    let mut values = vec![f64::NAN; len];
    for (original, valid) in data.original_to_valid.iter().enumerate() {
        if let Some(valid_idx) = valid {
            for cluster in 0..k {
                values[cluster * data.original_row_count + original] =
                    result.distances[*valid_idx][cluster];
            }
        }
    }
    Tensor::new(values, vec![data.original_row_count, k])
        .map_err(|err| internal(format!("kmeans: {err}")))
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, Tensor};

    use super::*;
    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let mut tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
        tensor.data.fill(f64::NAN);
        Value::Tensor(tensor)
    }

    fn outputs(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn kmeans_clusters_numeric_matrix_with_outputs() {
        random::reset_rng();
        let x = tensor(vec![0.0, 0.2, 9.8, 10.0, 0.0, 0.1, 9.9, 10.1], 4, 2);
        let _guard = crate::output_count::push_output_count(Some(4));
        let out = block_on(kmeans_builtin(
            x,
            Value::Num(2.0),
            vec![
                Value::from("Start"),
                tensor(vec![0.0, 10.0, 0.0, 10.0], 2, 2),
                Value::from("MaxIter"),
                Value::Num(20.0),
            ],
        ))
        .unwrap();
        let out = outputs(out);
        match &out[0] {
            Value::Tensor(idx) => assert_eq!(idx.data, vec![1.0, 1.0, 2.0, 2.0]),
            other => panic!("idx {other:?}"),
        }
        match &out[1] {
            Value::Tensor(c) => {
                assert_eq!(c.shape, vec![2, 2]);
                assert!((c.data[0] - 0.1).abs() < 1.0e-12);
                assert!((c.data[1] - 9.9).abs() < 1.0e-12);
                assert!((c.data[2] - 0.05).abs() < 1.0e-12);
                assert!((c.data[3] - 10.0).abs() < 1.0e-12);
            }
            other => panic!("centers {other:?}"),
        }
        match &out[2] {
            Value::Tensor(sumd) => {
                assert_eq!(sumd.shape, vec![2, 1]);
                assert!(sumd.data.iter().all(|v| v.is_finite()));
            }
            other => panic!("sumd {other:?}"),
        }
        match &out[3] {
            Value::Tensor(d) => assert_eq!(d.shape, vec![4, 2]),
            other => panic!("distances {other:?}"),
        }
    }

    #[test]
    fn kmeans_reads_typed_integer_inputs_start_and_options_exactly() {
        let mut statset = StructValue::new();
        statset.insert(
            "MaxIter",
            poisoned_int_tensor(IntegerStorage::U16(vec![20]), 1, 1),
        );
        statset.insert(
            "TolFun",
            poisoned_int_tensor(IntegerStorage::U8(vec![0]), 1, 1),
        );
        statset.insert(
            "UseParallel",
            poisoned_int_tensor(IntegerStorage::U8(vec![0]), 1, 1),
        );

        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(kmeans_builtin(
            poisoned_int_tensor(IntegerStorage::I16(vec![0, 0, 10, 10]), 4, 1),
            poisoned_int_tensor(IntegerStorage::U8(vec![2]), 1, 1),
            vec![
                Value::from("Start"),
                poisoned_int_tensor(IntegerStorage::I16(vec![0, 10]), 2, 1),
                Value::from("Replicates"),
                poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
                Value::from("Options"),
                Value::Struct(statset),
            ],
        ))
        .unwrap();
        let out = outputs(out);
        match &out[0] {
            Value::Tensor(idx) => assert_eq!(idx.data, vec![1.0, 1.0, 2.0, 2.0]),
            other => panic!("idx {other:?}"),
        }
        match &out[1] {
            Value::Tensor(centers) => assert_eq!(centers.data, vec![0.0, 10.0]),
            other => panic!("centers {other:?}"),
        }
    }

    #[test]
    fn kmeans_treats_vector_as_column_and_nan_rows_as_missing() {
        random::reset_rng();
        let x = tensor(vec![1.0, 2.0, f64::NAN, 20.0, 21.0], 1, 5);
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(kmeans_builtin(
            x,
            Value::Num(2.0),
            vec![Value::from("Start"), Value::from("sample")],
        ))
        .unwrap();
        let out = outputs(out);
        match &out[0] {
            Value::Tensor(idx) => {
                assert_eq!(idx.shape, vec![5, 1]);
                assert!(idx.data[2].is_nan());
                assert!(idx.data[0].is_finite());
                assert!(idx.data[4].is_finite());
            }
            other => panic!("idx {other:?}"),
        }
    }

    #[test]
    fn kmeans_supports_cityblock_and_replicates() {
        random::reset_rng();
        let x = tensor(vec![0.0, 1.0, 10.0, 11.0], 4, 1);
        let out = block_on(kmeans_builtin(
            x,
            Value::Num(2.0),
            vec![
                Value::from("Distance"),
                Value::from("cityblock"),
                Value::from("Replicates"),
                Value::Num(2.0),
                Value::from("Start"),
                Value::from("plus"),
            ],
        ))
        .unwrap();
        match out {
            Value::Tensor(idx) => assert_eq!(idx.shape, vec![4, 1]),
            other => panic!("idx {other:?}"),
        }
    }

    #[test]
    fn kmeans_infers_k_from_numeric_start_when_k_is_empty() {
        let x = tensor(vec![0.0, 0.5, 10.0, 10.5], 4, 1);
        let empty_k = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(kmeans_builtin(
            x,
            empty_k,
            vec![
                Value::from("Start"),
                tensor(vec![0.0, 10.0], 2, 1),
                Value::from("MaxIter"),
                Value::Num(5.0),
            ],
        ))
        .unwrap();
        let out = outputs(out);
        match &out[1] {
            Value::Tensor(c) => assert_eq!(c.shape, vec![2, 1]),
            other => panic!("centers {other:?}"),
        }
    }

    #[test]
    fn kmeans_accepts_3d_numeric_starts_as_replicates() {
        let x = tensor(vec![0.0, 0.2, 10.0, 10.2], 4, 1);
        let starts = Value::Tensor(Tensor::new(vec![0.0, 10.0, 0.2, 10.2], vec![2, 1, 2]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(3));
        let out = block_on(kmeans_builtin(
            x,
            Value::Num(2.0),
            vec![Value::from("Start"), starts],
        ))
        .unwrap();
        let out = outputs(out);
        match &out[2] {
            Value::Tensor(sumd) => assert_eq!(sumd.shape, vec![2, 1]),
            other => panic!("sumd {other:?}"),
        }
    }

    #[test]
    fn kmeans_supports_cluster_start_and_online_phase() {
        random::reset_rng();
        let x = tensor(vec![0.0, 0.1, 9.9, 10.0, 0.0, 0.2, 10.1, 10.0], 4, 2);
        let out = block_on(kmeans_builtin(
            x,
            Value::Num(2.0),
            vec![
                Value::from("Start"),
                Value::from("cluster"),
                Value::from("OnlinePhase"),
                Value::from("on"),
            ],
        ))
        .unwrap();
        match out {
            Value::Tensor(idx) => assert_eq!(idx.shape, vec![4, 1]),
            other => panic!("idx {other:?}"),
        }
    }

    #[test]
    fn kmeans_rejects_bad_options() {
        let x = tensor(vec![0.0, 1.0, 2.0], 3, 1);
        let err = block_on(kmeans_builtin(
            x.clone(),
            Value::Num(2.0),
            vec![Value::from("Distance"), Value::from("mahalanobis")],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:kmeans:InvalidArgument"));

        let err = block_on(kmeans_builtin(
            x,
            Value::Num(2.0),
            vec![
                Value::from("Distance"),
                Value::from("hamming"),
                Value::from("Start"),
                Value::from("uniform"),
            ],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:kmeans:InvalidArgument"));
    }
}
