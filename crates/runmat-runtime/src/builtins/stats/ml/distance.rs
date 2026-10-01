//! Pairwise distance and squareform helpers for Statistics and Machine Learning workflows.

use std::cmp::Ordering;

use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_builtins::{
    DistanceControlKind, DistanceMetricKind, DistanceSelectionKind, KnnMetric, KnnOption,
    KnnSearchMethod, DISTANCE_ERROR_INTERNAL, DISTANCE_ERROR_INVALID_ARGUMENT,
    KNN_GPU_FAST_DISTANCE_EXTENSION, KNN_GPU_INCLUDE_TIES_EXTENSION, KNN_GPU_NSMETHOD_EXTENSION,
    KNN_GPU_SORT_INDICES_EXTENSION, KNN_INTEGER_DATA_EXTENSION,
    KNN_INTEGER_FLOATING_CONTROL_EXTENSION, KNN_INTEGER_LOGICAL_CONTROL_EXTENSION,
    KNN_INTEGER_STRUCTURAL_EXTENSION, PDIST2_INTEGER_CONTROL_EXTENSION,
    PDIST2_INTEGER_DATA_EXTENSION, PDIST2_INTEGER_SELECTION_EXTENSION,
    PDIST_INTEGER_CONTROL_EXTENSION, PDIST_INTEGER_DATA_EXTENSION,
    SQUAREFORM_EXPLICIT_GPU_EXTENSION, SQUAREFORM_INTEGER_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, Tensor, Value};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const PDIST_NAME: &str = "pdist";
const PDIST2_NAME: &str = "pdist2";
const KNNSEARCH_NAME: &str = "knnsearch";
const SQUAREFORM_NAME: &str = "squareform";
const EPS: f64 = 1.0e-12;

mod squareform;
use squareform::{parse_squareform_force, squareform_compute};

fn distance_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    distance_descriptor_error(name, message, &DISTANCE_ERROR_INVALID_ARGUMENT)
}

fn with_distance_context(name: &'static str, mut error: RuntimeError) -> RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(name);
    }
    error
}

fn internal_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    distance_descriptor_error(name, message, &DISTANCE_ERROR_INTERNAL)
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

#[derive(Clone, Copy)]
enum PairwiseDistanceBuiltin {
    Pdist,
    Pdist2,
}

impl PairwiseDistanceBuiltin {
    fn name(self) -> &'static str {
        match self {
            Self::Pdist => PDIST_NAME,
            Self::Pdist2 => PDIST2_NAME,
        }
    }

    fn data_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Pdist => &PDIST_INTEGER_DATA_EXTENSION,
            Self::Pdist2 => &PDIST2_INTEGER_DATA_EXTENSION,
        }
    }

    fn control_extension(self) -> &'static BuiltinExtensionDescriptor {
        match self {
            Self::Pdist => &PDIST_INTEGER_CONTROL_EXTENSION,
            Self::Pdist2 => &PDIST2_INTEGER_CONTROL_EXTENSION,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct KnnOptions {
    k: usize,
    include_ties: bool,
}

#[runtime_builtin(
    name = "pdist",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn pdist_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_pdist_integer_boundaries(PairwiseDistanceBuiltin::Pdist, &x, None, &rest).await?;
    let metric_args = strip_cache_size_option(PDIST_NAME, rest).await?;
    parse_metric_name(PDIST_NAME, &metric_args)?;
    let x = value_to_matrix(PDIST_NAME, x).await?;
    let metric = parse_metric(PDIST_NAME, &x, None, metric_args).await?;
    let distances = condensed_distances(PDIST_NAME, &x, &metric)?;
    Ok(Value::Tensor(distances))
}

#[runtime_builtin(
    name = "pdist2",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn pdist2_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_pdist_integer_boundaries(PairwiseDistanceBuiltin::Pdist2, &x, Some(&y), &rest).await?;
    let (metric_args, selection) = split_pdist2_options(rest)?;
    let metric_args = strip_cache_size_option(PDIST2_NAME, metric_args).await?;
    parse_metric_name(PDIST2_NAME, &metric_args)?;
    if selection.is_none()
        && matches!(crate::output_count::current_output_count(), Some(count) if count > 1)
    {
        return Err(distance_error(
            PDIST2_NAME,
            "pdist2: second output is only available with Smallest or Largest",
        ));
    }
    let x = value_to_matrix(PDIST2_NAME, x).await?;
    let y = value_to_matrix(PDIST2_NAME, y).await?;
    if x.cols != y.cols {
        return Err(distance_error(
            PDIST2_NAME,
            "pdist2: X and Y must have the same number of columns",
        ));
    }
    let metric = parse_metric(PDIST2_NAME, &x, Some(&y), metric_args).await?;
    let distances = distance_matrix(PDIST2_NAME, &x, &y, &metric)?;
    match selection {
        Some(selection) => selected_distance_outputs(&distances, selection),
        None => match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![Value::Tensor(distances)],
            )),
            None => Ok(Value::Tensor(distances)),
        },
    }
}

async fn strip_cache_size_option(
    name: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    let mut metric_args = Vec::with_capacity(args.len());
    let mut index = 0usize;
    while index < args.len() {
        if keyword_of(&args[index])
            .as_deref()
            .and_then(DistanceControlKind::from_keyword)
            .is_some()
        {
            let Some(value) = args.get(index + 1) else {
                return Err(distance_error(
                    name,
                    format!("{name}: CacheSize requires a value"),
                ));
            };
            if keyword_of(value).is_some_and(|keyword| keyword.eq_ignore_ascii_case("maximal")) {
                index += 2;
                continue;
            }
            let cache_size = scalar_parameter(name, value, "CacheSize").await?;
            if !cache_size.is_finite() || cache_size <= 0.0 {
                return Err(distance_error(
                    name,
                    format!("{name}: CacheSize must be positive or 'maximal'"),
                ));
            }
            index += 2;
            continue;
        }
        metric_args.push(args[index].clone());
        index += 1;
    }
    Ok(metric_args)
}

async fn ensure_pdist_integer_boundaries(
    builtin: PairwiseDistanceBuiltin,
    x: &Value,
    y: Option<&Value>,
    rest: &[Value],
) -> BuiltinResult<()> {
    let name = builtin.name();
    for (value, role) in std::iter::once((x, "X")).chain(y.map(|value| (value, "Y"))) {
        if knn_typed_integer(value) {
            crate::compatibility::ensure_builtin_extension_enabled(builtin.data_extension(), name)?;
            ensure_distance_exact_f64(name, value, role).await?;
        }
    }
    for (index, value) in rest.iter().enumerate() {
        if !knn_typed_integer(value) {
            continue;
        }
        let selector = index
            .checked_sub(1)
            .and_then(|previous| keyword_of(&rest[previous]))
            .as_deref()
            .and_then(DistanceSelectionKind::from_keyword);
        if matches!(builtin, PairwiseDistanceBuiltin::Pdist2) && selector.is_some() {
            crate::compatibility::ensure_builtin_extension_enabled(
                &PDIST2_INTEGER_SELECTION_EXTENSION,
                name,
            )?;
            continue;
        }
        crate::compatibility::ensure_builtin_extension_enabled(builtin.control_extension(), name)?;
        ensure_distance_exact_f64(name, value, "numeric control").await?;
    }
    Ok(())
}

async fn ensure_distance_exact_f64(
    name: &'static str,
    value: &Value,
    role: &str,
) -> BuiltinResult<()> {
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value).await? {
        return Err(distance_error(
            name,
            format!("{name}: typed-integer {role} must be exactly representable as double"),
        ));
    }
    Ok(())
}

#[runtime_builtin(
    name = "knnsearch",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn knnsearch_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_knnsearch_extensions(&x, &y, &rest)?;
    let output_source = crate::builtins::common::gpu_helpers::select_resident_output_source(
        [&x, &y].into_iter().filter_map(knn_gpu_source),
        KNNSEARCH_NAME,
    )?;
    ensure_knnsearch_exact_boundaries(&x, &y, &rest).await?;
    let (metric_args, options) = parse_knnsearch_options(rest).await?;
    let x = value_to_matrix(KNNSEARCH_NAME, x).await?;
    let y = value_to_matrix(KNNSEARCH_NAME, y).await?;
    if x.cols != y.cols {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: X and Y must have the same number of columns",
        ));
    }
    if options.k > x.rows {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: K must be <= size(X,1)",
        ));
    }
    let metric = parse_metric(KNNSEARCH_NAME, &x, Some(&y), metric_args).await?;
    let value = knnsearch_outputs(&x, &y, &metric, options)?;
    restore_knnsearch_value(value, output_source.as_ref())
}

fn knn_gpu_source(value: &Value) -> Option<runmat_accelerate_api::GpuTensorHandle> {
    let Value::GpuTensor(handle) = value else {
        return None;
    };
    Some(handle.clone())
}

fn restore_knnsearch_value(
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    match value {
        Value::Tensor(tensor) => {
            let restored = crate::builtins::common::gpu_helpers::restore_class_preserving_value(
                source,
                Value::Tensor(tensor),
                KNNSEARCH_NAME,
            )?;
            if runmat_accelerate_api::handle_is_explicit(source)
                && !matches!(restored, Value::GpuTensor(_))
            {
                return Err(distance_error(
                    KNNSEARCH_NAME,
                    "knnsearch: provider cannot preserve explicit gpuArray output residency",
                ));
            }
            Ok(restored)
        }
        Value::Cell(cell) => {
            let shape = cell.shape.clone();
            let values = cell
                .data
                .into_iter()
                .map(|value| restore_knnsearch_value(value, Some(source)))
                .collect::<BuiltinResult<Vec<_>>>()?;
            CellArray::new_with_shape(values, shape)
                .map(Value::Cell)
                .map_err(|error| distance_error(KNNSEARCH_NAME, error))
        }
        Value::OutputList(values) => values
            .into_iter()
            .map(|value| restore_knnsearch_value(value, Some(source)))
            .collect::<BuiltinResult<Vec<_>>>()
            .map(Value::OutputList),
        other => Ok(other),
    }
}

fn knn_typed_integer(value: &Value) -> bool {
    crate::builtins::common::validation::value_has_native_integer_class(value)
}

fn explicit_knn_gpu(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
}

fn ensure_knnsearch_extensions(x: &Value, y: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if knn_typed_integer(x) || knn_typed_integer(y) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &KNN_INTEGER_DATA_EXTENSION,
            KNNSEARCH_NAME,
        )?;
    }
    let explicit_gpu =
        explicit_knn_gpu(x) || explicit_knn_gpu(y) || rest.iter().any(explicit_knn_gpu);
    let mut explicit_exhaustive = false;
    let mut idx = 0;
    while idx + 1 < rest.len() {
        let Some(option) = keyword_of(&rest[idx])
            .as_deref()
            .and_then(KnnOption::from_keyword)
        else {
            idx += 1;
            continue;
        };
        let value = &rest[idx + 1];
        if explicit_gpu {
            match option {
                KnnOption::IncludeTies if logical_scalar_is(value, true) => {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &KNN_GPU_INCLUDE_TIES_EXTENSION,
                        KNNSEARCH_NAME,
                    )?;
                }
                KnnOption::SortIndices if logical_scalar_is(value, false) => {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &KNN_GPU_SORT_INDICES_EXTENSION,
                        KNNSEARCH_NAME,
                    )?;
                }
                KnnOption::Distance
                    if keyword_of(value)
                        .as_deref()
                        .and_then(KnnMetric::from_keyword)
                        .is_some_and(|metric| metric.fast_alias) =>
                {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &KNN_GPU_FAST_DISTANCE_EXTENSION,
                        KNNSEARCH_NAME,
                    )?;
                }
                _ => {}
            }
        }
        match option {
            KnnOption::K | KnnOption::BucketSize if knn_typed_integer(value) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &KNN_INTEGER_STRUCTURAL_EXTENSION,
                    KNNSEARCH_NAME,
                )?;
            }
            KnnOption::P | KnnOption::Cov | KnnOption::Scale | KnnOption::CacheSize
                if knn_typed_integer(value) =>
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &KNN_INTEGER_FLOATING_CONTROL_EXTENSION,
                    KNNSEARCH_NAME,
                )?;
            }
            KnnOption::IncludeTies | KnnOption::SortIndices if knn_typed_integer(value) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &KNN_INTEGER_LOGICAL_CONTROL_EXTENSION,
                    KNNSEARCH_NAME,
                )?;
            }
            KnnOption::NsMethod => {
                explicit_exhaustive = keyword_of(value)
                    .as_deref()
                    .and_then(KnnSearchMethod::from_keyword)
                    == Some(KnnSearchMethod::Exhaustive);
            }
            _ => {}
        }
        idx += 2;
    }
    if explicit_gpu && !explicit_exhaustive {
        crate::compatibility::ensure_builtin_extension_enabled(
            &KNN_GPU_NSMETHOD_EXTENSION,
            KNNSEARCH_NAME,
        )?;
    }
    Ok(())
}

fn logical_scalar_is(value: &Value, expected: bool) -> bool {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .is_some_and(|value| matches!(value, 0 | 1) && (value != 0) == expected);
    }
    match value {
        Value::Bool(value) => *value == expected,
        Value::LogicalArray(value) if value.data.len() == 1 => (value.data[0] != 0) == expected,
        Value::Num(value) => (*value != 0.0) == expected,
        _ => false,
    }
}

async fn ensure_knnsearch_exact_boundaries(
    x: &Value,
    y: &Value,
    rest: &[Value],
) -> BuiltinResult<()> {
    for (value, role) in [(x, "X"), (y, "Y")] {
        ensure_knn_exact_f64(value, role).await?;
    }
    let mut idx = 0;
    while idx + 1 < rest.len() {
        if keyword_of(&rest[idx])
            .as_deref()
            .and_then(KnnOption::from_keyword)
            .is_some_and(|option| {
                matches!(
                    option,
                    KnnOption::P | KnnOption::Cov | KnnOption::Scale | KnnOption::CacheSize
                )
            })
        {
            ensure_knn_exact_f64(&rest[idx + 1], "floating option").await?;
        }
        idx += 2;
    }
    Ok(())
}

async fn ensure_knn_exact_f64(value: &Value, role: &str) -> BuiltinResult<()> {
    if knn_typed_integer(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        return Err(distance_error(
            KNNSEARCH_NAME,
            format!("knnsearch: typed-integer {role} must be exactly representable as double"),
        ));
    }
    Ok(())
}

#[runtime_builtin(
    name = "squareform",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn squareform_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SQUAREFORM_INTEGER_INPUT_EXTENSION,
            SQUAREFORM_NAME,
        )?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SQUAREFORM_EXPLICIT_GPU_EXTENSION,
            SQUAREFORM_NAME,
        )?;
    }
    let force = parse_squareform_force(rest)?;
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|error| with_distance_context(SQUAREFORM_NAME, error))?;
    match gathered {
        Value::LogicalArray(logical) => squareform::squareform_logical(logical, force),
        Value::Bool(value) => squareform::squareform_logical(
            runmat_value::LogicalArray::new(vec![u8::from(value)], vec![1, 1])
                .map_err(|error| internal_error(SQUAREFORM_NAME, format!("squareform: {error}")))?,
            force,
        ),
        value => {
            let tensor = host_value_to_matrix(SQUAREFORM_NAME, value)?;
            let output = squareform_compute(tensor, force)?;
            Ok(Value::Tensor(output))
        }
    }
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
        .map_err(|error| with_distance_context(name, error))?;
    host_value_to_matrix(name, gathered)
}

fn host_value_to_matrix(name: &'static str, gathered: Value) -> BuiltinResult<Tensor> {
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
        .map_err(|error| with_distance_context(name, error))
}

fn parse_metric_name(name: &'static str, args: &[Value]) -> BuiltinResult<DistanceMetricKind> {
    let Some(first) = args.first() else {
        return Ok(DistanceMetricKind::Euclidean);
    };
    let text = keyword_of(first)
        .ok_or_else(|| distance_error(name, format!("{name}: distance metric must be text")))?;
    DistanceMetricKind::from_keyword(&text).ok_or_else(|| {
        distance_error(
            name,
            format!("{name}: unsupported distance metric '{text}'"),
        )
    })
}

async fn parse_metric(
    name: &'static str,
    x: &Tensor,
    _y: Option<&Tensor>,
    args: Vec<Value>,
) -> BuiltinResult<DistanceMetric> {
    let kind = parse_metric_name(name, &args)?;
    let rest = if args.is_empty() {
        Vec::new()
    } else {
        args[1..].to_vec()
    };
    match kind {
        DistanceMetricKind::Euclidean => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Euclidean)
        }
        DistanceMetricKind::SquareEuclidean => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::SquareEuclidean)
        }
        DistanceMetricKind::Cityblock => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Cityblock)
        }
        DistanceMetricKind::Chebychev => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Chebychev)
        }
        DistanceMetricKind::Minkowski => {
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
        DistanceMetricKind::Seuclidean => {
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
        DistanceMetricKind::Mahalanobis => {
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
        DistanceMetricKind::Cosine => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Cosine)
        }
        DistanceMetricKind::Correlation => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Correlation)
        }
        DistanceMetricKind::Hamming => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Hamming)
        }
        DistanceMetricKind::Jaccard => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Jaccard)
        }
        DistanceMetricKind::Spearman => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Spearman)
        }
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
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            Ok(tensor::tensor_value_f64(&tensor, 0))
        }
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
    if tensor::tensor_element_len(&tensor) != expected_len {
        return Err(distance_error(
            name,
            format!("{name}: {label} length must match the number of columns"),
        ));
    }
    Ok(tensor::tensor_into_values_f64(tensor))
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
    Ok(tensor::tensor_into_values_f64(tensor))
}

fn split_pdist2_options(args: Vec<Value>) -> BuiltinResult<(Vec<Value>, Option<Selection>)> {
    let mut metric_args = Vec::with_capacity(args.len());
    let mut selection = None;
    let mut idx = 0usize;
    while idx < args.len() {
        let selector = keyword_of(&args[idx])
            .as_deref()
            .and_then(DistanceSelectionKind::from_keyword);
        if let Some(selector) = selector {
            if selection.is_some() || idx + 1 >= args.len() {
                return Err(distance_error(
                    PDIST2_NAME,
                    "pdist2: Smallest or Largest must be a single name-value pair",
                ));
            }
            let k = parse_positive_integer(PDIST2_NAME, &args[idx + 1], "pdist2 selection count")?;
            selection = Some(match selector {
                DistanceSelectionKind::Smallest => Selection::Smallest(k),
                DistanceSelectionKind::Largest => Selection::Largest(k),
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
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                distance_error(
                    name,
                    format!("{name}: {label} must be a positive integer scalar"),
                )
            });
    }
    let raw = match value {
        Value::Num(value) => *value,
        Value::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(array) if array.data.len() == 1 => f64::from(array.data[0] != 0),
        other => {
            return Err(distance_error(
                name,
                format!("{name}: {label} must be a positive integer scalar, got {other:?}"),
            ))
        }
    };
    if !raw.is_finite()
        || raw < 1.0
        || raw.fract().abs() > EPS
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
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
    let mut metric = KnnMetric {
        kind: DistanceMetricKind::Euclidean,
        fast_alias: false,
    };
    let mut metric_parameter: Option<Value> = None;
    let mut ns_method: Option<KnnSearchMethod> = None;
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
        let option = KnnOption::from_keyword(&name).ok_or_else(|| {
            distance_error(
                KNNSEARCH_NAME,
                format!("knnsearch: unsupported option '{name}'"),
            )
        })?;
        match option {
            KnnOption::K => k = parse_positive_integer(KNNSEARCH_NAME, &value, "K")?,
            KnnOption::IncludeTies => include_ties = parse_logical_scalar(&value, "IncludeTies")?,
            KnnOption::SortIndices => sort_indices = parse_logical_scalar(&value, "SortIndices")?,
            KnnOption::Distance => {
                let distance = keyword_of(&value).ok_or_else(|| {
                    distance_error(KNNSEARCH_NAME, "knnsearch: Distance must be text")
                })?;
                metric = KnnMetric::from_keyword(&distance).ok_or_else(|| {
                    distance_error(
                        KNNSEARCH_NAME,
                        format!("knnsearch: unsupported distance metric '{distance}'"),
                    )
                })?;
                metric_name = Some(Value::from(metric.kind.canonical_name()));
            }
            KnnOption::P | KnnOption::Cov | KnnOption::Scale => {
                if metric_parameter.is_some() {
                    return Err(distance_error(
                        KNNSEARCH_NAME,
                        "knnsearch: only one distance parameter may be specified",
                    ));
                }
                metric_parameter = Some(value);
            }
            KnnOption::NsMethod => {
                let method = keyword_of(&value).ok_or_else(|| {
                    distance_error(KNNSEARCH_NAME, "knnsearch: NSMethod must be text")
                })?;
                ns_method = Some(KnnSearchMethod::from_keyword(&method).ok_or_else(|| {
                    distance_error(
                        KNNSEARCH_NAME,
                        format!("knnsearch: unsupported NSMethod '{method}'"),
                    )
                })?);
            }
            KnnOption::CacheSize => {
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
            KnnOption::BucketSize => {
                parse_positive_integer(KNNSEARCH_NAME, &value, "BucketSize")?;
            }
        }
        idx += 2;
    }
    if ns_method == Some(KnnSearchMethod::KdTree) && !metric.allows_kdtree() {
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
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(distance_error(
                KNNSEARCH_NAME,
                format!("knnsearch: {label} must be logical scalar, got {value:?}"),
            )),
        };
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let raw = tensor::tensor_value_f64(tensor, 0);
            if raw == 0.0 || raw == 1.0 {
                Ok(raw != 0.0)
            } else {
                Err(distance_error(
                    KNNSEARCH_NAME,
                    format!("knnsearch: {label} must be logical scalar, got {value:?}"),
                ))
            }
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
    tensor::tensor_value_f64(tensor, col * tensor.rows + row)
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
    let requested = match selection {
        Selection::Smallest(k) | Selection::Largest(k) => k,
    };
    let k = requested.min(distances.rows);
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

#[cfg(test)]
mod tests;
