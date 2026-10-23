//! Pairwise distance and squareform helpers for Statistics and Machine Learning workflows.

use runmat_builtins::{BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_builtins::{
    DistanceControlKind, DistanceSelectionKind, KnnCacheSizeKeyword, KnnMetric, KnnOption,
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
use runmat_value::{Tensor, Value};

use crate::builtins::common::arg_tokens::scalar_keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const PDIST_NAME: &str = "pdist";
const PDIST2_NAME: &str = "pdist2";
const KNNSEARCH_NAME: &str = "knnsearch";
const SQUAREFORM_NAME: &str = "squareform";
const EPS: f64 = 1.0e-12;

mod options;
mod selection;
mod squareform;
#[cfg(test)]
mod tests;

use super::pairwise_metric::{
    condensed_distances, distance_matrix, observation_matrix_dimensions, parse_metric_kind,
    parse_metric_name, scalar_parameter, DistanceMetric, MetricCaller,
};
use options::{
    parse_knnsearch_options, preflight_knn_metric_parameter, split_pdist2_options,
    validate_numeric_cache_sizes,
};
#[cfg(test)]
use options::{parse_logical_scalar, parse_positive_integer};
use selection::{knnsearch_outputs, selected_distance_outputs};
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

fn distance_keyword(name: &'static str, value: &Value) -> BuiltinResult<Option<String>> {
    scalar_keyword_of(value, name).map_err(|source| {
        let mut error = distance_error(name, format!("{name}: text control is not valid UTF-16"));
        error.source = Some(Box::new(source));
        error
    })
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

#[cfg(test)]
async fn pdist_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    pdist_builtin_registered(None, x, rest.into()).await
}

#[cfg(test)]
async fn pdist2_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    pdist2_builtin_registered(None, x, y, rest.into()).await
}

#[runtime_builtin(
    name = "pdist",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn pdist_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_pdist_integer_boundaries(PairwiseDistanceBuiltin::Pdist, &x, None, rest.view()).await?;
    let metric_args = strip_cache_size_option(MetricCaller::Pdist, rest, runtime.as_ref()).await?;
    let kind = parse_metric_name(MetricCaller::Pdist, metric_args.view())?;
    let x = value_to_matrix(PDIST_NAME, x).await?;
    let metric = parse_metric_kind(
        MetricCaller::Pdist,
        &x,
        kind,
        metric_args.view().range(metric_args.len().min(1)..)?,
    )
    .await?;
    let distances = condensed_distances(MetricCaller::Pdist, &x, &metric)?;
    Ok(Value::Tensor(distances))
}

#[runtime_builtin(
    name = "pdist2",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::distance"
)]
async fn pdist2_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    y: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_pdist_integer_boundaries(PairwiseDistanceBuiltin::Pdist2, &x, Some(&y), rest.view())
        .await?;
    let (metric_args, selection) = split_pdist2_options(rest, runtime.as_ref()).await?;
    let metric_args =
        strip_cache_size_option(MetricCaller::Pdist2, metric_args, runtime.as_ref()).await?;
    let kind = parse_metric_name(MetricCaller::Pdist2, metric_args.view())?;
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
    if observation_matrix_dimensions(&x).1 != observation_matrix_dimensions(&y).1 {
        return Err(distance_error(
            PDIST2_NAME,
            "pdist2: X and Y must have the same number of columns",
        ));
    }
    let metric = parse_metric_kind(
        MetricCaller::Pdist2,
        &x,
        kind,
        metric_args.view().range(metric_args.len().min(1)..)?,
    )
    .await?;
    let distances = distance_matrix(MetricCaller::Pdist2, &x, &y, &metric)?;
    match selection {
        Some(selection) => selected_distance_outputs(&distances, selection),
        None => match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::output_list(Vec::new())),
            Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![Value::Tensor(distances)],
            )),
            None => Ok(Value::Tensor(distances)),
        },
    }
}

async fn strip_cache_size_option(
    caller: MetricCaller,
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let name = caller.name();
    let mut count = 0usize;
    let mut index = 0usize;
    while index < args.len() {
        if distance_keyword(name, &args[index])?
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
            if distance_keyword(name, value)?
                .as_deref()
                .and_then(KnnCacheSizeKeyword::from_keyword)
                == Some(KnnCacheSizeKeyword::Maximal)
            {
                index += 2;
                continue;
            }
            let cache_size = scalar_parameter(caller, value, "CacheSize").await?;
            if !cache_size.is_finite() || cache_size <= 0.0 {
                return Err(distance_error(
                    name,
                    format!("{name}: CacheSize must be positive or 'maximal'"),
                ));
            }
            index += 2;
            continue;
        }
        count += 1;
        index += 1;
    }
    if count == args.len() {
        return Ok(args);
    }
    let mut cursor = 0usize;
    let metric_args = args
        .try_map_preserving_roots_async(count, runtime, |_| {
            let value = (|| -> BuiltinResult<Value> {
                while distance_keyword(name, &args[cursor])?
                    .as_deref()
                    .and_then(DistanceControlKind::from_keyword)
                    .is_some()
                {
                    cursor += 2;
                }
                let value = args[cursor].clone();
                cursor += 1;
                Ok(value)
            })();
            std::future::ready(value)
        })
        .await?;
    Ok(metric_args)
}

async fn ensure_pdist_integer_boundaries(
    builtin: PairwiseDistanceBuiltin,
    x: &Value,
    y: Option<&Value>,
    rest: crate::call::FunctionArgumentView<'_>,
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
        let selector = match index.checked_sub(1) {
            Some(previous) => distance_keyword(name, &rest[previous])?,
            None => None,
        }
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
async fn knnsearch_builtin(
    x: Value,
    y: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let parsed = parse_knnsearch_options(rest.view())?;
    let columns = preflight_knnsearch_geometry(&x, &y, parsed.options.k)?;
    preflight_knn_metric_parameter(
        parsed.metric.kind,
        parsed.metric_parameter.as_ref(),
        columns,
    )?;
    ensure_knnsearch_extensions(&x, &y, rest.view())?;
    let output_source = crate::builtins::common::gpu_helpers::select_resident_output_source(
        [&x, &y].into_iter().filter_map(knn_gpu_source),
        KNNSEARCH_NAME,
    )?;
    ensure_knnsearch_exact_boundaries(&x, &y, rest.view()).await?;
    validate_numeric_cache_sizes(parsed.numeric_cache_sizes).await?;
    let x = value_to_matrix(KNNSEARCH_NAME, x).await?;
    let y = value_to_matrix(KNNSEARCH_NAME, y).await?;
    if observation_matrix_dimensions(&x).1 != observation_matrix_dimensions(&y).1 {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: X and Y must have the same number of columns",
        ));
    }
    if parsed.options.k > observation_matrix_dimensions(&x).0 {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: K must be <= size(X,1)",
        ));
    }
    let metric = parse_metric_kind(
        MetricCaller::Knnsearch,
        &x,
        parsed.metric.kind,
        crate::call::FunctionArgumentView::from_slice(parsed.metric_parameter.as_slice()),
    )
    .await?;
    let value = knnsearch_outputs(&x, &y, &metric, parsed.options)?;
    restore_knnsearch_value(value, output_source.as_ref())
}

fn preflight_knnsearch_geometry(x: &Value, y: &Value, k: usize) -> BuiltinResult<usize> {
    let (x_rows, x_columns) = preflight_observation_shape(x)?;
    let (_, y_columns) = preflight_observation_shape(y)?;
    if x_columns != y_columns {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: X and Y must have the same number of columns",
        ));
    }
    if k > x_rows {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: K must be <= size(X,1)",
        ));
    }
    Ok(x_columns)
}

fn preflight_observation_shape(value: &Value) -> BuiltinResult<(usize, usize)> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(distance_error(
                    KNNSEARCH_NAME,
                    "knnsearch: observations must have real numeric or logical storage",
                ));
            }
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok((1, 1)),
        _ => {
            return Err(distance_error(
                KNNSEARCH_NAME,
                "knnsearch: observations must be numeric arrays",
            ))
        }
    };
    let geometry = match shape {
        [] => Ok((1, 1)),
        [rows] => Ok((*rows, 1)),
        [rows, columns] => Ok((*rows, *columns)),
        _ => Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: input must be a numeric vector or 2-D matrix",
        )),
    }?;
    geometry
        .0
        .checked_mul(geometry.1)
        .ok_or_else(|| distance_error(KNNSEARCH_NAME, "knnsearch: observation shape overflow"))?;
    Ok(geometry)
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
    let mut protected_outputs = Vec::new();
    restore_knnsearch_value_protecting(value, source, &mut protected_outputs)
}

fn restore_knnsearch_value_protecting(
    value: Value,
    source: &runmat_accelerate_api::GpuTensorHandle,
    protected_outputs: &mut Vec<runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            let protected = protected_outputs.iter().collect::<Vec<_>>();
            let restored =
                crate::builtins::common::gpu_helpers::restore_class_preserving_value_protecting(
                    source,
                    Value::Tensor(tensor),
                    KNNSEARCH_NAME,
                    &protected,
                )?;
            if runmat_accelerate_api::handle_is_explicit(source)
                && !matches!(restored, Value::GpuTensor(_))
            {
                return Err(distance_error(
                    KNNSEARCH_NAME,
                    "knnsearch: provider cannot preserve explicit gpuArray output residency",
                ));
            }
            if let Value::GpuTensor(handle) = &restored {
                protected_outputs.push(handle.clone());
            }
            Ok(restored)
        }
        Value::Cell(mut cell) => {
            if !cell.data_owner().is_unique() {
                cell.make_data_unique_unbounded()
                    .map_err(|error| distance_error(KNNSEARCH_NAME, error.to_string()))?;
            }
            let slots = cell.try_data_mut_unique().ok_or_else(|| {
                distance_error(
                    KNNSEARCH_NAME,
                    "shared admitted Cell output requires an admitted replacement",
                )
            })?;
            for slot in slots {
                *slot = restore_knnsearch_value_protecting(
                    std::mem::replace(slot, Value::Num(0.0)),
                    source,
                    protected_outputs,
                )?;
            }
            Ok(Value::Cell(cell))
        }
        Value::OutputList(values) => runmat_value::ValueSequence::from_output_storage(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?
            .try_map(
                |value| restore_knnsearch_value_protecting(value, source, protected_outputs),
                crate::sequence::sequence_error_to_runtime,
            )
            .map(|sequence| Value::OutputList(sequence.into_output_storage())),
        other => Ok(other),
    }
}

fn knn_typed_integer(value: &Value) -> bool {
    crate::builtins::common::validation::value_has_native_integer_class(value)
}

fn explicit_knn_gpu(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
}

fn ensure_knnsearch_extensions(
    x: &Value,
    y: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
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
        let Some(option) = distance_keyword(KNNSEARCH_NAME, &rest[idx])?
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
                    if distance_keyword(KNNSEARCH_NAME, value)?
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
                explicit_exhaustive = distance_keyword(KNNSEARCH_NAME, value)?
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
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for (value, role) in [(x, "X"), (y, "Y")] {
        ensure_knn_exact_f64(value, role).await?;
    }
    let mut idx = 0;
    while idx + 1 < rest.len() {
        if distance_keyword(KNNSEARCH_NAME, &rest[idx])?
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
async fn squareform_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
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

async fn value_to_matrix(name: &'static str, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|error| with_distance_context(name, error))?;
    host_value_to_matrix(name, gathered)
}

fn host_value_to_matrix(name: &'static str, gathered: Value) -> BuiltinResult<Tensor> {
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| distance_error(name, format!("{name}: {err}")))?;
    if tensor.shape().len() > 2 {
        return Err(distance_error(
            name,
            format!("{name}: input must be a numeric vector or 2-D matrix"),
        ));
    }
    Ok(tensor)
}
