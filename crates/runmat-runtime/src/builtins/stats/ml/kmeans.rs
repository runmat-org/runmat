//! k-means clustering compatibility surface.

use runmat_builtins::{
    BuiltinErrorDescriptor, KmeansDisplay, KmeansDistance as Distance,
    KmeansEmptyAction as EmptyAction, KmeansOption, KmeansStart, KmeansStatsetField, KmeansToggle,
    KMEANS_ERROR_INTERNAL as ERROR_INTERNAL,
    KMEANS_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    KMEANS_INTEGER_CLUSTER_COUNT_EXTENSION, KMEANS_INTEGER_DATA_EXTENSION,
    KMEANS_INTEGER_ITERATION_CONTROL_EXTENSION, KMEANS_INTEGER_PARALLEL_TOGGLE_EXTENSION,
    KMEANS_INTEGER_START_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::random_args::keyword_of;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod compute;
mod geometry;
mod options;
mod output;
#[cfg(test)]
mod tests;

#[cfg(test)]
use crate::builtins::common::random;
use compute::run_kmeans;
use geometry::{
    preflight_cluster_count, preflight_observations, prepare_data, tensor_rows_cols,
    validate_options,
};
#[cfg(test)]
use options::{bool_option, is_empty_numeric, parse_positive_usize};
use options::{parse_k, parse_option_headers, parse_options};
use output::outputs_for_count;
#[cfg(test)]
use runmat_builtins::{KMEANS_EXTENSIONS, KMEANS_INTEGER_CAPABILITIES};

const NAME: &str = "kmeans";
const EPS: f64 = 1.0e-12;
const DEFAULT_MAX_ITER: usize = 100;
const MAX_WORK_CELLS: usize = 20_000_000;

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

fn with_kmeans_context(mut error: RuntimeError) -> RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(NAME);
    }
    error
}

#[derive(Clone, Debug)]
enum StartSpec {
    Named(KmeansStart),
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
            start: StartSpec::Named(KmeansStart::Plus),
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
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::kmeans"
)]
async fn kmeans_builtin(x: Value, k: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 4) {
        return Err(invalid("kmeans: too many output arguments; maximum is 4"));
    }
    parse_option_headers(&rest)?;
    preflight_observations(&x)?;
    preflight_cluster_count(&k)?;
    ensure_kmeans_extensions(&x, &k, &rest)?;
    ensure_exact_integer_boundary(&x, "observation data").await?;
    ensure_exact_start_boundary(&rest).await?;
    let output_source = value_gpu_source(&x);
    let x = gather(x).await?;
    let k = gather(k).await?;
    let rest = gather_all(rest).await?;
    let mut options = parse_options(rest)?;
    let data = prepare_data(x, options.distance)?;
    let k = parse_k(&k, &options.start)?;
    validate_options(&mut options, k, &data)?;
    let result = run_kmeans(&data, k, &options)?;
    let value = outputs_for_count(result, &data)?;
    restore_kmeans_value(value, output_source.as_ref())
}

fn value_gpu_source(value: &Value) -> Option<runmat_accelerate_api::GpuTensorHandle> {
    let Value::GpuTensor(handle) = value else {
        return None;
    };
    Some(handle.clone())
}

fn restore_kmeans_value(
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    match value {
        Value::Tensor(_) => restore_one_kmeans_output(value, source, &[]),
        Value::OutputList(values) => {
            let mut restored = Vec::with_capacity(values.len());
            for value in values {
                let protected = restored
                    .iter()
                    .filter_map(|value| match value {
                        Value::GpuTensor(handle) => Some(handle),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                restored.push(restore_one_kmeans_output(value, source, &protected)?);
            }
            Ok(Value::OutputList(restored))
        }
        other => Ok(other),
    }
}

fn restore_one_kmeans_output(
    value: Value,
    source: &runmat_accelerate_api::GpuTensorHandle,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Value> {
    let restored = match value {
        Value::Tensor(tensor) => {
            crate::builtins::common::gpu_helpers::restore_class_preserving_value_protecting(
                source,
                Value::Tensor(tensor),
                NAME,
                protected,
            )?
        }
        other => other,
    };
    if runmat_accelerate_api::handle_is_explicit(source) && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(internal(
            "kmeans: provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

fn typed_integer(value: &Value) -> bool {
    crate::builtins::common::validation::value_has_native_integer_class(value)
}

fn ensure_kmeans_extensions(x: &Value, k: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if typed_integer(x) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &KMEANS_INTEGER_DATA_EXTENSION,
            NAME,
        )?;
    }
    if typed_integer(k) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &KMEANS_INTEGER_CLUSTER_COUNT_EXTENSION,
            NAME,
        )?;
    }
    let mut idx = 0;
    while idx + 1 < rest.len() {
        let Some(name) = keyword_of(&rest[idx]) else {
            idx += 1;
            continue;
        };
        let value = &rest[idx + 1];
        let Some(option) = KmeansOption::from_keyword(name.trim()) else {
            idx += 2;
            continue;
        };
        match option {
            KmeansOption::Start if typed_integer(value) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &KMEANS_INTEGER_START_EXTENSION,
                    NAME,
                )?;
            }
            KmeansOption::MaxIter | KmeansOption::Replicates if typed_integer(value) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &KMEANS_INTEGER_ITERATION_CONTROL_EXTENSION,
                    NAME,
                )?;
            }
            KmeansOption::Options => {
                if let Value::Struct(options) = value {
                    for (field, value) in &options.fields {
                        if KmeansStatsetField::from_keyword(field) == KmeansStatsetField::MaxIter
                            && typed_integer(value)
                        {
                            crate::compatibility::ensure_builtin_extension_enabled(
                                &KMEANS_INTEGER_ITERATION_CONTROL_EXTENSION,
                                NAME,
                            )?;
                        }
                        if matches!(
                            KmeansStatsetField::from_keyword(field),
                            KmeansStatsetField::UseParallel | KmeansStatsetField::UseSubstreams
                        ) && typed_integer(value)
                        {
                            crate::compatibility::ensure_builtin_extension_enabled(
                                &KMEANS_INTEGER_PARALLEL_TOGGLE_EXTENSION,
                                NAME,
                            )?;
                        }
                    }
                }
            }
            _ => {}
        }
        idx += 2;
    }
    Ok(())
}

async fn ensure_exact_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    if typed_integer(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        return Err(invalid(format!(
            "kmeans: typed-integer {role} must be exactly representable as double"
        )));
    }
    Ok(())
}

async fn ensure_exact_start_boundary(rest: &[Value]) -> BuiltinResult<()> {
    let mut idx = 0;
    while idx + 1 < rest.len() {
        if keyword_of(&rest[idx]).and_then(KmeansOption::from_keyword) == Some(KmeansOption::Start)
        {
            ensure_exact_integer_boundary(&rest[idx + 1], "Start values").await?;
        }
        idx += 2;
    }
    Ok(())
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(with_kmeans_context)
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}
