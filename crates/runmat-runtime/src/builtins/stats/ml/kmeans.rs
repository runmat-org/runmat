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

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod admission;
mod compute;
mod geometry;
mod options;
mod output;
#[cfg(test)]
mod admission_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
use crate::builtins::common::random;
#[cfg(test)]
use admission::is_empty_option_value;
use admission::{BorrowedOperands, OptionHeader};
use compute::run_kmeans;
use geometry::{
    preflight_cluster_count, preflight_observations, prepare_data, tensor_rows_cols,
    validate_options,
};
#[cfg(test)]
use options::{bool_option, is_empty_numeric, parse_positive_usize};
use options::{parse_k, parse_options};
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
    let mut borrowed = BorrowedOperands::new();
    borrowed.protect_value(&x);
    borrowed.protect_value(&k);
    for value in &rest {
        borrowed.protect_value(value);
    }
    let headers = admission::parse_option_headers(rest)?;
    preflight_observations(&x)?;
    preflight_cluster_count(&k)?;
    ensure_kmeans_extensions(&x, &k, &headers)?;
    let options = admission::admit_option_values(headers)?;
    if !matches!(&k, Value::GpuTensor(_)) && !admission::is_empty_numeric(&k) {
        options::parse_positive_usize(&k, "k")?;
    }
    let output_source = value_gpu_source(&x);
    borrowed.record_consumed(&x);
    let x_was_integer = typed_integer(&x);
    let x = gather(x).await?;
    if x_was_integer && !admission::numeric_value_is_exact(&x) {
        return Err(invalid(
            "kmeans: typed-integer observation data must be exactly representable as double",
        ));
    }
    borrowed.record_consumed(&k);
    let k = gather(k).await?;
    let mut options = parse_options(options, &mut borrowed).await?;
    let data = prepare_data(x, options.distance)?;
    let k = parse_k(&k, &options.start)?;
    validate_options(&mut options, k, &data)?;
    let result = run_kmeans(&data, k, &options)?;
    let value = outputs_for_count(result, &data)?;
    restore_kmeans_value(value, output_source.as_ref(), &borrowed)
}

#[cfg(test)]
fn parse_option_headers(args: &[Value]) -> BuiltinResult<Vec<OptionHeader>> {
    admission::parse_option_headers(args.to_vec())
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
    borrowed: &BorrowedOperands,
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    match value {
        Value::Tensor(_) => {
            let protected = borrowed.protected().iter().collect::<Vec<_>>();
            let intent = borrowed.intent_sources().iter().collect::<Vec<_>>();
            restore_one_kmeans_output(value, source, &protected, &intent)
        }
        Value::OutputList(values) => {
            let mut restored = Vec::with_capacity(values.len());
            for value in values {
                let mut protected = borrowed.protected().iter().collect::<Vec<_>>();
                let intent = borrowed.intent_sources().iter().collect::<Vec<_>>();
                protected.extend(restored.iter().filter_map(|value| match value {
                    Value::GpuTensor(handle) => Some(handle),
                    _ => None,
                }));
                match restore_one_kmeans_output(value, source, &protected, &intent) {
                    Ok(value) => restored.push(value),
                    Err(error) => return Err(retire_staged_outputs(restored, error)),
                }
            }
            Ok(Value::OutputList(restored))
        }
        other => Ok(other),
    }
}

fn retire_staged_outputs(values: Vec<Value>, mut error: RuntimeError) -> RuntimeError {
    for value in values {
        if let Value::GpuTensor(handle) = value {
            error = crate::builtins::common::gpu_helpers::reject_gpu_output(handle, error);
        }
    }
    error
}

fn restore_one_kmeans_output(
    value: Value,
    source: &runmat_accelerate_api::GpuTensorHandle,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
    intent: &[&runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Value> {
    let restored = match value {
        Value::Tensor(tensor) => {
            crate::builtins::common::gpu_helpers::restore_class_preserving_value_with_operands(
                source,
                Value::Tensor(tensor),
                NAME,
                crate::builtins::common::gpu_helpers::RestoreOperandContext::new(protected, intent),
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

fn ensure_kmeans_extensions(x: &Value, k: &Value, options: &[OptionHeader]) -> BuiltinResult<()> {
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
    for option in options {
        match option.kind {
            KmeansOption::Start if typed_integer(&option.value) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &KMEANS_INTEGER_START_EXTENSION,
                    NAME,
                )?;
            }
            KmeansOption::MaxIter | KmeansOption::Replicates if typed_integer(&option.value) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &KMEANS_INTEGER_ITERATION_CONTROL_EXTENSION,
                    NAME,
                )?;
            }
            KmeansOption::Options => {
                if let Some(fields) = &option.statset_fields {
                    for field in fields {
                        match field.kind {
                            KmeansStatsetField::MaxIter if typed_integer(&field.value) => {
                                crate::compatibility::ensure_builtin_extension_enabled(
                                    &KMEANS_INTEGER_ITERATION_CONTROL_EXTENSION,
                                    NAME,
                                )?;
                            }
                            KmeansStatsetField::UseParallel | KmeansStatsetField::UseSubstreams
                                if typed_integer(&field.value) =>
                            {
                                crate::compatibility::ensure_builtin_extension_enabled(
                                    &KMEANS_INTEGER_PARALLEL_TOGGLE_EXTENSION,
                                    NAME,
                                )?;
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(with_kmeans_context)
}
