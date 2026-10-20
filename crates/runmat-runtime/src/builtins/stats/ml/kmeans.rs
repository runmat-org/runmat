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
mod cooperation;
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
use admission::{BorrowedOperands, OptionHeader};
#[cfg(test)]
use geometry::prepare_data;
use geometry::{
    preflight_cluster_count, preflight_observations, tensor_rows_cols, validate_options,
};
#[cfg(test)]
use options::{bool_option, is_empty_numeric, parse_positive_usize};
use options::{parse_k, parse_options};
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

fn reserve_borrowed<T>(
    values: &mut Vec<T>,
    additional: usize,
    message: &'static str,
) -> BuiltinResult<()> {
    values
        .try_reserve(additional)
        .map_err(|source| internal_with_source(message, source))
}

fn internal_with_source(
    message: impl Into<String>,
    source: std::collections::TryReserveError,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source);
    if let Some(identifier) = ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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
    context = true,
    builtin_path = "crate::builtins::stats::ml::kmeans"
)]
pub(crate) async fn kmeans_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    k: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    kmeans_in_runtime(runtime.as_ref(), x, k, rest).await
}

#[cfg(test)]
async fn kmeans_builtin(x: Value, k: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    kmeans_in_runtime(
        None,
        x,
        k,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

async fn kmeans_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    x: Value,
    k: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = cooperation::KmeansCooperation::new(runtime);
    cooperation.completed(0).await?;
    if crate::output_count::current_output_count().is_some_and(|count| count > 4) {
        return Err(invalid("kmeans: too many output arguments; maximum is 4"));
    }
    let mut borrowed = BorrowedOperands::new();
    borrowed.protect_value(&x)?;
    borrowed.protect_value(&k)?;
    for value in &rest {
        borrowed.protect_value(value)?;
    }
    let headers = admission::parse_option_headers(rest)?;
    preflight_observations(&x)?;
    preflight_cluster_count(&k)?;
    ensure_kmeans_extensions(&x, &k, &headers)?;
    let options = admission::admit_option_values_cooperative(headers, &mut cooperation).await?;
    if !matches!(&k, Value::GpuTensor(_)) && !admission::is_empty_numeric(&k) {
        options::parse_positive_usize(&k, "k")?;
    }
    geometry::preflight_effective_start(&x, &k, &options)?;
    let output_source = value_gpu_source(&x);
    borrowed.record_consumed(&x)?;
    let x_was_integer = typed_integer(&x);
    let x = gather(x).await?;
    if x_was_integer && !admission::numeric_value_is_exact_cooperative(&x, &mut cooperation).await?
    {
        return Err(invalid(
            "kmeans: typed-integer observation data must be exactly representable as double",
        ));
    }
    borrowed.record_consumed(&k)?;
    let k = gather(k).await?;
    let mut options = parse_options(options, &mut borrowed, &mut cooperation).await?;
    let data = geometry::prepare_data_cooperative(x, options.distance, &mut cooperation).await?;
    let k = parse_k(&k, &options.start)?;
    validate_options(&mut options, k, &data, &mut cooperation).await?;
    let result = compute::run_kmeans_cooperative(&data, k, &options, &mut cooperation).await?;
    let value = output::outputs_for_count_cooperative(result, &data, &mut cooperation).await?;
    cooperation.completed(0).await?;
    restore_kmeans_value_cooperative(value, output_source.as_ref(), &borrowed, &mut cooperation)
        .await
}

#[cfg(test)]
fn parse_option_headers(args: &[Value]) -> BuiltinResult<Vec<OptionHeader>> {
    admission::parse_option_headers(crate::builtins::common::test_support::builtin_arguments(
        args.to_vec(),
    ))
}

fn value_gpu_source(value: &Value) -> Option<runmat_accelerate_api::GpuTensorHandle> {
    let Value::GpuTensor(handle) = value else {
        return None;
    };
    Some(handle.clone())
}

#[cfg(test)]
fn restore_kmeans_value(
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    borrowed: &BorrowedOperands,
) -> BuiltinResult<Value> {
    futures::executor::block_on(restore_kmeans_value_cooperative(
        value,
        source,
        borrowed,
        &mut cooperation::KmeansCooperation::new(None),
    ))
}

async fn restore_kmeans_value_cooperative(
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    borrowed: &BorrowedOperands,
    cooperation: &mut cooperation::KmeansCooperation,
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    match value {
        Value::Tensor(_) => {
            let protected = borrowed_handle_refs(borrowed.protected())?;
            let intent = borrowed_handle_refs(borrowed.intent_sources())?;
            let restored = restore_one_kmeans_output(value, source, &protected, &intent)?;
            if let Err(error) = cooperation.completed(0).await {
                return Err(retire_staged_outputs(vec![restored], error));
            }
            Ok(restored)
        }
        Value::OutputList(values) => {
            let mut restored = Vec::new();
            reserve_borrowed(
                &mut restored,
                values.len(),
                "kmeans: restored outputs exceed available memory",
            )?;
            let values = values.try_into_owned().map_err(|_| {
                crate::sequence::sequence_error_to_runtime(
                    runmat_value::ValueSequenceError::SharedOutputStorage,
                )
            })?;
            for value in values {
                let mut protected = match borrowed_handle_refs(borrowed.protected()) {
                    Ok(handles) => handles,
                    Err(error) => return Err(retire_staged_outputs(restored, error)),
                };
                if let Err(error) = reserve_borrowed(
                    &mut protected,
                    restored.len(),
                    "kmeans: borrowed output handles exceed available memory",
                ) {
                    return Err(retire_staged_outputs(restored, error));
                }
                let intent = match borrowed_handle_refs(borrowed.intent_sources()) {
                    Ok(handles) => handles,
                    Err(error) => return Err(retire_staged_outputs(restored, error)),
                };
                protected.extend(restored.iter().filter_map(|value| match value {
                    Value::GpuTensor(handle) => Some(handle),
                    _ => None,
                }));
                match restore_one_kmeans_output(value, source, &protected, &intent) {
                    Ok(value) => restored.push(value),
                    Err(error) => return Err(retire_staged_outputs(restored, error)),
                }
                if let Err(error) = cooperation.completed(0).await {
                    return Err(retire_staged_outputs(restored, error));
                }
            }
            Ok(Value::output_list(restored))
        }
        other => Ok(other),
    }
}

fn borrowed_handle_refs(
    handles: &[runmat_accelerate_api::GpuTensorHandle],
) -> BuiltinResult<Vec<&runmat_accelerate_api::GpuTensorHandle>> {
    let mut refs = Vec::new();
    reserve_borrowed(
        &mut refs,
        handles.len(),
        "kmeans: borrowed output handles exceed available memory",
    )?;
    refs.extend(handles.iter());
    Ok(refs)
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
        match option {
            OptionHeader::Argument {
                kind: KmeansOption::Start,
                value,
            } if typed_integer(value) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &KMEANS_INTEGER_START_EXTENSION,
                    NAME,
                )?;
            }
            OptionHeader::Argument {
                kind: KmeansOption::MaxIter | KmeansOption::Replicates,
                value,
            } if typed_integer(value) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &KMEANS_INTEGER_ITERATION_CONTROL_EXTENSION,
                    NAME,
                )?;
            }
            OptionHeader::Statset(fields) => {
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
            _ => {}
        }
    }
    Ok(())
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    // Admission leaves only direct numeric operands here. Keep owned host
    // payloads in place; route resident operands through the shared owner-
    // validating download path.
    if matches!(&value, Value::GpuTensor(_)) {
        gather_if_needed_async(&value)
            .await
            .map_err(with_kmeans_context)
    } else {
        Ok(value)
    }
}
