//! t-SNE dimensionality reduction compatibility surface.

use runmat_builtins::{
    BuiltinErrorDescriptor, TsneAlgorithm as Algorithm, TsneCacheSizeKeyword, TsneDisplay,
    TsneDistance as DistanceMetric, TsneOption, TsneStatsetField,
    TSNE_ERROR_INTERNAL as ERROR_INTERNAL, TSNE_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    TSNE_INTEGER_DATA_EXTENSION, TSNE_INTEGER_OPTION_EXTENSION, TSNE_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod admission;
mod cooperation;
mod geometry;
mod metric;
mod optimizer;
mod options;
mod workspace;
#[cfg(test)]
mod tests;

use admission::{
    admit_options, ensure_exact_tsne_floating_options, ensure_exact_tsne_integer_value,
    materialize_options, option_headers, OptionHeader,
};
use optimizer::compute_tsne_cooperative;
#[cfg(test)]
use options::is_empty_numeric;
use options::parse_options;

const NAME: &str = "tsne";
const DEFAULT_MAX_ITER: usize = 1000;
const MAX_WORK_CELLS: usize = 4_000_000;
const MAX_EMBEDDING_DIMS: usize = 128;
const MAX_ITERATIONS: usize = 100_000;
const MAX_NUM_PRINT: usize = 1_000_000;
const MIN_PROB: f64 = 1.0e-12;

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

fn internal_resource(message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn with_tsne_context(mut error: RuntimeError) -> RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(NAME);
    }
    error
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::tsne"
)]
pub(crate) async fn tsne_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    tsne_in_runtime(runtime.as_ref(), x, rest).await
}

#[cfg(test)]
async fn tsne_builtin(x: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    tsne_in_runtime(
        None,
        x,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

async fn tsne_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let mut cooperation = cooperation::TsneCooperation::new(runtime);
    cooperation.completed(0).await?;
    preflight_observations(&x)?;
    let headers = option_headers(rest)?;
    ensure_tsne_extensions(&x, &headers)?;
    let admitted = admit_options(headers)?;
    ensure_exact_tsne_integer_value(&x, "observation data", &mut cooperation).await?;
    ensure_exact_tsne_floating_options(&admitted, &mut cooperation).await?;
    let x = gather(x).await?;
    ensure_exact_tsne_integer_value(&x, "observation data", &mut cooperation).await?;
    let admitted = materialize_options(admitted, &mut cooperation).await?;
    ensure_exact_tsne_floating_options(&admitted, &mut cooperation).await?;
    let options = parse_options(admitted)?;
    let tensor =
        tensor::value_into_tensor_for(NAME, x).map_err(|err| invalid(format!("tsne: {err}")))?;
    let result = compute_tsne_cooperative(tensor, options, &mut cooperation).await?;
    cooperation.completed(0).await?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![Value::Tensor(result.embedding)])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![Value::Tensor(result.embedding), Value::Num(result.loss)],
        )),
        None => Ok(Value::Tensor(result.embedding)),
    }
}

fn tsne_value_contains_typed_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        Value::Cell(cell) => cell.data().iter().any(tsne_value_contains_typed_integer),
        Value::Struct(value) => value.fields.values().any(tsne_value_contains_typed_integer),
        Value::OutputList(values) => values.iter().any(tsne_value_contains_typed_integer),
        _ => false,
    }
}

fn tsne_value_contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data().iter().any(tsne_value_contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(tsne_value_contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(tsne_value_contains_explicit_gpu),
        _ => false,
    }
}

fn ensure_tsne_extensions(x: &Value, rest: &[OptionHeader]) -> BuiltinResult<()> {
    if tsne_value_contains_typed_integer(x) {
        crate::compatibility::ensure_builtin_extension_enabled(&TSNE_INTEGER_DATA_EXTENSION, NAME)?;
    }
    if rest
        .iter()
        .any(|(_, value)| tsne_value_contains_typed_integer(value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TSNE_INTEGER_OPTION_EXTENSION,
            NAME,
        )?;
    }
    if tsne_value_contains_explicit_gpu(x)
        || rest
            .iter()
            .any(|(_, value)| tsne_value_contains_explicit_gpu(value))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TSNE_RESIDENT_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gather_if_needed_async(&value)
            .await
            .map_err(with_tsne_context),
        host => Ok(host),
    }
}

fn preflight_observations(value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) => {
            if handle.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real {
                return Err(invalid("tsne: X must have real numeric or logical storage"));
            }
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Ok(()),
        _ => {
            return Err(invalid(
                "tsne: X must be a real numeric vector or 2-D matrix",
            ))
        }
    };
    if shape.len() > 2 {
        return Err(invalid("tsne: X must be a 2-D numeric matrix"));
    }
    shape
        .iter()
        .try_fold(1usize, |size, dimension| size.checked_mul(*dimension))
        .ok_or_else(|| invalid("tsne: observation shape overflow"))?;
    Ok(())
}
