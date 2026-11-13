//! Outlier detection and filling helpers.

use runmat_builtins::{
    BuiltinErrorDescriptor, OutlierOption, FILLOUTLIERS_INTEGER_DATA_EXTENSION,
    FILLOUTLIERS_INTEGER_FILL_EXTENSION, FILLOUTLIERS_NUMERIC_MASK_EXTENSION,
    FILLOUTLIERS_RESIDENT_INPUT_EXTENSION, ISOUTLIER_GPU_DATA_VARIABLES_EXTENSION,
    ISOUTLIER_GPU_MOVMEDIAN_EXTENSION, ISOUTLIER_GPU_SAMPLE_POINTS_EXTENSION,
    ISOUTLIER_INTEGER_DATA_EXTENSION, ISOUTLIER_INTEGER_DIMENSION_EXTENSION,
    ISOUTLIER_INTEGER_THRESHOLD_EXTENSION, ISOUTLIER_INTEGER_WINDOW_EXTENSION,
    OUTLIER_ERROR_INTERNAL, OUTLIER_ERROR_INVALID_ARGUMENT, OUTLIER_ERROR_TOO_MANY_OUTPUTS,
    OUTLIER_ERROR_UNSUPPORTED,
};
#[cfg(test)]
use runmat_builtins::{ISOUTLIER_EXTENSIONS, ISOUTLIER_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, IntegerStorage, LogicalArray, NumericDType, Tensor, Value};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod detection;
mod fill;
mod options;
#[cfg(test)]
use detection::centered_window;
use detection::detect_outliers;
use fill::fill_outlier_tensor;
#[cfg(test)]
use fill::fill_slice;
use options::{parse_detection_options, parse_fill_method, preflight_host_options, DecodedOptions};

const ISOUTLIER_NAME: &str = "isoutlier";
const FILLOUTLIERS_NAME: &str = "filloutliers";
const MAD_SCALE: f64 = 1.4826;
const MAX_EXACT_INTEGER_F64: u128 = 1_u128 << 53;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutlierOperation {
    Detect,
    Fill,
}

impl OutlierOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::Detect => ISOUTLIER_NAME,
            Self::Fill => FILLOUTLIERS_NAME,
        }
    }
}

fn outlier_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.into()))
        .with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    outlier_error(builtin, &OUTLIER_ERROR_INVALID_ARGUMENT, detail)
}

fn unsupported(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    outlier_error(builtin, &OUTLIER_ERROR_UNSUPPORTED, detail)
}

fn internal_error(builtin: &'static str, detail: impl Into<String>) -> RuntimeError {
    outlier_error(builtin, &OUTLIER_ERROR_INTERNAL, detail)
}

fn too_many_outputs(builtin: &'static str, max_outputs: usize) -> RuntimeError {
    outlier_error(
        builtin,
        &OUTLIER_ERROR_TOO_MANY_OUTPUTS,
        format!("{builtin}: requested more than {max_outputs} outputs"),
    )
}

#[derive(Clone, Debug)]
enum DetectionMethod {
    Median,
    Mean,
    Quartiles,
    Percentiles(f64, f64),
    MovingMedian(usize),
    MovingMean(usize),
}

#[derive(Clone, Debug)]
struct DetectionOptions {
    method: DetectionMethod,
    method_was_specified: bool,
    dim: Option<usize>,
    threshold_factor: Option<f64>,
    forced_mask: Option<(Vec<u8>, Vec<usize>)>,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            method: DetectionMethod::Median,
            method_was_specified: false,
            dim: None,
            threshold_factor: None,
            forced_mask: None,
        }
    }
}

#[derive(Clone, Debug)]
enum FillMethod {
    Center,
    Clip,
    Constant(f64),
    Previous,
    Next,
    Nearest,
    Linear,
}

#[derive(Clone, Debug)]
struct DetectionResult {
    mask: Vec<u8>,
    mask_shape: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    center: Vec<f64>,
    lower_full: Vec<f64>,
    upper_full: Vec<f64>,
    center_full: Vec<f64>,
    threshold_shape: Vec<usize>,
}

#[cfg(test)]
pub(crate) async fn isoutlier_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    isoutlier_builtin_registered(None, value, rest.into()).await
}

#[cfg(test)]
pub(crate) async fn filloutliers_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest =
        crate::execution::RootedValues::new(rest).map_err(|error| error.into_runtime_error())?;
    filloutliers_builtin_registered(None, value, rest.into()).await
}

#[runtime_builtin(
    name = "isoutlier",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::outliers"
)]
pub(crate) async fn isoutlier_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let decoded = DecodedOptions::new(OutlierOperation::Detect, rest.view())?;
    ensure_isoutlier_extensions(&value, rest.view(), &decoded)?;
    preflight_host_options(OutlierOperation::Detect, rest.view(), &decoded).await?;
    ensure_isoutlier_exact_boundaries(&value, rest.view(), &decoded).await?;
    let output_source = isoutlier_gpu_source(&value);
    let tensor = value_to_tensor(ISOUTLIER_NAME, value).await?;
    let options = parse_detection_options(
        OutlierOperation::Detect,
        rest,
        decoded.all(),
        runtime.as_ref(),
    )
    .await?;
    let result = detect_outliers(&tensor, &options, ISOUTLIER_NAME)?;
    let value = outlier_outputs(ISOUTLIER_NAME, None, result)?;
    restore_isoutlier_value(value, output_source.as_ref())
}

fn isoutlier_gpu_source(value: &Value) -> Option<runmat_accelerate_api::GpuTensorHandle> {
    let Value::GpuTensor(handle) = value else {
        return None;
    };
    Some(handle.clone())
}

fn restore_isoutlier_value(
    value: Value,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    match value {
        Value::Tensor(tensor) => restore_isoutlier_array(Value::Tensor(tensor), source),
        Value::LogicalArray(logical) => {
            restore_isoutlier_array(Value::LogicalArray(logical), source)
        }
        Value::OutputList(values) => runmat_value::ValueSequence::from_output_storage(values)
            .map_err(crate::sequence::sequence_error_to_runtime)?
            .try_map(
                |value| restore_isoutlier_value(value, Some(source)),
                crate::sequence::sequence_error_to_runtime,
            )
            .map(|sequence| Value::OutputList(sequence.into_output_storage())),
        other => Ok(other),
    }
}

fn restore_isoutlier_array(
    value: Value,
    source: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    let restored = crate::builtins::common::gpu_helpers::restore_class_preserving_value(
        source,
        value,
        ISOUTLIER_NAME,
    )?;
    if runmat_accelerate_api::handle_is_explicit(source) && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(internal_error(
            ISOUTLIER_NAME,
            "isoutlier: provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

fn explicit_isoutlier_gpu(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
}

fn ensure_isoutlier_extensions(
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    decoded: &DecodedOptions,
) -> BuiltinResult<()> {
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ISOUTLIER_INTEGER_DATA_EXTENSION,
            ISOUTLIER_NAME,
        )?;
    }
    let explicit_gpu = explicit_isoutlier_gpu(value) || rest.iter().any(explicit_isoutlier_gpu);
    let mut idx = 0;
    while idx < rest.len() {
        if keyword_of(&rest[idx])?.is_some() {
            match decoded.at(idx) {
                Some(OutlierOption::MovingMedian | OutlierOption::MovingMean) => {
                    if decoded.at(idx) == Some(OutlierOption::MovingMedian) && explicit_gpu {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_GPU_MOVMEDIAN_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    if rest.get(idx + 1).is_some_and(is_typed_integer_value) {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_INTEGER_WINDOW_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    idx += 2;
                    continue;
                }
                Some(OutlierOption::SamplePoints) => {
                    if explicit_gpu {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_GPU_SAMPLE_POINTS_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    idx += 2;
                    continue;
                }
                Some(OutlierOption::DataVariables) => {
                    if explicit_gpu {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_GPU_DATA_VARIABLES_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    idx += 2;
                    continue;
                }
                Some(OutlierOption::Percentiles | OutlierOption::ThresholdFactor) => {
                    if rest.get(idx + 1).is_some_and(is_typed_integer_value) {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &ISOUTLIER_INTEGER_THRESHOLD_EXTENSION,
                            ISOUTLIER_NAME,
                        )?;
                    }
                    idx += 2;
                    continue;
                }
                _ => {}
            }
        } else if is_typed_integer_value(&rest[idx]) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ISOUTLIER_INTEGER_DIMENSION_EXTENSION,
                ISOUTLIER_NAME,
            )?;
        }
        idx += 1;
    }
    Ok(())
}

async fn ensure_isoutlier_exact_boundaries(
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    decoded: &DecodedOptions,
) -> BuiltinResult<()> {
    ensure_isoutlier_exact_f64(value, "input data").await?;
    let mut idx = 0;
    while idx + 1 < rest.len() {
        if decoded.at(idx).is_some_and(|option| {
            matches!(
                option,
                OutlierOption::Percentiles | OutlierOption::ThresholdFactor
            )
        }) {
            ensure_isoutlier_exact_f64(&rest[idx + 1], "threshold value").await?;
        }
        idx += 1;
    }
    Ok(())
}

async fn ensure_isoutlier_exact_f64(value: &Value, role: &str) -> BuiltinResult<()> {
    if is_typed_integer_value(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        return Err(invalid_argument(
            ISOUTLIER_NAME,
            format!("typed-integer {role} must be exactly representable as double"),
        ));
    }
    Ok(())
}

#[runtime_builtin(
    name = "filloutliers",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::outliers"
)]
pub(crate) async fn filloutliers_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let decoded = DecodedOptions::new(OutlierOperation::Fill, rest.view())?;
    ensure_filloutliers_extensions(&value, rest.view(), &decoded)?;
    preflight_host_options(OutlierOperation::Fill, rest.view(), &decoded).await?;
    let tensor = value_to_tensor(FILLOUTLIERS_NAME, value).await?;
    let rest = gather_values(rest, runtime.as_ref()).await?;
    if rest.is_empty() {
        return Err(invalid_argument(
            FILLOUTLIERS_NAME,
            "filloutliers: fill method is required",
        ));
    }
    let fill_method = parse_fill_method(rest.first().expect("fill method checked above"))?;
    let options = options::parse_gathered_detection_options(
        OutlierOperation::Fill,
        rest.view().range(1..)?,
        decoded.after_fill(),
    )?;
    let mut result = detect_outliers(&tensor, &options, FILLOUTLIERS_NAME)?;
    let (filled, filled_mask) = fill_outlier_tensor(&tensor, &result, &options, &fill_method)?;
    result.mask = filled_mask;
    outlier_outputs(FILLOUTLIERS_NAME, Some(filled), result)
}

fn ensure_filloutliers_extensions(
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
    decoded: &DecodedOptions,
) -> BuiltinResult<()> {
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLOUTLIERS_INTEGER_DATA_EXTENSION,
            FILLOUTLIERS_NAME,
        )?;
    }
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLOUTLIERS_RESIDENT_INPUT_EXTENSION,
            FILLOUTLIERS_NAME,
        )?;
    }
    if rest.first().is_some_and(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLOUTLIERS_INTEGER_FILL_EXTENSION,
            FILLOUTLIERS_NAME,
        )?;
    }
    let mut idx = 0;
    while idx + 1 < rest.len() {
        if decoded.at(idx) == Some(OutlierOption::OutlierLocations)
            && is_numeric_mask_value(&rest[idx + 1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FILLOUTLIERS_NUMERIC_MASK_EXTENSION,
                FILLOUTLIERS_NAME,
            )?;
        }
        idx += 1;
    }
    if rest
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLOUTLIERS_RESIDENT_INPUT_EXTENSION,
            FILLOUTLIERS_NAME,
        )?;
    }
    Ok(())
}

fn is_numeric_mask_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) | Value::GpuTensor(_)
    )
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

async fn gather_values(
    values: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    values
        .try_map_preserving_roots_async(values.len(), runtime, |index| {
            gather_if_needed_async(&values[index])
        })
        .await
}

async fn value_to_tensor(builtin: &'static str, value: Value) -> BuiltinResult<Tensor> {
    let value = gather_if_needed_async(&value).await.map_err(|err| {
        invalid_argument(builtin, format!("{builtin}: failed to gather input: {err}"))
    })?;
    tensor::value_into_tensor_for(builtin, value)
        .map_err(|err| invalid_argument(builtin, format!("{builtin}: {err}")))
}

fn outlier_outputs(
    builtin: &'static str,
    filled: Option<Value>,
    result: DetectionResult,
) -> BuiltinResult<Value> {
    let max_outputs = if filled.is_some() { 5 } else { 4 };
    if matches!(crate::output_count::current_output_count(), Some(n) if n > max_outputs) {
        return Err(too_many_outputs(builtin, max_outputs));
    }
    let mask_shape = result_shape_for_mask(&filled, &result)?;
    let mask_value = Value::LogicalArray(
        LogicalArray::new(result.mask, mask_shape)
            .map_err(|err| internal_error(builtin, format!("{builtin}: {err}")))?,
    );
    let lower = tensor_value(result.lower, result.threshold_shape.clone(), builtin)?;
    let upper = tensor_value(result.upper, result.threshold_shape.clone(), builtin)?;
    let center = tensor_value(result.center, result.threshold_shape, builtin)?;
    let outputs = if let Some(filled) = filled {
        vec![filled, mask_value, lower, upper, center]
    } else {
        vec![mask_value, lower, upper, center]
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        )),
        None => Ok(outputs.into_iter().next().unwrap_or(Value::Num(0.0))),
    }
}

fn result_shape_for_mask(
    filled: &Option<Value>,
    result: &DetectionResult,
) -> BuiltinResult<Vec<usize>> {
    if let Some(Value::Tensor(tensor)) = filled {
        Ok(tensor.shape().to_vec())
    } else {
        Ok(result.mask_shape.clone())
    }
}

fn tensor_value(data: Vec<f64>, shape: Vec<usize>, builtin: &'static str) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_error(builtin, format!("{builtin}: {err}")))
}

fn first_non_singleton(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|dim| *dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests;
