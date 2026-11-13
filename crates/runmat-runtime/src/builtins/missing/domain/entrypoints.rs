use super::detection::{any_missing, ismissing_value, missing_string_array, parse_size_args};
use super::filling::{fill_missing_value, FillOptions};
use super::moving::{moving_mad, MovingOptions};
use super::numeric::{
    is_empty_numeric_placeholder, is_numeric_data_like, pairwise_nan_min, scalar_usize,
};
use super::removal::{remove_missing_value, RemoveOptions};
use super::standardize::{indicator_set, numeric_tensor, standardize_missing_value};
use super::*;
use crate::context::RuntimeContext;
use crate::execution::RootedValue;

#[runtime_builtin(
    name = "missing",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn missing_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if !args.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MISSING_SHAPED_ARRAY_EXTENSION,
            "missing",
        )?;
    }
    let mut gathered = Vec::with_capacity(args.len());
    for argument in &args {
        gathered.push(
            gather_if_needed_async(argument)
                .await
                .map_err(|err| gather_error("missing", "arguments", err))?,
        );
    }
    let shape = parse_size_args(&gathered)?;
    missing_string_array(shape)
}

#[runtime_builtin(
    name = "ismissing",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn ismissing_builtin(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = &value {
        if runmat_accelerate_api::handle_is_explicit(handle) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ISMISSING_RESIDENT_INPUT_EXTENSION,
                "ismissing",
            )?;
        }
        if runmat_accelerate_api::handle_integer_type(handle).is_some() {
            return ismissing_resident_integer(handle);
        }
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| gather_error("ismissing", "input", err))?;
    ismissing_value(&value)
}

pub(super) fn ismissing_resident_integer(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    let integer = runmat_accelerate_api::handle_integer_type(handle)
        .expect("resident integer predicate requires integer metadata");
    let storage = runmat_accelerate_api::handle_storage(handle);
    if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err()
        || storage != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_precision(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
        || !gpu_helpers::gpu_class_metadata_matches(handle, None, Some(integer), false)
    {
        return Err(internal_error(
            "ismissing: resident integer metadata is contradictory",
        ));
    }
    Ok(Value::LogicalArray(LogicalArray::zeros(
        handle.shape().to_vec(),
    )))
}

#[runtime_builtin(
    name = "anymissing",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn anymissing_builtin(value: Value) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| gather_error("anymissing", "input", err))?;
    Ok(Value::Bool(any_missing(&value)?))
}

#[runtime_builtin(
    name = "standardizeMissing",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing",
    context = true
)]
pub(super) async fn standardize_missing_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STANDARDIZE_MISSING_INTEGER_DATA_EXTENSION,
            "standardizeMissing",
        )?;
    }
    let indicator = rest
        .first()
        .ok_or_else(|| invalid_argument("standardizeMissing: missing indicators argument"))?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(indicator) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STANDARDIZE_MISSING_EXPLICIT_GPU_INDICATOR_EXTENSION,
            "standardizeMissing",
        )?;
    }
    let indicator = gather_if_needed_async(indicator)
        .await
        .map_err(|err| gather_error("standardizeMissing", "indicators", err))?;
    let indicators = indicator_set(&indicator)?;
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| gather_error("standardizeMissing", "input", err))?;
    standardize_missing_value(runtime.as_ref(), value, &indicators)
}

#[runtime_builtin(
    name = "rmmissing",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing",
    context = true
)]
pub(super) async fn rmmissing_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_output_arity("rmmissing", 2)?;
    if rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RMMISSING_INTEGER_DIM_EXTENSION,
            "rmmissing",
        )?;
    }
    let options = RemoveOptions::parse(rest.view())?;
    let source = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let value = if let Some(handle) = source.as_ref() {
        let owner = gpu_helpers::require_resident_compute_owner(handle, || {
            invalid_argument("rmmissing: no provider owns the resident input")
        })?;
        gpu_helpers::download_value_preserving_residency_async(owner, handle)
            .await
            .map_err(|err| gather_error("rmmissing", "input", err))?
    } else {
        value
    };
    let (result, removed) = remove_missing_value(runtime.as_ref(), value, options)?;
    let (result, removed) = if let Some(source) = source.as_ref() {
        (
            gpu_helpers::restore_class_preserving_value(source, result, "rmmissing")?,
            gpu_helpers::restore_class_preserving_value(
                source,
                Value::LogicalArray(removed),
                "rmmissing",
            )?,
        )
    } else {
        (result, Value::LogicalArray(removed))
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![result])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![result, removed],
        )),
        None => Ok(result),
    }
}

#[runtime_builtin(
    name = "fillmissing",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing",
    context = true
)]
pub(super) async fn fillmissing_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_output_arity("fillmissing", 2)?;
    if is_real_typed_integer_value(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLMISSING_INTEGER_DATA_EXTENSION,
            "fillmissing",
        )?;
    } else if fillmissing_aggregate_contains_integer(&value)? {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILLMISSING_AGGREGATE_INTEGER_DATA_EXTENSION,
            "fillmissing",
        )?;
    }
    let options = FillOptions::parse(rest.view())?;
    options.preflight(&value)?;
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| gather_error("fillmissing", "input", err))?;
    let (result, mask) = fill_missing_value(runtime.as_ref(), value, &options)?;
    if matches!(result, Value::StringArray(_)) {
        return super::text_output::fill_outputs(runtime.as_ref(), result, mask);
    }
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![result])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![result, Value::LogicalArray(mask)],
        )),
        None => Ok(result),
    }
}

pub(super) fn fillmissing_aggregate_contains_integer(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Cell(cell) => {
            for child in cell.data() {
                if is_real_typed_integer_value(child)
                    || fillmissing_aggregate_contains_integer(child)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Value::Object(object) if is_tabular_object(object) => {
            let variables = table_variables(object)?;
            for child in variables.fields.values() {
                if is_real_typed_integer_value(child)
                    || fillmissing_aggregate_contains_integer(child)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

#[runtime_builtin(
    name = "nanmean",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanmean_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_output_arity("nanmean", 1)?;
    ensure_nan_integer_extension(&NANMEAN_INTEGER_EXTENSION, "nanmean", &value, rest.view())?;
    mean::mean_with_nan_policy(value, rest, Some(runmat_builtins::ReductionNaN::Omit)).await
}

#[runtime_builtin(
    name = "nansum",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing",
    context = true,
    call_observation = true
)]
pub(super) async fn nansum_builtin(
    runtime: Option<RuntimeContext>,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    ensure_output_arity("nansum", 1)?;
    ensure_nan_integer_extension(&NANSUM_INTEGER_EXTENSION, "nansum", &value, rest.view())?;
    sum::sum_for_invocation_observed(
        runtime.as_ref(),
        value,
        rest,
        None,
        runmat_types::SumInvocation::NanSum,
        observation,
    )
    .await
}

#[runtime_builtin(
    name = "nanmin",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanmin_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_output_arity("nanmin", 2)?;
    ensure_nan_integer_extension(&NANMIN_INTEGER_EXTENSION, "nanmin", &value, rest.view())?;
    if let Some(first) = rest.first() {
        if is_numeric_data_like(first) && !is_empty_numeric_placeholder(first) {
            ensure_output_arity("nanmin pairwise", 1)?;
            if rest.len() != 1 {
                return Err(invalid_argument(
                    "nanmin: pairwise form accepts exactly two numeric inputs",
                ));
            }
            return pairwise_nan_min(value, first.clone());
        }
    }
    min::min_reduction_with_nan_policy(value, rest, runmat_builtins::ReductionNaN::Omit).await
}

#[runtime_builtin(
    name = "nanmedian",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanmedian_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_output_arity("nanmedian", 1)?;
    ensure_nan_integer_extension(
        &NANMEDIAN_INTEGER_EXTENSION,
        "nanmedian",
        &value,
        rest.view(),
    )?;
    median::median_with_nan_policy(value, rest, Some(runmat_builtins::ReductionNaN::Omit)).await
}

#[runtime_builtin(
    name = "nanstd",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanstd_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_output_arity("nanstd", 1)?;
    ensure_nan_integer_control_extension(&NANSTD_INTEGER_EXTENSION, "nanstd", &value, rest.view())?;
    std_reduction::std_with_nan_policy(value, rest, Some(runmat_builtins::ReductionNaN::Omit)).await
}

#[runtime_builtin(
    name = "nanvar",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanvar_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_output_arity("nanvar", 1)?;
    ensure_nan_integer_control_extension(&NANVAR_INTEGER_EXTENSION, "nanvar", &value, rest.view())?;
    var::var_with_nan_policy(value, rest, Some(runmat_builtins::ReductionNaN::Omit)).await
}

#[runtime_builtin(
    name = "movmad",
    binding_variant = "default",
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn movmad_builtin(
    value: Value,
    window: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_output_arity("movmad", 1)?;
    let window = scalar_usize(&window, "movmad window")?;
    if window == 0 {
        return Err(invalid_argument("movmad: window length must be positive"));
    }
    let options = MovingOptions::parse(rest.view())?;
    let source = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    if source.is_some() && window > 31 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MOVMAD_GPU_LARGE_WINDOW_EXTENSION,
            "movmad",
        )?;
    }
    if let Some(handle) = source.as_ref() {
        gpu_helpers::require_resident_compute_owner(handle, || {
            invalid_argument("movmad: no provider owns the resident input")
        })?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| gather_error("movmad", "input", err))?;
    let tensor = numeric_tensor(value, "movmad")?;
    let result = moving_mad(tensor, window, options)?;
    match source {
        Some(source) => gpu_helpers::restore_class_preserving_value(&source, result, "movmad"),
        None => Ok(result),
    }
}

fn ensure_output_arity(builtin: &str, maximum: usize) -> BuiltinResult<()> {
    if crate::output_count::current_output_count().is_some_and(|count| count > maximum) {
        return Err(invalid_argument(format!(
            "{builtin}: at most {maximum} output{} supported",
            if maximum == 1 { " is" } else { "s are" }
        )));
    }
    Ok(())
}

pub(super) fn is_real_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

pub(super) fn ensure_nan_integer_extension(
    extension: &BuiltinExtensionDescriptor,
    builtin: &str,
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if is_real_typed_integer_value(value) || rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
}

pub(super) fn ensure_nan_integer_control_extension(
    extension: &BuiltinExtensionDescriptor,
    builtin: &str,
    value: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if !is_real_typed_integer_value(value) && rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
}
