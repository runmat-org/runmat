use super::detection::{any_missing, ismissing_value, missing_string_array, parse_size_args};
use super::filling::{fill_missing_value, FillOptions};
use super::moving::{moving_mad, MovingOptions};
use super::numeric::{is_numeric_data_like, pairwise_nan_min, scalar_text, scalar_usize};
use super::removal::{remove_missing_value, RemoveOptions};
use super::standardize::{indicator_set, numeric_tensor, standardize_missing_value};
use super::*;

#[runtime_builtin(
    name = "missing",
    category = "missing",
    summary = "Create MATLAB missing string scalars or arrays.",
    keywords = "missing,string,missing values",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::MISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::MISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::MISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn missing_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MISSING_SHAPED_ARRAY_EXTENSION,
            "missing",
        )?;
    }
    let packed = Value::OutputList(args);
    let gathered = gather_if_needed_async(&packed)
        .await
        .map_err(|err| invalid_argument(format!("missing: failed to gather arguments: {err}")))?;
    let args = match gathered {
        Value::OutputList(values) => values,
        _ => Vec::new(),
    };
    let shape = parse_size_args(&args)?;
    missing_string_array(shape)
}

#[runtime_builtin(
    name = "ismissing",
    category = "missing",
    summary = "Return a logical mask identifying missing values.",
    keywords = "ismissing,missing,NaN,NaT,string,table",
    accel = "cpu",
    type_resolver(logical_type),
    descriptor(crate::builtins::missing::ISMISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::ISMISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::ISMISSING_INTEGER_CAPABILITIES),
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
        .map_err(|err| invalid_argument(format!("ismissing: failed to gather input: {err}")))?;
    ismissing_value(&value)
}

pub(super) fn ismissing_resident_integer(
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    let integer = runmat_accelerate_api::handle_integer_type(handle)
        .expect("resident integer predicate requires integer metadata");
    let storage = runmat_accelerate_api::handle_storage(handle);
    if gpu_helpers::exact_provider_for_handle(handle).is_none()
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
        handle.shape.clone(),
    )))
}

#[runtime_builtin(
    name = "anymissing",
    category = "missing",
    summary = "Return true when an input contains at least one missing value.",
    keywords = "anymissing,missing,NaN,string,table",
    accel = "cpu",
    type_resolver(logical_type),
    descriptor(crate::builtins::missing::ANYMISSING_DESCRIPTOR),
    integer_capabilities(crate::builtins::missing::ANYMISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn anymissing_builtin(value: Value) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("anymissing: failed to gather input: {err}")))?;
    Ok(Value::Bool(any_missing(&value)?))
}

#[runtime_builtin(
    name = "standardizeMissing",
    category = "missing",
    summary = "Replace user-specified missing indicators with canonical missing values.",
    keywords = "standardizeMissing,missing,NaN,string,table",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::STANDARDIZE_MISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::STANDARDIZE_MISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::STANDARDIZE_MISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn standardize_missing_builtin(
    value: Value,
    rest: Vec<Value>,
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
    let value = gather_if_needed_async(&value).await.map_err(|err| {
        invalid_argument(format!("standardizeMissing: failed to gather input: {err}"))
    })?;
    let indicator = gather_if_needed_async(indicator).await.map_err(|err| {
        invalid_argument(format!(
            "standardizeMissing: failed to gather indicators: {err}"
        ))
    })?;
    let indicators = indicator_set(&indicator)?;
    standardize_missing_value(value, &indicators)
}

#[runtime_builtin(
    name = "rmmissing",
    category = "missing",
    summary = "Remove missing elements, rows, or columns.",
    keywords = "rmmissing,missing,NaN,string,table",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::RMMISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::RMMISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::RMMISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn rmmissing_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RMMISSING_INTEGER_DIM_EXTENSION,
            "rmmissing",
        )?;
    }
    let source = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let value = if let Some(handle) = source.as_ref() {
        let owner = gpu_helpers::exact_provider_for_handle(handle)
            .ok_or_else(|| invalid_argument("rmmissing: no provider owns the resident input"))?;
        gpu_helpers::download_value_preserving_residency_async(owner, handle)
            .await
            .map_err(|err| invalid_argument(format!("rmmissing: failed to gather input: {err}")))?
    } else {
        value
    };
    let options = RemoveOptions::parse(&rest)?;
    let (result, removed) = remove_missing_value(value, options)?;
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
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![result])),
        Some(n) => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![result, removed],
        )),
        None => Ok(result),
    }
}

#[runtime_builtin(
    name = "fillmissing",
    category = "missing",
    summary = "Fill missing entries using constant, neighbor, or summary methods.",
    keywords = "fillmissing,missing,NaN,string,table,previous,next,linear,constant",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::FILLMISSING_DESCRIPTOR),
    extensions(crate::builtins::missing::FILLMISSING_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::FILLMISSING_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn fillmissing_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("fillmissing: failed to gather input: {err}")))?;
    let options = FillOptions::parse(&rest)?;
    let (result, mask) = fill_missing_value(value, &options)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![result])),
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
            for child in &cell.data {
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
    category = "missing",
    summary = "Mean that ignores NaN values.",
    keywords = "nanmean,mean,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANMEAN_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANMEAN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanmean_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_extension(&NANMEAN_INTEGER_EXTENSION, "nanmean", &value, &rest)?;
    mean::mean_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nansum",
    category = "missing",
    summary = "Sum that ignores NaN values.",
    keywords = "nansum,sum,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANSUM_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANSUM_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nansum_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_extension(&NANSUM_INTEGER_EXTENSION, "nansum", &value, &rest)?;
    sum::sum_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nanmin",
    category = "missing",
    summary = "Minimum that ignores NaN values.",
    keywords = "nanmin,min,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANMIN_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANMIN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanmin_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_extension(&NANMIN_INTEGER_EXTENSION, "nanmin", &value, &rest)?;
    if let Some(first) = rest.first() {
        if is_numeric_data_like(first) {
            if rest.len() != 1 {
                return Err(invalid_argument(
                    "nanmin: pairwise form accepts exactly two numeric inputs",
                ));
            }
            return pairwise_nan_min(value, first.clone());
        }
    }
    min::min_builtin(value, nanmin_rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nanmedian",
    category = "missing",
    summary = "Median that ignores NaN values.",
    keywords = "nanmedian,median,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANMEDIAN_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANMEDIAN_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanmedian_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_extension(&NANMEDIAN_INTEGER_EXTENSION, "nanmedian", &value, &rest)?;
    median::median_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nanstd",
    category = "missing",
    summary = "Standard deviation that ignores NaN values.",
    keywords = "nanstd,std,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANSTD_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANSTD_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanstd_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_control_extension(
        &NANSTD_INTEGER_CONTROL_EXTENSION,
        "nanstd",
        &value,
        &rest,
    )?;
    std_reduction::std_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "nanvar",
    category = "missing",
    summary = "Variance that ignores NaN values.",
    keywords = "nanvar,var,omitnan,missing",
    accel = "reduction",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::NANVAR_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::NANVAR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn nanvar_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_nan_integer_control_extension(
        &NANVAR_INTEGER_CONTROL_EXTENSION,
        "nanvar",
        &value,
        &rest,
    )?;
    var::var_builtin(value, rest_with_omitnan(rest)).await
}

#[runtime_builtin(
    name = "movmad",
    category = "missing",
    summary = "Moving median absolute deviation over vectors and matrix dimensions.",
    keywords = "movmad,moving,median,absolute,deviation,missing",
    accel = "cpu",
    type_resolver(any_type),
    descriptor(crate::builtins::missing::NAN_AWARE_DESCRIPTOR),
    extensions(crate::builtins::missing::MOVMAD_EXTENSIONS),
    integer_capabilities(crate::builtins::missing::MOVMAD_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::missing"
)]
pub(super) async fn movmad_builtin(
    value: Value,
    window: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let window = scalar_usize(&window, "movmad window")?;
    let provider = match &value {
        Value::GpuTensor(handle) => runmat_accelerate_api::provider_for_handle(handle)
            .or_else(runmat_accelerate_api::provider),
        _ => None,
    };
    if provider.is_some() && window > 31 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &MOVMAD_GPU_LARGE_WINDOW_EXTENSION,
            "movmad",
        )?;
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_argument(format!("movmad: failed to gather input: {err}")))?;
    let tensor = numeric_tensor(value, "movmad")?;
    let options = MovingOptions::parse(&rest)?;
    let result = moving_mad(tensor, window, options)?;
    match (provider, result) {
        (Some(provider), Value::Tensor(tensor)) => {
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|err| internal_error(format!("movmad: failed to upload result: {err}")))?;
            Ok(Value::GpuTensor(handle))
        }
        (_, result) => Ok(result),
    }
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
    rest: &[Value],
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
    rest: &[Value],
) -> BuiltinResult<()> {
    if !is_real_typed_integer_value(value) && rest.iter().any(is_real_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    Ok(())
}

pub(super) fn rest_with_omitnan(mut rest: Vec<Value>) -> Vec<Value> {
    let insert_at = rest
        .iter()
        .position(|arg| scalar_text(arg).is_some_and(|text| text.eq_ignore_ascii_case("like")))
        .unwrap_or(rest.len());
    rest.insert(insert_at, Value::from("omitnan"));
    rest
}

pub(super) fn nanmin_rest_with_omitnan(mut rest: Vec<Value>) -> Vec<Value> {
    if rest.is_empty() {
        rest.push(Value::Tensor(
            Tensor::new(Vec::<f64>::new(), vec![0, 0]).expect("empty placeholder shape"),
        ));
    }
    rest.push(Value::from("omitnan"));
    rest
}
