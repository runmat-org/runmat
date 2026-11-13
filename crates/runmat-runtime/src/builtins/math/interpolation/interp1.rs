//! MATLAB-compatible `interp1` builtin for dense real numeric data.

use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostTensorView, ProviderInterp1Extrapolation,
    ProviderInterp1Method, ProviderInterp1Request,
};
#[cfg(test)]
use runmat_builtins::INTERP1_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, INTERP1_ERROR_INTERNAL, INTERP1_ERROR_INVALID_ARGUMENT,
    INTERP1_ERROR_INVALID_INPUT, INTERP1_INTEGER_EXTRAPOLATION_EXTENSION,
    INTERP1_INTEGER_QUERY_EXTENSION, INTERP1_INTEGER_SAMPLE_EXTENSION,
    INTERP1_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{
    build_makima_pp, build_pchip_pp, build_spline_pp, evaluate_linear_or_nearest, evaluate_pp,
    implicit_series_from_values, interp1_series_from_plan, is_spline_execution_failure,
    is_vector_shape, parse_extrapolation, parse_method, prepare_interp1_samples, query_points,
    Extrapolation, Interp1SamplePlan, InterpMethod, PpOutputLayout,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};

const NAME: &str = "interp1";

fn interp1_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn interp1_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    interp1_error_with_message(
        format!(
            "{}: {}",
            INTERP1_ERROR_INVALID_ARGUMENT.message,
            detail.as_ref()
        ),
        &INTERP1_ERROR_INVALID_ARGUMENT,
    )
}

fn interp1_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        interp1_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::interp1")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("interpolation-1d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("interp1")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Linear and nearest interpolation keep resident real Y/query inputs on the provider when X is implicit or host-validated; spline/pchip and resident-X validation fall back to host.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::interp1"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Interpolation is currently a runtime sink.",
};

#[runtime_builtin(
    name = "interp1",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::interpolation::interp1",
    context = true
)]
async fn interp1_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let explicit = args.len() >= 3 && !third_arg_is_option(args.view())?;
    if args.len() >= 2 {
        let data = [
            &args[0],
            &args[1],
            if explicit { &args[2] } else { &args[1] },
        ];
        gpu_helpers::require_resident_compute_values(&data, false, || {
            interp1_error_with_message(
                "interp1: resident input has no exact provider owner",
                &INTERP1_ERROR_INTERNAL,
            )
        })?;
    }
    let sample_plan = if explicit {
        preflight_interp1_explicit_x(&args[0]).await?;
        Some(
            prepare_interp1_samples(&args[0], NAME)
                .await
                .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INVALID_INPUT))?,
        )
    } else {
        None
    };
    preflight_interp1_inputs(args.view(), explicit).await?;
    if let Some(output) = try_interp1_gpu(args.view(), sample_plan.as_ref()).await? {
        return Ok(output);
    }
    let parsed = ParsedInterp1::parse(args, sample_plan)
        .await
        .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INVALID_INPUT))?;
    match parsed.method {
        InterpMethod::Linear | InterpMethod::Nearest => evaluate_linear_or_nearest(
            &parsed.series,
            &parsed.query,
            parsed.method,
            &parsed.extrap,
            NAME,
        )
        .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL)),
        InterpMethod::Spline => {
            let mut work = runtime
                .as_ref()
                .map(crate::context::RuntimeContext::cpu_cooperation);
            let pp = build_spline_pp(&parsed.series, NAME, &mut work)
                .await
                .map_err(|err| {
                    if is_spline_execution_failure(&err) {
                        crate::builtins::common::map_control_flow_with_builtin(err, NAME)
                    } else {
                        interp1_map_error(err, &INTERP1_ERROR_INTERNAL)
                    }
                })?;
            evaluate_pp(
                &pp,
                &parsed.query,
                &parsed.extrap_for_cubic(),
                PpOutputLayout::QueryLeading,
                NAME,
            )
            .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL))
        }
        InterpMethod::Pchip | InterpMethod::Cubic => {
            let pp = build_pchip_pp(&parsed.series, NAME)
                .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL))?;
            evaluate_pp(
                &pp,
                &parsed.query,
                &parsed.extrap_for_cubic(),
                PpOutputLayout::QueryLeading,
                NAME,
            )
            .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL))
        }
        InterpMethod::Makima => {
            let pp = build_makima_pp(&parsed.series, NAME)
                .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL))?;
            evaluate_pp(
                &pp,
                &parsed.query,
                &parsed.extrap_for_cubic(),
                PpOutputLayout::QueryLeading,
                NAME,
            )
            .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL))
        }
    }
}

#[cfg(test)]
async fn interp1_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&args, None)?;
    interp1_builtin_registered(None, args).await
}

async fn preflight_interp1_explicit_x(value: &Value) -> crate::BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64_async, value_has_logical_class,
        value_has_native_integer_class,
    };
    if value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP1_INTEGER_SAMPLE_EXTENSION,
            NAME,
        )?;
        if !native_integer_value_is_exact_f64_async(value).await? {
            return Err(interp1_error_with_message(
                "interp1: integer samples must be exactly representable as double",
                &INTERP1_ERROR_INVALID_INPUT,
            ));
        }
    }
    if value_has_logical_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP1_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

async fn preflight_interp1_inputs(
    args: crate::call::FunctionArgumentView<'_>,
    explicit_x_admitted: bool,
) -> crate::BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64_async, value_has_logical_class,
        value_has_native_integer_class,
    };
    if args.len() < 2 {
        return Ok(());
    }
    let explicit = args.len() >= 3 && !third_arg_is_option(args)?;
    let (samples, query) = if explicit {
        (args.range(..2).expect("interp1 samples"), &args[2])
    } else {
        (args.range(..1).expect("interp1 samples"), &args[1])
    };
    for value in samples.iter().skip(usize::from(explicit_x_admitted)) {
        if value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP1_INTEGER_SAMPLE_EXTENSION,
                NAME,
            )?;
            if !native_integer_value_is_exact_f64_async(value).await? {
                return Err(interp1_error_with_message(
                    "interp1: integer samples must be exactly representable as double",
                    &INTERP1_ERROR_INVALID_INPUT,
                ));
            }
        }
    }
    if value_has_native_integer_class(query) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP1_INTEGER_QUERY_EXTENSION,
            NAME,
        )?;
        if !native_integer_value_is_exact_f64_async(query).await? {
            return Err(interp1_error_with_message(
                "interp1: integer query coordinates must be exactly representable as double",
                &INTERP1_ERROR_INVALID_INPUT,
            ));
        }
    }
    if samples
        .iter()
        .chain(std::iter::once(query))
        .any(value_has_logical_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP1_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    let option_start = if explicit { 3 } else { 2 };
    for option in args.iter().skip(option_start) {
        if value_has_native_integer_class(option) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP1_INTEGER_EXTRAPOLATION_EXTENSION,
                NAME,
            )?;
            if !native_integer_value_is_exact_f64_async(option).await? {
                return Err(interp1_error_with_message(
                    "interp1: integer extrapolation value must be exactly representable as double",
                    &INTERP1_ERROR_INVALID_INPUT,
                ));
            }
        }
        if value_has_logical_class(option) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP1_LOGICAL_INPUT_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

struct GpuInterp1Series<'a> {
    x: std::borrow::Cow<'a, [f64]>,
    y: &'a GpuTensorHandle,
    sample_len: usize,
    series_count: usize,
    trailing_shape: Vec<usize>,
}

struct GpuInterp1Query {
    handle: GpuTensorHandle,
    shape: Vec<usize>,
    len: usize,
    owned: bool,
}

#[cfg(test)]
#[path = "interp1/owning_tests.rs"]
mod owning_tests;

async fn try_interp1_gpu(
    args: crate::call::FunctionArgumentView<'_>,
    explicit_plan: Option<&Interp1SamplePlan>,
) -> crate::BuiltinResult<Option<Value>> {
    if args.len() < 2 {
        return Ok(None);
    }

    let mut method = InterpMethod::Linear;
    let mut extrap = Extrapolation::Nan;
    let (series, query_value, options) = if args.len() == 2 || third_arg_is_option(args)? {
        let Some(series) = implicit_gpu_series(&args[0])? else {
            return Ok(None);
        };
        (
            series,
            &args[1],
            args.range(2..).expect("interp1 option suffix"),
        )
    } else {
        if matches!(&args[0], Value::GpuTensor(_)) {
            return Ok(None);
        }
        let Some(series) = explicit_gpu_series(explicit_plan.expect("explicit X plan"), &args[1])?
        else {
            return Ok(None);
        };
        (
            series,
            &args[2],
            args.range(3..).expect("interp1 option suffix"),
        )
    };

    for option in options {
        if let Some(parsed) = parse_extrapolation(option, NAME).await? {
            extrap = parsed;
            continue;
        }
        if let Some(parsed) = parse_method(option, NAME)? {
            method = parsed;
            continue;
        }
        return Ok(None);
    }
    if !matches!(method, InterpMethod::Linear | InterpMethod::Nearest) {
        return Ok(None);
    }

    let provider = gpu_helpers::require_resident_compute_owner(series.y, || {
        interp1_error_with_message(
            "interp1: no acceleration provider owns the resident Y input",
            &INTERP1_ERROR_INTERNAL,
        )
    })?;
    let series_shape = series.y.shape().to_vec();
    let _series_residency = gpu_helpers::ResidencyRestoreGuard::new(series.y);
    let resident_query = match query_value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    };
    let _query_residency = resident_query.map(gpu_helpers::ResidencyRestoreGuard::new);
    let series_precision = runmat_accelerate_api::handle_precision(series.y);
    if series_precision != Some(provider.precision())
        || !valid_interp1_gpu_input(series.y, &series_shape, series_precision, provider, &[])
    {
        return Err(interp1_error_with_message(
            "interp1: resident Y has contradictory floating metadata",
            &INTERP1_ERROR_INTERNAL,
        ));
    }
    let x_shape = [1, series.sample_len];
    let x_upload = provider.upload(&HostTensorView {
        data: &series.x,
        shape: &x_shape,
    });
    let x_handle = match x_upload {
        Ok(handle) => handle,
        Err(error) => {
            let mut failure = interp1_error_with_message(
                format!("interp1: X upload failed: {error}"),
                &INTERP1_ERROR_INTERNAL,
            );
            failure = crate::runtime_error::with_provider_execution_source(failure, error.into());
            return Err(failure);
        }
    };
    let mut x_protected = vec![series.y];
    if let Some(handle) = resident_query {
        x_protected.push(handle);
    }
    if !valid_interp1_gpu_input(series.y, &series_shape, series_precision, provider, &[])
        || !valid_interp1_gpu_input(
            &x_handle,
            &x_shape,
            series_precision,
            provider,
            &x_protected,
        )
    {
        return Err(gpu_helpers::reject_gpu_output(
            x_handle,
            interp1_error_with_message(
                "interp1: provider returned an invalid X upload",
                &INTERP1_ERROR_INTERNAL,
            ),
        ));
    }

    let query_result = gpu_query_points(query_value, provider).await;
    let query = match query_result {
        Ok(Some(query)) => query,
        Ok(None) => return Ok(None),
        Err(err) => return Err(err),
    };
    let owned_query_protected = [series.y, &x_handle];
    let borrowed_query_protected = [&x_handle];
    let query_protected: &[&GpuTensorHandle] = if query.owned {
        &owned_query_protected
    } else {
        &borrowed_query_protected
    };
    if !valid_interp1_gpu_input(series.y, &series_shape, series_precision, provider, &[])
        || !valid_interp1_gpu_input(&x_handle, &x_shape, series_precision, provider, &[series.y])
        || !valid_interp1_gpu_input(
            &query.handle,
            &query.shape,
            series_precision,
            provider,
            query_protected,
        )
    {
        return Err(gpu_helpers::reject_gpu_output(
            query.handle,
            interp1_error_with_message(
                "interp1: invalid resident Xq input",
                &INTERP1_ERROR_INTERNAL,
            ),
        ));
    }

    let output_shape =
        interp1_gpu_output_shape(&query.shape, series.series_count, &series.trailing_shape);
    let request = ProviderInterp1Request {
        x: &x_handle,
        y: series.y,
        xq: &query.handle,
        sample_len: series.sample_len,
        series_count: series.series_count,
        query_len: query.len,
        output_shape: &output_shape,
        method: provider_interp1_method(method),
        extrapolation: provider_interp1_extrapolation(&extrap),
        extrapolation_value: match extrap {
            Extrapolation::Value(value) => value,
            _ => f64::NAN,
        },
    };
    let result = provider.interp1(&request).await;
    let valid = result.as_ref().is_ok_and(|output| {
        valid_interp1_gpu_output(
            output,
            &output_shape,
            series.y,
            &x_handle,
            &query.handle,
            provider,
            series_precision,
        )
    });
    match result {
        Ok(output) if valid => {
            let output = gpu_helpers::propagate_owned_output_provenance(output, x_protected)?;
            gpu_helpers::resident_gpu_value(output).map(Some)
        }
        Ok(output) => Err(gpu_helpers::reject_gpu_output(
            output,
            interp1_error_with_message(
                "interp1: provider returned an invalid interpolation result",
                &INTERP1_ERROR_INTERNAL,
            ),
        )),
        Err(error)
            if gpu_helpers::provider_hook_allows_host_fallback(&error)
                || error.to_string() == "interp1 not supported by provider" =>
        {
            Ok(None)
        }
        Err(error) => {
            let mut failure =
                build_runtime_error(format!("interp1: provider execution failed: {error}"))
                    .with_builtin(NAME)
                    .with_identifier(
                        INTERP1_ERROR_INTERNAL
                            .identifier
                            .expect("interp1 internal descriptor identifier"),
                    )
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build();
            failure = crate::runtime_error::with_provider_execution_source(failure, error.into());
            Err(failure)
        }
    }
}

fn valid_interp1_gpu_output(
    output: &GpuTensorHandle,
    output_shape: &[usize],
    y: &GpuTensorHandle,
    x: &GpuTensorHandle,
    xq: &GpuTensorHandle,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
) -> bool {
    output.shape() == output_shape
        && output.device_id() == y.device_id()
        && ![y, x, xq]
            .iter()
            .any(|input| input.shares_allocation_with(output))
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output) == expected_precision
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::gpu_class_metadata_matches(output, expected_precision, None, false)
}

fn valid_interp1_gpu_input(
    handle: &GpuTensorHandle,
    expected_shape: &[usize],
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    protected: &[&GpuTensorHandle],
) -> bool {
    handle.shape() == expected_shape
        && handle.device_id() == provider.device_id()
        && !protected
            .iter()
            .any(|input| input.shares_allocation_with(handle))
        && gpu_helpers::exact_provider_for_handle(handle)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(handle) == expected_precision
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && gpu_helpers::gpu_class_metadata_matches(handle, expected_precision, None, false)
}

fn implicit_gpu_series(value: &Value) -> crate::BuiltinResult<Option<GpuInterp1Series<'_>>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Ok(None);
    }
    let Some((sample_len, series_count, trailing_shape)) = gpu_y_layout(handle, None)? else {
        return Ok(None);
    };
    let x = (1..=sample_len)
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    Ok(Some(GpuInterp1Series {
        x: x.into(),
        y: handle,
        sample_len,
        series_count,
        trailing_shape,
    }))
}

fn explicit_gpu_series<'a>(
    plan: &'a Interp1SamplePlan,
    y_value: &'a Value,
) -> crate::BuiltinResult<Option<GpuInterp1Series<'a>>> {
    let Value::GpuTensor(handle) = y_value else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Ok(None);
    }
    // The provider hook indexes Y in its original order. A distinct but
    // nonascending X must take the host path, which permutes X and Y together.
    if !plan.is_ascending() {
        return Ok(None);
    }
    let x = std::borrow::Cow::Borrowed(plan.x());
    let Some((sample_len, series_count, trailing_shape)) = gpu_y_layout(handle, Some(x.len()))?
    else {
        return Ok(None);
    };
    Ok(Some(GpuInterp1Series {
        x,
        y: handle,
        sample_len,
        series_count,
        trailing_shape,
    }))
}

fn gpu_y_layout(
    handle: &GpuTensorHandle,
    explicit_sample_len: Option<usize>,
) -> crate::BuiltinResult<Option<(usize, usize, Vec<usize>)>> {
    let len = handle
        .shape()
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| {
            interp1_error_with_message("interp1: Y size overflow", &INTERP1_ERROR_INVALID_INPUT)
        })?;
    let shape = tensor::default_shape_for(handle.shape(), len);
    let sample_len = explicit_sample_len
        .unwrap_or_else(|| shape.iter().copied().find(|dim| *dim > 1).unwrap_or(len));
    if sample_len < 2 {
        return Ok(None);
    }
    if len != sample_len && !len.is_multiple_of(sample_len) {
        return Ok(None);
    }
    let (series_count, trailing_shape) = if len == sample_len && is_vector_shape(&shape) {
        (1, Vec::new())
    } else if shape.first().copied() == Some(sample_len) {
        let series = len / sample_len;
        let trailing = if shape.len() > 1 {
            shape[1..].to_vec()
        } else {
            vec![series]
        };
        (series, trailing)
    } else if len == sample_len {
        (1, Vec::new())
    } else {
        return Ok(None);
    };
    Ok(Some((sample_len, series_count, trailing_shape)))
}

async fn gpu_query_points(
    value: &Value,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> crate::BuiltinResult<Option<GpuInterp1Query>> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
                || runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
            {
                return Ok(None);
            }
            let len = handle
                .shape()
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
                .ok_or_else(|| {
                    interp1_error_with_message(
                        "interp1: Xq size overflow",
                        &INTERP1_ERROR_INVALID_INPUT,
                    )
                })?;
            if !gpu_helpers::exact_provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            {
                return Err(interp1_error_with_message(
                    "interp1: resident Xq is not owned by the Y provider",
                    &INTERP1_ERROR_INTERNAL,
                ));
            }
            Ok(Some(GpuInterp1Query {
                handle: handle.clone(),
                shape: tensor::default_shape_for(handle.shape(), len),
                len,
                owned: false,
            }))
        }
        other => {
            let query = query_points(other.clone(), NAME).await?;
            let uploaded = provider
                .upload(&HostTensorView {
                    data: &query.values,
                    shape: &query.shape,
                })
                .map_err(|err| {
                    let mut failure = interp1_error_with_message(
                        format!("interp1: Xq upload failed: {err}"),
                        &INTERP1_ERROR_INTERNAL,
                    );
                    failure =
                        crate::runtime_error::with_provider_execution_source(failure, err.into());
                    failure
                })?;
            Ok(Some(GpuInterp1Query {
                handle: uploaded,
                shape: query.shape,
                len: query.values.len(),
                owned: true,
            }))
        }
    }
}

fn interp1_gpu_output_shape(
    query_shape: &[usize],
    series_count: usize,
    trailing_shape: &[usize],
) -> Vec<usize> {
    if series_count == 1 {
        return query_shape.to_vec();
    }
    let mut shape = query_shape.to_vec();
    if trailing_shape.is_empty() {
        shape.push(series_count);
    } else {
        shape.extend(trailing_shape.iter().copied());
    }
    shape
}

fn provider_interp1_method(method: InterpMethod) -> ProviderInterp1Method {
    match method {
        InterpMethod::Linear => ProviderInterp1Method::Linear,
        InterpMethod::Nearest => ProviderInterp1Method::Nearest,
        _ => unreachable!("provider fast path is limited to direct methods"),
    }
}

fn provider_interp1_extrapolation(extrap: &Extrapolation) -> ProviderInterp1Extrapolation {
    match extrap {
        Extrapolation::Nan => ProviderInterp1Extrapolation::Nan,
        Extrapolation::Extrapolate => ProviderInterp1Extrapolation::Extrapolate,
        Extrapolation::Value(_) => ProviderInterp1Extrapolation::Value,
    }
}

struct ParsedInterp1 {
    series: super::pp::NumericSeries,
    query: super::pp::QueryPoints,
    method: InterpMethod,
    extrap: Extrapolation,
}

impl ParsedInterp1 {
    async fn parse(
        mut args: crate::call::BuiltinArguments,
        explicit_plan: Option<Interp1SamplePlan>,
    ) -> crate::BuiltinResult<Self> {
        if args.len() < 2 {
            return Err(interp1_invalid_argument(
                "expected at least Y and Xq arguments",
            ));
        }

        let mut method = InterpMethod::Linear;
        let mut extrap = Extrapolation::Nan;
        let mut explicit_extrapolation = false;
        let (series, query) = if args.len() == 2 || third_arg_is_option(args.view())? {
            let y = args.pop_front().expect("Y argument");
            let xq = args.pop_front().expect("Xq argument");
            let series = implicit_series_from_values(y, NAME).await?;
            let query = query_points(xq, NAME).await?;
            (series, query)
        } else {
            let _x = args.pop_front().expect("X argument");
            let y = args.pop_front().expect("Y argument");
            let xq = args.pop_front().expect("Xq argument");
            let series =
                interp1_series_from_plan(explicit_plan.expect("explicit X plan"), y, NAME).await?;
            let query = query_points(xq, NAME).await?;
            (series, query)
        };

        for option in &args {
            if let Some(parsed) = parse_extrapolation(option, NAME).await? {
                extrap = parsed;
                explicit_extrapolation = true;
                continue;
            }
            if let Some(parsed) = parse_method(option, NAME)? {
                method = parsed;
                continue;
            }
            return Err(interp1_error_with_message(
                "interp1: unsupported interpolation option",
                &INTERP1_ERROR_INVALID_ARGUMENT,
            ));
        }

        if !explicit_extrapolation
            && matches!(
                method,
                InterpMethod::Spline | InterpMethod::Pchip | InterpMethod::Makima
            )
        {
            extrap = Extrapolation::Extrapolate;
        }

        Ok(Self {
            series,
            query,
            method,
            extrap,
        })
    }

    fn extrap_for_cubic(&self) -> Extrapolation {
        self.extrap.clone()
    }
}

fn third_arg_is_option(args: crate::call::FunctionArgumentView<'_>) -> crate::BuiltinResult<bool> {
    Ok(args
        .get(2)
        .map(crate::builtins::common::random_args::keyword_of)
        .transpose()?
        .flatten()
        .is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor};

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    fn int_row(storage: IntegerStorage) -> Value {
        let len = storage.len();
        let tensor = Tensor::new_integer(storage, vec![1, len]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(interp1_builtin(args))
    }

    #[test]
    fn interp1_linear_midpoints() {
        let result = run(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[10.0, 20.0, 40.0]),
            row(&[1.5, 2.5]),
        ])
        .expect("interp1");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.materialize_f64(), vec![15.0, 30.0]);
    }

    #[test]
    fn interp1_pairs_distinct_unsorted_x_with_each_sample_channel() {
        let y = Value::Tensor(
            Tensor::new(vec![30.0, 10.0, 20.0, 300.0, 100.0, 200.0], vec![3, 2])
                .expect("two sample channels"),
        );
        let result = run(vec![
            row(&[3.0, 1.0, 2.0]),
            y,
            row(&[0.0, 1.5, 3.5]),
            Value::String("linear".to_string().into()),
            Value::String("extrap".to_string().into()),
        ])
        .expect("distinct unsorted interp1");
        let Value::Tensor(tensor) = result else {
            panic!("expected query-by-channel output");
        };
        assert_eq!(tensor.shape(), &[1, 3, 2]);
        assert_eq!(
            tensor.materialize_f64(),
            vec![0.0, 15.0, 35.0, 0.0, 150.0, 350.0]
        );

        let nd = run(vec![
            row(&[3.0, 1.0, 2.0]),
            Value::Tensor(
                Tensor::new(vec![30.0, 10.0, 20.0, 300.0, 100.0, 200.0], vec![3, 1, 2])
                    .expect("N-D sample channels"),
            ),
            row(&[1.5, 2.5]),
        ])
        .expect("N-D paired sample ordering");
        let Value::Tensor(tensor) = nd else {
            panic!("expected N-D query-by-channel output");
        };
        assert_eq!(tensor.shape(), &[1, 2, 1, 2]);
        assert_eq!(tensor.materialize_f64(), vec![15.0, 25.0, 150.0, 250.0]);

        let descending = run(vec![
            row(&[3.0, 2.0, 1.0]),
            row(&[30.0, 20.0, 10.0]),
            row(&[0.0, 1.5, 3.5]),
            Value::String("linear".to_string().into()),
            Value::String("extrap".to_string().into()),
        ])
        .expect("descending interp1");
        let Value::Tensor(tensor) = descending else {
            panic!("expected query-shaped descending output");
        };
        assert_eq!(tensor.materialize_f64(), vec![0.0, 15.0, 35.0]);

        for method in ["pchip", "spline"] {
            let unordered = run(vec![
                row(&[3.0, 1.0, 4.0, 2.0]),
                row(&[9.0, 1.0, 16.0, 4.0]),
                Value::Num(2.5),
                Value::String(method.to_string().into()),
            ])
            .expect("higher-order unordered interpolation");
            let ordered = run(vec![
                row(&[1.0, 2.0, 3.0, 4.0]),
                row(&[1.0, 4.0, 9.0, 16.0]),
                Value::Num(2.5),
                Value::String(method.to_string().into()),
            ])
            .expect("higher-order ordered reference");
            assert_eq!(unordered, ordered, "paired ordering for {method}");
        }

        let duplicate = run(vec![row(&[-0.0, 0.0]), row(&[1.0, 2.0]), Value::Num(0.5)])
            .expect_err("signed-zero duplicate points must be rejected");
        assert!(duplicate.message().contains("distinct"));
        let nonfinite = run(vec![
            row(&[1.0, f64::INFINITY]),
            row(&[10.0, 20.0]),
            Value::Num(1.5),
        ])
        .expect_err("nonfinite sample point must be rejected");
        assert!(nonfinite.message().contains("finite"));
    }

    #[test]
    fn interp1_nonascending_host_x_with_resident_y_uses_paired_host_path() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[30.0, 10.0, 20.0],
                    shape: &[1, 3],
                })
                .expect("upload unordered samples");
            let result = run(vec![
                row(&[3.0, 1.0, 2.0]),
                Value::GpuTensor(y.clone()),
                row(&[1.5, 2.5]),
            ])
            .expect("host ordering of resident Y");
            let Value::Tensor(tensor) = result else {
                panic!("nonascending X must not reach the ascending-only provider hook");
            };
            assert_eq!(tensor.materialize_f64(), vec![15.0, 25.0]);
            let duplicate = run(vec![
                row(&[2.0, 1.0, 2.0]),
                Value::GpuTensor(y.clone()),
                Value::Num(1.5),
            ])
            .expect_err("duplicate host X must fail before resident Y materialization");
            assert!(duplicate.message().contains("distinct"));
            y.release().expect("release borrowed Y");

            let x = provider
                .upload(&HostTensorView {
                    data: &[3.0, 1.0, 2.0],
                    shape: &[1, 3],
                })
                .expect("upload unordered X");
            let result = run(vec![
                Value::GpuTensor(x.clone()),
                row(&[30.0, 10.0, 20.0]),
                row(&[1.5, 2.5]),
            ])
            .expect("resident X host validation and ordering");
            let Value::Tensor(tensor) = result else {
                panic!("resident X follows host ordering admission");
            };
            assert_eq!(tensor.materialize_f64(), vec![15.0, 25.0]);
            x.release().expect("release borrowed X");
        });
    }

    #[test]
    fn interp1_nearest() {
        let result = run(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[10.0, 20.0, 40.0]),
            row(&[1.2, 2.8]),
            Value::String("nearest".to_string().into()),
        ])
        .expect("interp1");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.materialize_f64(), vec![10.0, 40.0]);
    }

    #[test]
    fn interp1_makima_uses_modified_slopes_and_preserves_query_shape() {
        let result = run(vec![
            row(&[0.0, 1.0, 2.0, 3.0, 4.0]),
            row(&[0.0, 0.0, 0.0, 1.0, 0.0]),
            row(&[2.5, 4.0]),
            Value::String("makima".to_string().into()),
        ])
        .expect("makima interp1");
        let Value::Tensor(tensor) = result else {
            panic!("expected query-shaped tensor");
        };
        assert_eq!(tensor.shape(), &[1, 2]);
        assert!((tensor.materialize_f64()[0] - 39.0 / 88.0).abs() < 1e-12);
        assert_eq!(tensor.materialize_f64()[1], 0.0);

        let extrapolated = run(vec![
            row(&[0.0, 1.0]),
            row(&[0.0, 1.0]),
            Value::Num(-1.0),
            Value::String("makima".to_string().into()),
        ])
        .expect("two-point makima extrapolation");
        assert_eq!(extrapolated, Value::Num(-1.0));

        let filled = run(vec![
            row(&[0.0, 1.0]),
            row(&[0.0, 1.0]),
            Value::Num(-1.0),
            Value::String("makima".to_string().into()),
            Value::Num(99.0),
        ])
        .expect("explicit makima fill");
        assert_eq!(filled, Value::Num(99.0));
    }

    #[test]
    fn interp1_cubic_keeps_nan_default_distinct_from_pchip() {
        let samples = || {
            vec![
                row(&[0.0, 1.0, 2.0]),
                row(&[0.0, 1.0, 4.0]),
                Value::Num(-1.0),
            ]
        };
        let mut cubic = samples();
        cubic.push(Value::String("cubic".to_string().into()));
        let Value::Num(default_cubic) = run(cubic).expect("cubic default") else {
            panic!("expected scalar cubic output");
        };
        assert!(default_cubic.is_nan());

        let mut pchip = samples();
        pchip.push(Value::String("pchip".to_string().into()));
        let Value::Num(default_pchip) = run(pchip).expect("pchip default") else {
            panic!("expected scalar pchip output");
        };
        assert!(default_pchip.is_finite());

        let mut explicit = samples();
        explicit.push(Value::String("cubic".to_string().into()));
        explicit.push(Value::String("extrap".to_string().into()));
        assert_eq!(
            run(explicit).expect("explicit cubic extrapolation"),
            Value::Num(default_pchip)
        );

        let mut filled = samples();
        filled.push(Value::String("cubic".to_string().into()));
        filled.push(Value::Num(17.0));
        assert_eq!(run(filled).expect("explicit cubic fill"), Value::Num(17.0));
    }

    #[test]
    fn interp1_reads_typed_integer_x_y_and_query_exactly() {
        let args = || {
            vec![
                int_row(IntegerStorage::I16(vec![1, 2, 3])),
                int_row(IntegerStorage::U16(vec![10, 20, 40])),
                int_row(IntegerStorage::I16(vec![1, 2])),
            ]
        };
        let error = run(args()).expect_err("compatible mode rejects integer samples");
        assert_eq!(
            error.identifier(),
            INTERP1_INTEGER_SAMPLE_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run(args()).expect("RunMat integer interpolation");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.materialize_f64(), vec![10.0, 20.0]);
    }

    #[test]
    fn interp1_gpu_implicit_x_linear_keeps_output_resident() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0, 40.0],
                    shape: &[1, 3],
                })
                .expect("upload y");
            let result = run(vec![Value::GpuTensor(y.clone()), row(&[1.5, 2.5])]).expect("interp1");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpu output");
            };
            assert_eq!(handle.shape(), vec![1, 2]);
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![15.0, 30.0]);
            y.release().expect("release borrowed y");
        });
    }

    #[test]
    fn interp1_gpu_explicit_x_resident_query_preserves_series_shape() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0, 40.0, 100.0, 200.0, 400.0],
                    shape: &[3, 2],
                })
                .expect("upload y");
            let xq = provider
                .upload(&HostTensorView {
                    data: &[1.5, 2.5],
                    shape: &[1, 2],
                })
                .expect("upload xq");
            let result = run(vec![
                row(&[1.0, 2.0, 3.0]),
                Value::GpuTensor(y.clone()),
                Value::GpuTensor(xq.clone()),
            ])
            .expect("interp1");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpu output");
            };
            assert_eq!(handle.shape(), vec![1, 2, 2]);
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![15.0, 30.0, 150.0, 300.0]);
            y.release().expect("release borrowed y");
            xq.release().expect("release borrowed query");
        });
    }

    #[test]
    fn interp1_gpu_explicit_query_makes_output_explicit() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0, 40.0],
                    shape: &[1, 3],
                })
                .expect("upload y");
            let xq = provider
                .upload(&HostTensorView {
                    data: &[1.5],
                    shape: &[1, 1],
                })
                .expect("upload xq");
            let xq = xq
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("admit provenance view");
            let result = run(vec![
                Value::GpuTensor(y.clone()),
                Value::GpuTensor(xq.clone()),
            ])
            .expect("interp1");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_provenance(&output),
                Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            );
            y.release().expect("release fixture input");
            xq.release().expect("release fixture input");
            output.release().expect("release fixture input");
        });
    }

    #[test]
    fn interp1_gpu_allows_same_resident_handle_for_y_and_query() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &[1, 3],
                })
                .expect("upload y/query");
            let metadata = y.wire_descriptor();
            let result = run(vec![
                Value::GpuTensor(y.clone()),
                Value::GpuTensor(y.clone()),
            ])
            .expect("same-handle Y/Xq interpolation");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            assert_eq!(output.shape(), vec![1, 3]);
            assert_eq!(y.wire_descriptor(), metadata);
            let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 3.0]);
            y.release().expect("release fixture input");
        });
    }

    #[test]
    fn interp1_gpu_input_validation_rejects_alias_and_bad_class_metadata() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0],
                    shape: &[1, 2],
                })
                .expect("upload y");
            let precision = runmat_accelerate_api::handle_precision(&y);
            assert!(!valid_interp1_gpu_input(
                &y,
                &[1, 2],
                precision,
                provider,
                &[&y],
            ));
            let other = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[1, 2]),
                    shape: &[1, 2],
                })
                .expect("upload other");
            assert!(!valid_interp1_gpu_input(
                &other,
                &[1, 2],
                precision,
                provider,
                &[&y],
            ));
            y.release().expect("release fixture input");
            other.release().expect("release fixture input");
        });
    }

    #[test]
    fn interp1_gpu_scalar_fill_value_uses_provider_path() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0],
                    shape: &[1, 2],
                })
                .expect("upload y");
            let result = run(vec![
                Value::GpuTensor(y.clone()),
                Value::Num(0.0),
                Value::String("linear".to_string().into()),
                Value::Num(99.0),
            ])
            .expect("interp1");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpu output");
            };
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![99.0]);
            y.release().expect("release borrowed y");
        });
    }

    #[test]
    fn interp1_integer_extrapolation_is_independently_gated() {
        let args = || {
            vec![
                row(&[10.0, 20.0]),
                Value::Num(0.0),
                Value::String("linear".to_string().into()),
                Value::Int(runmat_value::IntValue::I16(99)),
            ]
        };
        let error = run(args()).expect_err("compatible mode rejects integer extrapolation");
        assert_eq!(
            error.identifier(),
            INTERP1_INTEGER_EXTRAPOLATION_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(run(args()).expect("RunMat extrapolation"), Value::Num(99.0));
    }

    #[test]
    fn interp1_default_out_of_range_is_nan() {
        let result =
            run(vec![row(&[1.0, 2.0]), row(&[10.0, 20.0]), Value::Num(0.0)]).expect("interp1");
        let Value::Num(value) = result else {
            panic!("expected scalar");
        };
        assert!(value.is_nan());
    }

    #[test]
    fn interp1_extrapolates_when_requested() {
        let result = run(vec![
            row(&[1.0, 2.0]),
            row(&[10.0, 20.0]),
            Value::Num(0.0),
            Value::String("extrap".to_string().into()),
        ])
        .expect("interp1");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn interp1_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = INTERP1_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"Vq = interp1(Y, Xq)"));
        assert!(labels.contains(&"Vq = interp1(X, Y, Xq)"));
        assert!(labels.contains(&"Vq = interp1(X, Y, Xq, method, extrap)"));
    }

    #[test]
    fn interp1_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = INTERP1_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.INTERP1.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.INTERP1.INVALID_INPUT"));
        assert!(codes.contains(&"RM.INTERP1.INTERNAL"));
    }

    #[test]
    fn interp1_too_few_args_uses_stable_identifier() {
        let err = run(vec![row(&[1.0, 2.0])]).expect_err("expected interp1 argument error");
        assert_eq!(err.identifier(), INTERP1_ERROR_INVALID_ARGUMENT.identifier);
    }
}
