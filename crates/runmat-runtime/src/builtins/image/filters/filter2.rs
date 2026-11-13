//! MATLAB-compatible `filter2` builtin implementing 2-D correlation and convolution.
//!
//! The implementation builds on the shared filter kernel so that host
//! and GPU behaviour stay perfectly aligned. By default the builtin performs a
//! correlation and returns an output the same size as the input image, matching
//! MathWorks MATLAB semantics.

use runmat_accelerate_api::{GpuTensorHandle, ImfilterMode, ImfilterOptions, ImfilterShape};
use runmat_builtins::{
    BuiltinErrorDescriptor, Filter2Option, FILTER2_ERROR_INTERNAL, FILTER2_ERROR_INVALID_INPUT,
    FILTER2_ERROR_INVALID_OPTION, FILTER2_INTEGER_GPU_EXTENSION, FILTER2_LOGICAL_GPU_EXTENSION,
    FILTER2_MODE_EXTENSION, FILTER2_ND_EXTENSION, FILTER2_VARIADIC_OPTIONS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use super::kernel::{apply_imfilter_tensor, filter_output_shape};
use super::placement;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod host_route;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::filters::filter2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "filter2",
    op_kind: GpuOpKind::Custom("filter2"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("imfilter")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to the provider `imfilter` hook with zero padding and correlation/convolution options.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::filters::filter2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "filter2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not a fusion candidate; executes via the dedicated filtering pipeline.",
};

const FILTER2_BUILTIN: &str = "filter2";

fn filter2_descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    detail: Option<&str>,
) -> RuntimeError {
    let message = match detail {
        Some(detail) => format!("{}: {}", error.message, detail),
        None => error.message.to_string(),
    };
    let mut builder = build_runtime_error(message).with_builtin(FILTER2_BUILTIN);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn filter2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    filter2_descriptor_error(error, Some(detail.as_ref()))
}

fn filter2_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let mut mapped = filter2_error_with_detail(fallback, err.message());
        mapped.source = Some(Box::new(err));
        mapped
    }
}

#[runtime_builtin(
    name = "filter2",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::filters::filter2"
)]
async fn filter2_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    kernel: Value,
    image: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    ensure_filter2_extensions(&kernel, &image, rest.view())?;
    // Invalid controls must fail before a resident integer/logical operand is
    // gathered, or a floating kernel is uploaded to the image owner.
    let options = parse_filter2_options(rest.view())?;
    preflight_filter2_geometry(&kernel, &image, options.shape)?;
    let output_single = value_is_single(&kernel) || value_is_single(&image);
    let image_precision_mismatch = matches!(&image, Value::GpuTensor(handle)
    if output_single
        && runmat_accelerate_api::provider_for_handle(handle).is_some_and(|provider| {
            provider.precision() != runmat_accelerate_api::ProviderPrecision::F32
        }));
    if is_nonfloating_input(&kernel) || is_nonfloating_input(&image) || image_precision_mismatch {
        return host_route::run(runtime.as_ref(), kernel, image, &options).await;
    }
    match (kernel, image) {
        (Value::GpuTensor(kernel_handle), Value::GpuTensor(image_handle)) => {
            filter2_gpu(
                runtime.as_ref(),
                Value::GpuTensor(kernel_handle),
                image_handle,
                &options,
            )
            .await
        }
        (Value::GpuTensor(kernel_handle), image_value) => {
            host_route::run(
                runtime.as_ref(),
                Value::GpuTensor(kernel_handle),
                image_value,
                &options,
            )
            .await
        }
        (kernel_value, Value::GpuTensor(image_handle)) => {
            filter2_gpu(runtime.as_ref(), kernel_value, image_handle, &options).await
        }
        (kernel_value, image_value) => {
            host_route::run(runtime.as_ref(), kernel_value, image_value, &options).await
        }
    }
}

fn value_rank(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape().len()),
        Value::LogicalArray(array) => Some(array.shape.len()),
        Value::GpuTensor(handle) => Some(handle.shape().len()),
        _ => None,
    }
}

fn filter2_value_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Tensor(value) => Some(value.shape().to_vec()),
        Value::LogicalArray(value) => Some(value.shape.to_vec()),
        Value::GpuTensor(value) => Some(value.shape().to_vec()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(vec![1, 1]),
        _ => None,
    }
}

/// A prepared H can retain its converted host tensor and a cached copy while
/// the provider hook runs. The caller's original host payload is excluded.
fn prepared_host_kernel_prefix(value: &Value) -> runmat_execution::ResourceByteBound {
    use runmat_execution::{ResourceBoundUnavailable, ResourceByteBound};
    filter2_value_shape(value)
        .and_then(|shape| {
            shape.into_iter().try_fold(1_u64, |total, extent| {
                total.checked_mul(u64::try_from(extent).ok()?)
            })
        })
        .and_then(|count| count.checked_mul(if value_is_single(value) { 4 } else { 8 }))
        .and_then(|bytes| bytes.checked_mul(2))
        .map(ResourceByteBound::Known)
        .unwrap_or(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        ))
}

fn projected_filter2_kernel(
    value: &Value,
) -> Option<runmat_accelerate_api::ProviderRepresentation> {
    use runmat_accelerate_api::{
        ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency,
        ProviderStorage,
    };
    let element_type = match value {
        Value::Num(_) => ProviderElementType::F64,
        Value::Tensor(tensor) => match tensor.numeric_dtype() {
            NumericDType::F32 => ProviderElementType::F32,
            NumericDType::F64 => ProviderElementType::F64,
            _ => return None,
        },
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_integer_type(handle).is_none() =>
        {
            match runmat_accelerate_api::handle_precision(handle)? {
                runmat_accelerate_api::ProviderPrecision::F32 => ProviderElementType::F32,
                runmat_accelerate_api::ProviderPrecision::F64 => ProviderElementType::F64,
            }
        }
        _ => return None,
    };
    Some(ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: filter2_value_shape(value)?
            .into_iter()
            .map(|extent| u64::try_from(extent).ok())
            .collect::<Option<Vec<_>>>()?,
        residency: ProviderResidency::Device,
    })
}

fn preflight_filter2_geometry(
    kernel: &Value,
    image: &Value,
    shape: ImfilterShape,
) -> BuiltinResult<()> {
    if let (Some(kernel_shape), Some(image_shape)) =
        (filter2_value_shape(kernel), filter2_value_shape(image))
    {
        filter_output_shape(&image_shape, &kernel_shape, shape, FILTER2_BUILTIN)
            .map_err(|error| filter2_map_error(error.into(), &FILTER2_ERROR_INVALID_INPUT))?;
    }
    Ok(())
}

fn is_nonfloating_input(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
        }
        Value::Tensor(tensor) => !matches!(
            tensor.numeric_dtype(),
            NumericDType::F32 | NumericDType::F64
        ),
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        _ => false,
    }
}

fn value_is_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_none()
            && !runmat_accelerate_api::handle_is_logical(handle)
            && runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn ensure_filter2_extensions(
    kernel: &Value,
    image: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if rest.len() > 1 {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILTER2_VARIADIC_OPTIONS_EXTENSION,
            FILTER2_BUILTIN,
        )?;
    }
    if rest.iter().any(|value| {
        tensor::value_to_string(value).is_some_and(|text| {
            matches!(text.trim().to_ascii_lowercase().as_str(), "conv" | "corr")
        })
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILTER2_MODE_EXTENSION,
            FILTER2_BUILTIN,
        )?;
    }
    if value_rank(kernel).is_some_and(|rank| rank > 2)
        || value_rank(image).is_some_and(|rank| rank > 2)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILTER2_ND_EXTENSION,
            FILTER2_BUILTIN,
        )?;
    }
    for value in [kernel, image] {
        if let Value::GpuTensor(handle) = value {
            if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FILTER2_INTEGER_GPU_EXTENSION,
                    FILTER2_BUILTIN,
                )?;
            }
            if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FILTER2_LOGICAL_GPU_EXTENSION,
                    FILTER2_BUILTIN,
                )?;
            }
        }
    }
    Ok(())
}

async fn gather_filter2_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle)
            .await
            .map(Value::Tensor)
            .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT)),
        other => Ok(other),
    }
}

fn filter2_host(
    kernel_value: Value,
    image_value: Value,
    options: &ImfilterOptions,
) -> BuiltinResult<Value> {
    let kernel_tensor = tensor::value_into_tensor_for(FILTER2_BUILTIN, kernel_value)
        .map_err(|err| filter2_error_with_detail(&FILTER2_ERROR_INVALID_INPUT, err))?;
    let image_tensor = tensor::value_into_tensor_for(FILTER2_BUILTIN, image_value)
        .map_err(|err| filter2_error_with_detail(&FILTER2_ERROR_INVALID_INPUT, err))?;
    let output_single = kernel_tensor.numeric_dtype() == NumericDType::F32
        || image_tensor.numeric_dtype() == NumericDType::F32;
    let result = apply_imfilter_tensor(&image_tensor, &kernel_tensor, options, FILTER2_BUILTIN)
        .map_err(|err| filter2_map_error(err.into(), &FILTER2_ERROR_INVALID_INPUT))?;
    if output_single {
        let shape = result.shape().to_vec();
        let result = Tensor::from_f32(
            result
                .materialize_f64()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            shape,
        )
        .map_err(|err| filter2_error_with_detail(&FILTER2_ERROR_INTERNAL, err))?;
        return Ok(tensor::tensor_into_value(result));
    }
    Ok(tensor::tensor_into_value(result))
}

async fn filter2_gpu(
    runtime: Option<&crate::context::RuntimeContext>,
    kernel_value: Value,
    image_handle: GpuTensorHandle,
    options: &ImfilterOptions,
) -> BuiltinResult<Value> {
    let kernel_clone = kernel_value.clone();
    image_handle
        .allocation_lease()
        .validate_live()
        .map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                filter2_error_with_detail(
                    &FILTER2_ERROR_INVALID_INPUT,
                    "X allocation is unavailable",
                ),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
    let invalid_owner = || {
        filter2_error_with_detail(
            &FILTER2_ERROR_INVALID_INPUT,
            "no exact provider owns the resident X/H operands",
        )
    };
    if let Value::GpuTensor(kernel) = &kernel_value {
        gpu_helpers::require_resident_compute_owners(
            &[&image_handle, kernel],
            false,
            invalid_owner,
        )?;
    } else {
        gpu_helpers::require_resident_compute_owner(&image_handle, invalid_owner)?;
    }
    let provider = match gpu_helpers::exact_provider_for_handle(&image_handle) {
        Some(p) => p,
        None => {
            return host_route::run(
                runtime,
                kernel_clone,
                Value::GpuTensor(image_handle),
                options,
            )
            .await;
        }
    };

    let original_kernel = match &kernel_value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    };
    let borrowed_kernel = original_kernel.is_some_and(|handle| {
        gpu_helpers::exact_provider_for_handle(handle)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
    });
    let (prepared_host_scratch, prepared_host_result) = placement::conditional_host_bounds(
        FILTER2_BUILTIN,
        image_handle.shape(),
        &filter2_value_shape(&kernel_value).unwrap_or_default(),
        options,
        value_is_single(&Value::GpuTensor(image_handle.clone())),
        value_is_single(&kernel_value),
        true,
    );
    let prepared_fallback_readbacks = [&image_handle];
    let prepared_selection = if borrowed_kernel {
        None
    } else if let Some(projected) = projected_filter2_kernel(&kernel_value) {
        placement::select_prepared(
            placement::FilterRouteRequest {
                runtime,
                semantic_operation: FILTER2_BUILTIN,
                image: &image_handle,
                options,
                conditional: Some(placement::ConditionalFilterRoute::Host(
                    placement::ConditionalHostRoute {
                        result_bytes: prepared_host_result,
                        operation_scratch: prepared_host_scratch,
                        readbacks: &prepared_fallback_readbacks,
                    },
                )),
            },
            original_kernel,
            projected,
            prepared_host_kernel_prefix(&kernel_value),
        )?
    } else {
        None
    };

    let mut uploaded_kernel: Option<GpuTensorHandle> = None;
    let mut kernel_tensor_cache: Option<Tensor> = None;

    let kernel_handle = match kernel_value {
        Value::GpuTensor(handle) => {
            handle.allocation_lease().validate_live().map_err(|error| {
                crate::runtime_error::with_provider_execution_source(
                    filter2_error_with_detail(
                        &FILTER2_ERROR_INVALID_INPUT,
                        "H allocation is unavailable",
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                )
            })?;
            if gpu_helpers::exact_provider_for_handle(&handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            {
                handle
            } else {
                let tensor = runmat_accelerate_api::observation::observe_provider_future(
                    prepared_selection
                        .as_ref()
                        .map(|selected| selected.selection.recorder()),
                    gpu_helpers::gather_tensor_async(&handle),
                )
                .await
                .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
                let upload = runmat_accelerate_api::observation::with_provider_observation(
                    prepared_selection
                        .as_ref()
                        .map(|selected| selected.selection.recorder()),
                    || upload_filter2_kernel(provider, &tensor, &image_handle, Some(&handle)),
                )?;
                match upload {
                    Some(uploaded) => {
                        kernel_tensor_cache = Some(tensor);
                        uploaded_kernel = Some(uploaded.clone());
                        uploaded
                    }
                    None => {
                        let image_tensor =
                            runmat_accelerate_api::observation::observe_provider_future(
                                prepared_selection
                                    .as_ref()
                                    .map(|selected| selected.selection.recorder()),
                                gpu_helpers::gather_tensor_async(&image_handle),
                            )
                            .await
                            .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
                        let result = filter2_host(
                            Value::Tensor(tensor),
                            Value::Tensor(image_tensor),
                            options,
                        );
                        if let Some(prepared) = prepared_selection {
                            if result.is_ok() {
                                prepared.selection.finish_preparation_host_fallback();
                            } else {
                                prepared.selection.finish(Err(
                                    runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                                ));
                            }
                        }
                        return result;
                    }
                }
            }
        }
        other => {
            let tensor = tensor::value_into_tensor_for(FILTER2_BUILTIN, other)
                .map_err(|err| filter2_error_with_detail(&FILTER2_ERROR_INVALID_INPUT, err))?;
            let upload = runmat_accelerate_api::observation::with_provider_observation(
                prepared_selection
                    .as_ref()
                    .map(|selected| selected.selection.recorder()),
                || upload_filter2_kernel(provider, &tensor, &image_handle, None),
            )?;
            match upload {
                Some(uploaded) => {
                    kernel_tensor_cache = Some(tensor);
                    uploaded_kernel = Some(uploaded.clone());
                    uploaded
                }
                None => {
                    let image_tensor = runmat_accelerate_api::observation::observe_provider_future(
                        prepared_selection
                            .as_ref()
                            .map(|selected| selected.selection.recorder()),
                        gpu_helpers::gather_tensor_async(&image_handle),
                    )
                    .await
                    .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
                    let result =
                        filter2_host(Value::Tensor(tensor), Value::Tensor(image_tensor), options);
                    if let Some(prepared) = prepared_selection {
                        if result.is_ok() {
                            prepared.selection.finish_preparation_host_fallback();
                        } else {
                            prepared.selection.finish(Err(
                                runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                            ));
                        }
                    }
                    return result;
                }
            }
        }
    };

    let kernel_handle_clone = kernel_handle.clone();
    let expected_shape = filter_output_shape(
        image_handle.shape(),
        kernel_handle.shape(),
        options.shape,
        FILTER2_BUILTIN,
    )
    .map_err(|error| filter2_map_error(error.into(), &FILTER2_ERROR_INVALID_INPUT))?;

    if let Some(prepared) = prepared_selection.as_ref() {
        if !placement::prepared_hook_matches_realized(
            &prepared.query,
            &image_handle,
            &kernel_handle,
            options,
            prepared.expected_kernel_requested_bytes,
        ) {
            return Err(filter2_error_with_detail(
                &FILTER2_ERROR_INTERNAL,
                "prepared H did not match the admitted provider representation",
            ));
        }
    }
    let direct_fallback_readbacks = [&image_handle, &kernel_handle];
    let (direct_host_scratch, direct_host_result) = placement::conditional_host_bounds(
        FILTER2_BUILTIN,
        image_handle.shape(),
        kernel_handle.shape(),
        options,
        value_is_single(&Value::GpuTensor(image_handle.clone())),
        value_is_single(&Value::GpuTensor(kernel_handle.clone())),
        false,
    );
    let mut selected = if let Some(prepared) = prepared_selection {
        Some(prepared.selection)
    } else if uploaded_kernel.is_none() {
        placement::select_direct(
            placement::FilterRouteRequest {
                runtime,
                semantic_operation: FILTER2_BUILTIN,
                image: &image_handle,
                options,
                conditional: Some(placement::ConditionalFilterRoute::Host(
                    placement::ConditionalHostRoute {
                        result_bytes: direct_host_result,
                        operation_scratch: direct_host_scratch,
                        readbacks: &direct_fallback_readbacks,
                    },
                )),
            },
            &kernel_handle,
            false,
        )?
    } else {
        None
    };
    let provider_result = runmat_accelerate_api::observation::observe_provider_future(
        selected.as_ref().map(|selection| selection.recorder()),
        provider.imfilter(&image_handle, &kernel_handle, options),
    )
    .await;
    match provider_result {
        Ok(output) => {
            drop(uploaded_kernel);
            let valid = output.shape() == expected_shape
                && output.device_id() == image_handle.device_id()
                && output.allocation_lease().validate_live().is_ok()
                && gpu_helpers::exact_provider_for_handle(&output).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
                && !output.shares_allocation_with(&image_handle)
                && !output.shares_allocation_with(&kernel_handle)
                && runmat_accelerate_api::handle_storage(&output)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                && runmat_accelerate_api::handle_integer_type(&output).is_none()
                && !runmat_accelerate_api::handle_is_logical(&output)
                && runmat_accelerate_api::handle_precision(&output) == Some(provider.precision())
                && gpu_helpers::gpu_class_metadata_matches(
                    &output,
                    Some(provider.precision()),
                    None,
                    false,
                )
                && selected
                    .as_mut()
                    .is_none_or(|selection| selection.validate_direct_provider_output(&output));
            if !valid {
                if let Some(selection) = selected {
                    selection.finish(Err(runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed));
                }
                let mut error = filter2_error_with_detail(
                    &FILTER2_ERROR_INTERNAL,
                    "provider returned an invalid filter2 result",
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
            let result = gpu_helpers::propagate_owned_output_provenance(output, [&image_handle])
                .and_then(gpu_helpers::resident_gpu_value);
            if let Some(selection) = selected {
                selection.finish(if result.is_ok() { Ok(true) } else { Err(runmat_accelerate_api::observation::ProviderObservationFailure::PublicationFailed) });
            }
            result
        }
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
            drop(uploaded_kernel);
            let fallback = async {
                let image_tensor = gpu_helpers::gather_tensor_async(&image_handle)
                    .await
                    .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?;
                let kernel_tensor = if let Some(ref tensor) = kernel_tensor_cache {
                    tensor.clone()
                } else {
                    gpu_helpers::gather_tensor_async(&kernel_handle_clone)
                        .await
                        .map_err(|err| filter2_map_error(err, &FILTER2_ERROR_INVALID_INPUT))?
                };
                filter2_host(
                    Value::Tensor(kernel_tensor),
                    Value::Tensor(image_tensor),
                    options,
                )
            };
            let result = runmat_accelerate_api::observation::observe_provider_future(
                selected.as_ref().map(|selection| selection.recorder()),
                fallback,
            )
            .await;
            if let Some(selection) = selected {
                if result.is_ok() {
                    selection.finish_conditional_host_fallback();
                } else {
                    selection.finish(Err(runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed));
                }
            }
            result
        }
        Err(error) => {
            if let Some(selection) = selected {
                selection.finish(Err(
                    runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                ));
            }
            Err(crate::runtime_error::with_provider_execution_source(
                filter2_error_with_detail(
                    &FILTER2_ERROR_INTERNAL,
                    format!("provider execution failed: {error}"),
                ),
                error.into(),
            ))
        }
    }
}

fn upload_filter2_kernel(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: &Tensor,
    image: &GpuTensorHandle,
    resident_kernel: Option<&GpuTensorHandle>,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    use runmat_accelerate_api::NumericUploadError;
    let protected = std::iter::once(image)
        .chain(resident_kernel)
        .collect::<Vec<_>>();
    match gpu_helpers::upload_tensor_protecting_typed(provider, tensor, &protected) {
        Ok(handle) => Ok(Some(handle)),
        Err(NumericUploadError::Provider(error)) if error.is_unsupported() => Ok(None),
        Err(error) => Err(crate::runtime_error::with_provider_execution_source(
            filter2_error_with_detail(
                &FILTER2_ERROR_INTERNAL,
                format!("failed to upload H to X's provider: {error}"),
            ),
            runmat_accelerate_api::ProviderExecutionError::from_source(error),
        )),
    }
}

fn parse_filter2_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ImfilterOptions> {
    let mut options = ImfilterOptions::default();
    for value in args {
        let Some(text) = tensor::value_to_string(value) else {
            return Err(filter2_error_with_detail(
                &FILTER2_ERROR_INVALID_OPTION,
                "expected string option ('same', 'full', 'valid', 'conv', 'corr')",
            ));
        };
        match Filter2Option::from_name(&text) {
            Some(Filter2Option::Same) => options.shape = ImfilterShape::Same,
            Some(Filter2Option::Full) => options.shape = ImfilterShape::Full,
            Some(Filter2Option::Valid) => options.shape = ImfilterShape::Valid,
            Some(Filter2Option::Convolution) => options.mode = ImfilterMode::Convolution,
            Some(Filter2Option::Correlation) => options.mode = ImfilterMode::Correlation,
            None => {
                return Err(filter2_error_with_detail(
                    &FILTER2_ERROR_INVALID_OPTION,
                    format!(
                        "unknown option '{}' (supported: 'same', 'full', 'valid', 'conv', 'corr')",
                        text.trim().to_ascii_lowercase()
                    ),
                ))
            }
        }
    }
    Ok(options)
}

#[cfg(test)]
#[path = "filter2/tests.rs"]
pub(crate) mod tests;
