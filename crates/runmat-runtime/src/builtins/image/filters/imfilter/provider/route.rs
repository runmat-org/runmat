//! Typed provider-route choice and prepared H projection. No provider effects
//! occur here; the selected whole-call receipt is created before preparation.

use super::{
    floating_accumulation, gpu_helpers, placement, BuiltinResult, FilterAccumulation,
    GpuTensorHandle, GpuTensorStorage, ImageOutputClass, ImfilterOptions, NumericDType,
    ProviderPrecision, Value, IMFILTER_BUILTIN,
};

#[derive(Default)]
pub(super) struct ProviderRouteSelection {
    pub selected: Option<crate::context::provider_operation::SelectedProviderOperation>,
    pub planned_kernel: Option<PlannedKernel>,
}

pub(super) struct PlannedKernel {
    pub query: runmat_accelerate_api::ProviderFeasibilityQuery,
    pub requested_bytes: u64,
}

fn projected_public_kernel(
    value: &Value,
    precision: ProviderPrecision,
) -> Option<(
    runmat_accelerate_api::ProviderRepresentation,
    runmat_execution::ResourceByteBound,
)> {
    use runmat_accelerate_api::{
        ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency,
        ProviderStorage,
    };
    let shape = match value {
        Value::Num(_) => vec![1, 1],
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 => {
            tensor.shape().to_vec()
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F64) =>
        {
            handle.shape().to_vec()
        }
        _ => return None,
    };
    let len = shape.iter().try_fold(1_u64, |count, &extent| {
        count.checked_mul(u64::try_from(extent).ok()?)
    })?;
    let scratch = len
        .checked_mul(if precision == ProviderPrecision::F32 {
            4
        } else {
            0
        })
        .and_then(|bytes| bytes.checked_add(u64::from(matches!(value, Value::Num(_))) * 8))
        .map(runmat_execution::ResourceByteBound::Known)
        .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        ));
    Some((
        ProviderRepresentation {
            element_type: ProviderElementType::from(precision),
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape
                .into_iter()
                .map(|extent| u64::try_from(extent).ok())
                .collect::<Option<Vec<_>>>()?,
            residency: ProviderResidency::Device,
        },
        scratch,
    ))
}

pub(super) fn select_provider_route(
    runtime: Option<&crate::context::RuntimeContext>,
    image: &GpuTensorHandle,
    kernel: &Value,
    options: &ImfilterOptions,
    output_class: ImageOutputClass,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<ProviderRouteSelection> {
    let Some(image_precision) = output_class.provider_precision() else {
        return Ok(ProviderRouteSelection::default());
    };
    let required_kernel_precision = match floating_accumulation(
        image_precision,
        runmat_accelerate_api::handle_is_explicit(image),
    ) {
        FilterAccumulation::HostDouble => ProviderPrecision::F64,
        FilterAccumulation::GpuSingle => ProviderPrecision::F32,
    };
    if !valid_public_kernel(kernel) {
        return Ok(ProviderRouteSelection::default());
    }
    if let Value::GpuTensor(handle) = kernel {
        if required_kernel_precision == ProviderPrecision::F64
            && gpu_helpers::exact_provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        {
            let (scratch, result_bytes) = placement::conditional_host_bounds(
                IMFILTER_BUILTIN,
                image.shape(),
                handle.shape(),
                options,
                image_precision == ProviderPrecision::F32,
                false,
                false,
            );
            let fallback_readbacks = [image];
            let explicit = runmat_accelerate_api::handle_is_explicit(image);
            let upload_scratch = placement::host_scratch_excluding_result(scratch, result_bytes)?;
            let conditional = if explicit {
                placement::ConditionalFilterRoute::Provider(placement::ConditionalProviderRoute {
                    operation_scratch: upload_scratch,
                    readbacks: &fallback_readbacks,
                })
            } else {
                placement::ConditionalFilterRoute::Host(placement::ConditionalHostRoute {
                    result_bytes,
                    operation_scratch: scratch,
                    readbacks: &fallback_readbacks,
                })
            };
            return Ok(ProviderRouteSelection {
                selected: placement::select_direct(
                    placement::FilterRouteRequest {
                        runtime,
                        semantic_operation: IMFILTER_BUILTIN,
                        image,
                        options,
                        conditional: Some(conditional),
                    },
                    handle,
                    true,
                )?,
                planned_kernel: None,
            });
        }
    }
    let Some((projected, scratch)) = projected_public_kernel(kernel, required_kernel_precision)
    else {
        return Ok(ProviderRouteSelection::default());
    };
    let original = match kernel {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    };
    let kernel_shape = match kernel {
        Value::Num(_) => vec![1, 1],
        Value::Tensor(tensor) => tensor.shape().to_vec(),
        Value::GpuTensor(handle) => handle.shape().to_vec(),
        _ => return Ok(ProviderRouteSelection::default()),
    };
    let (fallback_scratch, result_bytes) = placement::conditional_host_bounds(
        IMFILTER_BUILTIN,
        image.shape(),
        &kernel_shape,
        options,
        image_precision == ProviderPrecision::F32,
        false,
        true,
    );
    let fallback_readbacks = [image];
    let explicit = runmat_accelerate_api::handle_is_explicit(image);
    let upload_scratch = placement::host_scratch_excluding_result(fallback_scratch, result_bytes)?;
    let conditional = if explicit {
        placement::ConditionalFilterRoute::Provider(placement::ConditionalProviderRoute {
            operation_scratch: upload_scratch,
            readbacks: &fallback_readbacks,
        })
    } else {
        placement::ConditionalFilterRoute::Host(placement::ConditionalHostRoute {
            result_bytes,
            operation_scratch: fallback_scratch,
            readbacks: &fallback_readbacks,
        })
    };
    Ok(
        match placement::select_prepared(
            placement::FilterRouteRequest {
                runtime,
                semantic_operation: IMFILTER_BUILTIN,
                image,
                options,
                conditional: Some(conditional),
            },
            original,
            projected,
            scratch,
        )? {
            Some(prepared) => ProviderRouteSelection {
                selected: Some(prepared.selection),
                planned_kernel: Some(PlannedKernel {
                    query: prepared.query,
                    requested_bytes: prepared.expected_kernel_requested_bytes,
                }),
            },
            None => ProviderRouteSelection::default(),
        },
    )
}

pub(super) fn valid_public_kernel(kernel: &Value) -> bool {
    match kernel {
        Value::Num(_) => true,
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F64,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F64)
                && runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_integer_type(handle).is_none()
                && gpu_helpers::gpu_class_metadata_matches(
                    handle,
                    Some(ProviderPrecision::F64),
                    None,
                    false,
                )
        }
        _ => false,
    }
}
