//! One pre-effect-selected native integer GPU child candidate. An unsupported
//! exact provider proposal leaves the session stream unchanged and permits the
//! separately admitted host-mediated candidate; an effectful failure cannot.

use super::gpu;
use crate::builtins::common::gpu_helpers::query_metadata;
use crate::builtins::common::{
    gpu_helpers, random_args::validate_constructor_gpu_output_protecting,
};
use crate::context::host_allocation::{
    provider_registration_host_quote, HostAllocationError, HostAllocationPlan,
    HostAllocationReceipt,
};
use crate::context::host_effect::HostEffectAllocationError;
use crate::context::provider_operation::{
    select_exact_provider_operation, ExactProviderOperation, ProviderPhysicalCalibration,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::observation::ProviderObservationFailure;
use runmat_accelerate_api::random::{
    GpuBoundedIntegerDomain, GpuIntegerRequest, GpuIntegerWorkload, GpuRandomReservation,
    PROVIDER_GPU_INTEGER_OPERATION,
};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderElementType,
    ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderQueryMetadataFootprint, ProviderRepresentation, ProviderResidency, ProviderStorage,
    ProviderWorkload,
};
use runmat_execution::{Digest, ResourceByteBound};

fn error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("randi").build()
}

fn provider_error(source: runmat_accelerate_api::ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        error(format!(
            "randi: native GPU integer provider failed: {source}"
        )),
        source,
    )
}

fn output_element(class: NumericElementType) -> Option<ProviderElementType> {
    match class {
        NumericElementType::F32 => Some(ProviderElementType::F32),
        NumericElementType::F64 => Some(ProviderElementType::F64),
        NumericElementType::I32 => Some(ProviderElementType::I32),
        NumericElementType::U32 => Some(ProviderElementType::U32),
        _ => None,
    }
}

fn validate_output(
    provider: &'static dyn AccelProvider,
    source: &GpuTensorHandle,
    output: GpuTensorHandle,
    shape: &[usize],
    class: NumericElementType,
    expected_bytes: u64,
) -> BuiltinResult<GpuTensorHandle> {
    let mut output = validate_constructor_gpu_output_protecting(
        "randi",
        provider,
        output,
        shape,
        GpuTensorStorage::Real,
        class.precision(),
        class.integer_type(),
        false,
        &[source],
    )?;
    if output.physical().requested_bytes() != expected_bytes {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            error("randi: native output exceeds physical proposal"),
        ));
    }
    if let Err(cause) = gpu_helpers::propagate_output_provenance(&mut output, [source]) {
        return Err(gpu_helpers::reject_gpu_output(output, cause));
    }
    Ok(output)
}

/// `Ok(None)` means the native candidate was rejected without effects; the
/// caller may then choose its separately admitted host-mediated route.
pub(crate) fn try_fixed_work(
    source: &GpuTensorHandle,
    domain: GpuBoundedIntegerDomain,
    shape: &[usize],
    class: NumericElementType,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(output_element) = output_element(class) else {
        return Ok(None);
    };
    let len = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or_else(|| error("randi: output extent overflow"))?
    };
    let Some(provider) = gpu_helpers::exact_provider_for_handle(source) else {
        return Ok(None);
    };
    source
        .allocation_lease()
        .validate_live()
        .map_err(|cause| error(format!("randi: prototype lease: {cause}")))?;
    let snapshot = gpu::snapshot()?;
    let count = u64::try_from(len).map_err(|_| error("randi: child range exceeds u64"))?;
    let reservation = GpuRandomReservation {
        algorithm: snapshot.algorithm,
        seed: snapshot.seed,
        first: snapshot.position,
        count,
        normal_transform: snapshot.normal_transform,
    };
    let request = GpuIntegerRequest {
        shape,
        class,
        domain,
        reservation,
    };
    if request.fixed_work_32().is_none() {
        return Ok(None);
    }
    let Some(physical) = provider
        .propose_gpu_integer(&request)
        .map_err(provider_error)?
    else {
        return Ok(None);
    };
    let runtime = crate::context::legacy::active();
    let footprint = ProviderQueryMetadataFootprint {
        operation: PROVIDER_GPU_INTEGER_OPERATION,
        input_ranks: &[],
        output_ranks: &[shape.len()],
        preparation_shape_ranks: &[],
        preparation_representation_slots: 0,
    };
    let query_bytes = query_metadata::requested_bytes("randi", &footprint)?;
    let _query_receipt = match query_metadata::admit(runtime.as_ref(), query_bytes) {
        Ok(receipt) => receipt,
        Err(HostAllocationError::Resource(HostEffectAllocationError::ExceedsCapacity)) => {
            return Ok(None);
        }
        Err(cause) => {
            return Err(query_metadata::admission_error(
                "randi",
                runtime.as_ref(),
                cause,
            ))
        }
    };
    let query = ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(PROVIDER_GPU_INTEGER_OPERATION),
        family: ProviderOperationFamily::Library,
        inputs: Vec::new(),
        outputs: vec![ProviderRepresentation {
            element_type: output_element,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: shape
                .iter()
                .map(|&dim| u64::try_from(dim))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| error("randi: output dimension overflow"))?,
            residency: ProviderResidency::Device,
        }],
        workload: ProviderWorkload {
            gpu_integer: Some(GpuIntegerWorkload {
                domain,
                reservation,
            }),
            requested_outputs: Some(1),
            ..ProviderWorkload::default()
        },
    };
    if !footprint.matches(&query) {
        return Err(error("randi: integer query metadata plan changed"));
    }
    let derived_views = if runmat_accelerate_api::handle_is_explicit(source) {
        1
    } else {
        2
    };
    let registration = provider_registration_host_quote(provider, shape.len(), 1, derived_views)
        .map_err(|source| error(format!("randi: metadata quote: {source}")))?;
    let finite_host = runtime
        .as_ref()
        .and_then(|runtime| runtime.service_ports().parallel())
        .and_then(|service| service.placement_resources().memory_available_bytes)
        .is_some();
    if finite_host && registration.is_none() {
        return Ok(None);
    }
    let host_bytes = registration
        .map(|quote| quote.preparation_bytes.checked_add(quote.retained_bytes))
        .unwrap_or(Some(0))
        .map(|bytes| bytes.max(query_bytes))
        .ok_or_else(|| error("randi: host metadata extent overflow"))?;
    let facts = Digest::sha256_serialized(&(
        &query,
        source.view_identity(),
        (
            physical.output_buffer_bytes,
            physical.uniform_buffer_bytes,
            physical.uniform_allocations,
            physical.dispatches,
            physical.queue_submissions,
        ),
    ))
    .expect("typed GPU integer facts serialize");
    let calibration = ProviderPhysicalCalibration::from_typed(&(
        &query.outputs,
        reservation.algorithm,
        reservation.normal_transform,
        domain.span,
        (
            physical.output_buffer_bytes,
            physical.uniform_buffer_bytes,
            physical.uniform_allocations,
            physical.dispatches,
            physical.queue_submissions,
        ),
    ));
    let selected = select_exact_provider_operation(
        runtime.as_ref(),
        ExactProviderOperation {
            provider,
            query: &query,
            semantic_operation: "randi",
            variant: "random.gpu_child.integer.fixed32.v1",
            domain_facts: facts,
            calibration_facts: Some(calibration),
            host_scratch: ResourceByteBound::Known(host_bytes),
            provider_host_scratch: ResourceByteBound::Known(0),
            host_cpu_millicores: 1_000,
            already_live_bytes: 0,
            readbacks: &[],
            produced_readback: None,
            internal_input_stages: None,
            output_queue_slots: Some(physical.queue_submissions),
            conditional_host_fallback: None,
            conditional_host_scratch: ResourceByteBound::Known(0),
            conditional_provider_upload: None,
            provider_host_mediated: false,
        },
    )?;
    let Some(mut selected) = selected else {
        return Ok(None);
    };
    drop(query);
    drop(_query_receipt);
    let receipt = if let (Some(runtime), Some(registration)) = (runtime.as_ref(), registration) {
        let admitted =
            HostAllocationPlan::new(registration.preparation_bytes, registration.retained_bytes)
                .and_then(|plan| HostAllocationReceipt::admit(Some(runtime), plan));
        match admitted {
            Ok(receipt) => receipt,
            Err(source) => {
                selected.finish(Err(ProviderObservationFailure::ResourceLimit));
                return Err(error(format!("randi: host metadata admission: {source}")));
            }
        }
    } else {
        None
    };
    let permit = if let (Some(runtime), Some(receipt)) = (runtime.as_ref(), receipt.as_ref()) {
        let retained = match receipt.retained_view_permit(runtime) {
            Ok(permit) => permit,
            Err(source) => {
                selected.finish(Err(ProviderObservationFailure::ResourceLimit));
                return Err(error(format!("randi: metadata permit: {source}")));
            }
        };
        match provider.prepare_output_host_metadata(Box::new(retained), 1) {
            Ok(permit) => Some(permit),
            Err(source) => {
                selected.finish(Err(ProviderObservationFailure::ProviderFailed));
                return Err(provider_error(source));
            }
        }
    } else {
        None
    };
    let result = gpu::with_reserved_attempt(len, "randi", |live| {
        if live != reservation {
            return (
                Err(error("randi: child stream changed after selection")),
                false,
            );
        }
        match runmat_accelerate_api::observation::with_provider_observation(
            Some(selected.recorder()),
            || provider.execute_gpu_integer(&request, permit.as_deref()),
        ) {
            Ok(output) => {
                let output = match validate_output(
                    provider,
                    source,
                    output,
                    shape,
                    class,
                    physical.output_buffer_bytes,
                ) {
                    Ok(output) => output,
                    Err(cause) => {
                        selected.finish(Err(ProviderObservationFailure::PublicationFailed));
                        return (Err(cause), true);
                    }
                };
                if !selected.validate_direct_provider_output(&output) {
                    selected.finish(Err(ProviderObservationFailure::PublicationFailed));
                    return (
                        Err(gpu_helpers::reject_gpu_output(
                            output,
                            error("randi: native output violates selected bound"),
                        )),
                        true,
                    );
                }
                selected.finish(Ok(true));
                (Ok(Some(output)), true)
            }
            Err(source) if source.is_unsupported() => {
                selected.finish(Err(ProviderObservationFailure::ProviderFailed));
                (Ok(None), false)
            }
            Err(source) => {
                selected.finish(Err(ProviderObservationFailure::ProviderFailed));
                (Err(provider_error(source)), true)
            }
        }
    });
    result
}
