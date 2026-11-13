//! Exact selected execution of one explicit GPU child-stream range. The
//! session owns counters, the provider owns device work, and the existing
//! whole-work and host-allocation services own finite admission.

use crate::builtins::common::gpu_helpers::query_metadata;
use crate::builtins::common::random::gpu;
use crate::builtins::common::random_args::validate_constructor_gpu_output_protecting;
use crate::context::host_allocation::{
    provider_registration_host_quote, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::provider_operation::{
    select_exact_provider_operation, ExactProviderOperation, ProviderPhysicalCalibration,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::observation::ProviderObservationFailure;
use runmat_accelerate_api::random::{
    GpuRandomDistribution, GpuRandomRequest, GpuRandomWorkload, PROVIDER_GPU_RANDOM_OPERATION,
};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderElementType,
    ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderPrecision, ProviderQueryMetadataFootprint, ProviderRepresentation, ProviderResidency,
    ProviderStorage, ProviderWorkload,
};
use runmat_execution::{Digest, ResourceByteBound};

fn fail(label: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(label).build()
}

fn provider_failure(
    label: &'static str,
    source: runmat_accelerate_api::ProviderExecutionError,
) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        fail(
            label,
            format!("{label}: GPU random provider failed: {source}"),
        ),
        source,
    )
}

fn validate_output(
    label: &'static str,
    provider: &'static dyn AccelProvider,
    output: GpuTensorHandle,
    shape: &[usize],
    storage: GpuTensorStorage,
    precision: ProviderPrecision,
    prototype: Option<&GpuTensorHandle>,
    expected_bytes: u64,
) -> BuiltinResult<GpuTensorHandle> {
    let protected_storage;
    let protected: &[&GpuTensorHandle] = if let Some(input) = prototype {
        protected_storage = [input];
        &protected_storage
    } else {
        &[]
    };
    let mut output = validate_constructor_gpu_output_protecting(
        label,
        provider,
        output,
        shape,
        storage,
        Some(precision),
        None,
        false,
        protected,
    )?;
    if let Some(input) = prototype {
        if let Err(error) =
            crate::builtins::common::gpu_helpers::propagate_output_provenance(&mut output, [input])
        {
            return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                output, error,
            ));
        }
    }
    if output.physical().requested_bytes() != expected_bytes {
        return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
            output,
            fail(
                label,
                format!("{label}: GPU random output exceeds its physical proposal"),
            ),
        ));
    }
    Ok(output)
}

/// The resident prototype is borrowed for owner/alias validation only. It
/// cannot issue a stream range or silently select a different device.
pub(crate) fn explicit_real(
    label: &'static str,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
    precision: ProviderPrecision,
    distribution: GpuRandomDistribution,
    prototype: Option<&GpuTensorHandle>,
) -> BuiltinResult<GpuTensorHandle> {
    explicit(
        label,
        provider,
        shape,
        precision,
        GpuTensorStorage::Real,
        distribution,
        prototype,
    )
}

pub(crate) fn explicit_complex(
    label: &'static str,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
    precision: ProviderPrecision,
    distribution: GpuRandomDistribution,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<GpuTensorHandle> {
    explicit(
        label,
        provider,
        shape,
        precision,
        GpuTensorStorage::ComplexInterleaved,
        distribution,
        Some(prototype),
    )
}

fn explicit(
    label: &'static str,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
    precision: ProviderPrecision,
    storage: GpuTensorStorage,
    distribution: GpuRandomDistribution,
    prototype: Option<&GpuTensorHandle>,
) -> BuiltinResult<GpuTensorHandle> {
    let count = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |n, dim| n.checked_mul(*dim))
            .ok_or_else(|| fail(label, format!("{label}: output size overflow")))?
    };
    let count = count
        .checked_mul(match storage {
            GpuTensorStorage::Real => 1,
            GpuTensorStorage::ComplexInterleaved => 2,
        })
        .ok_or_else(|| fail(label, format!("{label}: component count overflow")))?;
    let runtime = crate::context::legacy::active();
    gpu::with_reserved_attempt(count, label, |reservation| {
        let mut consumed = false;
        let mut operation = || -> BuiltinResult<GpuTensorHandle> {
            let request = GpuRandomRequest {
                shape,
                precision,
                storage,
                distribution,
                reservation,
            };
            let physical = provider
                .propose_gpu_random(&request)
                .map_err(|source| provider_failure(label, source))?
                .ok_or_else(|| {
                    fail(
                        label,
                        format!("{label}: provider has no exact GPU random route"),
                    )
                })?;
            let footprint = ProviderQueryMetadataFootprint {
                operation: PROVIDER_GPU_RANDOM_OPERATION,
                input_ranks: &[],
                output_ranks: &[shape.len()],
                preparation_shape_ranks: &[],
                preparation_representation_slots: 0,
            };
            let query_bytes = query_metadata::requested_bytes(label, &footprint)?;
            // This claim precedes every owned Vec/String in the feasibility
            // query and remains live through selection, then retires before
            // the separately admitted output-publication metadata.
            let _query_receipt =
                query_metadata::admit(runtime.as_ref(), query_bytes).map_err(|source| {
                    query_metadata::admission_error(label, runtime.as_ref(), source)
                })?;
            let output_shape = shape
                .iter()
                .map(|extent| u64::try_from(*extent))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| fail(label, format!("{label}: output dimension overflow")))?;
            let output = ProviderRepresentation {
                element_type: match precision {
                    ProviderPrecision::F32 => ProviderElementType::F32,
                    ProviderPrecision::F64 => ProviderElementType::F64,
                },
                storage: match storage {
                    GpuTensorStorage::Real => ProviderStorage::DenseReal,
                    GpuTensorStorage::ComplexInterleaved => {
                        ProviderStorage::DenseComplexInterleaved
                    }
                },
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: output_shape,
                residency: ProviderResidency::Device,
            };
            let query = ProviderFeasibilityQuery {
                operation: ProviderOperationIdentity::new(PROVIDER_GPU_RANDOM_OPERATION),
                family: ProviderOperationFamily::Library,
                inputs: Vec::new(),
                outputs: vec![output],
                workload: ProviderWorkload {
                    gpu_random: Some(GpuRandomWorkload {
                        distribution,
                        reservation,
                    }),
                    requested_outputs: Some(1),
                    ..ProviderWorkload::default()
                },
            };
            if !footprint.matches(&query) {
                return Err(fail(label, format!("{label}: query metadata plan changed")));
            }
            let derived_views = if prototype
                .is_some_and(|input| !runmat_accelerate_api::handle_is_explicit(input))
            {
                2
            } else {
                1
            };
            let registration =
                provider_registration_host_quote(provider, shape.len(), 1, derived_views).map_err(
                    |source| fail(label, format!("{label}: host metadata quote: {source}")),
                )?;
            let finite_host = runtime
                .as_ref()
                .and_then(|runtime| runtime.service_ports().parallel())
                .and_then(|service| service.placement_resources().memory_available_bytes)
                .is_some();
            if finite_host && registration.is_none() {
                return Err(fail(
                    label,
                    format!("{label}: provider has no finite host metadata bound"),
                ));
            }
            let host_bytes = registration
                .map(|quote| quote.preparation_bytes.checked_add(quote.retained_bytes))
                .unwrap_or(Some(0))
                .map(|bytes| bytes.max(query_bytes))
                .ok_or_else(|| fail(label, format!("{label}: host metadata extent overflow")))?;
            let facts = Digest::sha256_serialized(&(
                &query,
                prototype.map(GpuTensorHandle::view_identity),
                (
                    physical.output_buffer_bytes,
                    physical.uniform_buffer_bytes,
                    physical.uniform_allocations,
                    physical.dispatches,
                    physical.queue_submissions,
                ),
            ))
            .expect("typed GPU random facts serialize");
            let calibration = ProviderPhysicalCalibration::from_typed(&(
                &query.outputs,
                distribution,
                reservation.algorithm,
                reservation.normal_transform,
                (
                    physical.output_buffer_bytes,
                    physical.uniform_buffer_bytes,
                    physical.uniform_allocations,
                    physical.dispatches,
                    physical.queue_submissions,
                ),
            ));
            // Direct builtin unit tests intentionally have no RuntimeContext;
            // their genuine in-process provider still executes the exact
            // request, but this is not a production finite-admission route.
            #[cfg(test)]
            if runtime.is_none() {
                consumed = true;
                let output = match provider.execute_gpu_random(&request, None) {
                    Ok(output) => output,
                    Err(source) => {
                        consumed = !source.is_unsupported();
                        return Err(provider_failure(label, source));
                    }
                };
                return validate_output(
                    label,
                    provider,
                    output,
                    shape,
                    storage,
                    precision,
                    prototype,
                    physical.output_buffer_bytes,
                );
            }
            let mut selected = select_exact_provider_operation(
                runtime.as_ref(),
                ExactProviderOperation {
                    provider,
                    query: &query,
                    semantic_operation: label,
                    variant: "random.gpu_child.explicit.v1",
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
            )?
            .ok_or_else(|| fail(label, format!("{label}: no admitted provider random route")))?;
            // The selected receipt now owns the exact operation identity;
            // the request-local feasibility metadata is not needed during
            // output registration or provider execution.
            drop(query);
            drop(_query_receipt);
            let receipt =
                if let (Some(runtime), Some(registration)) = (runtime.as_ref(), registration) {
                    let admitted = HostAllocationPlan::new(
                        registration.preparation_bytes,
                        registration.retained_bytes,
                    )
                    .and_then(|plan| HostAllocationReceipt::admit(Some(runtime), plan));
                    match admitted {
                        Ok(receipt) => receipt,
                        Err(source) => {
                            selected.finish(Err(ProviderObservationFailure::ResourceLimit));
                            return Err(fail(label, format!("{label}: {source}")));
                        }
                    }
                } else {
                    None
                };
            let permit =
                if let (Some(runtime), Some(receipt)) = (runtime.as_ref(), receipt.as_ref()) {
                    let permit = match receipt.retained_view_permit(runtime) {
                        Ok(permit) => permit,
                        Err(source) => {
                            selected.finish(Err(ProviderObservationFailure::ResourceLimit));
                            return Err(fail(label, format!("{label}: {source}")));
                        }
                    };
                    match provider.prepare_output_host_metadata(Box::new(permit), 1) {
                        Ok(permit) => Some(permit),
                        Err(source) => {
                            selected.finish(Err(ProviderObservationFailure::ProviderFailed));
                            return Err(provider_failure(label, source));
                        }
                    }
                } else {
                    None
                };
            consumed = true;
            let output = match provider.execute_gpu_random(&request, permit.as_deref()) {
                Ok(output) => output,
                Err(source) => {
                    if source.is_unsupported() {
                        consumed = false;
                    }
                    selected.finish(Err(ProviderObservationFailure::ProviderFailed));
                    // Only certified pre-effect Unsupported can release the
                    // counter range; an unproven failure cannot be replayed.
                    return Err(provider_failure(label, source));
                }
            };
            let output = match validate_output(
                label,
                provider,
                output,
                shape,
                storage,
                precision,
                prototype,
                physical.output_buffer_bytes,
            ) {
                Ok(output) => output,
                Err(error) => {
                    selected.finish(Err(ProviderObservationFailure::PublicationFailed));
                    return Err(error);
                }
            };
            if !selected.validate_direct_provider_output(&output) {
                selected.finish(Err(ProviderObservationFailure::PublicationFailed));
                return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                    output,
                    fail(
                        label,
                        format!("{label}: GPU random output violates the selected physical bound"),
                    ),
                ));
            }
            selected.finish(Ok(true));
            Ok(output)
        };
        let result = operation();
        (result, consumed)
    })
}
