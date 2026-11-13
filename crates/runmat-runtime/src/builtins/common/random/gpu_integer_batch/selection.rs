//! Query construction and the one pre-effect whole-work decision for bounded
//! GPU integer batches. No child-stream range is committed here.

use super::super::gpu;
use super::{error, provider_error};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::gpu_helpers::{host_upload::GpuHostCandidate, query_metadata};
use crate::context::host_allocation::{
    provider_registration_host_quote, HostAllocationError, HostAllocationPlan,
};
use crate::context::host_effect::HostEffectAllocationError;
use crate::context::provider_operation::{
    select_exact_provider_or_host_operation, ExactHostUploadAlternative, ExactProviderOperation,
    ExactProviderOrHostSelection, ProviderPhysicalCalibration, SelectedProviderOperation,
};
use crate::BuiltinResult;
use runmat_accelerate_api::observation::ProviderObservationCapability;
use runmat_accelerate_api::random::{
    GpuBoundedIntegerDomain, GpuIntegerBatchRequest, GpuIntegerBatchWorkload, GpuIntegerRequest,
    GpuIntegerWorkload, GpuRandomReservation, PROVIDER_GPU_INTEGER_BATCH_OPERATION,
};
use runmat_accelerate_api::{
    GpuTensorHandle, NumericElementType, ProviderElementType, ProviderFeasibilityQuery,
    ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
    ProviderQueryMetadataFootprint, ProviderRepresentation, ProviderResidency, ProviderStorage,
    ProviderWorkload,
};
use runmat_execution::{Digest, ResourceByteBound, WholeWorkCandidate};

fn output_element(class: NumericElementType) -> Option<ProviderElementType> {
    match class {
        NumericElementType::F32 => Some(ProviderElementType::F32),
        NumericElementType::F64 => Some(ProviderElementType::F64),
        NumericElementType::I32 => Some(ProviderElementType::I32),
        NumericElementType::U32 => Some(ProviderElementType::U32),
        _ => None,
    }
}

fn representation(shape: Vec<u64>, element_type: ProviderElementType) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: ProviderResidency::Device,
    }
}

pub(super) fn batch_query(
    shape: &[usize],
    class: NumericElementType,
    domain: GpuBoundedIntegerDomain,
    reservation: GpuRandomReservation,
    first_attempt: u64,
    pending: bool,
) -> BuiltinResult<ProviderFeasibilityQuery> {
    let element_type = output_element(class)
        .ok_or_else(|| error("randi: unsupported native GPU integer class"))?;
    let output_shape = shape
        .iter()
        .map(|&extent| u64::try_from(extent))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| error("randi: output dimension overflow"))?;
    let outputs = vec![
        representation(output_shape.clone(), element_type),
        representation(output_shape, ProviderElementType::U32),
    ];
    Ok(ProviderFeasibilityQuery {
        operation: ProviderOperationIdentity::new(PROVIDER_GPU_INTEGER_BATCH_OPERATION),
        family: ProviderOperationFamily::Library,
        inputs: if pending { outputs.clone() } else { Vec::new() },
        outputs,
        workload: ProviderWorkload {
            gpu_integer: Some(GpuIntegerWorkload {
                domain,
                reservation,
            }),
            gpu_integer_batch: Some(GpuIntegerBatchWorkload {
                first_attempt,
                attempts_per_item: 8,
                pending,
            }),
            requested_outputs: Some(2),
            ..ProviderWorkload::default()
        },
    })
}

pub(super) fn batch_calibration(
    query: &ProviderFeasibilityQuery,
    reservation: GpuRandomReservation,
    domain: GpuBoundedIntegerDomain,
    pending_remaining: Option<u64>,
    physical: runmat_accelerate_api::random::GpuIntegerBatchPhysicalProposal,
) -> ProviderPhysicalCalibration {
    ProviderPhysicalCalibration::from_typed(&(
        &query.outputs,
        reservation.algorithm,
        reservation.normal_transform,
        domain.span,
        pending_remaining,
        physical,
    ))
}

pub(crate) struct SelectedInitialNativeBatch {
    pub(crate) selected: SelectedProviderOperation,
    pub(crate) snapshot: gpu::GpuRandomSnapshot,
    pub(crate) physical: runmat_accelerate_api::random::GpuIntegerBatchPhysicalProposal,
}

pub(crate) struct InitialIntegerRoute {
    pub(crate) selection: ExactProviderOrHostSelection,
    pub(crate) snapshot: gpu::GpuRandomSnapshot,
    pub(crate) physical: runmat_accelerate_api::random::GpuIntegerBatchPhysicalProposal,
}

/// Available host and native candidates enter one whole-work decision before
/// any range commit. Failure to prepare an optional host-upload query does
/// not suppress a legal native candidate, or vice versa.
pub(crate) fn select_initial_route(
    source: &GpuTensorHandle,
    domain: GpuBoundedIntegerDomain,
    shape: &[usize],
    class: NumericElementType,
    host: Option<&GpuHostCandidate>,
) -> BuiltinResult<Option<InitialIntegerRoute>> {
    if output_element(class).is_none() {
        return Ok(None);
    }
    let Some(provider) = gpu_helpers::exact_provider_for_handle(source) else {
        return Ok(None);
    };
    let count = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |n, &extent| n.checked_mul(extent))
            .ok_or_else(|| error("randi: output extent overflow"))?
    };
    let snapshot = gpu::snapshot()?;
    let reservation = GpuRandomReservation {
        algorithm: snapshot.algorithm,
        seed: snapshot.seed,
        first: snapshot.position,
        count: u64::try_from(count).map_err(|_| error("randi: child range exceeds u64"))?,
        normal_transform: snapshot.normal_transform,
    };
    let integer = GpuIntegerRequest {
        shape,
        class,
        domain,
        reservation,
    };
    if integer.rejection_domain_32().is_none() {
        return Ok(None);
    }
    let batch = GpuIntegerBatchRequest {
        integer,
        first_attempt: 0,
        attempts_per_item: 8,
        pending: None,
    };
    let Some(physical) = provider
        .propose_gpu_integer_batch(batch)
        .map_err(provider_error)?
    else {
        return Ok(None);
    };
    let runtime = crate::context::legacy::active();
    let footprint = ProviderQueryMetadataFootprint {
        operation: PROVIDER_GPU_INTEGER_BATCH_OPERATION,
        input_ranks: &[],
        output_ranks: &[shape.len(), shape.len()],
        preparation_shape_ranks: &[],
        preparation_representation_slots: 0,
    };
    let query_bytes = query_metadata::requested_bytes("randi", &footprint)?;
    let _query_receipt = match query_metadata::admit(runtime.as_ref(), query_bytes) {
        Ok(receipt) => receipt,
        Err(HostAllocationError::Resource(HostEffectAllocationError::ExceedsCapacity)) => {
            // Query preparation belongs to the optional native route. The
            // independently legal host candidate remains available.
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
    let query = batch_query(shape, class, domain, reservation, 0, false)?;
    if !footprint.matches(&query) {
        return Err(error("randi: batch query metadata plan changed"));
    }
    if !matches!(
        provider.query_feasibility(&query),
        runmat_accelerate_api::ProviderFeasibility::Supported { .. }
    ) {
        return Ok(None);
    }
    if !matches!(
        provider.operation_observation_capability(&query),
        ProviderObservationCapability::Supported(_)
    ) {
        // A missing native observation contract excludes only the native
        // candidate. The caller can still select the independently quoted
        // host-compute/upload route before committing a stream range.
        return Ok(None);
    }
    let register = provider_registration_host_quote(provider, shape.len(), 2, 1)
        .map_err(|cause| error(format!("randi: batch metadata quote: {cause}")))?;
    let Some(register) = register else {
        return Ok(None);
    };
    let selection_query_peak = query_bytes
        .checked_add(host.map_or(0, |host| host.query_metadata_bytes))
        .ok_or_else(|| error("randi: paired query metadata overflow"))?;
    let registration_bytes = register
        .preparation_bytes
        .checked_add(register.retained_bytes)
        .ok_or_else(|| error("randi: batch host metadata overflow"))?;
    let host_bytes = selection_query_peak.max(registration_bytes);
    let readback_bytes =
        HostAllocationPlan::retained_metadata_bytes(physical.readback_control_host_bytes)
            .map_err(|cause| error(format!("randi: count control quote: {cause}")))?;
    let provider_host_bytes = readback_bytes
        .checked_add(physical.pending_view_workspace_bytes)
        .ok_or_else(|| error("randi: provider host workspace overflow"))?;
    let facts = Digest::sha256_serialized(&(&query, source.view_identity(), physical))
        .expect("typed integer route serializes");
    let host_candidate = host
        .map(|host| -> BuiltinResult<WholeWorkCandidate> {
            let mut candidate = host.candidate.clone();
            let selected_host_preparation = selection_query_peak
                .checked_add(host.publication_metadata_bytes)
                .ok_or_else(|| error("randi: selected upload metadata overflow"))?;
            candidate.resources.scratch_bytes = match candidate.resources.scratch_bytes {
                ResourceByteBound::Known(bytes) => {
                    ResourceByteBound::Known(bytes.max(selected_host_preparation))
                }
                other => other,
            };
            candidate.route_facts =
                Digest::sha256_serialized(&(candidate.route_facts, query_bytes))
                    .expect("typed shared preparation serializes");
            Ok(candidate)
        })
        .transpose()?;
    let selection = select_exact_provider_or_host_operation(
        runtime.as_ref(),
        ExactProviderOperation {
            provider,
            query: &query,
            semantic_operation: "randi",
            variant: "random.gpu_child.integer.batch.v1",
            domain_facts: facts,
            calibration_facts: Some(batch_calibration(
                &query,
                reservation,
                domain,
                None,
                physical,
            )),
            host_scratch: ResourceByteBound::Known(host_bytes),
            provider_host_scratch: ResourceByteBound::Known(provider_host_bytes),
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
        host.zip(host_candidate)
            .map(|(host, candidate)| ExactHostUploadAlternative {
                candidate,
                query: &host.query,
                output: host.output.clone(),
                requested_bytes: host.requested_bytes,
            }),
    )?;
    Ok(selection.map(|selection| InitialIntegerRoute {
        selection,
        snapshot,
        physical,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::random::{GpuNormalTransform, GpuRandomAlgorithm};

    #[test]
    fn calibration_reuses_physical_work_without_reusing_stream_authority() {
        let reservation = GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 0,
            first: 0,
            count: 4,
            normal_transform: GpuNormalTransform::BoxMuller,
        };
        let domain = GpuBoundedIntegerDomain::new(1, 7).expect("valid domain");
        let query = batch_query(
            &[2, 2],
            NumericElementType::I32,
            domain,
            reservation,
            0,
            false,
        )
        .expect("valid batch query");
        let physical = runmat_accelerate_api::random::GpuIntegerBatchPhysicalProposal {
            output_buffer_bytes: 16,
            accepted_buffer_bytes: 16,
            count_buffer_bytes: 4,
            uniform_buffer_bytes: 64,
            staging_buffer_bytes: 4,
            readback_control_host_bytes: 128,
            pending_view_workspace_bytes: 0,
            uniform_allocations: 1,
            queue_submissions: 2,
        };
        let changed_range = GpuRandomReservation {
            seed: 123,
            first: 8192,
            ..reservation
        };
        let changed_query = batch_query(
            &[2, 2],
            NumericElementType::I32,
            domain,
            changed_range,
            128,
            false,
        )
        .expect("valid changed request");
        assert_ne!(query, changed_query, "the live request must remain exact");
        assert_eq!(
            batch_calibration(&query, reservation, domain, None, physical),
            batch_calibration(&changed_query, changed_range, domain, None, physical)
        );
        assert_ne!(
            batch_calibration(&query, reservation, domain, None, physical),
            batch_calibration(&query, reservation, domain, Some(2), physical)
        );
    }
}
