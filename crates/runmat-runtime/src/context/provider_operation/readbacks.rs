//! Authenticate original readback owners and compose typed stage/host demands.

use super::types::{
    ExactConditionalProviderUpload, ExactProviderOperation, ExactProviderReadbackPhase,
    PreparedProducedReadback,
};
use crate::context::host_presentation::combine_host_scratch;
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{
    provider_for_handle, GpuHandleProvenance, GpuTensorLayout, ProviderCapabilitySnapshot,
    ProviderFeasibilityQuery,
};
use runmat_execution::{
    ProviderInputStageDemand, ProviderInputStageDemands, ProviderResourceSnapshot,
    ResourceByteBound,
};

pub(super) struct PreparedReadbacks {
    pub(super) provider_resources: Vec<ProviderResourceSnapshot>,
    pub(super) source_capabilities: Vec<(u32, ProviderCapabilitySnapshot)>,
    pub(super) readback_facts: Vec<(
        ProviderFeasibilityQuery,
        ResourceByteBound,
        GpuTensorLayout,
        Option<GpuHandleProvenance>,
    )>,
    pub(super) input_stages: Option<ProviderInputStageDemands>,
    pub(super) preparation_host_workspace: ResourceByteBound,
    pub(super) conditional_readback_host_workspace: ResourceByteBound,
    pub(super) host_scratch: ResourceByteBound,
    pub(super) conditional_provider_upload: Option<ExactConditionalProviderUpload>,
    pub(super) produced_readback: Option<PreparedProducedReadback>,
}

pub(super) fn prepare_readbacks(
    request: &ExactProviderOperation<'_>,
    provider_output_bytes: Option<u64>,
    parallel_memory_available_bytes: Option<u64>,
) -> BuiltinResult<Option<PreparedReadbacks>> {
    let provider = request.provider;
    let mut provider_resources = vec![provider.placement_resources()];
    let mut source_capabilities = Vec::with_capacity(request.readbacks.len());
    let mut readback_facts = Vec::with_capacity(request.readbacks.len());
    let mut input_stages = Vec::new();
    let mut preparation_host_workspace = ResourceByteBound::Known(0);
    let mut conditional_readback_host_workspace = ResourceByteBound::Known(0);
    for readback in request.readbacks {
        let source = readback.source;
        let Some(source_provider) = provider_for_handle(source) else {
            return Err(
                build_runtime_error("provider input readback has no exact owner")
                    .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                    .build(),
            );
        };
        source.allocation_lease().validate_live().map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                build_runtime_error("provider input readback allocation is unavailable")
                    .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                    .build(),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
        if source.device_id() != source_provider.device_id() {
            return Err(
                build_runtime_error("provider input readback owner device is invalid")
                    .with_identifier("RunMat:Placement:InvalidReadbackOwner")
                    .build(),
            );
        }
        let source_resource = source_provider.placement_resources();
        super::admission::validate_live(&source_resource)?;
        let Some(download) = ProviderFeasibilityQuery::exact_numeric_download(source) else {
            if super::admission::finite_budget(&source_resource, parallel_memory_available_bytes) {
                return Err(build_runtime_error(
                    "provider input has no exact native readback under a finite resource budget",
                )
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .build());
            }
            return Ok(None);
        };
        if super::admission::estimate(
            source_provider.query_numeric_download_feasibility(source, &download),
            &source_resource,
            parallel_memory_available_bytes,
        )?
        .is_none()
        {
            if super::admission::finite_budget(&source_resource, parallel_memory_available_bytes) {
                return Err(build_runtime_error(
                    "provider input readback has no proven route under a finite resource budget",
                )
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .build());
            }
            return Ok(None);
        }
        let workspace = source_provider.download_numeric_host_workspace_bound(source);
        match readback.phase {
            ExactProviderReadbackPhase::Preparation => {
                preparation_host_workspace =
                    combine_host_scratch(preparation_host_workspace, workspace)?;
            }
            ExactProviderReadbackPhase::Conditional => {
                conditional_readback_host_workspace =
                    combine_host_scratch(conditional_readback_host_workspace, workspace)?;
            }
        }
        let stage_peak = sum_resource_bounds(
            readback.prior_created_bytes,
            source_provider.download_numeric_provider_scratch_bound(source),
        )?;
        input_stages.push(ProviderInputStageDemand {
            device_id: source_provider.device_id(),
            incremental_peak_bytes: stage_peak,
            queue_slots: source_provider.download_numeric_queue_slots(source),
        });
        source_capabilities.push((
            source_provider.device_id(),
            source_provider.capability_snapshot(),
        ));
        readback_facts.push((
            download,
            readback.prior_created_bytes,
            source.view().layout(),
            runmat_accelerate_api::handle_provenance(source),
        ));
        if !provider_resources
            .iter()
            .any(|resource| resource.device_id == source_resource.device_id)
        {
            provider_resources.push(source_resource);
        }
    }
    if let Some(internal) = request.internal_input_stages {
        if !internal.is_canonical() {
            return Err(
                build_runtime_error("provider input stage set is not canonical")
                    .with_identifier("RunMat:Placement:InvalidInputStages")
                    .build(),
            );
        }
        if internal
            .iter()
            .any(|stage| stage.device_id != provider.device_id())
        {
            return Err(
                build_runtime_error("provider internal stage has a foreign owner")
                    .with_identifier("RunMat:Placement:InvalidInputStages")
                    .build(),
            );
        }
        input_stages.extend(internal.iter());
    }
    let input_stages = if input_stages.is_empty() {
        None
    } else {
        Some(
            ProviderInputStageDemands::sequential(input_stages).ok_or_else(|| {
                build_runtime_error("provider input stages exceed the resource set")
                    .with_identifier("RunMat:Placement:TooManyProviders")
                    .build()
            })?,
        )
    };
    let produced_readback = if let Some(spec) = request.produced_readback {
        let Some(output) = request.query.outputs.get(spec.output_index) else {
            return Err(
                build_runtime_error("provider-produced readback output index is invalid")
                    .with_identifier("RunMat:Placement:InvalidReadbackOutput")
                    .build(),
            );
        };
        let Some(evidence) = provider.produced_numeric_readback_evidence(output) else {
            if provider_resources.iter().any(|snapshot| {
                super::admission::finite_budget(snapshot, parallel_memory_available_bytes)
            }) {
                return Err(build_runtime_error("provider-produced readback has no physical bound under a finite resource budget")
                    .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never).build());
            }
            return Ok(None);
        };
        if output.checked_byte_len() != Some(evidence.source_requested_bytes)
            || evidence.transfer_bytes != evidence.source_requested_bytes
            || evidence.queue_slots == 0
            || provider_output_bytes.is_none()
        {
            return Err(build_runtime_error(
                "provider-produced readback has no exact physical result",
            )
            .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
            .build());
        }
        Some(PreparedProducedReadback {
            output: output.clone(),
            evidence,
            public_host_bytes: spec.public_host_bytes,
        })
    } else {
        None
    };
    let prefix_host_workspace =
        combine_host_scratch(request.host_scratch, preparation_host_workspace)?;
    let direct_host_peak = combine_host_scratch(
        combine_host_scratch(prefix_host_workspace, request.provider_host_scratch)?,
        match produced_readback.as_ref() {
            Some(readback) => combine_host_scratch(
                readback.evidence.host_workspace_bytes,
                request
                    .produced_readback
                    .expect("prepared readback has a request")
                    .conversion_host_workspace,
            )?,
            None => ResourceByteBound::Known(0),
        },
    )?;
    let host_scratch = if request.conditional_host_fallback.is_some() {
        let fallback = combine_host_scratch(
            combine_host_scratch(prefix_host_workspace, conditional_readback_host_workspace)?,
            request.conditional_host_scratch,
        )?;
        peak_resource_bounds(direct_host_peak, fallback)
    } else {
        direct_host_peak
    };
    let conditional_provider_upload = if let Some(upload) = request.conditional_provider_upload {
        Some(ExactConditionalProviderUpload {
            host_scratch_bytes: combine_host_scratch(
                combine_host_scratch(prefix_host_workspace, conditional_readback_host_workspace)?,
                upload.host_scratch_bytes,
            )?,
            ..upload
        })
    } else {
        None
    };
    Ok(Some(PreparedReadbacks {
        provider_resources,
        source_capabilities,
        readback_facts,
        input_stages,
        preparation_host_workspace,
        conditional_readback_host_workspace,
        host_scratch,
        conditional_provider_upload,
        produced_readback,
    }))
}

fn sum_resource_bounds(
    left: ResourceByteBound,
    right: ResourceByteBound,
) -> BuiltinResult<ResourceByteBound> {
    match (left, right) {
        (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => left
            .checked_add(right)
            .map(ResourceByteBound::Known)
            .ok_or_else(|| {
                build_runtime_error("provider input stage byte bound overflows")
                    .with_identifier("RunMat:Placement:ResourceOverflow")
                    .build()
            }),
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => Ok(ResourceByteBound::Unavailable(reason)),
    }
}

fn peak_resource_bounds(left: ResourceByteBound, right: ResourceByteBound) -> ResourceByteBound {
    match (left, right) {
        (ResourceByteBound::Known(left), ResourceByteBound::Known(right)) => {
            ResourceByteBound::Known(left.max(right))
        }
        (ResourceByteBound::Unavailable(reason), _)
        | (_, ResourceByteBound::Unavailable(reason)) => ResourceByteBound::Unavailable(reason),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::physical_test_support::OwnedPhysicalProvider;
    use runmat_accelerate_api::{
        ProviderElementType, ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity,
        ProviderRepresentation, ProviderResidency, ProviderStorage, ProviderWorkload,
    };
    use runmat_execution::ResourceBoundUnavailable;

    #[test]
    fn missing_produced_index_evidence_cannot_skip_finite_readback_admission() {
        let provider = Box::leak(Box::new(OwnedPhysicalProvider::new()));
        let output = ProviderRepresentation {
            element_type: ProviderElementType::F32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![1, 1],
            residency: ProviderResidency::Device,
        };
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("min"),
            family: ProviderOperationFamily::Reduction,
            inputs: vec![],
            outputs: vec![output.clone(), output],
            workload: ProviderWorkload::default(),
        };
        let request = ExactProviderOperation {
            provider,
            query: &query,
            semantic_operation: "min",
            variant: "missing_readback_fixture",
            domain_facts: runmat_execution::Digest::sha256(b"readback_fixture"),
            calibration_facts: None,
            host_scratch: ResourceByteBound::Known(0),
            provider_host_scratch: ResourceByteBound::Known(0),
            host_cpu_millicores: 1000,
            already_live_bytes: 0,
            readbacks: &[],
            produced_readback: Some(super::super::types::ExactProducedProviderReadback {
                output_index: 1,
                public_host_bytes: ResourceByteBound::Known(8),
                conversion_host_workspace: ResourceByteBound::Known(0),
            }),
            internal_input_stages: None,
            output_queue_slots: None,
            conditional_host_fallback: None,
            conditional_host_scratch: ResourceByteBound::Known(0),
            conditional_provider_upload: None,
            provider_host_mediated: false,
        };
        assert!(prepare_readbacks(&request, Some(8), None)
            .unwrap()
            .is_none());
        let error = match prepare_readbacks(&request, Some(8), Some(1024)) {
            Err(error) => error,
            Ok(_) => panic!("finite readback requires real physical evidence"),
        };
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);

        let source = runmat_execution::cancellation::CancellationSource::new();
        source.cancel(runmat_execution::CancellationReason::User);
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        let error = match super::super::select_exact_provider_operation(Some(&runtime), request) {
            Err(error) => error,
            Ok(_) => panic!("cancelled route cannot become no-plan"),
        };
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    }

    #[test]
    fn provider_input_stage_overflow_is_a_resource_error_not_a_missing_selection() {
        let error = sum_resource_bounds(
            ResourceByteBound::Known(u64::MAX),
            ResourceByteBound::Known(1),
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:ResourceOverflow")
        );
    }

    #[test]
    fn provider_input_stage_unknown_never_becomes_zero() {
        let unknown = ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound);
        assert_eq!(sum_resource_bounds(16.into(), unknown).unwrap(), unknown);
    }

    #[test]
    fn conditional_host_peak_is_maximum_of_sequential_branches() {
        assert_eq!(peak_resource_bounds(24.into(), 32.into()), 32.into());
        assert_eq!(peak_resource_bounds(40.into(), 32.into()), 40.into());
        let unknown = ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoScratchBound);
        assert_eq!(peak_resource_bounds(40.into(), unknown), unknown);
    }
}
