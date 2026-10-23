//! One physical candidate for the exact move-owned staged session. A missing
//! whole-program quote remains unpriced; this owner never sums per-hook costs.

use super::*;
use crate::placement::{
    plan_local_multi, plan_local_multi_bounded, tensor_representation, LocalCpuAdmission,
    LocalCpuCandidate, LocalMultiPlacementRequest, LocalPhysicalCandidate, LocalPlacementOutcome,
    LocalPublicationSemantics, PlacementPolicy, PreparedPhysicalAuthority,
    TotalCandidateObservationContract,
};
use runmat_accelerate_api::{
    ProviderElementType, ProviderImageStageWorkload, ProviderLayout, ProviderOperationFamily,
    ProviderPreparationRoute, ProviderRepresentation, ProviderResidency, ProviderStorage,
    ProviderWorkload,
};
use runmat_execution::{
    CandidateCostEvidence, CandidateCostUnavailable, CandidatePreparationState, Digest,
    ExecutionCandidateKind,
};

struct ImageSelectionParts<'a> {
    provider: &'static dyn AccelProvider,
    authority: PreparedPhysicalAuthority<'a>,
    semantic_identity: Digest,
    source: ProviderRepresentation,
    prepared: ProviderRepresentation,
    adaptation: Option<ProviderPreparationRoute>,
    upload_bytes: u64,
    program: &'a ProviderImageStageWorkload,
    outputs: Vec<ProviderRepresentation>,
}

fn selected_outputs(
    means: &ordered_mean::ImageOrderedMeanPlans,
    program: &ProviderImageStageWorkload,
) -> ProviderExecutionResult<Vec<ProviderRepresentation>> {
    let (mean, _) = means.public_mean_carrier().ok_or_else(|| {
        ProviderExecutionError::from(anyhow!("selected image mean has no public carrier"))
    })?;
    let (sigma, _) = means.public_sigma_carrier().ok_or_else(|| {
        ProviderExecutionError::from(anyhow!("selected image sigma has no public carrier"))
    })?;
    let output_program = program.output_program.as_ref().ok_or_else(|| {
        ProviderExecutionError::from(anyhow!("selected image output program is absent"))
    })?;
    let output_count = output_program
        .calls
        .iter()
        .filter(|call| call.checkpoint)
        .count()
        .checked_add(2)
        .ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("selected image output count overflows"))
        })?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(output_count)
        .map_err(ProviderExecutionError::from_source)?;
    outputs.extend([mean, sigma]);
    outputs.extend(
        output_program
            .calls
            .iter()
            .filter(|call| call.checkpoint)
            .map(|call| output_representation(call.output_precision, call.output_shape)),
    );
    Ok(outputs)
}

fn output_representation(precision: ProviderPrecision, shape: [u64; 3]) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type: ProviderElementType::from(precision),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.into(),
        residency: ProviderResidency::Device,
    }
}

impl PreparedImageStages {
    /// Placement consumes this prepared session's physical bounds and exact
    /// provider snapshot. No fabricated host-execution candidate is offered:
    /// declining the optimization resumes the original selected VM statements.
    pub(crate) fn select(
        &mut self,
        input: &Value,
        runtime: &runmat_runtime::context::RuntimeContext,
        decline: Option<&runmat_runtime::context::VmFusionDeclineRight>,
        completion: Option<&runmat_runtime::context::VmFusionCompletionRight>,
        finite: bool,
    ) -> ProviderExecutionResult<(LocalPlacementOutcome, TotalCandidateObservationContract)> {
        let Value::GpuTensor(handle) = input else {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image stages require their original resident input"
            )));
        };
        if handle.view_identity() != self.input_view_identity()
            || !self.matches_pre_effect_snapshot(self.session.provider)?
        {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image stage source or provider snapshot changed"
            )));
        }
        let means = self.session.selected_mean_plans.as_ref().ok_or_else(|| {
            ProviderExecutionError::from(anyhow!("selected image moments have no physical plan"))
        })?;
        let source =
            tensor_representation(input, self.session.provider.precision()).ok_or_else(|| {
                ProviderExecutionError::from(anyhow!(
                    "selected image input has no physical representation"
                ))
            })?;
        let (outcome, contract, receipt) = select_image_candidate(
            ImageSelectionParts {
                provider: self.session.provider,
                authority: PreparedPhysicalAuthority::ImageStages {
                    plan: self,
                    tail: self.session.provider.region_queue_tail_resources(),
                },
                semantic_identity: self.semantic_identity(),
                source: source.clone(),
                prepared: source,
                adaptation: None,
                upload_bytes: 0,
                program: &self.session.program,
                outputs: selected_outputs(means, &self.session.program)?,
            },
            input,
            runtime,
            decline,
            completion,
            finite,
        )?;
        self.retain_placement_preparation(receipt);
        Ok((outcome, contract))
    }
}

impl HostSingleImageSource {
    pub(crate) fn select(
        &mut self,
        input: &Value,
        runtime: &runmat_runtime::context::RuntimeContext,
        decline: Option<&runmat_runtime::context::VmFusionDeclineRight>,
        completion: Option<&runmat_runtime::context::VmFusionCompletionRight>,
        finite: bool,
    ) -> ProviderExecutionResult<(LocalPlacementOutcome, TotalCandidateObservationContract)> {
        if !self.matches_source_and_provider(self.plan.provider, input)? {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image host source or provider snapshot changed"
            )));
        }
        let (outcome, contract, receipt) = select_image_candidate(
            ImageSelectionParts {
                provider: self.plan.provider,
                authority: PreparedPhysicalAuthority::ImageHostStages(self),
                semantic_identity: self.semantic_identity(),
                source: self.source_representation(),
                prepared: self.route.prepared().clone(),
                adaptation: Some(self.preparation_route()),
                upload_bytes: self.upload.output_requested_bytes,
                program: &self.plan.program,
                outputs: selected_outputs(&self.plan.means, &self.plan.program)?,
            },
            input,
            runtime,
            decline,
            completion,
            finite,
        )?;
        self.retain_placement_preparation(receipt);
        Ok((outcome, contract))
    }
}

fn select_image_candidate(
    body: ImageSelectionParts<'_>,
    input: &Value,
    runtime: &runmat_runtime::context::RuntimeContext,
    decline: Option<&runmat_runtime::context::VmFusionDeclineRight>,
    completion: Option<&runmat_runtime::context::VmFusionCompletionRight>,
    finite: bool,
) -> ProviderExecutionResult<(
    LocalPlacementOutcome,
    TotalCandidateObservationContract,
    Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
)> {
    let data_inputs = [input];
    let no_quote = CandidateCostEvidence::Unavailable(CandidateCostUnavailable::NoEstimate);
    #[cfg(feature = "wgpu")]
    let original_tail = body
        .provider
        .region_queue_tail_resources()
        .filter(|_| body.provider.wgpu_device_access().is_some());
    #[cfg(not(feature = "wgpu"))]
    let original_tail = None;
    // A finite whole-route policy cannot admit an original VM region with
    // deferred per-instruction resource checks as a comparable aggregate
    // candidate. Keep the decline right: if the provider is rejected, the VM
    // still resumes its ordinary source instructions under their own owners.
    let original = decline.filter(|_| !finite).and_then(|decline| {
        let tail = original_tail?;
        let completion =
            completion.filter(|right| decline.same_installed_region(right, runtime))?;
        let source = decline.semantic_identity(runtime)?.digest();
        let route = runmat_execution::Digest::sha256_serialized(&(
            "image.original-vm-region.v1",
            source,
            body.semantic_identity,
            body.provider.device_id(),
        ))
        .ok()?;
        Some((route, completion, tail))
    });
    let (cpu, cpu_admission) = if let Some((route, _, tail)) = original {
        (
            LocalCpuCandidate::original_vm_region(route, body.provider.device_id(), tail, no_quote),
            LocalCpuAdmission::VmRegionAtUse,
        )
    } else {
        (
            LocalCpuCandidate::host("cpu.shared.fusion.image_normalize".into(), no_quote),
            LocalCpuAdmission::Unavailable,
        )
    };
    let contract = if original.is_some() {
        TotalCandidateObservationContract::FullCall
    } else {
        TotalCandidateObservationContract::PreparationBodyCleanup
    };
    let request = LocalMultiPlacementRequest {
        operation: "fusion.image_normalize",
        semantic_identity: Some(body.semantic_identity),
        observation_contract: contract,
        prepared_authority: None,
        vm_right: original.map(|(_, right, _)| right),
        vm_decline: decline,
        source_decline: crate::placement::LocalSourceDecline::None,
        family: ProviderOperationFamily::Fusion,
        cpu,
        cpu_admission,
        publication: LocalPublicationSemantics::SourceOrderedVm,
        physical: vec![LocalPhysicalCandidate {
            authority: Some(body.authority),
            physical_identity: Some(body.semantic_identity),
            provider_kind: ExecutionCandidateKind::ProviderFusion,
            data_inputs: &data_inputs,
            source_inputs: Some(vec![body.source]),
            prepared_inputs: Some(vec![body.prepared]),
            input_adaptation: body.adaptation,
            outputs: body.outputs,
            workload: ProviderWorkload {
                image_stages: Some(body.program.clone()),
                ..Default::default()
            },
            preparation: CandidatePreparationState::Cold,
            preparation_program: None,
            shader_requirements: None,
            fallback: no_quote,
            required_upload_bytes: Some(body.upload_bytes),
            required_download_bytes: 0,
            downstream_materialization: false,
        }],
        compare_profitability: true,
        runtime: Some(runtime),
    };
    let (outcome, receipt) = if finite {
        plan_local_multi_bounded(body.provider, request, PlacementPolicy::default())?
    } else {
        (
            plan_local_multi(body.provider, request, PlacementPolicy::default())
                .map_err(|error| ProviderExecutionError::from_source(*error))?,
            None,
        )
    };
    Ok((outcome, contract, receipt))
}
