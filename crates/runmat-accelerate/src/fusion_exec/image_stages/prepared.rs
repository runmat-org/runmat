//! Move-owned selected image preparation. A resource proposal cannot be
//! detached from the recipes and input lease whose requests it bounds.

use super::*;
use runmat_execution::{
    CandidatePeakLiveDemand, CandidateResourceDemand, Digest, ProviderResourceSnapshot,
    ResourceByteBound,
};

pub(crate) struct RegionTailAdmission {
    pub(crate) claim:
        Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
}
pub(crate) fn admit_region_tail(
    runtime: &runmat_runtime::context::RuntimeContext,
    provider: &dyn AccelProvider,
) -> ProviderExecutionResult<Option<RegionTailAdmission>> {
    use runmat_runtime::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
    let Some(bytes) = provider
        .region_queue_tail_resources()
        .and_then(|tail| tail.completion_control_host_bytes)
    else {
        // Unknown optional completion must withhold observation, not execute an
        // unquoted callback or turn unavailable host storage into zero bytes.
        return Ok(None);
    };
    let retained = HostAllocationPlan::retained_metadata_bytes(bytes)
        .map_err(ProviderExecutionError::from_source)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(ProviderExecutionError::from_source)?,
    )
    .map_err(ProviderExecutionError::from_source)?;
    let claim = receipt
        .as_ref()
        .map(|receipt| receipt.claim_retained_metadata(runtime, bytes))
        .transpose()
        .map_err(ProviderExecutionError::from_source)?;
    Ok(Some(RegionTailAdmission { claim }))
}

/// One shared physical audit owner. VM/native callers retain their distinct
/// source completion rights; this allocation grants neither of those rights.
pub(crate) fn admit_region_audit(
    runtime: &runmat_runtime::context::RuntimeContext,
    provider: &'static dyn AccelProvider,
) -> ProviderExecutionResult<Option<runmat_accelerate_api::observation::SourceRegionRouteAudit>> {
    use runmat_accelerate_api::observation::SourceRegionRouteAudit;
    use runmat_runtime::context::host_allocation::{
        HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    };
    let Some(access) = provider.wgpu_device_access().cloned() else {
        return Ok(None);
    };
    let bytes = SourceRegionRouteAudit::backing_bytes()
        .ok_or_else(|| ProviderExecutionError::from_source(HostAllocationError::Overflow))?;
    let retained = HostAllocationPlan::retained_metadata_bytes(bytes)
        .map_err(ProviderExecutionError::from_source)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(ProviderExecutionError::from_source)?,
    )
    .map_err(ProviderExecutionError::from_source)?;
    let claim = receipt
        .as_ref()
        .map(|receipt| receipt.claim_retained_metadata(runtime, bytes))
        .transpose()
        .map_err(ProviderExecutionError::from_source)?;
    SourceRegionRouteAudit::try_new(provider, access, claim)
        .map(Some)
        .map_err(ProviderExecutionError::from_source)
}
pub(super) fn charge_region_tail_host_demand(
    demand: &mut CandidateResourceDemand,
    tail: runmat_accelerate_api::ProviderRegionTailResources,
) -> Option<()> {
    use runmat_execution::ResourceBoundUnavailable;
    let tail = tail.completion_control_host_bytes.and_then(|bytes| {
        runmat_runtime::context::host_allocation::HostAllocationPlan::retained_metadata_bytes(bytes)
            .ok()
    });
    demand.provider_host_workspace = match (demand.provider_host_workspace, tail) {
        (Some(ResourceByteBound::Known(existing)), Some(tail)) => {
            Some(ResourceByteBound::Known(existing.checked_add(tail)?))
        }
        (Some(unavailable @ ResourceByteBound::Unavailable(_)), _) => Some(unavailable),
        (_, None) => Some(ResourceByteBound::Unavailable(
            ResourceBoundUnavailable::NoScratchBound,
        )),
        // The enclosing candidate's unmodeled host domain remains unmodeled.
        (None, Some(_)) => None,
    };
    Some(())
}

pub(crate) struct PreparedImageStages {
    pub(super) session: ImageNormalizeStageSession,
    semantic_identity: Digest,
    resources: resources::ImageStagePhysicalUpperBound,
    provider_snapshot: ProviderResourceSnapshot,
    provider_metadata_bytes: Option<u64>,
    provider_power_workspace_bytes: Option<u64>,
    _placement_host_preparation:
        Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
}

impl PreparedImageStages {
    pub(super) fn retain_placement_preparation(
        &mut self,
        receipt: Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
    ) {
        self._placement_host_preparation = receipt;
    }

    pub(super) fn from_selected_host_upload(
        session: ImageNormalizeStageSession,
        semantic_identity: Digest,
        resources: resources::ImageStagePhysicalUpperBound,
        placement_preparation: Option<
            runmat_runtime::context::host_allocation::HostAllocationReceipt,
        >,
    ) -> ProviderExecutionResult<Self> {
        let provider_snapshot = session.provider.placement_resources();
        provider_snapshot
            .validate_typed()
            .map_err(|error| ProviderExecutionError::from_source(error).after_effect())?;
        if provider_snapshot.device_id != session.provider.device_id() || provider_snapshot.lost {
            return Err(ProviderExecutionError::from(anyhow!(
                "selected image provider changed after host upload"
            ))
            .after_effect());
        }
        Ok(Self {
            session,
            semantic_identity,
            resources,
            provider_snapshot,
            provider_metadata_bytes: None,
            provider_power_workspace_bytes: None,
            _placement_host_preparation: placement_preparation,
        })
    }

    pub(crate) fn physical_work(
        &self,
    ) -> Option<runmat_accelerate_api::ProviderImageStagePhysicalWork> {
        let work = runmat_accelerate_api::ProviderImageStagePhysicalWork {
            requested_bytes: self.resources.requested_bytes,
            dispatches: self.resources.dispatches,
            queue_submissions: self.resources.queue_submissions?,
        };
        work.validate().then_some(work)
    }

    pub(crate) fn observation_capability(
        &self,
        provider: &dyn AccelProvider,
    ) -> runmat_accelerate_api::observation::ProviderObservationCapability {
        use runmat_accelerate_api::observation::ProviderObservationCapability;
        if !runmat_accelerate_api::same_provider_instance(self.session.provider, provider) {
            return ProviderObservationCapability::Unsupported;
        }
        self.physical_work()
            .map_or(ProviderObservationCapability::Unsupported, |work| {
                provider.image_stage_observation_capability(work)
            })
    }

    pub(crate) fn resources_with_observation(
        &self,
        observation: Option<runmat_accelerate_api::observation::ProviderObservationOverhead>,
    ) -> Result<CandidateResourceDemand, Box<runmat_runtime::RuntimeError>> {
        let Some(observation) = observation else {
            return Ok(self.resource_demand());
        };
        let bytes = observation
            .requested_total_peak_bytes()
            .ok_or_else(observation_overflow)?;
        self.resource_demand()
            .with_provider_observation_peak(bytes)
            .ok_or_else(observation_overflow)
    }

    pub(crate) fn resources_with_observation_and_tail(
        &self,
        observation: Option<runmat_accelerate_api::observation::ProviderObservationOverhead>,
        tail: Option<runmat_accelerate_api::ProviderRegionTailResources>,
    ) -> Result<CandidateResourceDemand, Box<runmat_runtime::RuntimeError>> {
        let mut demand = self.resources_with_observation(observation)?;
        if let Some(tail) = tail {
            demand.queue_slots = demand.queue_slots.max(tail.queue_slots);
            demand = demand
                .with_provider_observation_peak(tail.provider_scratch_bytes)
                .ok_or_else(observation_overflow)?;
            charge_region_tail_host_demand(&mut demand, tail).ok_or_else(observation_overflow)?;
        }
        Ok(demand)
    }

    pub(crate) fn resource_demand_with_tail(
        &self,
        tail: Option<runmat_accelerate_api::ProviderRegionTailResources>,
    ) -> Option<CandidateResourceDemand> {
        let mut demand = self.resource_demand();
        if let Some(tail) = tail {
            demand.queue_slots = demand.queue_slots.max(tail.queue_slots);
            demand = demand.with_provider_observation_peak(tail.provider_scratch_bytes)?;
            charge_region_tail_host_demand(&mut demand, tail)?;
        }
        Some(demand)
    }

    pub(crate) fn prepare(
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        invocation: &ImageSourceInvocation,
    ) -> ProviderExecutionResult<Option<Self>> {
        let Some(proof) = ImageArithmeticProof::for_selected(invocation) else {
            log::debug!("selected image physical preparation declined: source arithmetic proof");
            return Ok(None);
        };
        let Some(session) =
            ImageNormalizeStageSession::begin_selected(provider, input, invocation, proof)?
        else {
            log::debug!("selected image physical preparation declined: stage session");
            return Ok(None);
        };
        let Some(resources) = session.selected_physical_upper_bound() else {
            log::debug!("selected image physical preparation declined: physical upper bound");
            return Ok(None);
        };
        let provider_snapshot = provider.placement_resources();
        provider_snapshot
            .validate_typed()
            .map_err(ProviderExecutionError::from_source)?;
        if provider_snapshot.device_id != provider.device_id() || provider_snapshot.lost {
            log::debug!("selected image physical preparation declined: provider snapshot");
            return Ok(None);
        }
        Ok(Some(Self {
            session,
            semantic_identity: invocation.semantic_identity(),
            resources,
            provider_snapshot,
            provider_metadata_bytes: None,
            provider_power_workspace_bytes: None,
            _placement_host_preparation: None,
        }))
    }

    pub(crate) fn semantic_identity(&self) -> Digest {
        self.semantic_identity
    }

    pub(crate) fn input_view_identity(&self) -> runmat_accelerate_api::GpuHandleIdentity {
        self.session.input.view_identity()
    }

    /// Conservative complete-region resource envelope from the exact prepared
    /// hook sequence. Each new allocation and per-hook uniform is charged at
    /// most once. Retention deliberately uses the full sum until the VM's
    /// ordered live-out/prefix receipt can narrow it. The hooks execute
    /// serially, so total dispatch count is not concurrent queue occupancy.
    pub(crate) fn resource_demand(&self) -> CandidateResourceDemand {
        CandidateResourceDemand {
            cpu_millicores: 0,
            retained_bytes: ResourceByteBound::Known(self.resources.requested_bytes),
            scratch_bytes: ResourceByteBound::Known(0),
            queue_slots: u32::from(self.resources.dispatches != 0),
            provider_host_workspace: self
                .provider_metadata_bytes
                .and_then(|metadata| metadata.checked_add(self.provider_power_workspace_bytes?))
                .map(ResourceByteBound::Known),
            provider_input_stages: None,
            peak_live: Some(CandidatePeakLiveDemand {
                peak_bytes: self.resources.requested_bytes,
                already_live_bytes: 0,
            }),
        }
    }

    pub(super) fn set_provider_metadata_bytes(&mut self, bytes: u64) {
        self.provider_metadata_bytes = Some(bytes);
    }

    pub(super) fn set_provider_power_workspace_bytes(&mut self, bytes: u64) {
        self.provider_power_workspace_bytes = Some(bytes);
    }

    pub(super) fn finite_power_workspace_bytes(&self) -> Option<u64> {
        self.session
            .selected_power_plans
            .as_ref()?
            .provider_host_workspace_bytes()
    }

    pub(super) fn finite_metadata_registration_count(&self) -> Option<usize> {
        let means = self.session.selected_mean_plans.as_ref()?;
        if !means.admitted_casts_supported() {
            return None;
        }
        means
            .registered_output_count()?
            .checked_add(
                self.session
                    .selected_binary_plans
                    .as_ref()?
                    .registered_output_count(),
            )?
            .checked_add(
                self.session
                    .selected_real_plans
                    .as_ref()?
                    .registered_output_count(),
            )?
            .checked_add(
                self.session
                    .selected_power_plans
                    .as_ref()?
                    .registered_output_count()?,
            )
    }

    pub(super) fn finite_metadata_derived_view_count(&self) -> Option<usize> {
        2usize
            .checked_add(
                self.session
                    .selected_mean_plans
                    .as_ref()?
                    .alias_view_count(),
            )?
            .checked_add(
                self.session
                    .selected_real_plans
                    .as_ref()?
                    .alias_view_count(),
            )
    }

    /// The selector must consume the snapshot held by this same preparation.
    /// A concurrent provider resource/epoch change requires a new selection,
    /// never a detached reuse of an earlier physical estimate.
    pub(crate) fn matches_pre_effect_snapshot(
        &self,
        provider: &dyn AccelProvider,
    ) -> ProviderExecutionResult<bool> {
        self.session
            .input
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        let current = provider.placement_resources();
        current
            .validate_typed()
            .map_err(ProviderExecutionError::from_source)?;
        Ok(
            runmat_accelerate_api::same_provider_instance(self.session.provider, provider)
                && current == self.provider_snapshot,
        )
    }

    /// Once the first hook has run, this session's own allocations advance
    /// the provider resource epoch. Retain exact owner/source-liveness and
    /// device-loss checks; the original snapshot is only an admission input,
    /// never a post-effect equality condition.
    pub(crate) fn selected_owner_is_live(&self) -> ProviderExecutionResult<bool> {
        self.session
            .input
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        let current = self.session.provider.placement_resources();
        current
            .validate_typed()
            .map_err(ProviderExecutionError::from_source)?;
        Ok(current.device_id == self.provider_snapshot.device_id && !current.lost)
    }

    pub(crate) fn session_mut(&mut self) -> &mut ImageNormalizeStageSession {
        &mut self.session
    }
}

fn observation_overflow() -> Box<runmat_runtime::RuntimeError> {
    Box::new(
        runmat_runtime::build_runtime_error("image observation resource extent overflows")
            .with_identifier("RunMat:Placement:ResourceOverflow")
            .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
            .build(),
    )
}
