//! Move-owned selected image preparation. A resource proposal cannot be
//! detached from the recipes and input lease whose requests it bounds.

use super::*;
use runmat_execution::{
    CandidatePeakLiveDemand, CandidateResourceDemand, Digest, ProviderResourceSnapshot,
    ResourceByteBound,
};

pub(crate) struct PreparedImageStages {
    pub(super) session: ImageNormalizeStageSession,
    semantic_identity: Digest,
    resources: resources::ImageStagePhysicalUpperBound,
    provider_snapshot: ProviderResourceSnapshot,
}

impl PreparedImageStages {
    pub(crate) fn prepare(
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        invocation: &ImageSourceInvocation,
    ) -> ProviderExecutionResult<Option<Self>> {
        let Some(proof) = ImageArithmeticProof::for_selected(invocation) else {
            return Ok(None);
        };
        let Some(session) =
            ImageNormalizeStageSession::begin_selected(provider, input, invocation, proof)?
        else {
            return Ok(None);
        };
        let Some(resources) = session.selected_physical_upper_bound() else {
            return Ok(None);
        };
        let provider_snapshot = provider.placement_resources();
        provider_snapshot
            .validate_typed()
            .map_err(ProviderExecutionError::from_source)?;
        if provider_snapshot.device_id != provider.device_id() || provider_snapshot.lost {
            return Ok(None);
        }
        Ok(Some(Self {
            session,
            semantic_identity: invocation.semantic_identity(),
            resources,
            provider_snapshot,
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
            provider_host_workspace: None,
            provider_input_stages: None,
            peak_live: Some(CandidatePeakLiveDemand {
                peak_bytes: self.resources.requested_bytes,
                already_live_bytes: 0,
            }),
        }
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
