//! Physical completion evidence for source-ordered native MIR execution.
//! The native owner verifies traversal/publications and consumes its own right.
use super::*;
use runmat_accelerate_api::{AccelProvider, ProviderExecutionError, ProviderExecutionResult};
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::numeric_region::service::{
    NativeNumericRegionRight, NativeOriginalRegionObservation, SourceRegionRouteAudit,
};
use std::sync::Arc;

pub(super) struct NativeOriginalObservation {
    provider: &'static dyn AccelProvider,
    feedback: Box<LocalPlacementFeedback>,
    audit: SourceRegionRouteAudit,
    tail: super::super::image_stages::RegionTailAdmission,
    identity: (
        runmat_types::RegionId,
        runmat_runtime::native::NativeSiteRequest,
    ),
    _backing: Option<Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
}

impl NativeOriginalObservation {
    pub(super) fn admit(
        runtime: &RuntimeContext,
        provider: &'static dyn AccelProvider,
        right: &NativeNumericRegionRight,
        feedback: Option<Box<LocalPlacementFeedback>>,
    ) -> ProviderExecutionResult<Option<Box<dyn NativeOriginalRegionObservation>>> {
        let Some(feedback) = feedback else {
            return Ok(None);
        };
        if feedback.kind != runmat_execution::ExecutionCandidateKind::OriginalNativeRegion
            || !feedback.is_current()
            || !right.is_current(runtime)
            || feedback.signature.region != Some(right.identity().0)
        {
            return Ok(None);
        }
        let Some(audit) = super::super::image_stages::admit_region_audit(runtime, provider)? else {
            return Ok(None);
        };
        let Some(tail) = super::super::image_stages::admit_region_tail(runtime, provider)? else {
            return Ok(None);
        };
        let bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| ProviderExecutionError::from_source(HostAllocationError::Overflow))?;
        let retained = HostAllocationPlan::retained_metadata_bytes(bytes)
            .map_err(ProviderExecutionError::from_source)?;
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, retained).map_err(ProviderExecutionError::from_source)?,
        )
        .map_err(ProviderExecutionError::from_source)?;
        let backing = receipt
            .as_ref()
            .map(|receipt| receipt.claim_retained_metadata(runtime, bytes))
            .transpose()
            .map_err(ProviderExecutionError::from_source)?;
        Ok(Some(Box::new(Self {
            provider,
            feedback,
            audit,
            tail,
            identity: right.identity(),
            _backing: backing,
        })))
    }
}

impl NativeOriginalRegionObservation for NativeOriginalObservation {
    fn audit(&self) -> Option<SourceRegionRouteAudit> {
        Some(self.audit.clone())
    }

    fn complete(
        self: Box<Self>,
        runtime: RuntimeContext,
        right: NativeNumericRegionRight,
    ) -> Result<
        runmat_runtime::execution::RuntimeContinuationFuture<'static, ()>,
        runmat_runtime::RuntimeError,
    > {
        let admission = runtime.clone();
        runmat_runtime::execution::admit_native_continuation_with(&admission, || async move {
            if right.identity() != self.identity
                || !right.is_current(&runtime)
                || !self.feedback.is_current()
                || !self.audit.exact_single_provider()
            {
                return Ok(());
            }
            let snapshot = self.provider.placement_resources();
            if snapshot.device_id != self.provider.device_id() || snapshot.lost {
                return Ok(());
            }
            if self
                .provider
                .observe_region_queue_tail(self.provider.device_id(), self.tail.claim.clone())
                .await
                .is_err()
            {
                return Ok(());
            }
            if right.is_current(&runtime)
                && self.feedback.is_current()
                && self.audit.exact_single_provider()
            {
                self.feedback
                    .observe(duration_ns_saturating(right.started().elapsed()), true);
            }
            Ok(())
        })
    }
}
