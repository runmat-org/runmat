//! Distinct original-bytecode route. It never authorizes a host fallback or
//! executes source instructions; VM owns those and their ordered publications.
use super::*;
use crate::placement::LocalPlacementFeedback;
use runmat_accelerate_api::observation::VmRegionRouteAudit;
use runmat_runtime::context::VmFusionCompletionRight;
use runmat_time::{duration_ns_saturating, Instant};
use std::sync::Arc;

pub struct ImageVmOriginalRoute {
    provider: &'static dyn AccelProvider,
    device_id: u32,
    feedback: Option<Box<LocalPlacementFeedback>>,
    right: VmFusionCompletionRight,
    audit: Arc<VmRegionRouteAudit>,
    started: Instant,
}

impl ImageVmOriginalRoute {
    pub(super) fn new(
        provider: &'static dyn AccelProvider,
        feedback: Option<Box<LocalPlacementFeedback>>,
        right: VmFusionCompletionRight,
        started: Instant,
    ) -> Option<Self> {
        let access = provider.wgpu_device_access()?.clone();
        Some(Self {
            provider,
            device_id: provider.device_id(),
            feedback,
            right,
            audit: VmRegionRouteAudit::new(provider, access),
            started,
        })
    }

    pub fn audit(&self) -> Arc<VmRegionRouteAudit> {
        Arc::clone(&self.audit)
    }

    /// Called only after the VM verified every source store and advanced past
    /// the installed region. The queue tail is optional measurement work: an
    /// error, cancellation, owner change or unsupported driver withholds the
    /// sample without changing completed source semantics.
    pub async fn complete(self: Box<Self>) {
        let Some(feedback) = self.feedback else {
            return;
        };
        if !self.audit.exact_single_provider()
            || !feedback.is_current()
            || feedback.runtime.is_cancelled()
        {
            return;
        }
        let Some(installation) = self.right.completion_lease(&feedback.runtime) else {
            return;
        };
        let resources = self.provider.placement_resources();
        if resources.device_id != self.device_id || resources.lost {
            return;
        }
        if self
            .provider
            .observe_vm_region_queue_tail(self.device_id)
            .await
            .is_err()
        {
            return;
        }
        if feedback.runtime.is_cancelled() || !feedback.is_current() || !installation.is_current() {
            return;
        }
        feedback.observe(duration_ns_saturating(self.started.elapsed()), true);
        drop(installation);
    }
}
