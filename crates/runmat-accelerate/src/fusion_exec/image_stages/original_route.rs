//! Distinct original-bytecode route. It never authorizes a host fallback or
//! executes source instructions; VM owns those and their ordered publications.
use super::*;
use crate::placement::LocalPlacementFeedback;
use runmat_accelerate_api::observation::SourceRegionRouteAudit;
use runmat_runtime::context::VmFusionCompletionRight;
use runmat_time::{duration_ns_saturating, Instant};

pub struct ImageVmOriginalRoute {
    provider: &'static dyn AccelProvider,
    device_id: u32,
    feedback: Option<Box<LocalPlacementFeedback>>,
    right: VmFusionCompletionRight,
    audit: SourceRegionRouteAudit,
    started: Instant,
    tail_control: Option<super::prepared::RegionTailAdmission>,
}

impl ImageVmOriginalRoute {
    pub(super) fn new(
        runtime: &runmat_runtime::context::RuntimeContext,
        provider: &'static dyn AccelProvider,
        feedback: Option<Box<LocalPlacementFeedback>>,
        right: VmFusionCompletionRight,
        started: Instant,
    ) -> Option<Self> {
        let audit = super::prepared::admit_region_audit(runtime, provider).ok()??;
        let tail_control = match &feedback {
            Some(feedback) => {
                super::prepared::admit_region_tail(&feedback.runtime, provider).ok()?
            }
            None => None,
        };
        Some(Self {
            provider,
            device_id: provider.device_id(),
            feedback,
            right,
            audit,
            started,
            tail_control,
        })
    }

    pub fn audit(&self) -> SourceRegionRouteAudit {
        self.audit.clone()
    }

    /// Called only after the VM verified every source store and advanced past
    /// the installed region. The queue tail is optional measurement work: an
    /// error, cancellation, owner change or unsupported driver withholds the
    /// sample without changing completed source semantics.
    pub async fn complete(self: Box<Self>) {
        let Some(feedback) = self.feedback else {
            return;
        };
        let Some(tail_control) = self.tail_control else {
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
            .observe_region_queue_tail(self.device_id, tail_control.claim)
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
