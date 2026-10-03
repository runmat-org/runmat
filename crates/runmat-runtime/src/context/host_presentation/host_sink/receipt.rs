//! Completion of a host-result readback route. Awaited provider downloads
//! have finished before `finish`; dropping this owner abandons feedback.

use crate::context::{RuntimeContext, RuntimeWholeWorkSelection};
use runmat_execution::PlacementFeedback;
use runmat_time::Instant;

pub(crate) struct HostSinkReceipt {
    runtime: RuntimeContext,
    selected: RuntimeWholeWorkSelection,
    started: Instant,
}

impl HostSinkReceipt {
    pub(super) fn new(runtime: RuntimeContext, selected: RuntimeWholeWorkSelection) -> Self {
        Self {
            runtime,
            selected,
            started: Instant::now(),
        }
    }

    /// Call only after the awaited readbacks and final host value are complete.
    /// Dropping before this point abandons the sample and probe lease.
    pub(crate) fn finish(self, published: bool) {
        if self.runtime.is_cancelled() {
            return;
        }
        if let Some(service) = self.runtime.service_ports().placement() {
            let _ = service.observe(PlacementFeedback {
                signature: self.selected.decision.signature.clone(),
                candidate: self.selected.decision.selected.candidate.clone(),
                kind: self.selected.decision.selected.kind,
                total_elapsed_ns: runmat_time::duration_ns_saturating(self.started.elapsed())
                    .max(1),
                succeeded: published,
            });
        }
    }
}
