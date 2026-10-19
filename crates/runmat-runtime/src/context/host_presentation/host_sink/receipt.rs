//! Completion of a host-result readback route. Awaited provider downloads
//! have finished before `finish`; dropping this owner abandons feedback.

use crate::context::{RuntimeContext, RuntimeManagedWholeWorkSelection, RuntimeWholeWorkSelection};
use runmat_execution::WholeWorkObservationOutcome;
use runmat_time::Instant;

pub(crate) struct HostSinkReceipt {
    runtime: RuntimeContext,
    selected: HostSinkSelection,
    started: Instant,
}

enum HostSinkSelection {
    Portable(RuntimeWholeWorkSelection),
    Managed(RuntimeManagedWholeWorkSelection),
}

impl HostSinkReceipt {
    pub(crate) fn new(runtime: RuntimeContext, selected: RuntimeWholeWorkSelection) -> Self {
        Self {
            runtime,
            selected: HostSinkSelection::Portable(selected),
            started: Instant::now(),
        }
    }

    /// A managed host route retains its precharged observation lease through
    /// the awaited readback and source publication. No deferred source or
    /// portable profile object is allocated at completion.
    pub(crate) fn new_managed(
        runtime: RuntimeContext,
        selected: RuntimeManagedWholeWorkSelection,
    ) -> Self {
        Self {
            runtime,
            selected: HostSinkSelection::Managed(selected),
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
            let elapsed = runmat_time::duration_ns_saturating(self.started.elapsed()).max(1);
            let outcome = if published {
                WholeWorkObservationOutcome::Success
            } else {
                WholeWorkObservationOutcome::OperationalFailure
            };
            match &self.selected {
                HostSinkSelection::Portable(selected) => {
                    let _ = service.observe_selected_whole_work(selected, elapsed, outcome);
                }
                HostSinkSelection::Managed(selected) if selected.observation_is_current() => {
                    let _ = service.observe_managed_whole_work(selected, elapsed, outcome);
                }
                HostSinkSelection::Managed(_) => {}
            }
        }
    }
}
