//! Cancel independently inside vertex, protected-segment and triangle searches.
use runmat_meshing_core::MeshingCancellationSignal;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct CancelDuringCandidateWalk {
    probes: AtomicUsize,
    cancel_at: usize,
}
impl CancelDuringCandidateWalk {
    pub(super) fn new(cancel_at: usize) -> Self {
        Self {
            probes: AtomicUsize::new(0),
            cancel_at,
        }
    }
    pub(super) fn observed(&self) -> usize {
        self.probes.load(Ordering::Relaxed)
    }
}
impl MeshingCancellationSignal for CancelDuringCandidateWalk {
    fn is_cancelled(&self) -> bool {
        std::path::Path::new(std::panic::Location::caller().file())
            .ends_with(std::path::Path::new("insertion/candidate.rs"))
            && self.probes.fetch_add(1, Ordering::Relaxed) + 1 == self.cancel_at
    }
}
