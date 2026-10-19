//! Retain the first interruption across domain-to-codec adapters.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
#[derive(Clone, Copy)]
enum Source<'a> {
    Meshing(&'a dyn MeshingCancellationSignal),
    Codec(&'a dyn runmat_canonical_codec::CanonicalCodecControl),
}
pub(in crate::cdt::volume_mesh) struct Control<'a> {
    source: Source<'a>,
    interrupted: AtomicBool,
}
impl<'a> Control<'a> {
    pub fn new(source: &'a dyn MeshingCancellationSignal) -> Self {
        Self {
            source: Source::Meshing(source),
            interrupted: AtomicBool::new(false),
        }
    }
    pub fn from_codec(source: &'a dyn runmat_canonical_codec::CanonicalCodecControl) -> Self {
        Self {
            source: Source::Codec(source),
            interrupted: AtomicBool::new(false),
        }
    }
    pub fn classify(
        &self,
        mut error: DelaunayVolumeMeshCodecError,
    ) -> DelaunayVolumeMeshCodecError {
        if self.interrupted.load(Ordering::Relaxed) {
            error.kind = DelaunayVolumeMeshCodecErrorKind::Cancelled;
        }
        error
    }
}
impl runmat_canonical_codec::CanonicalCodecControl for Control<'_> {
    #[track_caller]
    fn is_cancelled(&self) -> bool {
        if self.interrupted.load(Ordering::Relaxed) {
            return true;
        }
        let cancelled = match self.source {
            Source::Meshing(source) => source.is_cancelled(),
            Source::Codec(source) => source.is_cancelled(),
        };
        if cancelled {
            self.interrupted.store(true, Ordering::Relaxed);
        }
        cancelled
    }
}
