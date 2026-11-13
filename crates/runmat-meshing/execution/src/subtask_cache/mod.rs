//! Face-local numerical reuse. Host adapters own persistence and capacity policy.
use runmat_geometry_core::{GeometryEvaluationControl, GeometryEvaluationError};
use runmat_meshing_core::StableDigest;
use runmat_meshing_surface::{
    AcceptedExactFace, ExactBoundaryNodeCoordinates, ExactFaceBoundary, ExactFaceMesh,
    ExactFaceMeshReuse, ExactFacePartitionError, ExactFacePartitionErrorKind,
};
use std::sync::Arc;

mod codec_error;
pub(crate) mod diagnostic_chunks;
mod face_records;
pub(crate) mod identity;
pub(crate) mod record;
mod record_chunks;
mod reuse;
pub(crate) use reuse::SurfaceFaceReuse;

#[derive(Debug)]
pub enum MeshingSubtaskCacheError {
    Control(GeometryEvaluationError),
    Storage(String),
}

#[derive(Debug)]
pub struct MeshingSubtaskCacheEntry<'a> {
    pub bytes: Vec<u8>,
    pub storage: runmat_geometry_core::GeometryAllocationReservation<'a>,
    /// Admitted before allocating either the encoded record or its decoded carrier.
    pub materialization_resident_bytes: u64,
}

/// Opaque, bounded byte persistence. Keys, admissibility and face authority stay in meshing.
pub trait MeshingSubtaskCacheStore: Send + Sync + std::fmt::Debug {
    fn implementation_identity(&self) -> StableDigest;
    /// Diagnostic provenance is volume-specific; it never authorizes stage reuse.
    fn volume_implementation_identity(&self) -> Option<StableDigest> {
        None
    }
    fn maximum_entry_bytes(&self) -> u64;
    fn admit(
        &self,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<bool, MeshingSubtaskCacheError> {
        Ok(true)
    }
    fn read<'a>(
        &self,
        key: StableDigest,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Option<MeshingSubtaskCacheEntry<'a>>, MeshingSubtaskCacheError>;
    fn write(
        &self,
        key: StableDigest,
        bytes: &[u8],
        materialization_resident_bytes: u64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<bool, MeshingSubtaskCacheError>;
}

fn invalid(reason: impl Into<String>) -> ExactFacePartitionError {
    ExactFacePartitionError {
        kind: ExactFacePartitionErrorKind::Reuse,
        reason: reason.into(),
    }
}
fn control_error(error: GeometryEvaluationError) -> ExactFacePartitionError {
    ExactFacePartitionError {
        kind: ExactFacePartitionErrorKind::Coordinates(error.kind),
        reason: error.reason,
    }
}
fn store_error(error: MeshingSubtaskCacheError) -> ExactFacePartitionError {
    match error {
        MeshingSubtaskCacheError::Control(error) => control_error(error),
        MeshingSubtaskCacheError::Storage(reason) => invalid(reason),
    }
}

#[cfg(test)]
pub(crate) mod tests;
