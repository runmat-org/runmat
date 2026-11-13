//! Numerical ownership of accepted face reuse; persistence belongs to the host.
use super::{ExactFacePartitionError, ExactFacePartitionErrorKind};
use crate::{ExactBoundaryNodeCoordinates, ExactFaceBoundary, ExactFaceMesh};
use runmat_meshing_core::MeshingCancellationSignal;

/// Construction is restricted to the numerical pipeline after chart acceptance.
/// This is an in-process producer capability, not a deserializable proof.
pub struct AcceptedExactFace<'a> {
    mesh: &'a ExactFaceMesh,
}

impl<'a> AcceptedExactFace<'a> {
    pub(super) fn new(mesh: &'a ExactFaceMesh) -> Self {
        Self { mesh }
    }

    pub fn mesh(&self) -> &'a ExactFaceMesh {
        self.mesh
    }

    /// Storage required before allocating the borrowed encoding dictionaries.
    pub fn encoding_preparation_storage_bytes(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<u64, ExactFacePartitionError> {
        crate::ExactFacePartitionEncoding::single_preparation_storage_bytes(self.mesh, cancellation)
            .map_err(|error| super::failure(ExactFacePartitionErrorKind::Batch(error.kind), error))
    }

    /// Prepare once under the host's reservation, then reuse for profiling and encoding.
    pub fn prepare_encoding(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<crate::ExactFacePartitionEncoding<'a>, ExactFacePartitionError> {
        crate::ExactFacePartitionEncoding::single(self.mesh, cancellation)
            .map_err(|error| super::failure(ExactFacePartitionErrorKind::Batch(error.kind), error))
    }

    /// None means this optional cache entry exceeds its byte allowance.
    pub fn encoding_profile(
        &self,
        maximum_encoded_bytes: usize,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Option<runmat_canonical_codec::CanonicalEncodingProfile>, ExactFacePartitionError>
    {
        crate::surface_mesh::profile_accepted_face(self.mesh, maximum_encoded_bytes, cancellation)
            .map_err(|error| super::failure(ExactFacePartitionErrorKind::Batch(error.kind), error))
    }

    /// Borrow the original mesh while encoding; never clone its node/triangle inventory.
    pub fn encode(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, ExactFacePartitionError> {
        self.encode_bounded(256 * 1024 * 1024, cancellation)
    }

    pub fn encode_bounded(
        &self,
        maximum_encoded_bytes: usize,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, ExactFacePartitionError> {
        crate::surface_mesh::encode_accepted_face(self.mesh, maximum_encoded_bytes, cancellation)
            .map_err(|error| super::failure(ExactFacePartitionErrorKind::Batch(error.kind), error))
    }
}

/// The adapter must bind the complete numerical context and original local boundary.
/// Restored meshes still pass the ordinary partition and shared-curve admission.
pub trait ExactFaceMeshReuse {
    fn restore(
        &mut self,
        boundary: &ExactFaceBoundary,
        coordinates: &ExactBoundaryNodeCoordinates<'_>,
    ) -> Result<Option<ExactFaceMesh>, ExactFacePartitionError>;

    fn retain(
        &mut self,
        boundary: &ExactFaceBoundary,
        coordinates: &ExactBoundaryNodeCoordinates<'_>,
        accepted: AcceptedExactFace<'_>,
    ) -> Result<(), ExactFacePartitionError>;
}

pub(super) struct NoFaceReuse;
impl ExactFaceMeshReuse for NoFaceReuse {
    fn restore(
        &mut self,
        _: &ExactFaceBoundary,
        _: &ExactBoundaryNodeCoordinates<'_>,
    ) -> Result<Option<ExactFaceMesh>, ExactFacePartitionError> {
        Ok(None)
    }

    fn retain(
        &mut self,
        _: &ExactFaceBoundary,
        _: &ExactBoundaryNodeCoordinates<'_>,
        _: AcceptedExactFace<'_>,
    ) -> Result<(), ExactFacePartitionError> {
        Ok(())
    }
}
