//! Physical cell/face geometry only: no boundary law, material, remeshing or solver policy.
//! Affine Tet10 is admitted as geometry, not as a quadratic finite-volume field.

mod derive;
mod geometry;
#[cfg(test)]
mod tests;

use runmat_execution::Digest;
use runmat_meshing_core::{ElementOrder, PersistentEntityId, StableDigest};
use thiserror::Error;

use super::AnalysisProjection;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteVolumeProjectionPolicy {
    pub max_cells: usize,
    pub max_face_incidences: usize,
    pub max_boundary_faces: usize,
    pub max_scratch_bytes: usize,
    pub max_traversal_operations: usize,
}

impl Default for FiniteVolumeProjectionPolicy {
    fn default() -> Self {
        Self {
            max_cells: 1_000_000,
            max_face_incidences: 4_000_000,
            max_boundary_faces: 4_000_000,
            max_scratch_bytes: 1024 * 1024 * 1024,
            max_traversal_operations: 1_000_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniteVolumeReservation {
    pub cell_count: usize,
    pub face_incidence_count: usize,
    pub boundary_face_count: usize,
    /// Conservative peak for this projection's output and auxiliary owned storage.
    pub scratch_bytes: usize,
    /// Conservative comparison/visit bound including ordered-map and sort work.
    pub traversal_operations: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FiniteVolumeTopology {
    pub mesh_digest: StableDigest,
    pub projection_digest: Digest,
    pub geometry_digest: Digest,
    pub policy: FiniteVolumeProjectionPolicy,
    pub reservation: FiniteVolumeReservation,
    pub cells: Vec<FiniteVolumeCell>,
    pub faces: Vec<FiniteVolumeFace>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FiniteVolumeCell {
    pub element_identity: StableDigest,
    pub element_id: u64,
    pub source_element_index: usize,
    pub geometry_order: ElementOrder,
    pub region: PersistentEntityId,
    pub volume_m3: f64,
    pub centroid_m: [f64; 3],
    pub faces: [FiniteVolumeIncidence; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniteVolumeIncidence {
    pub face_index: usize,
    /// +1 at owner, -1 at neighbor, multiplying the face area vector.
    pub sign: i8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FiniteVolumeFace {
    pub corner_identities: [StableDigest; 3],
    pub centroid_m: [f64; 3],
    pub owner_area_vector_m2: [f64; 3],
    pub area_m2: runmat_types::physical::Area,
    pub owner: usize,
    pub neighbor: Option<usize>,
    /// Original canonical face, including interface roles when present. Exterior is mandatory.
    /// Role and exact provenance remain accessible through the source projection, unchanged.
    pub boundary_source: Option<FiniteVolumeBoundarySource>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniteVolumeBoundarySource {
    pub projection_boundary_index: usize,
    pub face_id: u64,
    pub face_identity: StableDigest,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FiniteVolumeProjectionError {
    #[error("finite-volume geometry reservation overflow")]
    Overflow,
    #[error("finite-volume geometry reservation exceeds {0}")]
    Limit(&'static str),
    #[error("invalid canonical finite-volume geometry: {0}")]
    Invalid(&'static str),
    #[error("curved Tet10 finite-volume geometry is not supported (element {0})")]
    CurvedTet10(u64),
    #[error("finite-volume projection does not bind exactly to the supplied source")]
    SourceMismatch,
}

impl FiniteVolumeTopology {
    /// Count-only preflight over borrowed raw source records, before projection allocation.
    /// This grants no validity authority; derive/validate_for still authenticate geometry.
    pub fn reserve_mesh(
        source: &runmat_meshing_core::SolverMeshArtifact,
        policy: FiniteVolumeProjectionPolicy,
    ) -> Result<FiniteVolumeReservation, FiniteVolumeProjectionError> {
        derive::reserve_mesh(source, policy)
    }

    pub fn derive(
        source: &AnalysisProjection,
        policy: FiniteVolumeProjectionPolicy,
    ) -> Result<Self, FiniteVolumeProjectionError> {
        derive::derive(source, policy)
    }

    /// Revalidate externally constructible/mutable records before using geometry in a kernel.
    /// This is geometry validation, not a physics solve or a repaired source mesh.
    pub fn validate_for(
        &self,
        source: &AnalysisProjection,
    ) -> Result<(), FiniteVolumeProjectionError> {
        if self != &Self::derive(source, self.policy)? {
            return Err(FiniteVolumeProjectionError::SourceMismatch);
        }
        Ok(())
    }

    pub fn reserve(
        source: &AnalysisProjection,
        policy: FiniteVolumeProjectionPolicy,
    ) -> Result<FiniteVolumeReservation, FiniteVolumeProjectionError> {
        derive::reserve(source, policy)
    }
}
