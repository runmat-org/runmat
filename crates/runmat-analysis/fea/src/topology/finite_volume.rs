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
/// Each limit is `None` when unbounded: a cap is a count, never a sentinel value.
pub struct FiniteVolumeProjectionPolicy {
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub max_cells: Option<usize>,
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub max_face_incidences: Option<usize>,
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub max_boundary_faces: Option<usize>,
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub max_scratch_bytes: Option<usize>,
    #[serde(default)]
    #[serde(with = "runmat_types::wire_integer::option")]
    pub max_traversal_operations: Option<usize>,
}

/// No fixed size caps by default: derivation memory is charged to the calling phase's
/// resident admission, which the authored resource budget bounds.
impl Default for FiniteVolumeProjectionPolicy {
    fn default() -> Self {
        Self {
            max_cells: None,
            max_face_incidences: None,
            max_boundary_faces: None,
            max_scratch_bytes: None,
            max_traversal_operations: None,
        }
    }
}

impl FiniteVolumeProjectionPolicy {
    /// Limits that admit exactly this projection's complete cell complex. Work and scratch
    /// are governed by the calling phase's resident admission, not by a fixed cell cap.
    pub fn covering(source: &AnalysisProjection) -> Self {
        Self {
            max_cells: Some(source.elements.len()),
            max_face_incidences: Some(source.elements.len().saturating_mul(4)),
            max_boundary_faces: Some(source.boundary_faces.len()),
            max_scratch_bytes: None,
            max_traversal_operations: None,
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

/// The source volume elements that become cells. A region subset makes every face it shares
/// with an excluded region an exterior face of the cell complex; that face must be an original
/// conformal-interface face, so its exact CAD provenance remains the boundary authority.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FiniteVolumeCellSelection {
    AllElements,
    /// Sorted, unique mesh region identities.
    Regions(Vec<PersistentEntityId>),
}

impl FiniteVolumeCellSelection {
    pub fn regions(
        mut regions: Vec<PersistentEntityId>,
    ) -> Result<Self, FiniteVolumeProjectionError> {
        regions.sort();
        regions.dedup();
        if regions.is_empty() {
            return Err(FiniteVolumeProjectionError::Invalid(
                "empty region selection",
            ));
        }
        Ok(Self::Regions(regions))
    }
    pub fn includes(&self, region: &PersistentEntityId) -> bool {
        match self {
            Self::AllElements => true,
            Self::Regions(regions) => regions.binary_search(region).is_ok(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FiniteVolumeTopology {
    pub mesh_digest: StableDigest,
    pub projection_digest: Digest,
    pub geometry_digest: Digest,
    pub policy: FiniteVolumeProjectionPolicy,
    pub selection: FiniteVolumeCellSelection,
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
        derive::derive(source, policy, FiniteVolumeCellSelection::AllElements)
    }

    /// Cells of the selected mesh regions only, such as the fluid of a conjugate model.
    pub fn derive_selected(
        source: &AnalysisProjection,
        policy: FiniteVolumeProjectionPolicy,
        selection: FiniteVolumeCellSelection,
    ) -> Result<Self, FiniteVolumeProjectionError> {
        derive::derive(source, policy, selection)
    }

    /// Revalidate externally constructible/mutable records before using geometry in a kernel.
    /// This is geometry validation, not a physics solve or a repaired source mesh.
    pub fn validate_for(
        &self,
        source: &AnalysisProjection,
    ) -> Result<(), FiniteVolumeProjectionError> {
        if self != &Self::derive_selected(source, self.policy, self.selection.clone())? {
            return Err(FiniteVolumeProjectionError::SourceMismatch);
        }
        Ok(())
    }

    /// Count-only reservation for the cells `selection` admits; the boundary-face count is
    /// the whole projection's, an upper bound for any region subset.
    pub fn reserve(
        source: &AnalysisProjection,
        policy: FiniteVolumeProjectionPolicy,
        selection: &FiniteVolumeCellSelection,
    ) -> Result<FiniteVolumeReservation, FiniteVolumeProjectionError> {
        derive::reserve(source, policy, selection)
    }
}
