//! All potential face-pair couplings, admitted before coordinate/sparse allocation.
//! This support is independent of the current active set; it is not whole-task admission.
use super::*;
use crate::algebra::{SparsePattern, SparsePatternError};
use thiserror::Error;
#[derive(Clone, Copy, Debug)]
pub struct ContactSupportBudget {
    pub maximum_face_pairs: usize,
    pub maximum_coordinates: usize,
}
#[derive(Debug, Error)]
pub enum ContactSupportError {
    #[error("contact symbolic support exceeds its pair/coordinate budget")]
    Budget,
    #[error("contact symbolic node map or dimensions are invalid")]
    InvalidMap,
    #[error("contact symbolic support cancelled")]
    Cancelled,
    #[error(transparent)]
    Sparse(#[from] SparsePatternError),
}
pub struct ContactTangentSupport {
    pattern: SparsePattern,
    face_pairs: usize,
    coordinates: usize,
}
impl ContactTangentSupport {
    /// Node rows follow the original projection inventory, supplied by shared structural layout.
    /// All three translations are coupled; no SPD/symmetry claim follows from this graph.
    pub fn derive(
        surfaces: &CanonicalContactSurfaces<'_>,
        node_rows: &[[usize; 3]],
        dof_count: usize,
        budget: ContactSupportBudget,
    ) -> Result<Self, ContactSupportError> {
        if crate::is_fea_cancelled() {
            return Err(ContactSupportError::Cancelled);
        }
        let projection = surfaces.projection();
        if node_rows.len() != projection.nodes.len()
            || node_rows.iter().flatten().any(|r| *r >= dof_count)
        {
            return Err(ContactSupportError::InvalidMap);
        }
        let face_pairs = surfaces
            .primary_face_indices()
            .len()
            .checked_mul(surfaces.secondary_face_indices().len())
            .ok_or(ContactSupportError::Budget)?;
        if face_pairs > budget.maximum_face_pairs {
            return Err(ContactSupportError::Budget);
        }
        let mut coordinates = 0usize;
        for &primary in surfaces.primary_face_indices() {
            for &secondary in surfaces.secondary_face_indices() {
                if crate::is_fea_cancelled() {
                    return Err(ContactSupportError::Cancelled);
                }
                let nodes = projection.boundary_faces[primary]
                    .node_indices
                    .len()
                    .checked_add(projection.boundary_faces[secondary].node_indices.len())
                    .ok_or(ContactSupportError::Budget)?;
                let rows = nodes.checked_mul(3).ok_or(ContactSupportError::Budget)?;
                coordinates = coordinates
                    .checked_add(rows.checked_mul(rows).ok_or(ContactSupportError::Budget)?)
                    .ok_or(ContactSupportError::Budget)?;
                if coordinates > budget.maximum_coordinates {
                    return Err(ContactSupportError::Budget);
                }
            }
        }
        let mut entries = Vec::new();
        entries.try_reserve_exact(coordinates).map_err(|_| {
            SparsePatternError::StorageAllocation {
                elements: coordinates,
            }
        })?;
        for &primary in surfaces.primary_face_indices() {
            for &secondary in surfaces.secondary_face_indices() {
                if crate::is_fea_cancelled() {
                    return Err(ContactSupportError::Cancelled);
                }
                let nodes = || {
                    projection.boundary_faces[primary]
                        .node_indices
                        .iter()
                        .chain(&projection.boundary_faces[secondary].node_indices)
                };
                for &a in nodes() {
                    for &row in &node_rows[a] {
                        for &b in nodes() {
                            for &column in &node_rows[b] {
                                entries.push((row, column));
                            }
                        }
                    }
                }
            }
        }
        if crate::is_fea_cancelled() {
            return Err(ContactSupportError::Cancelled);
        }
        let pattern = SparsePattern::from_coordinates(dof_count, dof_count, entries)?;
        if crate::is_fea_cancelled() {
            return Err(ContactSupportError::Cancelled);
        }
        Ok(Self {
            pattern,
            face_pairs,
            coordinates,
        })
    }
    pub fn pattern(&self) -> &SparsePattern {
        &self.pattern
    }
    pub fn face_pair_count(&self) -> usize {
        self.face_pairs
    }
    pub fn coordinate_count(&self) -> usize {
        self.coordinates
    }
    /// Requested CSR heap, including its shared owners; no allocator, codec,
    /// borrowed source or solver allowance.
    pub fn resident_bytes(&self) -> Result<u64, ContactSupportError> {
        self.pattern.retained_storage_bytes().map_err(Into::into)
    }
    pub fn construction_resident_bytes(&self) -> Result<u64, ContactSupportError> {
        let scratch = self
            .coordinates
            .checked_mul(std::mem::size_of::<(usize, usize)>())
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(ContactSupportError::Budget)?;
        self.resident_bytes()?
            .checked_add(scratch)
            .ok_or(ContactSupportError::Budget)
    }
}
