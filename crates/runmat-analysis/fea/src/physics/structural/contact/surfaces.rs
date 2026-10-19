//! Borrow the shared projection and authored interface; retain only canonical face indexes.
use crate::topology::{
    AnalysisProjection, BoundarySearchError, OrientedBoundarySurface, ProjectedBoundaryFace,
};
use runmat_analysis_core::{
    AnalysisInterface, AnalysisInterfaceKind, EntityDimension, RegionTarget,
};
use runmat_meshing_core::BoundaryFaceRole;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ContactSurfaceError {
    #[error("contact surface selection cancelled")]
    Cancelled,
    #[error("contact needs a contact interface and two face targets")]
    InvalidInterface,
    #[error("contact target resolves to no physical boundary faces")]
    EmptySurface,
    #[error("contact surfaces overlap or have conflicting side roles")]
    SurfaceConflict,
    #[error("contact surface inventory exceeds addressable storage")]
    Capacity,
    #[error("contact face selection exceeds its declared inventory bounds")]
    Budget,
    #[error(transparent)]
    Geometry(#[from] BoundarySearchError),
}

pub struct CanonicalContactSurfaces<'a> {
    projection: &'a AnalysisProjection,
    interface: &'a AnalysisInterface,
    primary: Vec<usize>,
    secondary: Vec<usize>,
}
pub(super) fn counts(
    projection: &AnalysisProjection,
    interface: &AnalysisInterface,
) -> Result<[usize; 2], ContactSurfaceError> {
    if crate::is_fea_cancelled() {
        return Err(ContactSurfaceError::Cancelled);
    }
    if !matches!(interface.kind, AnalysisInterfaceKind::Contact(_))
        || interface.primary.entity_dimension != EntityDimension::Face
        || interface.secondary.entity_dimension != EntityDimension::Face
    {
        return Err(ContactSurfaceError::InvalidInterface);
    }
    let mut counts = [0_usize; 2];
    for face in &projection.boundary_faces {
        if crate::is_fea_cancelled() {
            return Err(ContactSurfaceError::Cancelled);
        }
        for (index, target) in [&interface.primary, &interface.secondary]
            .iter()
            .enumerate()
        {
            if matches(face, target) {
                counts[index] = counts[index]
                    .checked_add(1)
                    .ok_or(ContactSurfaceError::Capacity)?;
            }
        }
    }
    if counts.contains(&0) {
        return Err(ContactSurfaceError::EmptySurface);
    }
    Ok(counts)
}
fn indexes(
    projection: &AnalysisProjection,
    target: &RegionTarget,
    count: usize,
    primary: bool,
) -> Result<Vec<usize>, ContactSurfaceError> {
    let mut indexes = Vec::new();
    indexes
        .try_reserve_exact(count)
        .map_err(|_| ContactSurfaceError::Capacity)?;
    for (index, face) in projection.boundary_faces.iter().enumerate() {
        if crate::is_fea_cancelled() {
            return Err(ContactSurfaceError::Cancelled);
        }
        if !matches(face, target) {
            continue;
        }
        if face.role == BoundaryFaceRole::ConformalInterface
            || (primary && face.role == BoundaryFaceRole::ContactSecondary)
            || (!primary && face.role == BoundaryFaceRole::ContactPrimary)
        {
            return Err(ContactSurfaceError::SurfaceConflict);
        }
        indexes.push(index);
    }
    indexes.sort_unstable_by_key(|&index| projection.boundary_faces[index].stable_identity);
    Ok(indexes)
}
impl<'a> CanonicalContactSurfaces<'a> {
    /// Requested heap for both index inventories; excludes borrowed input, stack and allocator overhead.
    pub fn resident_bytes(
        projection: &AnalysisProjection,
        interface: &AnalysisInterface,
    ) -> Result<u64, ContactSurfaceError> {
        let [a, b] = counts(projection, interface)?;
        let count = a.checked_add(b).ok_or(ContactSurfaceError::Capacity)?;
        crate::topology::storage::slots::<usize>(count).map_err(|_| ContactSurfaceError::Capacity)
    }
    pub fn derive(
        projection: &'a AnalysisProjection,
        interface: &'a AnalysisInterface,
    ) -> Result<Self, ContactSurfaceError> {
        Self::derive_bounded(projection, interface, usize::MAX, usize::MAX)
    }
    pub fn derive_bounded(
        projection: &'a AnalysisProjection,
        interface: &'a AnalysisInterface,
        maximum_primary_faces: usize,
        maximum_secondary_faces: usize,
    ) -> Result<Self, ContactSurfaceError> {
        let [a, b] = counts(projection, interface)?;
        if a > maximum_primary_faces || b > maximum_secondary_faces {
            return Err(ContactSurfaceError::Budget);
        }
        crate::topology::storage::slots::<usize>(
            a.checked_add(b).ok_or(ContactSurfaceError::Capacity)?,
        )
        .map_err(|_| ContactSurfaceError::Capacity)?;
        let primary = indexes(projection, &interface.primary, a, true)?;
        let secondary = indexes(projection, &interface.secondary, b, false)?;
        if primary.iter().any(|index| {
            secondary
                .binary_search_by_key(&projection.boundary_faces[*index].stable_identity, |&i| {
                    projection.boundary_faces[i].stable_identity
                })
                .is_ok()
        }) {
            return Err(ContactSurfaceError::SurfaceConflict);
        }
        if crate::is_fea_cancelled() {
            return Err(ContactSurfaceError::Cancelled);
        }
        Ok(Self {
            projection,
            interface,
            primary,
            secondary,
        })
    }
    pub(super) fn projection(&self) -> &'a AnalysisProjection {
        self.projection
    }
    pub fn interface(&self) -> &'a AnalysisInterface {
        self.interface
    }
    pub fn primary_face_indices(&self) -> &[usize] {
        &self.primary
    }
    pub fn secondary_face_indices(&self) -> &[usize] {
        &self.secondary
    }
    /// Immutable primary geometry for this explicit current configuration.
    /// Whole-task admission and physical point/frame lineage remain the procedure's authority.
    pub fn primary_search_index(
        &self,
        positions_m: &[[f64; 3]],
        maximum_faces: u64,
    ) -> Result<crate::topology::BoundarySurfaceIndex, crate::topology::BoundaryPairingError> {
        crate::topology::BoundarySurfaceIndex::derive(
            self.projection,
            &self.primary,
            positions_m,
            maximum_faces,
        )
    }
    /// Current coordinates are supplied explicitly; never reuse reference normals as world normals.
    pub fn primary_snapshot(
        &self,
        local_index: usize,
        positions_m: &[[f64; 3]],
    ) -> Result<OrientedBoundarySurface, ContactSurfaceError> {
        let index = *self
            .primary
            .get(local_index)
            .ok_or(ContactSurfaceError::InvalidInterface)?;
        Ok(self
            .projection
            .boundary_surface_snapshot(index, positions_m)?)
    }
}

#[cfg(test)]
mod tests;

pub(super) fn matches(face: &ProjectedBoundaryFace, target: &RegionTarget) -> bool {
    face.provenance
        .iter()
        .any(|entity| target.matches_persistent_entity(entity))
}
