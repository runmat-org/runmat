//! Compile original secondary reference points ONCE; bindings never regenerate a face rule per point.
use super::*;
pub(super) struct ReferencePoint {
    pub(super) face: usize,
    pub(super) ordinal: usize,
    pub(super) point: BoundaryQuadraturePoint,
}
pub struct ContactReferenceInventory<'a> {
    pub(super) projection: &'a AnalysisProjection,
    pub(super) interface: &'a AnalysisInterface,
    pub(super) primary_faces: Vec<usize>,
    pub(super) points: Vec<ReferencePoint>,
}
impl<'a> ContactReferenceInventory<'a> {
    pub fn derive(
        surfaces: &CanonicalContactSurfaces<'a>,
        maximum_points: usize,
    ) -> Result<Self, ContactHistoryError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if maximum_points == 0 {
            return Err(ContactHistoryError::ReferenceBudget);
        }
        let projection = surfaces.projection();
        let mut count = 0usize;
        let mut exceeded = false;
        for &face in surfaces.secondary_face_indices() {
            if crate::is_fea_cancelled() {
                return Err(BoundarySearchError::Cancelled.into());
            }
            projection
                .visit_boundary_quadrature(&projection.boundary_faces[face], |_| {
                    if count == maximum_points {
                        exceeded = true;
                    } else {
                        count += 1;
                    }
                })
                .map_err(|_| ContactHistoryError::InvalidScope)?;
        }
        if exceeded {
            return Err(ContactHistoryError::ReferenceBudget);
        }
        if count == 0 {
            return Err(ContactHistoryError::InvalidScope);
        }
        let mut points = Vec::new();
        points
            .try_reserve_exact(count)
            .map_err(|_| BoundarySearchError::Capacity)?;
        for &face in surfaces.secondary_face_indices() {
            if crate::is_fea_cancelled() {
                return Err(BoundarySearchError::Cancelled.into());
            }
            let mut ordinal = 0;
            projection
                .visit_boundary_quadrature(&projection.boundary_faces[face], |point| {
                    if points.len() == count {
                        exceeded = true;
                    } else {
                        points.push(ReferencePoint {
                            face,
                            ordinal,
                            point,
                        });
                    }
                    ordinal += 1;
                })
                .map_err(|_| ContactHistoryError::InvalidScope)?;
        }
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled.into());
        }
        if exceeded || points.len() != count {
            return Err(ContactHistoryError::InvalidScope);
        }
        let mut primary_faces = Vec::new();
        primary_faces
            .try_reserve_exact(surfaces.primary_face_indices().len())
            .map_err(|_| BoundarySearchError::Capacity)?;
        primary_faces.extend_from_slice(surfaces.primary_face_indices());
        Ok(Self {
            projection,
            interface: surfaces.interface(),
            primary_faces,
            points,
        })
    }
    pub fn primary_face_indices(&self) -> &[usize] {
        &self.primary_faces
    }
    pub fn secondary_face_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.points
            .iter()
            .filter(|p| p.ordinal == 0)
            .map(|p| p.face)
    }
    pub fn point_count(&self) -> usize {
        self.points.len()
    }
    pub fn resident_bytes(&self) -> Result<u64, ContactHistoryError> {
        Self::resident_bytes_for_counts(self.primary_faces.len(), self.points.len())
    }
    pub(crate) fn resident_bytes_for_counts(
        primary_count: usize,
        point_count: usize,
    ) -> Result<u64, ContactHistoryError> {
        let points = crate::topology::storage::slots::<ReferencePoint>(point_count)
            .map_err(|_| ContactHistoryError::InvalidScope)?;
        let faces = crate::topology::storage::slots::<usize>(primary_count)
            .map_err(|_| ContactHistoryError::InvalidScope)?;
        points
            .checked_add(faces)
            .ok_or(ContactHistoryError::InvalidScope)
    }
    pub fn construction_resident_bytes(&self) -> Result<u64, ContactHistoryError> {
        let mut scratch = 0;
        for p in &self.points {
            scratch = scratch.max(
                AnalysisProjection::boundary_quadrature_scratch_resident_bytes(
                    self.projection.boundary_faces[p.face].order,
                )
                .map_err(|_| ContactHistoryError::InvalidScope)?,
            );
        }

        self.resident_bytes()?
            .checked_add(scratch)
            .ok_or(ContactHistoryError::InvalidScope)
    }
}
