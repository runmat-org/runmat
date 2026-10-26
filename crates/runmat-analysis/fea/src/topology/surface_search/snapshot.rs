//! Snapshot canonical face positions explicitly; outward side follows reference incidence.
use super::*;
use crate::topology::AnalysisProjection;
use runmat_meshing_core::StableDigest;

#[derive(Clone, Copy, Debug)]
pub struct OrientedBoundarySurface {
    identity: StableDigest,
    geometry: BoundarySurface,
    orientation: f64,
}
impl OrientedBoundarySurface {
    pub fn closest_point_stationary(
        &self,
        query: [f64; 3],
        search: BoundarySearchPolicy,
        refinement: BoundaryStationaryPolicy,
    ) -> Result<StationaryBoundaryProjection, BoundarySearchError> {
        let initial = self.closest_point(query, search)?;
        self.refine_candidate(query, initial, refinement)
    }
    pub(in crate::topology::surface_search) fn refine_candidate(
        &self,
        query: [f64; 3],
        mut candidate: ClosestBoundaryPoint,
        policy: BoundaryStationaryPolicy,
    ) -> Result<StationaryBoundaryProjection, BoundarySearchError> {
        candidate.point.unit_normal = candidate.point.unit_normal.map(|x| x * self.orientation);
        candidate.signed_normal_offset_m *= self.orientation;
        let mut result = self.geometry.refine_candidate(query, candidate, policy)?;
        result.closest.point.unit_normal = result
            .closest
            .point
            .unit_normal
            .map(|x| x * self.orientation);
        result.closest.signed_normal_offset_m *= self.orientation;
        Ok(result)
    }
    pub fn stationary_projection_mixed_variation(
        &self,
        query: [f64; 3],
        barycentric: [f64; 3],
        directions: [BoundaryProjectionDirection<'_>; 2],
        policy: BoundaryProjectionDifferentialPolicy,
    ) -> Result<BoundaryProjectionMixedVariation, BoundarySearchError> {
        Ok(self
            .geometry
            .stationary_projection_mixed_variation(query, barycentric, directions, policy)?
            .orient(self.orientation))
    }
    pub fn stationary_projection_variation(
        &self,
        query: [f64; 3],
        barycentric: [f64; 3],
        query_increment_m: [f64; 3],
        node_increments_m: &[[f64; 3]],
        policy: BoundaryProjectionDifferentialPolicy,
    ) -> Result<BoundaryProjectionVariation, BoundarySearchError> {
        Ok(self
            .geometry
            .stationary_projection_variation(
                query,
                barycentric,
                query_increment_m,
                node_increments_m,
                policy,
            )?
            .orient(self.orientation))
    }
    pub fn bounds(&self) -> Result<BoundarySurfaceBounds, BoundarySearchError> {
        self.geometry.bounds()
    }
    pub fn face_identity(&self) -> StableDigest {
        self.identity
    }
    pub fn evaluate(
        &self,
        barycentric: [f64; 3],
    ) -> Result<BoundarySurfacePoint, BoundarySearchError> {
        let mut point = self.geometry.evaluate(barycentric)?;
        point.unit_normal = point.unit_normal.map(|n| n * self.orientation);
        Ok(point)
    }
    pub fn closest_point(
        &self,
        query: [f64; 3],
        policy: BoundarySearchPolicy,
    ) -> Result<ClosestBoundaryPoint, BoundarySearchError> {
        let mut closest = self.geometry.closest_point(query, policy)?;
        closest.point.unit_normal = closest.point.unit_normal.map(|n| n * self.orientation);
        closest.signed_normal_offset_m *= self.orientation;
        Ok(closest)
    }
}
impl AnalysisProjection {
    /// Current positions must follow this projection's canonical node inventory.
    /// The caller binds that inventory to physical state and admits whole-task storage.
    /// Reference geometry is supplied explicitly by passing original nodal positions.
    /// This validates the face map, not volume deformation, contact law or equilibrium.
    pub fn boundary_surface_snapshot(
        &self,
        face_index: usize,
        positions_m: &[[f64; 3]],
    ) -> Result<OrientedBoundarySurface, BoundarySearchError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled);
        }
        if positions_m.len() != self.nodes.len() {
            return Err(BoundarySearchError::InvalidInput);
        }
        let face = self
            .boundary_faces
            .get(face_index)
            .ok_or(BoundarySearchError::InvalidInput)?;
        let expected = match face.order {
            BoundaryTriangleOrder::Tri3 => 3,
            BoundaryTriangleOrder::Tri6 => 6,
        };
        if face.node_indices.len() != expected {
            return Err(BoundarySearchError::InvalidGeometry);
        }
        let mut coordinates = [[0.0; 3]; 6];
        for (local, &index) in face.node_indices.iter().enumerate() {
            coordinates[local] = *positions_m
                .get(index)
                .ok_or(BoundarySearchError::InvalidGeometry)?;
        }
        self.boundary_surface_snapshot_local(face_index, &coordinates[..expected])
    }
    /// Exact local node inventory in the canonical face's original order.
    /// Uses the same reference-side authority as the full-configuration snapshot.
    pub fn boundary_surface_snapshot_local(
        &self,
        face_index: usize,
        coordinates: &[[f64; 3]],
    ) -> Result<OrientedBoundarySurface, BoundarySearchError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled);
        }
        let face = self
            .boundary_faces
            .get(face_index)
            .ok_or(BoundarySearchError::InvalidInput)?;
        let expected = match face.order {
            BoundaryTriangleOrder::Tri3 => 3,
            BoundaryTriangleOrder::Tri6 => 6,
        };
        if face.node_indices.len() != expected {
            return Err(BoundarySearchError::InvalidGeometry);
        }
        if coordinates.len() != face.node_indices.len() {
            return Err(BoundarySearchError::InvalidInput);
        }
        let mut reference = [[0.0; 3]; 3];
        for (local, point) in reference.iter_mut().enumerate() {
            *point = self
                .nodes
                .get(face.node_indices[local])
                .ok_or(BoundarySearchError::InvalidGeometry)?
                .coordinates_m
                .canonical_value();
        }
        let normal = cross(
            subtract(reference[1], reference[0]),
            subtract(reference[2], reference[0]),
        );
        let side = dot(normal, face.oriented_area_vector_m2);
        if !side.is_finite() || side == 0.0 {
            return Err(BoundarySearchError::InvalidGeometry);
        }
        Ok(OrientedBoundarySurface {
            identity: face.stable_identity,
            geometry: BoundarySurface::derive(face.order, coordinates)?,
            orientation: if side > 0.0 { 1.0 } else { -1.0 },
        })
    }
}
