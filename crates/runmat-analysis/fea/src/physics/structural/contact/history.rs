//! Original point/frame carrier. Present-state validation is not global convergence or past proof.
use super::*;
use crate::topology::{
    AnalysisProjection, BoundaryProjectionDifferentialPolicy, BoundaryQuadraturePoint,
    BoundarySearchError, OrientedBoundarySurface,
};
use runmat_analysis_core::{AnalysisInterface, AnalysisInterfaceKind};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};
use thiserror::Error;
mod configuration;
mod inventory;
mod prepared;
pub use configuration::{ContactConfigurationPoint, ContactConfigurationState};
pub(crate) use prepared::scope_construction_bytes;
pub use prepared::ContactPreparedPoint;
mod kinematics;
mod trial;
mod validation;
pub use inventory::ContactReferenceInventory;
pub use trial::{ContactPointTangentAction, ContactPointTrial};
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum ContactHistoryError {
    #[error(transparent)]
    Geometry(#[from] BoundarySearchError),
    #[error(transparent)]
    Point(#[from] ContactPointError),
    #[error("invalid contact material point or scope")]
    InvalidScope,
    #[error("original contact reference inventory exceeds its declared point budget")]
    ReferenceBudget,
    #[error("contact original history/configuration/frame does not match its scope")]
    InvalidHistory,
    #[error("contact accepted ordinal exceeds its inventory")]
    OrdinalOverflow,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactMaterialPointHistory {
    schema: u16,
    scope: Digest,
    ordinal: u64,
    primary_face: StableDigest,
    primary_barycentric: [f64; 3],
    primary_coordinates_m: [[f64; 3]; 6],
    secondary_coordinates_m: [[f64; 3]; 6],
    frame: ContactPointFrame,
    relative_material_vector_m: [f64; 3],
    point: ContactPointState,
}
impl ContactMaterialPointHistory {
    pub fn scope_identity(&self) -> Digest {
        self.scope
    }
    pub fn ordinal(&self) -> u64 {
        self.ordinal
    }
    pub fn primary_face_identity(&self) -> StableDigest {
        self.primary_face
    }
    pub fn primary_barycentric(&self) -> [f64; 3] {
        self.primary_barycentric
    }
    pub fn frame(&self) -> &ContactPointFrame {
        &self.frame
    }
    pub fn point(&self) -> &ContactPointState {
        &self.point
    }
    pub fn primary_coordinates_m(&self) -> &[[f64; 3]; 6] {
        &self.primary_coordinates_m
    }
    pub fn secondary_coordinates_m(&self) -> &[[f64; 3]; 6] {
        &self.secondary_coordinates_m
    }
}
struct Configuration {
    primary: [[f64; 3]; 6],
    secondary: [[f64; 3]; 6],
    relative: [f64; 3],
    frame: ContactPointFrame,
    gap: f64,
    query: [f64; 3],
    surface: OrientedBoundarySurface,
}
pub struct ContactPointBinding<'a> {
    projection: &'a AnalysisProjection,
    interface: &'a AnalysisInterface,
    primary_faces: &'a [usize],
    secondary_face: usize,
    point_ordinal: usize,
    reference: BoundaryQuadraturePoint,
    law: PenaltyContactLaw,
    policy: BoundaryProjectionDifferentialPolicy,
    scope: Digest,
}
impl<'a> ContactPointBinding<'a> {
    pub fn derive(
        inventory: &'a ContactReferenceInventory<'_>,
        point_index: usize,
        law: PenaltyContactLaw,
        policy: BoundaryProjectionDifferentialPolicy,
    ) -> Result<Self, ContactHistoryError> {
        ContactPreparedPoint::derive(inventory, point_index, law, policy)?.bind(inventory)
    }
    pub fn scope_identity(&self) -> Digest {
        self.scope
    }
    pub fn point_ordinal(&self) -> usize {
        self.point_ordinal
    }
    pub fn reference_point(&self) -> &BoundaryQuadraturePoint {
        &self.reference
    }
    pub fn interface(&self) -> &AnalysisInterface {
        self.interface
    }
    pub fn initial_history(
        &self,
        primary_face_index: usize,
        primary_barycentric: [f64; 3],
        positions_m: &[[f64; 3]],
    ) -> Result<ContactMaterialPointHistory, ContactHistoryError> {
        let Configuration {
            primary,
            secondary,
            relative,
            frame,
            gap,
            ..
        } = self.configuration(primary_face_index, primary_barycentric, positions_m)?;
        let point = self
            .law
            .update(
                runmat_types::physical::Length::new(gap),
                runmat_types::physical::Quantity::new([0.0; 2]),
                &self.law.initial_state(),
            )?
            .trial_state;
        Ok(ContactMaterialPointHistory {
            schema: 1,
            scope: self.scope,
            ordinal: 0,
            primary_face: self.projection.boundary_faces[primary_face_index].stable_identity,
            primary_barycentric,
            primary_coordinates_m: primary,
            secondary_coordinates_m: secondary,
            frame,
            relative_material_vector_m: relative,
            point,
        })
    }
    fn coordinates(
        &self,
        face: usize,
        positions: &[[f64; 3]],
    ) -> Result<[[f64; 3]; 6], ContactHistoryError> {
        if positions.len() != self.projection.nodes.len() {
            return Err(ContactHistoryError::InvalidScope);
        }
        let mut local = [[0.0; 3]; 6];
        for (out, &index) in local
            .iter_mut()
            .zip(&self.projection.boundary_faces[face].node_indices)
        {
            *out = *positions
                .get(index)
                .ok_or(ContactHistoryError::InvalidScope)?;
        }
        Ok(local)
    }
    fn primary_index(&self, identity: StableDigest) -> Result<usize, ContactHistoryError> {
        self.primary_faces
            .iter()
            .copied()
            .find(|&i| self.projection.boundary_faces[i].stable_identity == identity)
            .ok_or(ContactHistoryError::InvalidHistory)
    }
    fn configuration(
        &self,
        primary_face: usize,
        barycentric: [f64; 3],
        positions: &[[f64; 3]],
    ) -> Result<Configuration, ContactHistoryError> {
        if !self.primary_faces.contains(&primary_face) {
            return Err(ContactHistoryError::InvalidScope);
        }
        let primary = self.coordinates(primary_face, positions)?;
        let secondary = self.coordinates(self.secondary_face, positions)?;
        let surface = self
            .projection
            .boundary_surface_snapshot(primary_face, positions)?;
        let query = self
            .projection
            .boundary_surface_snapshot(self.secondary_face, positions)?
            .evaluate(self.reference.reference_barycentric())?
            .position_m;
        let point = surface.evaluate(barycentric)?;
        surface.stationary_projection_variation(
            query,
            barycentric,
            [0.0; 3],
            &[[0.0; 3]; 6][..point.shape_values().len()],
            self.policy,
        )?;
        let relative = std::array::from_fn(|i| query[i] - point.position_m[i]);
        let gap = relative
            .iter()
            .zip(point.unit_normal)
            .map(|(a, b)| a * b)
            .sum();
        let frame = ContactPointFrame::from_surface(&point)?;
        Ok(Configuration {
            primary,
            secondary,
            relative,
            frame,
            gap,
            query,
            surface,
        })
    }
}

#[cfg(test)]
mod tests;
