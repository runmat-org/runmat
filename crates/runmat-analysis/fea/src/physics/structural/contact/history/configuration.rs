//! Complete original global configuration once, with compact original point/frame records.
//! A configuration carrier is not a converged body state or restart authorization.
use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactConfigurationPoint {
    scope: Digest,
    primary_face: StableDigest,
    primary_barycentric: [f64; 3],
    frame: ContactPointFrame,
    relative_material_vector_m: [f64; 3],
    point: ContactPointState,
}
impl ContactConfigurationPoint {
    pub fn scope_identity(&self) -> Digest {
        self.scope
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
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactConfigurationState {
    pub(in crate::physics::structural::contact) schema: u16,
    pub(in crate::physics::structural::contact) scope: Digest,
    pub(in crate::physics::structural::contact) ordinal: u64,
    pub(in crate::physics::structural::contact) positions_m: Vec<[f64; 3]>,
    pub(in crate::physics::structural::contact) points: Vec<ContactConfigurationPoint>,
}
impl ContactConfigurationState {
    pub fn scope_identity(&self) -> Digest {
        self.scope
    }
    pub fn ordinal(&self) -> u64 {
        self.ordinal
    }
    pub fn positions_m(&self) -> &[[f64; 3]] {
        &self.positions_m
    }
    pub fn point_history(&self) -> &[ContactConfigurationPoint] {
        &self.points
    }
}
impl ContactPointBinding<'_> {
    pub fn configuration_record(
        &self,
        history: &ContactMaterialPointHistory,
        expected_ordinal: u64,
        positions_m: &[[f64; 3]],
    ) -> Result<ContactConfigurationPoint, ContactHistoryError> {
        self.validate_history(history, expected_ordinal, positions_m)?;
        Ok(ContactConfigurationPoint {
            scope: history.scope,
            primary_face: history.primary_face,
            primary_barycentric: history.primary_barycentric,
            frame: history.frame,
            relative_material_vector_m: history.relative_material_vector_m,
            point: history.point.clone(),
        })
    }
    pub fn material_history(
        &self,
        record: &ContactConfigurationPoint,
        ordinal: u64,
        positions_m: &[[f64; 3]],
    ) -> Result<ContactMaterialPointHistory, ContactHistoryError> {
        let index = self.primary_index(record.primary_face)?;
        let history = ContactMaterialPointHistory {
            schema: 1,
            scope: record.scope,
            ordinal,
            primary_face: record.primary_face,
            primary_barycentric: record.primary_barycentric,
            primary_coordinates_m: self.coordinates(index, positions_m)?,
            secondary_coordinates_m: self.coordinates(self.secondary_face, positions_m)?,
            frame: record.frame,
            relative_material_vector_m: record.relative_material_vector_m,
            point: record.point.clone(),
        };
        self.validate_history(&history, ordinal, positions_m)?;
        Ok(history)
    }
    pub fn current_query_position_m(
        &self,
        positions_m: &[[f64; 3]],
    ) -> Result<[f64; 3], ContactHistoryError> {
        Ok(self
            .projection
            .boundary_surface_snapshot(self.secondary_face, positions_m)?
            .evaluate(self.reference.reference_barycentric())?
            .position_m)
    }
    pub fn secondary_node_indices(&self) -> &[usize] {
        &self.projection.boundary_faces[self.secondary_face].node_indices
    }
    pub fn primary_node_indices(
        &self,
        face: StableDigest,
    ) -> Result<&[usize], ContactHistoryError> {
        Ok(&self.projection.boundary_faces[self.primary_index(face)?].node_indices)
    }
}
