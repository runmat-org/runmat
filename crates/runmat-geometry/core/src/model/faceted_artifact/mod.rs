mod codec;
mod construction;
mod validation;

use serde::{Deserialize, Serialize};

use super::{GeometryDocument, PersistentEntityId};

pub use codec::{decode_faceted_solid, encode_faceted_solid};
pub use construction::{admit_faceted_solid, build_faceted_solid_closure};

pub const FACETED_SOLID_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacetedSolid {
    pub schema_version: u16,
    pub vertices: Vec<FacetedVertex>,
    pub triangles: Vec<FacetedTriangle>,
    pub shells: Vec<FacetedShell>,
}

impl FacetedSolid {
    pub fn validate_against(
        &self,
        model: &super::FacetedSolidModel,
    ) -> Result<(), super::GeometryContractError> {
        validation::validate_faceted_solid(self, model)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacetedVertex {
    pub id: PersistentEntityId,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacetedTriangle {
    pub id: PersistentEntityId,
    pub vertex_indices: [u32; 3],
    pub shell_id: PersistentEntityId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacetedShell {
    pub id: PersistentEntityId,
    pub orientation: FacetedShellOrientation,
    pub triangle_indices: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FacetedShellOrientation {
    Outward,
    Inward,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncodedFacetedSolidClosure {
    pub document: GeometryDocument,
    pub solid: FacetedSolid,
    pub solid_bytes: Vec<u8>,
}

#[cfg(test)]
mod tests;
