mod codec;
mod payload;
mod validation;

use serde::{Deserialize, Serialize};

pub use codec::{decode_geometry_document, encode_geometry_document};
pub use payload::{
    DisplayTessellationRef, ExactBRepModel, ExactGeometryCapabilities, FacetedSolidModel,
    GeometryModel, GeometryObjectRef, DISPLAY_TESSELLATION_MEDIA_TYPE, EXACT_BREP_MEDIA_TYPE,
    FACETED_SOLID_MEDIA_TYPE,
};

use super::{GeometryContractError, GeometryTolerancePolicy, UnitSystem};

pub const GEOMETRY_DOCUMENT_SCHEMA_VERSION: u16 = 2;
pub const GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GeometryDigest([u8; 32]);

impl GeometryDigest {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub(crate) fn validate_nonzero(&self, field: &str) -> Result<(), GeometryContractError> {
        if *self == Self::ZERO {
            return Err(GeometryContractError::invalid(
                field,
                "digest must not be all zeroes",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometrySourceFormat {
    Step,
    Iges,
    Brep,
    NativeCad,
    Stl,
    Obj,
    Ply,
    Gltf,
}

impl GeometrySourceFormat {
    pub const fn is_exact(self) -> bool {
        matches!(self, Self::Step | Self::Iges | Self::Brep | Self::NativeCad)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometrySourceIdentity {
    pub content_digest: GeometryDigest,
    pub format: GeometrySourceFormat,
    pub importer_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_version: Option<String>,
    pub source_units: UnitSystem,
    pub meters_per_source_unit: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRevisionIdentity {
    pub revision: u64,
    pub persistent_mapping_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_document_digest: Option<GeometryDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryHealingPolicy {
    pub algorithm_version: String,
    pub sew: bool,
    pub repair_orientation: bool,
    pub consolidate_duplicates: bool,
    pub repair_tolerance_scale_gaps: bool,
    pub simplify_short_edges_and_sliver_faces: bool,
}

impl GeometryHealingPolicy {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        super::analysis_identity::validate_token(
            "geometry healing algorithm version",
            &self.algorithm_version,
            128,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryDocument {
    pub schema_version: u16,
    pub source: GeometrySourceIdentity,
    pub revision: GeometryRevisionIdentity,
    pub tolerance: GeometryTolerancePolicy,
    pub healing: GeometryHealingPolicy,
    pub model: GeometryModel,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub display_tessellations: Vec<DisplayTessellationRef>,
}

impl GeometryDocument {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        validation::validate_document(self)
    }

    pub const fn is_exact(&self) -> bool {
        matches!(self.model, GeometryModel::ExactBRep { .. })
    }

    pub const fn primary_artifact(&self) -> &GeometryObjectRef {
        self.model.primary_artifact()
    }
}

#[cfg(test)]
mod tests;
