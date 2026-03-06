use serde::{Deserialize, Serialize};

use crate::diagnostics::Diagnostic;

use super::{MeshDescriptor, Region, SourceGeometry, TessellationProfile, UnitSystem};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryAsset {
    pub geometry_id: String,
    pub source: GeometrySource,
    pub source_geometry: SourceGeometry,
    pub tessellation_profile: TessellationProfile,
    pub units: UnitSystem,
    pub revision: u32,
    pub meshes: Vec<MeshDescriptor>,
    pub regions: Vec<Region>,
    pub diagnostics: Vec<Diagnostic>,
}

impl GeometryAsset {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.units == UnitSystem::Unspecified {
            return Err("geometry units must be specified");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometrySource {
    pub path: String,
    pub sha256: String,
    pub importer_version: String,
}
