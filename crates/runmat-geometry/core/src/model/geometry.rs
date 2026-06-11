use serde::{Deserialize, Serialize};

use crate::diagnostics::Diagnostic;

use super::{MeshDescriptor, Region, SourceGeometry, SurfaceMesh, TessellationProfile, UnitSystem};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryAsset {
    pub geometry_id: String,
    pub source: GeometrySource,
    pub source_geometry: SourceGeometry,
    pub tessellation_profile: TessellationProfile,
    pub units: UnitSystem,
    pub revision: u32,
    pub meshes: Vec<MeshDescriptor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub surface_meshes: Vec<SurfaceMesh>,
    pub regions: Vec<Region>,
    pub diagnostics: Vec<Diagnostic>,
}

impl GeometryAsset {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.units == UnitSystem::Unspecified {
            return Err("geometry units must be specified");
        }
        for surface_mesh in &self.surface_meshes {
            surface_mesh.validate()?;
            if !self
                .meshes
                .iter()
                .any(|mesh| mesh.mesh_id == surface_mesh.mesh_id)
            {
                return Err("surface mesh must reference a declared mesh descriptor");
            }
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
