use serde::{Deserialize, Serialize};

use crate::diagnostics::Diagnostic;
use crate::selection::EntityKind;

use super::{
    MeshDescriptor, Region, RegionEntityMapping, SourceGeometry, SurfaceMesh, TessellationProfile,
    UnitSystem,
};

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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub region_entity_mappings: Vec<RegionEntityMapping>,
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
        for mapping in &self.region_entity_mappings {
            if !self
                .regions
                .iter()
                .any(|region| region.region_id == mapping.region_id)
            {
                return Err("region entity mapping must reference a declared region");
            }
            let Some(mesh) = self
                .meshes
                .iter()
                .find(|mesh| mesh.mesh_id == mapping.mesh_id)
            else {
                return Err("region entity mapping must reference a declared mesh");
            };
            let entity_count = match mapping.entity_kind {
                EntityKind::Node => mesh.vertex_count,
                EntityKind::Edge => 0,
                EntityKind::Face | EntityKind::Element => mesh.element_count,
            };
            for range in &mapping.ranges {
                if range.count == 0 {
                    return Err("region entity mapping ranges must not be empty");
                }
                let Some(end) = range.end_exclusive() else {
                    return Err("region entity mapping range overflows");
                };
                if end > entity_count {
                    return Err("region entity mapping range out of bounds");
                }
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
