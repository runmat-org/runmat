//! Canonical geometry domain model for RunMat.

pub mod diagnostics;
pub mod model;
pub mod selection;

pub use diagnostics::{Diagnostic, DiagnosticSeverity};
pub use model::{
    AssemblyNode, EntityIdRange, GeometryAsset, GeometrySource, MaterialEvidence,
    MaterialEvidenceConfidence, MeshDescriptor, MeshKind, Region, RegionEntityMapping,
    SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
};
pub use selection::{EntityKind, EntityRef};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_asset() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_test".to_string(),
            source: GeometrySource {
                path: "/models/part.stl".to_string(),
                sha256: "abc123".to_string(),
                importer_version: "stl/v1".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Mesh,
                assembly: None,
                material_evidence: vec![],
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 1,
            meshes: vec![MeshDescriptor {
                mesh_id: "mesh_1".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 3,
                element_count: 1,
            }],
            surface_meshes: vec![SurfaceMesh::new(
                "mesh_1",
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                vec![[0, 1, 2]],
            )],
            regions: vec![Region {
                region_id: "region_a".to_string(),
                name: "body".to_string(),
                tag: None,
            }],
            region_entity_mappings: vec![RegionEntityMapping::all_faces("region_a", "mesh_1", 1)],
            diagnostics: vec![],
        }
    }

    #[test]
    fn entity_identity_stable_within_revision() {
        let first = EntityRef {
            geometry_id: "geo_test".to_string(),
            geometry_revision: 1,
            mesh_id: "mesh_1".to_string(),
            entity_kind: EntityKind::Face,
            entity_id: 42,
        };
        let second = EntityRef { ..first.clone() };
        assert_eq!(first, second);
    }

    #[test]
    fn geometry_asset_round_trips_via_json() {
        let asset = sample_asset();
        let json = serde_json::to_string(&asset).expect("serialize");
        let decoded: GeometryAsset = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, asset);
    }

    #[test]
    fn unit_metadata_must_be_present() {
        let mut asset = sample_asset();
        asset.units = UnitSystem::Unspecified;
        let error = asset
            .validate()
            .expect_err("expected unspecified units to fail");
        assert_eq!(error, "geometry units must be specified");
    }
}
