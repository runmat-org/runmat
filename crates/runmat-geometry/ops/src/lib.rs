//! Pure geometry operations.

pub mod bounds;
pub mod quality;
pub mod queries;
pub mod stats;

pub use bounds::{compute_axis_aligned_bounds, AxisAlignedBounds};
pub use quality::{evaluate_quality, QualityReport};
pub use queries::{find_region, QueryError};
pub use stats::{compute_stats, GeometryStats};

#[cfg(test)]
mod tests {
    use runmat_geometry_core::{
        GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, SourceGeometry,
        SourceGeometryKind, SurfaceMesh, TessellationProfile, UnitSystem,
    };

    use crate::{compute_axis_aligned_bounds, compute_stats, evaluate_quality};

    fn sample() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo".to_string(),
            source: GeometrySource {
                path: "/x.stl".to_string(),
                sha256: "hash".to_string(),
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
                mesh_id: "mesh".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 3,
                element_count: 1,
            }],
            surface_meshes: vec![SurfaceMesh::new(
                "mesh",
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                vec![[0, 1, 2]],
            )],
            regions: vec![],
            region_entity_mappings: vec![],
            diagnostics: vec![],
        }
    }

    #[test]
    fn stats_are_computed() {
        let stats = compute_stats(&sample());
        assert_eq!(stats.mesh_count, 1);
        assert_eq!(stats.total_vertices, 3);
        assert_eq!(stats.total_elements, 1);
    }

    #[test]
    fn bounds_are_deterministic() {
        let bounds = compute_axis_aligned_bounds(&sample());
        assert_eq!(bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.max, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn quality_reports_units_warning_when_unspecified() {
        let mut asset = sample();
        asset.units = UnitSystem::Unspecified;
        let report = evaluate_quality(&asset);
        assert!(report
            .warnings
            .iter()
            .any(|message| message.contains("units")));
    }
}
