use super::*;
use runmat_geometry_core::{
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
    UnitSystem,
};

#[test]
fn extracts_deterministic_closed_shell_topology() {
    let topology = extract_source_topology(&cube_geometry()).expect("topology should extract");

    assert_eq!(topology.vertices.len(), 8);
    assert_eq!(topology.faces.len(), 12);
    assert_eq!(topology.edges.len(), 18);
    assert_eq!(
        topology.region_ids,
        vec!["root".to_string(), "tip".to_string()]
    );
    assert!(topology
        .edges
        .iter()
        .all(|edge| edge.adjacent_face_ids.len() == 2));
    assert!(topology.faces.iter().all(|face| face.area_m2 > 0.0));
    assert!(topology
        .faces
        .iter()
        .all(|face| super::geometry::norm(face.unit_normal) > 0.999999));
}

#[test]
fn topology_converts_geometry_units_to_meters() {
    let mut geometry = cube_geometry();
    geometry.units = UnitSystem::Millimeter;

    let topology = extract_source_topology(&geometry).expect("topology should extract");

    assert_eq!(topology.bounds_max_m, [0.001, 0.001, 0.001]);
    assert!(topology.edges.iter().any(|edge| {
        (edge.length_m - 0.001).abs() < 1.0e-12
            && edge.region_ids.iter().any(|region| region == "root")
    }));
}

fn cube_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_topology_cube".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_cube.step".to_string(),
            sha256: "generic-cube".to_string(),
            importer_version: "test".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        units: UnitSystem::Meter,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: "cube_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 8,
            element_count: 12,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "cube_surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
        )],
        regions: vec![
            Region {
                region_id: "root".to_string(),
                name: "root".to_string(),
                tag: None,
                cad_ownership: None,
            },
            Region {
                region_id: "tip".to_string(),
                name: "tip".to_string(),
                tag: None,
                cad_ownership: None,
            },
        ],
        region_entity_mappings: vec![
            RegionEntityMapping::new(
                "root",
                "cube_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(0, 6)],
            ),
            RegionEntityMapping::new(
                "tip",
                "cube_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(6, 6)],
            ),
        ],
        diagnostics: Vec::new(),
    }
}
