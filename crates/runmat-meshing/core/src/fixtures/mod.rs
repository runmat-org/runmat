mod solver_mesh;

pub use solver_mesh::canonical_tetrahedron_solver_mesh;

use runmat_geometry_core::{
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
    UnitSystem,
};

pub fn through_hole_plate_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_root_meshing_through_hole_plate".to_string(),
        source: GeometrySource {
            path: "/fixtures/generic_through_hole_plate.step".to_string(),
            sha256: "generic-through-hole-plate".to_string(),
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
            mesh_id: "through_hole_plate_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 16,
            element_count: 32,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "through_hole_plate_surface",
            vec![
                [0.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [3.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
                [1.0, 2.0, 1.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            vec![
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
                [8, 13, 9],
                [8, 12, 13],
                [9, 14, 10],
                [9, 13, 14],
                [10, 15, 11],
                [10, 14, 15],
                [11, 12, 8],
                [11, 15, 12],
                [0, 8, 9],
                [0, 9, 1],
                [1, 9, 10],
                [1, 10, 2],
                [2, 10, 11],
                [2, 11, 3],
                [3, 11, 8],
                [3, 8, 0],
                [4, 5, 13],
                [4, 13, 12],
                [5, 6, 14],
                [5, 14, 13],
                [6, 7, 15],
                [6, 15, 14],
                [7, 4, 12],
                [7, 12, 15],
            ],
        )],
        regions: vec![Region {
            region_id: "body".to_string(),
            name: "body".to_string(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "body",
            "through_hole_plate_surface",
            32,
        )],
        diagnostics: Vec::new(),
    }
}

pub fn split_material_through_hole_plate_geometry() -> GeometryAsset {
    let mut geometry = through_hole_plate_geometry();
    geometry.geometry_id = "geo_root_meshing_split_material_through_hole_plate".to_string();
    geometry.source.path = "/fixtures/generic_split_material_through_hole_plate.step".to_string();
    geometry.source.sha256 = "generic-split-material-through-hole-plate".to_string();
    geometry.regions = vec![
        Region {
            region_id: "region_base".to_string(),
            name: "base".to_string(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        },
        Region {
            region_id: "region_cap".to_string(),
            name: "cap".to_string(),
            tag: Some("material".to_string()),
            cad_ownership: None,
        },
    ];
    geometry.region_entity_mappings = vec![
        RegionEntityMapping::new(
            "region_base",
            "through_hole_plate_surface",
            EntityKind::Face,
            vec![
                EntityIdRange::new(0, 4),
                EntityIdRange::new(8, 4),
                EntityIdRange::new(16, 4),
                EntityIdRange::new(24, 4),
            ],
        ),
        RegionEntityMapping::new(
            "region_cap",
            "through_hole_plate_surface",
            EntityKind::Face,
            vec![
                EntityIdRange::new(4, 4),
                EntityIdRange::new(12, 4),
                EntityIdRange::new(20, 4),
                EntityIdRange::new(28, 4),
            ],
        ),
    ];
    geometry
}
