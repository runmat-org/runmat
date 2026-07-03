use super::*;

pub(super) fn cube_geometry() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo_tet_cube".to_string(),
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
        revision: 3,
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
                region_id: "region_fixed".to_string(),
                name: "fixed".to_string(),
                tag: Some("fixed".to_string()),
                cad_ownership: None,
            },
            Region {
                region_id: "region_load".to_string(),
                name: "load".to_string(),
                tag: Some("load".to_string()),
                cad_ownership: None,
            },
        ],
        region_entity_mappings: vec![
            RegionEntityMapping::all_faces("region_fixed", "cube_surface", 2),
            RegionEntityMapping::new(
                "region_load",
                "cube_surface",
                runmat_geometry_core::EntityKind::Face,
                vec![runmat_geometry_core::EntityIdRange::new(2, 2)],
            ),
        ],
        diagnostics: Vec::new(),
    }
}

pub(super) fn thin_box_geometry() -> GeometryAsset {
    let mut geometry = cube_geometry();
    geometry.geometry_id = "geo_tet_thin_box".to_string();
    for vertex in geometry.surface_meshes[0].vertices.iter_mut().skip(4) {
        vertex[2] = 0.2;
    }
    geometry
}

pub(super) fn tetrahedron_geometry() -> GeometryAsset {
    let mut geometry = cube_geometry();
    geometry.geometry_id = "geo_tet_tetrahedron".to_string();
    geometry.meshes[0].vertex_count = 4;
    geometry.meshes[0].element_count = 4;
    geometry.surface_meshes = vec![SurfaceMesh::new(
        "cube_surface",
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    )];
    geometry.region_entity_mappings = vec![
        RegionEntityMapping::all_faces("region_fixed", "cube_surface", 4),
        RegionEntityMapping::new(
            "region_load",
            "cube_surface",
            runmat_geometry_core::EntityKind::Face,
            vec![runmat_geometry_core::EntityIdRange::new(2, 2)],
        ),
    ];
    geometry
}
pub(super) fn boundary_projection_fixture() -> (
    BoundaryMeshInput,
    Vec<AnalysisMeshNode>,
    Vec<AnalysisVolumeElement>,
    Vec<AnalysisBoundaryFace>,
) {
    let input = BoundaryMeshInput {
        mesh_id: "projection_surface".to_string(),
        source_geometry_id: "geo_projection_surface".to_string(),
        source_geometry_revision: 1,
        source_geometry_sha256: None,
        vertices: vec![[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [0.0, 10.0, 0.0]],
        triangles: vec![BoundaryMeshTriangle {
            triangle_id: 0,
            node_ids: [0, 1, 2],
            region_ids: vec!["region_surface".to_string()],
            provenance: Vec::new(),
        }],
        bounds_min_m: [-10.0, -10.0, 0.0],
        bounds_max_m: [10.0, 10.0, 2.0],
        region_ids: vec!["region_surface".to_string()],
    };
    let nodes = vec![
        AnalysisMeshNode {
            node_id: 1,
            coordinates_m: [0.0, 0.0, 0.5],
            provenance: Vec::new(),
        },
        AnalysisMeshNode {
            node_id: 2,
            coordinates_m: [1.0, 0.0, 0.5],
            provenance: Vec::new(),
        },
        AnalysisMeshNode {
            node_id: 3,
            coordinates_m: [0.0, 1.0, 0.5],
            provenance: Vec::new(),
        },
        AnalysisMeshNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, 2.0],
            provenance: Vec::new(),
        },
    ];
    let volume_elements = vec![AnalysisVolumeElement {
        element_id: "tet_1".to_string(),
        kind: VolumeElementKind::Tet4,
        node_ids: vec![1, 2, 3, 4],
        material_region_id: "region_surface".to_string(),
        provenance: Vec::new(),
    }];
    let boundary_faces = vec![AnalysisBoundaryFace {
        face_id: "bf_1".to_string(),
        kind: BoundaryElementKind::Tri3,
        node_ids: vec![1, 2, 3],
        adjacent_volume_element_ids: vec!["tet_1".to_string()],
        region_ids: vec!["region_surface".to_string()],
        provenance: Vec::new(),
    }];
    (input, nodes, volume_elements, boundary_faces)
}
pub(super) fn all_nodes_are_referenced(mesh: &AnalysisMeshArtifact) -> bool {
    mesh.nodes.iter().all(|node| {
        mesh.volume_elements
            .iter()
            .any(|element| element.node_ids.contains(&node.node_id))
            || mesh
                .boundary_faces
                .iter()
                .any(|face| face.node_ids.contains(&node.node_id))
    })
}
pub(super) fn unique_axis_coordinates(mesh: &AnalysisMeshArtifact, axis: usize) -> Vec<f64> {
    let mut coordinates = mesh
        .nodes
        .iter()
        .map(|node| node.coordinates_m[axis])
        .collect::<Vec<_>>();
    coordinates.sort_by(f64::total_cmp);
    coordinates.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-12);
    coordinates
}
pub(super) fn tet_centroid(node_ids: [u32; 4], nodes: &[AnalysisMeshNode]) -> [f64; 3] {
    let points = tet_points(node_ids, nodes).expect("test tet nodes should resolve");
    [
        (points[0][0] + points[1][0] + points[2][0] + points[3][0]) * 0.25,
        (points[0][1] + points[1][1] + points[2][1] + points[3][1]) * 0.25,
        (points[0][2] + points[1][2] + points[2][2] + points[3][2]) * 0.25,
    ]
}
