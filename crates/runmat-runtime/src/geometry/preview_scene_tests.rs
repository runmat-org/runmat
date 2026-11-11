//! A CAD preview scene shades like a CAD design scene: each face keeps its own normals (a crease
//! stays sharp), small parts in metres keep their true normals, and feature edges draw in the CAD
//! edge color.
use super::*;
use runmat_geometry_core::{CadRegionOwnership, MeshDescriptor, RegionEntityMapping, SurfaceMesh};

/// Two faces of a 1 mm bracket in metres, folded at 90 degrees along a shared edge whose vertices
/// both faces use: the floor faces +Z, the wall faces +Y.
fn folded_bracket() -> GeometryAsset {
    let mm = 0.001;
    let face = |id: u64| Region {
        region_id: format!("face_{:06}", id + 1),
        name: format!("Face {}", id + 1),
        tag: Some("occt_face".to_string()),
        cad_ownership: Some(CadRegionOwnership {
            face_id: Some(id),
            ..Default::default()
        }),
    };
    GeometryAsset {
        geometry_id: "geo_folded".to_string(),
        source: GeometrySource {
            unit_evidence: Default::default(),
            path: "/models/folded.step".to_string(),
            sha256: "folded".to_string(),
            importer_version: "cad/occt/step/v1".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: vec![],
            exchange: None,
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        coordinate_units: None,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: "mesh_1".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 6,
            element_count: 4,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "mesh_1",
            vec![
                [0.0, 0.0, 0.0],
                [mm, 0.0, 0.0],
                [mm, mm, 0.0],
                [0.0, mm, 0.0],
                [0.0, 0.0, mm],
                [mm, 0.0, mm],
            ],
            // Floor (+Z) then wall (+Y); vertices 0 and 1 are on the crease.
            vec![[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 1]],
        )],
        regions: vec![face(0), face(1)],
        region_entity_mappings: vec![
            RegionEntityMapping::new(
                "face_000001",
                "mesh_1",
                EntityKind::Face,
                vec![EntityIdRange::new(0, 2)],
            ),
            RegionEntityMapping::new(
                "face_000002",
                "mesh_1",
                EntityKind::Face,
                vec![EntityIdRange::new(2, 2)],
            ),
        ],
        diagnostics: vec![],
    }
}

fn scene(xray: bool) -> runmat_plot::GeometryScene {
    geometry_preview_scene(
        &folded_bracket(),
        "Preview",
        GeometryPreviewSceneOptions {
            xray,
            ..GeometryPreviewSceneOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn preview_faces_keep_their_own_normals_across_a_crease_at_metre_scale() {
    let scene = scene(false);
    let surface = scene.chunks.iter().find(|c| !c.wireframe_only).unwrap();
    let indices = surface.indices.as_deref().unwrap();
    let normal = |corner: usize| surface.vertices[indices[corner] as usize].normal;
    for corner in 0..6 {
        assert_eq!(normal(corner), [0.0, 0.0, 1.0], "floor corner {corner}");
    }
    for corner in 6..12 {
        let n = normal(corner);
        assert!(
            (n[1] - 1.0).abs() < 1e-6 && n[0].abs() < 1e-6 && n[2].abs() < 1e-6,
            "wall corner {corner}: {n:?}"
        );
    }
    // The crease's two vertices are copied once per face.
    assert_eq!(surface.vertices.len(), 8);
}

#[test]
fn preview_feature_edges_use_the_cad_edge_color() {
    let edge_color = |scene: &runmat_plot::GeometryScene| {
        let edges = scene.chunks.iter().find(|c| c.wireframe_only).unwrap();
        assert!(!edges.vertices.is_empty());
        edges.vertices[0].color
    };
    assert_eq!(edge_color(&scene(false)), CAD_FEATURE_EDGE_COLOR.to_array());
    assert_eq!(
        edge_color(&scene(true)),
        CAD_XRAY_FEATURE_EDGE_COLOR.to_array()
    );
}
