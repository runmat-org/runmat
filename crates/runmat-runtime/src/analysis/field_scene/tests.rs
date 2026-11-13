use super::*;
use crate::analysis::{AnalysisRenderMesh, AnalysisRenderTopologySource};
use runmat_plot::{GeometrySceneDisplayMode, GeometryScenePresentation};

#[test]
fn large_surface_is_split_into_bounded_scene_chunks() {
    let topology = AnalysisRenderTopology {
        schema_version: "test".into(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        volume_element_count: 1,
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "mesh".into(),
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]; MAX_TRIANGLES_PER_TILE + 1],
            regions: vec![],
            vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
            triangle_volume_element_indices: vec![Some(0); MAX_TRIANGLES_PER_TILE + 1],
        }],
    };
    let (scene, _) =
        scene_from_topology("run", &topology, None, None, "undeformed", 1.0, false).unwrap();
    let surfaces = scene
        .chunks
        .iter()
        .filter(|chunk| chunk.render_data.pipeline_type == PipelineType::Triangles)
        .collect::<Vec<_>>();
    assert_eq!(surfaces.len(), 2);
    assert!(surfaces
        .iter()
        .all(|chunk| chunk.triangle_count() <= MAX_TRIANGLES_PER_TILE));
}

#[test]
fn selected_nodal_field_recolors_surface_and_preserves_wireframe_edges() {
    let topology = AnalysisRenderTopology {
        schema_version: "test".into(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        volume_element_count: 1,
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "mesh".into(),
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
            regions: vec![],
            vertex_volume_node_indices: vec![Some(0), Some(1), Some(2)],
            triangle_volume_element_indices: vec![Some(0)],
        }],
    };
    let field = FieldData::inline(
        AnalysisFieldLocation::Node,
        1,
        vec![0.0, 1.0, 2.0],
        "temperature",
        "K",
    );
    let (scene, scale) = scene_from_topology(
        "run",
        &topology,
        Some(&field),
        None,
        "undeformed",
        1.0,
        false,
    )
    .unwrap();
    assert_eq!(scale, 0.0);
    assert_eq!(scene.chunks.len(), 2);
    assert_ne!(
        scene.chunks[0].vertices[0].color,
        scene.chunks[0].vertices[2].color
    );
    assert_eq!(
        scene
            .overlay
            .as_ref()
            .unwrap()
            .scalar_legend
            .as_ref()
            .unwrap()
            .unit,
        "K"
    );
    assert!(scene.title.is_none());
    let wireframe = scene.nodes_with_presentation(&GeometryScenePresentation {
        display_mode: GeometrySceneDisplayMode::Wireframe,
        ..GeometryScenePresentation::default()
    });
    assert!(wireframe[0]
        .render_data
        .as_ref()
        .unwrap()
        .draw_calls
        .is_empty());
    assert!(!wireframe[1]
        .render_data
        .as_ref()
        .unwrap()
        .draw_calls
        .is_empty());
    let displacement = FieldData::inline(
        AnalysisFieldLocation::Node,
        3,
        vec![0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
        "structural.displacement",
        "m",
    );
    let (deformed, true_scale) = scene_from_topology(
        "run",
        &topology,
        Some(&field),
        Some(&displacement),
        "true",
        1.0,
        false,
    )
    .unwrap();
    assert_eq!(true_scale, 1.0);
    assert_ne!(
        deformed.chunks[0].vertices[1].position,
        scene.chunks[0].vertices[1].position
    );
    let (undeformed, _) = scene_from_topology(
        "run",
        &topology,
        Some(&field),
        Some(&displacement),
        "undeformed",
        1.0,
        false,
    )
    .unwrap();
    assert_eq!(
        undeformed.chunks[0].vertices[1].position,
        scene.chunks[0].vertices[1].position
    );
    assert_eq!(
        undeformed.chunks[0].vertices[1].color,
        deformed.chunks[0].vertices[1].color
    );
    let (exaggerated, exaggerated_scale) = scene_from_topology(
        "run",
        &topology,
        Some(&field),
        Some(&displacement),
        "exaggerated",
        2.0,
        false,
    )
    .unwrap();
    assert!(exaggerated_scale >= 2.0);
    assert_eq!(
        exaggerated.chunks[0].vertices[1].color,
        scene.chunks[0].vertices[1].color
    );
    let (outlined, _) = scene_from_topology(
        "run",
        &topology,
        Some(&field),
        Some(&displacement),
        "true",
        1.0,
        true,
    )
    .unwrap();
    assert_eq!(outlined.chunks.len(), deformed.chunks.len() + 1);
    assert_eq!(
        outlined
            .chunks
            .iter()
            .find(|chunk| chunk.label.as_deref() == Some("Undeformed outline"))
            .unwrap()
            .vertices[0]
            .position,
        [0.0, 0.0, 0.0]
    );
}

#[test]
fn integration_point_values_are_averaged_per_element() {
    let field = FieldData::inline(
        AnalysisFieldLocation::Element,
        1,
        vec![1.0, 3.0, 5.0, 7.0],
        "integration point temperature",
        "K",
    );
    assert_eq!(field_magnitude(&field, 0, 2).unwrap(), 2.0);
    assert_eq!(field_magnitude(&field, 1, 2).unwrap(), 6.0);
}

#[test]
fn element_colours_follow_owner_indices_without_nodal_smoothing() {
    let topology = AnalysisRenderTopology {
        schema_version: "test".into(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        volume_element_count: 2,
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "reversed-owners".into(),
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [1, 3, 2]],
            regions: vec![],
            vertex_volume_node_indices: vec![Some(0), Some(1), Some(2), Some(3)],
            triangle_volume_element_indices: vec![Some(1), Some(0)],
        }],
    };
    let field = FieldData::inline(
        AnalysisFieldLocation::Element,
        1,
        vec![2.0, 8.0],
        "von Mises stress",
        "Pa",
    );
    let (scene, _) = scene_from_topology(
        "run",
        &topology,
        Some(&field),
        None,
        "undeformed",
        1.0,
        false,
    )
    .unwrap();
    let surface = &scene.chunks[0];
    for vertex in &surface.vertices[..3] {
        assert_eq!(vertex.color, scalar_color(8.0, 2.0, 8.0));
    }
    for vertex in &surface.vertices[3..6] {
        assert_eq!(vertex.color, scalar_color(2.0, 2.0, 8.0));
    }
    // Shared geometry nodes do not blend the two distinct element values.
    assert_eq!(surface.vertices[1].position, surface.vertices[3].position);
    assert_ne!(surface.vertices[1].color, surface.vertices[3].color);
}

/// Closed unit-cube surface, `n` x `n` quads per face, wound outward.
fn dense_cube_topology(n: usize) -> AnalysisRenderTopology {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut index = std::collections::HashMap::new();
    let mut vertex = |p: [f64; 3], vertices: &mut Vec<[f64; 3]>| -> u32 {
        let key = p.map(|v| (v * 1.0e6).round() as i64);
        *index.entry(key).or_insert_with(|| {
            vertices.push(p);
            (vertices.len() - 1) as u32
        })
    };
    for axis in 0..3 {
        for side in [0.0, 1.0] {
            for i in 0..n {
                for j in 0..n {
                    let point = |a: usize, b: usize| {
                        let mut p = [0.0; 3];
                        p[axis] = side;
                        p[(axis + 1) % 3] = a as f64 / n as f64;
                        p[(axis + 2) % 3] = b as f64 / n as f64;
                        p
                    };
                    let q = [
                        point(i, j),
                        point(i + 1, j),
                        point(i + 1, j + 1),
                        point(i, j + 1),
                    ]
                    .map(|p| vertex(p, &mut vertices));
                    // (axis+1, axis+2) is right-handed about +axis; flip on the low side.
                    if side == 1.0 {
                        triangles.push([q[0], q[1], q[2]]);
                        triangles.push([q[0], q[2], q[3]]);
                    } else {
                        triangles.push([q[0], q[2], q[1]]);
                        triangles.push([q[0], q[3], q[2]]);
                    }
                }
            }
        }
    }
    let vertex_count = vertices.len();
    let triangle_count = triangles.len();
    AnalysisRenderTopology {
        schema_version: "test".into(),
        source: AnalysisRenderTopologySource::AnalysisMesh,
        volume_element_count: 1,
        meshes: vec![AnalysisRenderMesh {
            mesh_id: "cube".into(),
            vertices,
            triangles,
            regions: vec![],
            vertex_volume_node_indices: (0..vertex_count).map(Some).collect(),
            triangle_volume_element_indices: vec![Some(0); triangle_count],
        }],
    }
}

// witness: FEA-STRUCTURAL-PUBLIC-001/result_publication_defects @
//   linear_structural/tet10/numerical_kernel; coverage: partial;
//   boundary: a dense poster scene draws only its boundary triangles and, by default, feature
//   edges (the cube's 12 creases) instead of the full mesh wireframe; rendering itself is not
//   covered.
#[test]
fn dense_poster_scene_draws_boundary_triangles_and_feature_edges_only() {
    let n = 50;
    let topology = dense_cube_topology(n);
    let (scene, _) =
        scene_from_topology("run", &topology, None, None, "undeformed", 1.0, false).unwrap();
    let triangles = scene
        .chunks
        .iter()
        .filter(|chunk| chunk.render_data.pipeline_type == PipelineType::Triangles)
        .map(|chunk| chunk.triangle_count())
        .sum::<usize>();
    assert_eq!(triangles, 12 * n * n);
    assert!(triangles > FULL_MESH_EDGE_TRIANGLE_LIMIT);
    let lines = scene
        .chunks
        .iter()
        .filter(|chunk| chunk.render_data.pipeline_type == PipelineType::Lines)
        .collect::<Vec<_>>();
    assert!(lines
        .iter()
        .all(|chunk| chunk.label.as_deref() == Some("Feature edges")));
    let segments = lines
        .iter()
        .map(|chunk| chunk.vertices.len() / 2)
        .sum::<usize>();
    assert_eq!(
        segments,
        12 * n,
        "only the cube's 12 crease edges are drawn"
    );
}

/// A result view of a coarse mesh draws its field with feature edges only: the full wireframe
/// (one line per mesh edge) reads as dots over the field at poster size. The mesh view keeps it.
#[test]
fn result_views_swap_the_full_wireframe_for_feature_edges() {
    let n = 4;
    let topology = dense_cube_topology(n);
    let (mut scene, _) =
        scene_from_topology("run", &topology, None, None, "undeformed", 1.0, false).unwrap();
    let segments = |scene: &GeometryScene, label: &str| {
        scene
            .chunks
            .iter()
            .filter(|chunk| chunk.render_data.pipeline_type == PipelineType::Lines)
            .filter(|chunk| chunk.label.as_deref() == Some(label))
            .map(|chunk| chunk.vertices.len() / 2)
            .sum::<usize>()
    };
    assert!(segments(&scene, "Solver mesh edges") > 12 * n);
    use_result_view_edges(&mut scene);
    assert_eq!(segments(&scene, "Solver mesh edges"), 0);
    assert_eq!(segments(&scene, "Feature edges"), 12 * n);
    // Applying it again changes nothing.
    let chunks = scene.chunks.len();
    use_result_view_edges(&mut scene);
    assert_eq!(scene.chunks.len(), chunks);
}
