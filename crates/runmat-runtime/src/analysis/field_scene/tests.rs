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
