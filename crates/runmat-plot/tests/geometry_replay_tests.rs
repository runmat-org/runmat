//! Scene replay must retain visible results, not just successful deserialization.
use glam::{Vec3, Vec4};
use runmat_plot::{
    core::{Material, PipelineType, Vertex},
    event::FigureScene,
    geometry_scene::{GeometrySceneCompleteness, GeometrySceneOverlay, GeometrySceneScalarLegend},
    GeometryScene, GeometrySceneChunk,
};
#[test]
fn geometry_replay_preserves_field_colors_edges_and_legend() {
    let vertices = vec![
        Vertex::new(Vec3::ZERO, Vec4::new(1.0, 0.0, 0.0, 1.0)),
        Vertex::new(Vec3::X, Vec4::new(0.0, 1.0, 0.0, 1.0)),
        Vertex::new(Vec3::Z, Vec4::new(0.0, 0.0, 1.0, 1.0)),
    ];
    let faces = GeometrySceneChunk::indexed_triangles(
        "faces",
        vertices,
        vec![0, 1, 2],
        Material::default(),
    );
    let mut data = faces.render_data.clone();
    data.pipeline_type = PipelineType::Lines;
    data.indices = Some(vec![0, 1, 1, 2]);
    let lines = GeometrySceneChunk::from_render_data("edges", data);
    let mut original = GeometryScene::new("original", 1, vec![faces, lines]);
    original.overlay = Some(GeometrySceneOverlay {
        source_name: None,
        status: GeometrySceneCompleteness::Complete,
        quality_label: None,
        format: None,
        source_label: None,
        allow_create_fea_study: false,
        byte_count: None,
        mesh_count: 1,
        vertex_count: 3,
        triangle_count: 1,
        progress_percent: None,
        scalar_legend: Some(GeometrySceneScalarLegend {
            label: "Stress".into(),
            unit: "Pa".into(),
            minimum: 0.0,
            maximum: 177.0,
            limit: None,
            note: None,
        }),
        region_count: 0,
        mapped_region_count: 0,
        assembly_nodes: vec![],
        regions: vec![],
        warnings: vec![],
    });
    let bytes = serde_json::to_vec(&FigureScene::from_geometry_scene(&original)).unwrap();
    let restored = serde_json::from_slice::<FigureScene>(&bytes)
        .unwrap()
        .into_geometry_scene("cold", 2)
        .unwrap();
    assert_eq!(restored.chunks.len(), 2);
    assert_eq!(restored.title, original.title);
    for (before, after) in original.chunks[0]
        .vertices
        .iter()
        .zip(&restored.chunks[0].vertices)
    {
        assert_eq!(before.position, after.position);
        assert_eq!(before.color, after.color);
    }
    assert_eq!(restored.chunks[1].indices, original.chunks[1].indices);
    assert_eq!(
        restored.chunks[1].render_data.pipeline_type,
        PipelineType::Lines
    );
    let legend = restored.overlay.unwrap().scalar_legend.unwrap();
    assert_eq!(legend.label, "Stress");
    assert_eq!(legend.unit, "Pa");
    assert_eq!(legend.maximum, 177.0);
}

#[test]
fn malformed_saved_edges_are_rejected() {
    let scene = runmat_plot::plots::Figure::new();
    let mut saved = FigureScene::capture(&scene);
    saved
        .geometry_lines
        .push(runmat_plot::event::GeometryLines {
            positions: vec![[0.0; 3]],
            colors: vec![[1.0; 4]],
            indices: Some(vec![0, 4]),
            mesh_id: None,
            label: None,
            visible: true,
            wireframe_only: false,
        });
    assert!(saved.into_geometry_scene("invalid", 1).is_err());
}
