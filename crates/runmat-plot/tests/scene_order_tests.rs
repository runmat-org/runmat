//! Draw order: see-through surfaces (glass, a clear lid) draw after every opaque surface, so
//! they blend over what is behind them; depth-free overlays draw last.

use glam::{Mat4, Vec4};
use runmat_plot::core::{
    AlphaMode, BoundingBox, Material, PipelineType, RenderData, Scene, SceneNode,
};

fn node(name: &str, pipeline_type: PipelineType, albedo: Vec4, alpha_mode: AlphaMode) -> SceneNode {
    SceneNode {
        id: 0,
        name: name.into(),
        transform: Mat4::IDENTITY,
        visible: true,
        cast_shadows: false,
        receive_shadows: false,
        axes_index: 0,
        parent: None,
        children: Vec::new(),
        render_data: Some(RenderData {
            pipeline_type,
            vertices: Vec::new(),
            indices: None,
            gpu_vertices: None,
            bounds: None,
            material: Material {
                albedo,
                alpha_mode,
                ..Material::default()
            },
            draw_calls: Vec::new(),
            image: None,
        }),
        bounds: BoundingBox::default(),
        lod_levels: Vec::new(),
        current_lod: 0,
        instances: None,
    }
}

#[test]
fn see_through_surfaces_draw_after_opaque_ones_and_overlays_last() {
    let mut scene = Scene::new();
    let solid = Vec4::new(0.8, 0.8, 0.8, 1.0);
    scene.add_node(node(
        "housing",
        PipelineType::Triangles,
        solid,
        AlphaMode::Opaque,
    ));
    scene.add_node(node(
        "lid",
        PipelineType::Triangles,
        Vec4::new(0.55, 0.68, 0.78, 0.35),
        AlphaMode::Opaque,
    ));
    scene.add_node(node(
        "label",
        PipelineType::LinesNoDepth,
        solid,
        AlphaMode::Opaque,
    ));
    scene.add_node(node(
        "glass",
        PipelineType::Triangles,
        solid,
        AlphaMode::Blend,
    ));
    scene.add_node(node(
        "board",
        PipelineType::Triangles,
        solid,
        AlphaMode::Opaque,
    ));
    scene.add_node(node("edges", PipelineType::Lines, solid, AlphaMode::Opaque));
    let order: Vec<&str> = scene
        .get_visible_nodes()
        .iter()
        .map(|n| n.name.as_str())
        .collect();
    assert_eq!(
        order,
        ["housing", "board", "edges", "lid", "glass", "label"]
    );
}
