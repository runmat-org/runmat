//! Mesh plot tests.

use super::*;

fn triangle_mesh() -> MeshPlot {
    MeshPlot::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![[0, 1, 2]],
    )
    .expect("mesh should be valid")
}

fn square_mesh() -> MeshPlot {
    MeshPlot::new(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        vec![[0, 1, 2], [0, 2, 3]],
    )
    .expect("mesh should be valid")
}

#[test]
fn scalar_field_colors_mesh_vertices() {
    let mut mesh = triangle_mesh();
    let mut field = MeshScalarField::new(
        "fea.structural.von_mises",
        MeshFieldLocation::Vertex,
        vec![0.0, 0.5, 1.0],
    );
    field.color_limits = Some([0.0, 1.0]);
    field.alpha = 0.75;
    mesh.set_scalar_field(Some(field))
        .expect("scalar field should be accepted");

    let render = mesh.render_data();
    assert_eq!(render.vertices.len(), 3);
    assert_ne!(render.vertices[0].color, render.vertices[2].color);
    assert!((render.vertices[0].color[3] - 0.75).abs() < f32::EPSILON);
}

#[test]
fn deformation_updates_render_positions_and_bounds() {
    let mut mesh = triangle_mesh();
    let mut deformation = MeshDeformation::new(
        "fea.structural.displacement",
        vec![Vec3::ZERO, Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO],
    );
    deformation.scale = 0.5;
    mesh.set_deformation(Some(deformation))
        .expect("deformation should be accepted");

    let bounds = mesh.bounds();
    assert_eq!(bounds.max.z, 1.0);
    let render = mesh.render_data();
    assert!(render
        .vertices
        .iter()
        .any(|vertex| (vertex.position[2] - 1.0).abs() < f32::EPSILON));
}

#[test]
fn vector_field_generates_line_glyphs() {
    let mut mesh = triangle_mesh();
    let mut field = MeshVectorField::new(
        "fea.em.flux_density",
        MeshFieldLocation::Vertex,
        vec![Vec3::X, Vec3::ZERO, Vec3::Y],
    );
    field.scale = 0.25;
    mesh.set_vector_field(Some(field))
        .expect("vector field should be accepted");

    let render = mesh.vector_render_data().expect("vector glyphs");
    assert_eq!(render.pipeline_type, PipelineType::Lines);
    assert_eq!(render.vertices.len(), 4);
}

#[test]
fn feature_edge_mode_suppresses_internal_edges_with_same_group() {
    let mut mesh = square_mesh();
    mesh.set_edge_mode(MeshEdgeMode::Feature);
    mesh.set_feature_edge_groups(Some(vec![7, 7]))
        .expect("feature groups should be accepted");

    let render = mesh.edge_render_data().expect("feature edges");

    assert_eq!(render.pipeline_type, PipelineType::Lines);
    assert_eq!(render.vertices.len(), 8);
}

#[test]
fn feature_edge_mode_keeps_edges_between_groups() {
    let mut mesh = square_mesh();
    mesh.set_edge_mode(MeshEdgeMode::Feature);
    mesh.set_feature_edge_groups(Some(vec![7, 9]))
        .expect("feature groups should be accepted");

    let render = mesh.edge_render_data().expect("feature edges");

    assert_eq!(render.pipeline_type, PipelineType::Lines);
    assert_eq!(render.vertices.len(), 10);
}

#[test]
fn triangle_colors_generate_per_face_vertices() {
    let mut mesh = square_mesh();
    mesh.set_triangle_colors(Some(vec![
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
    ]))
    .expect("triangle colors should be accepted");

    let render = mesh.render_data();

    assert_eq!(render.vertices.len(), 6);
    assert_eq!(render.vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(render.vertices[3].color, [0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn vertex_colors_preserve_indexed_mesh_geometry() {
    let mut mesh = square_mesh();
    mesh.set_vertex_colors(Some(vec![
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        Vec4::new(1.0, 1.0, 0.0, 1.0),
    ]))
    .expect("vertex colors should be accepted");

    let render = mesh.render_data();

    assert_eq!(render.vertices.len(), 4);
    assert_eq!(render.indices.as_ref().unwrap().len(), 6);
    assert_eq!(render.vertices[2].color, [0.0, 0.0, 1.0, 1.0]);
}
#[test]
fn vertex_normals_preserve_orientation_across_coordinate_scales() {
    for scale in [1.0e-9, 1.0e-3, 1.0, 1.0e9] {
        let vertices = [Vec3::ZERO, Vec3::Y * scale, Vec3::Z * scale];
        for normal in vertex_normals(&vertices, &[[0, 1, 2]]) {
            assert!(
                (normal - Vec3::X).length() < 1.0e-6,
                "scale {scale}: {normal:?}"
            );
        }
    }
}

#[test]
fn vertex_normals_keep_area_weighting_and_finite_degenerate_fallback() {
    let vertices = [Vec3::ZERO, Vec3::Y, Vec3::Z, Vec3::X * 2.0];
    let normals = vertex_normals(&vertices, &[[0, 1, 2], [0, 2, 3], [0, 0, 0]]);
    assert!((normals[0] - Vec3::new(1.0, 2.0, 0.0).normalize()).length() < 1.0e-6);
    assert_eq!(vertex_normals(&[Vec3::ZERO], &[[0, 0, 0]]), vec![Vec3::Z]);
}

#[test]
fn small_vectors_remain_visible_after_display_scaling() {
    for location in [MeshFieldLocation::Vertex, MeshFieldLocation::Triangle] {
        let mut mesh = square_mesh();
        let count = match location {
            MeshFieldLocation::Vertex => 4,
            MeshFieldLocation::Triangle => 2,
        };
        let mut field =
            MeshVectorField::new("displacement", location, vec![Vec3::X * 1.0e-6; count]);
        field.scale = 1.0e6;
        mesh.set_vector_field(Some(field))
            .expect("valid vector field");
        let data = mesh
            .vector_render_data()
            .expect("small vectors must render");
        assert_eq!(data.vertices.len(), count * 2);
        for segment in data.vertices.chunks_exact(2) {
            let displacement =
                Vec3::from_array(segment[1].position) - Vec3::from_array(segment[0].position);
            assert!((displacement - Vec3::X).length() < 1.0e-6);
        }
    }
}
