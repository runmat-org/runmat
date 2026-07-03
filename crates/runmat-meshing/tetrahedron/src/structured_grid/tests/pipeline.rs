use super::common::*;
use super::*;

#[test]
fn structured_tetrahedron_mesher_generates_valid_analysis_mesh() {
    let geometry = cube_geometry();
    let mesh = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("cube should produce an analysis mesh");

    validate_analysis_mesh(&mesh, Default::default()).expect("analysis mesh should validate");
    assert_eq!(mesh.schema_version, ANALYSIS_MESH_SCHEMA_VERSION);
    assert_eq!(mesh.nodes.len(), 27);
    assert_eq!(mesh.volume_elements.len(), 48);
    assert_eq!(mesh.boundary_faces.len(), 48);
    assert!(mesh.quality.mean_boundary_projection_error_m <= 1.0e-12);
    assert!(mesh.quality.max_boundary_projection_error_m <= 1.0e-12);
    assert!(mesh.boundary_faces.iter().any(|face| {
        face.region_ids
            .iter()
            .any(|region| region == "region_fixed")
    }));
    assert!(mesh
        .boundary_faces
        .iter()
        .any(|face| face.region_ids.iter().any(|region| region == "region_load")));
    assert!(mesh
        .quality
        .elements
        .iter()
        .all(|quality| quality.volume_m3 > 0.0));
    assert!(mesh
        .boundary_faces
        .iter()
        .all(|face| !face.adjacent_volume_element_ids.is_empty()));
    assert!(mesh.boundary_faces.iter().all(|face| {
        face.adjacent_volume_element_ids.iter().all(|element_id| {
            mesh.volume_elements
                .iter()
                .any(|element| element.element_id == *element_id)
        })
    }));
    assert!(mesh.volume_elements.iter().all(|element| {
        tetrahedron_volume(
            [
                element.node_ids[0],
                element.node_ids[1],
                element.node_ids[2],
                element.node_ids[3],
            ],
            &mesh.nodes,
        ) > 0.0
    }));
}
#[test]
fn target_size_controls_structured_tetrahedron_density() {
    let geometry = cube_geometry();
    let coarse = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            kind: MeshKindRequest::Solid,
            target_size: MeshTargetSize::LengthM(1.0),
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("coarse mesh should generate");
    let fine = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            kind: MeshKindRequest::Solid,
            target_size: MeshTargetSize::LengthM(0.25),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("fine mesh should generate");

    assert!(fine.volume_elements.len() > coarse.volume_elements.len());
    assert!(fine.nodes.len() > coarse.nodes.len());
}
#[test]
fn structured_tetrahedron_mesher_carves_cells_outside_closed_surface() {
    let geometry = tetrahedron_geometry();
    let mesh = generate_analysis_mesh(
        &geometry,
        VolumeMeshingOptions {
            backend: MeshBackendKind::StructuredGridTetrahedron,
            kind: MeshKindRequest::Solid,
            target_size: MeshTargetSize::LengthM(0.25),
            max_elements: 10_000,
            ..VolumeMeshingOptions::default()
        },
    )
    .expect("tetrahedron should produce an analysis mesh");

    validate_analysis_mesh(&mesh, Default::default()).expect("carved mesh should validate");
    assert!(!mesh.volume_elements.is_empty());
    assert!(mesh.volume_elements.len() < 4 * 4 * 4 * 6);
    assert!(mesh.nodes.len() < 5 * 5 * 5);
    assert!(all_nodes_are_referenced(&mesh));
    assert!(mesh.quality.mean_boundary_projection_error_m.is_finite());
    assert!(mesh.quality.max_boundary_projection_error_m.is_finite());
    assert!(mesh.quality.max_boundary_projection_error_m > 0.0);
    assert!(mesh.volume_elements.iter().all(|element| {
        let centroid = tetrahedron_centroid(
            [
                element.node_ids[0],
                element.node_ids[1],
                element.node_ids[2],
                element.node_ids[3],
            ],
            &mesh.nodes,
        );
        point_inside_closed_surface(
            &BoundaryMeshInput::from_geometry(&geometry).expect("boundary input"),
            centroid,
        )
    }));
    assert!(mesh.boundary_faces.len() < 6 * 4 * 4 * 2);
    assert!(mesh
        .boundary_faces
        .iter()
        .all(|face| !face.adjacent_volume_element_ids.is_empty()));
}
