use super::*;
use runmat_meshing_core::{
    contracts::{
        AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode,
        AnalysisMeshProvenance, AnalysisVolumeElement, BoundaryElementKind, MeshBackendSummary,
        VolumeElementKind,
    },
    quality::AnalysisMeshQualityReport,
    size::field::MeshSizingField,
};

#[test]
fn maps_element_scalar_values_to_boundary_faces() {
    let mesh = field_mapping_mesh();

    let values = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0, 20.0])
        .expect("boundary scalar mapping should succeed");

    assert_eq!(
        values,
        vec![
            BoundaryFaceScalarValue {
                face_id: "bf1".to_string(),
                value: 10.0,
            },
            BoundaryFaceScalarValue {
                face_id: "bf2".to_string(),
                value: 15.0,
            },
        ]
    );
}

#[test]
fn maps_nodal_vector_values_to_boundary_nodes() {
    let mesh = field_mapping_mesh();

    let values = map_nodal_vector_field_to_boundary_nodes(&mesh, &nodal_vector_values())
        .expect("boundary node vector mapping should succeed");

    assert_eq!(
        values,
        vec![
            BoundaryNodeVectorValue {
                node_id: 1,
                value: [1.0, 0.0, 0.0],
            },
            BoundaryNodeVectorValue {
                node_id: 2,
                value: [2.0, 0.0, 0.0],
            },
            BoundaryNodeVectorValue {
                node_id: 3,
                value: [3.0, 0.0, 0.0],
            },
            BoundaryNodeVectorValue {
                node_id: 4,
                value: [4.0, 0.0, 0.0],
            },
        ]
    );
}

#[test]
fn maps_nodal_vector_values_to_boundary_faces() {
    let mesh = field_mapping_mesh();

    let values = map_nodal_vector_field_to_boundary_faces(&mesh, &nodal_vector_values())
        .expect("boundary face vector mapping should succeed");

    assert_eq!(
        values,
        vec![
            BoundaryFaceVectorValue {
                face_id: "bf1".to_string(),
                value: [7.0 / 3.0, 0.0, 0.0],
            },
            BoundaryFaceVectorValue {
                face_id: "bf2".to_string(),
                value: [2.0, 0.0, 0.0],
            },
        ]
    );
}

#[test]
fn rejects_unmapped_boundary_faces() {
    let mut mesh = field_mapping_mesh();
    mesh.boundary_faces[0].adjacent_volume_element_ids.clear();

    let err = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0, 20.0])
        .expect_err("missing adjacency should fail");

    assert_eq!(
        err,
        FieldMappingError::BoundaryFaceMissingAdjacentVolume {
            face_id: "bf1".to_string(),
        }
    );
}

#[test]
fn rejects_element_field_length_mismatch() {
    let mesh = field_mapping_mesh();

    let err = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0])
        .expect_err("field length mismatch should fail");

    assert_eq!(
        err,
        FieldMappingError::ElementFieldLengthMismatch {
            element_value_count: 1,
            volume_element_count: 2,
        }
    );
}

#[test]
fn rejects_node_vector_field_length_mismatch() {
    let mesh = field_mapping_mesh();

    let err = map_nodal_vector_field_to_boundary_nodes(&mesh, &[[1.0, 0.0, 0.0]])
        .expect_err("node field length mismatch should fail");

    assert_eq!(
        err,
        FieldMappingError::NodeVectorFieldLengthMismatch {
            node_value_count: 1,
            node_count: 5,
        }
    );
}

#[test]
fn rejects_nonfinite_node_vector_values() {
    let mesh = field_mapping_mesh();
    let mut values = nodal_vector_values();
    values[2][1] = f64::INFINITY;

    let err = map_nodal_vector_field_to_boundary_faces(&mesh, &values)
        .expect_err("nonfinite node vector should fail");

    assert_eq!(
        err,
        FieldMappingError::NonFiniteNodeVectorValue {
            node_index: 2,
            component_index: 1,
        }
    );
}

#[test]
fn rejects_nonfinite_element_values() {
    let mesh = field_mapping_mesh();

    let err = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0, f64::NAN])
        .expect_err("nonfinite element value should fail");

    assert_eq!(
        err,
        FieldMappingError::NonFiniteElementValue { element_index: 1 }
    );
}

#[test]
fn rejects_boundary_faces_referencing_unknown_volume_elements() {
    let mut mesh = field_mapping_mesh();
    mesh.boundary_faces[0].adjacent_volume_element_ids = vec!["missing".to_string()];

    let err = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0, 20.0])
        .expect_err("unknown adjacent volume element should fail");

    assert_eq!(
        err,
        FieldMappingError::BoundaryFaceReferencesUnknownVolume {
            face_id: "bf1".to_string(),
            volume_element_id: "missing".to_string(),
        }
    );
}

#[test]
fn rejects_boundary_faces_referencing_unknown_nodes() {
    let mut mesh = field_mapping_mesh();
    mesh.boundary_faces[0].node_ids = vec![1, 2, 99];

    let err = map_nodal_vector_field_to_boundary_faces(&mesh, &nodal_vector_values())
        .expect_err("unknown boundary face node should fail");

    assert_eq!(
        err,
        FieldMappingError::BoundaryFaceReferencesUnknownNode {
            face_id: "bf1".to_string(),
            node_id: 99,
        }
    );
}

#[test]
fn rejects_boundary_edges_referencing_unknown_nodes() {
    let mut mesh = field_mapping_mesh();
    mesh.boundary_edges.push(AnalysisBoundaryEdge {
        edge_id: "be1".to_string(),
        node_ids: [1, 99],
        adjacent_boundary_face_ids: Vec::new(),
        region_ids: Vec::new(),
        provenance: Vec::new(),
    });

    let err = map_nodal_vector_field_to_boundary_nodes(&mesh, &nodal_vector_values())
        .expect_err("unknown boundary edge node should fail");

    assert_eq!(
        err,
        FieldMappingError::BoundaryEdgeReferencesUnknownNode {
            edge_id: "be1".to_string(),
            node_id: 99,
        }
    );
}

#[test]
fn rejects_boundary_faces_without_nodes() {
    let mut mesh = field_mapping_mesh();
    mesh.boundary_faces[0].node_ids.clear();

    let err = map_nodal_vector_field_to_boundary_faces(&mesh, &nodal_vector_values())
        .expect_err("empty boundary face should fail");

    assert_eq!(
        err,
        FieldMappingError::BoundaryFaceHasNoNodes {
            face_id: "bf1".to_string(),
        }
    );
}

fn nodal_vector_values() -> Vec<[f64; 3]> {
    vec![
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ]
}

fn field_mapping_mesh() -> AnalysisMeshArtifact {
    AnalysisMeshArtifact {
        schema_version: "analysis-mesh/v1".to_string(),
        mesh_id: "field_mapping_fixture".to_string(),
        nodes: vec![
            AnalysisMeshNode {
                node_id: 1,
                coordinates_m: [0.0, 0.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 2,
                coordinates_m: [1.0, 0.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 3,
                coordinates_m: [0.0, 1.0, 0.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 4,
                coordinates_m: [0.0, 0.0, 1.0],
                provenance: Vec::new(),
            },
            AnalysisMeshNode {
                node_id: 5,
                coordinates_m: [0.0, 0.0, -1.0],
                provenance: Vec::new(),
            },
        ],
        volume_elements: vec![
            AnalysisVolumeElement {
                element_id: "e1".to_string(),
                kind: VolumeElementKind::Tetrahedron4,
                node_ids: vec![1, 2, 3, 4],
                material_region_id: "mat".to_string(),
                provenance: Vec::new(),
            },
            AnalysisVolumeElement {
                element_id: "e2".to_string(),
                kind: VolumeElementKind::Tetrahedron4,
                node_ids: vec![1, 3, 2, 5],
                material_region_id: "mat".to_string(),
                provenance: Vec::new(),
            },
        ],
        boundary_faces: vec![
            AnalysisBoundaryFace {
                face_id: "bf1".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 4],
                adjacent_volume_element_ids: vec!["e1".to_string()],
                region_ids: Vec::new(),
                provenance: Vec::new(),
            },
            AnalysisBoundaryFace {
                face_id: "bf2".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 3],
                adjacent_volume_element_ids: vec!["e1".to_string(), "e2".to_string()],
                region_ids: Vec::new(),
                provenance: Vec::new(),
            },
        ],
        boundary_edges: Vec::new(),
        quality: AnalysisMeshQualityReport::default(),
        sizing: MeshSizingField::default(),
        field_topology: Vec::new(),
        backend: MeshBackendSummary::default(),
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "fixture".to_string(),
            source_geometry_id: "field_mapping_fixture".to_string(),
            source_geometry_revision: 1,
            source_geometry_sha256: None,
        },
    }
}
