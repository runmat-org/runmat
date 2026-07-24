use super::*;

pub(super) fn assert_incomplete_recovery(
    error: TetrahedronRecoveryError,
    expected_missing_item_count: usize,
    expected_missing_source_face_item_count: usize,
    expected_missing_source_edge_item_count: usize,
    expected_missing_material_interface_item_count: usize,
) -> StageEvidence {
    let TetrahedronRecoveryError::IncompleteRecovery {
        missing_item_count,
        missing_source_face_item_count,
        missing_source_edge_item_count,
        missing_material_interface_item_count,
        recovery_evidence,
    } = error
    else {
        panic!("expected incomplete recovery error, got {error:?}");
    };

    assert_eq!(missing_item_count, expected_missing_item_count);
    assert_eq!(
        missing_source_face_item_count,
        expected_missing_source_face_item_count
    );
    assert_eq!(
        missing_source_edge_item_count,
        expected_missing_source_edge_item_count
    );
    assert_eq!(
        missing_material_interface_item_count,
        expected_missing_material_interface_item_count
    );
    recovery_evidence
}

pub(super) fn tetrahedron_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "tetrahedron_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [0.0, 0.0, 1.0]),
        ],
        facets: vec![
            facet("facet_1", ["0", "2", "1"], "face_1"),
            facet("facet_2", ["0", "1", "3"], "face_2"),
            facet("facet_3", ["1", "2", "3"], "face_3"),
            facet("facet_4", ["2", "0", "3"], "face_4"),
        ],
        protected_edges: vec![PlcProtectedEdge {
            edge_id: entity(MeshingStage::ProtectedBoundaryComplex, "plc_edge_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            ],
            source_edge_id: entity(MeshingStage::CurveMesh, "edge_1"),
            cad_curve_boundary: None,
        }],
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

pub(super) fn cad_curve_tetrahedron_plc() -> ProtectedBoundaryComplex {
    let mut plc = tetrahedron_plc();
    plc.protected_edges[0].cad_curve_boundary = Some(cad_curve_boundary());
    plc
}

pub(super) fn cad_curve_boundary() -> PlcProtectedEdgeCadCurveBoundary {
    PlcProtectedEdgeCadCurveBoundary {
        cad_edge_id: "cad_edge_1".to_string(),
        imported_curve_id: Some(1),
        evaluator_id: Some("cad_curve_1".to_string()),
        evaluator_supports_point_evaluation: true,
        evaluator_supports_projection: true,
        evaluator_supports_tangent: true,
        evaluator_supports_curvature: true,
        evaluator_sample_count: 2,
        live_query_backed: true,
        live_query_sample_count: 1,
        rejected_evaluator_sample_count: 0,
        curvature_sample_count: 1,
        curvature_limited_target_size_m: Some(0.5),
        boundary_segment_count: 1,
    }
}

pub(super) fn tetrahedron_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "tetrahedron".to_string(),
        tetrahedron_generation_family: "unknown".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [0.0, 0.0, 1.0],
            ),
        ],
        elements: vec![Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            ],
            material_region_id: "solid_body".to_string(),
        }],
        boundary_faces: vec![
            boundary_face("facet_1", ["0", "2", "1"], "face_1"),
            boundary_face("facet_2", ["0", "1", "3"], "face_2"),
            boundary_face("facet_3", ["1", "2", "3"], "face_3"),
            boundary_face("facet_4", ["2", "0", "3"], "face_4"),
        ],
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

pub(super) fn boundary_diagonal_flip_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "boundary_diagonal_flip_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [1.0, 1.0, 0.0]),
            plc_node("4", [0.5, 0.5, 1.0]),
        ],
        facets: vec![
            facet("facet_1", ["0", "1", "2"], "face_1"),
            facet("facet_2", ["0", "3", "1"], "face_2"),
            facet("facet_3", ["0", "2", "4"], "face_3"),
            facet("facet_4", ["0", "4", "3"], "face_4"),
            facet("facet_5", ["1", "3", "4"], "face_5"),
            facet("facet_6", ["1", "4", "2"], "face_6"),
        ],
        protected_edges: vec![PlcProtectedEdge {
            edge_id: entity(MeshingStage::ProtectedBoundaryComplex, "plc_edge_1"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
            ],
            source_edge_id: entity(MeshingStage::CurveMesh, "edge_1"),
            cad_curve_boundary: None,
        }],
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

pub(super) fn source_face_boundary_diagonal_flip_plc() -> ProtectedBoundaryComplex {
    let mut plc = boundary_diagonal_flip_plc();
    plc.complex_id = "source_face_boundary_diagonal_flip_plc".to_string();
    plc.protected_edges = Vec::new();
    plc
}

pub(super) fn boundary_diagonal_flip_tetrahedron_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "boundary_diagonal_flip_tetrahedron".to_string(),
        tetrahedron_generation_family: "unknown".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [1.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                [0.5, 0.5, 1.0],
            ),
        ],
        elements: vec![
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_1"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "solid_body".to_string(),
            },
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_2"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "solid_body".to_string(),
            },
        ],
        boundary_faces: vec![
            boundary_face("old_facet_1", ["0", "2", "3"], "old_face_1"),
            boundary_face("old_facet_2", ["1", "3", "2"], "old_face_2"),
            boundary_face("facet_3", ["0", "2", "4"], "face_3"),
            boundary_face("facet_4", ["0", "4", "3"], "face_4"),
            boundary_face("facet_5", ["1", "3", "4"], "face_5"),
            boundary_face("facet_6", ["1", "4", "2"], "face_6"),
        ],
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

pub(super) fn two_region_bipyramid_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "two_region_bipyramid_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [1.0, 1.0, 0.0]),
            plc_node("4", [0.5, 0.5, 1.0]),
        ],
        facets: vec![
            facet_with_material("facet_a_1", ["0", "2", "3"], "face_a_1", "region_a"),
            facet_with_material("facet_a_2", ["0", "4", "2"], "face_a_2", "region_a"),
            facet_with_material("facet_a_3", ["0", "3", "4"], "face_a_3", "region_a"),
            facet_with_material("facet_b_1", ["1", "3", "2"], "face_b_1", "region_b"),
            facet_with_material("facet_b_2", ["1", "4", "3"], "face_b_2", "region_b"),
            facet_with_material("facet_b_3", ["1", "2", "4"], "face_b_3", "region_b"),
        ],
        protected_edges: Vec::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

pub(super) fn two_region_bipyramid_plc_with_region_a_protected_edge() -> ProtectedBoundaryComplex {
    let mut plc = two_region_bipyramid_plc();
    plc.complex_id = "two_region_bipyramid_with_protected_edge_plc".to_string();
    plc.protected_edges = vec![PlcProtectedEdge {
        edge_id: entity(
            MeshingStage::ProtectedBoundaryComplex,
            "plc_edge_region_a_0_2",
        ),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "2"),
        ],
        source_edge_id: entity(MeshingStage::CurveMesh, "edge_region_a_0_2"),
        cad_curve_boundary: None,
    }];
    plc
}

pub(super) fn two_region_bipyramid_tetrahedron_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "two_region_bipyramid_tetrahedron".to_string(),
        tetrahedron_generation_family: "unknown".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [1.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                [0.5, 0.5, 1.0],
            ),
        ],
        elements: vec![
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_a"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "region_a".to_string(),
            },
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_b"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "region_b".to_string(),
            },
        ],
        boundary_faces: vec![
            boundary_face("facet_a_1", ["0", "2", "3"], "face_a_1"),
            boundary_face("facet_a_2", ["0", "4", "2"], "face_a_2"),
            boundary_face("facet_a_3", ["0", "3", "4"], "face_a_3"),
            boundary_face("facet_b_1", ["1", "3", "2"], "face_b_1"),
            boundary_face("facet_b_2", ["1", "4", "3"], "face_b_2"),
            boundary_face("facet_b_3", ["1", "2", "4"], "face_b_3"),
        ],
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

pub(super) fn two_element_material_partition_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "two_element_material_partition_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [1.0, 1.0, 0.0]),
            plc_node("4", [0.5, 0.5, 1.0]),
        ],
        facets: vec![
            facet_with_material("facet_a_1", ["0", "1", "2"], "face_a_1", "region_a"),
            facet_with_material("facet_a_2", ["0", "3", "1"], "face_a_2", "region_a"),
            facet_with_material("facet_a_3", ["0", "2", "4"], "face_a_3", "region_a"),
            facet_with_material("facet_a_4", ["0", "4", "3"], "face_a_4", "region_a"),
            facet_with_material("facet_a_5", ["1", "3", "4"], "face_a_5", "region_a"),
            facet_with_material("facet_a_6", ["1", "4", "2"], "face_a_6", "region_a"),
        ],
        protected_edges: Vec::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

pub(super) fn two_element_material_partition_seed_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "two_element_material_partition_seed".to_string(),
        tetrahedron_generation_family: "unknown".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [1.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                [0.5, 0.5, 1.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "5"),
                [3.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "6"),
                [4.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "7"),
                [3.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "8"),
                [3.0, 0.0, 1.0],
            ),
        ],
        elements: vec![Tetrahedron4Element {
            element_id: entity(MeshingStage::TetrahedronMesh, "support_tetrahedron"),
            node_ids: [
                entity(MeshingStage::ProtectedBoundaryComplex, "5"),
                entity(MeshingStage::ProtectedBoundaryComplex, "6"),
                entity(MeshingStage::ProtectedBoundaryComplex, "7"),
                entity(MeshingStage::ProtectedBoundaryComplex, "8"),
            ],
            material_region_id: "unrelated_region".to_string(),
        }],
        boundary_faces: Vec::new(),
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

pub(super) fn add_unclassified_region_a_boundary_neighbor(mesh: &mut TetrahedronMesh) {
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::ProtectedBoundaryComplex, "5"),
        [0.5, 0.5, -1.0],
    ));
    mesh.elements.push(Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_unclassified"),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            entity(MeshingStage::ProtectedBoundaryComplex, "5"),
        ],
        material_region_id: "unclassified".to_string(),
    });
}

pub(super) fn replace_region_b_with_unclassified_region_a_interior_neighbor(
    mesh: &mut TetrahedronMesh,
) {
    mesh.nodes.push(tetrahedron_node(
        entity(MeshingStage::ProtectedBoundaryComplex, "5"),
        [0.5, 0.5, -1.0],
    ));
    mesh.elements[1] = Tetrahedron4Element {
        element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_unclassified"),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, "2"),
            entity(MeshingStage::ProtectedBoundaryComplex, "3"),
            entity(MeshingStage::ProtectedBoundaryComplex, "4"),
            entity(MeshingStage::ProtectedBoundaryComplex, "5"),
        ],
        material_region_id: "unclassified".to_string(),
    };
    mesh.boundary_faces
        .retain(|face| face.source_face_id.id.starts_with("face_a"));
}

pub(super) fn interior_material_interface_propagation_plc() -> ProtectedBoundaryComplex {
    ProtectedBoundaryComplex {
        complex_id: "interior_material_interface_propagation_plc".to_string(),
        nodes: vec![
            plc_node("0", [0.0, 0.0, 0.0]),
            plc_node("1", [1.0, 0.0, 0.0]),
            plc_node("2", [0.0, 1.0, 0.0]),
            plc_node("3", [0.0, 0.0, 1.0]),
            plc_node("4", [0.0, 0.0, -1.0]),
            plc_node("5", [2.0, 0.0, 0.0]),
            plc_node("6", [2.0, 1.0, 0.0]),
            plc_node("7", [2.0, 0.0, 1.0]),
        ],
        facets: vec![
            facet_with_material("facet_a", ["0", "1", "2"], "face_a", "region_a"),
            facet_with_material("facet_b", ["5", "6", "7"], "face_b", "region_b"),
        ],
        protected_edges: Vec::new(),
        validation: PlcValidationSummary {
            watertight: true,
            manifold: true,
            shell_nesting_classified: true,
            material_interfaces_classified: true,
        },
        evidence: StageEvidence::complete(MeshingStage::ProtectedBoundaryComplex),
    }
}

pub(super) fn interior_material_interface_propagation_mesh() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "interior_material_interface_propagation_tetrahedron".to_string(),
        tetrahedron_generation_family: "unknown".to_string(),
        nodes: vec![
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                [0.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                [1.0, 0.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                [0.0, 1.0, 0.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                [0.0, 0.0, 1.0],
            ),
            tetrahedron_node(
                entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                [0.0, 0.0, -1.0],
            ),
        ],
        elements: vec![
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_seed"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "2"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                ],
                material_region_id: "unclassified".to_string(),
            },
            Tetrahedron4Element {
                element_id: entity(MeshingStage::TetrahedronMesh, "tetrahedron_interior"),
                node_ids: [
                    entity(MeshingStage::ProtectedBoundaryComplex, "0"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "1"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "3"),
                    entity(MeshingStage::ProtectedBoundaryComplex, "4"),
                ],
                material_region_id: "unclassified".to_string(),
            },
        ],
        boundary_faces: Vec::new(),
        recovery_complete: false,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

pub(super) fn plc_node(id: &str, coordinates_m: [f64; 3]) -> PlcNode {
    PlcNode {
        node_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        coordinates_m,
    }
}

pub(super) fn facet(id: &str, node_ids: [&str; 3], source_face_id: &str) -> PlcFacet {
    facet_with_material(id, node_ids, source_face_id, "solid_body")
}

pub(super) fn facet_with_material(
    id: &str,
    node_ids: [&str; 3],
    source_face_id: &str,
    material_interface_id: &str,
) -> PlcFacet {
    PlcFacet {
        facet_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        node_ids: [
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
            entity(MeshingStage::ProtectedBoundaryComplex, node_ids[2]),
        ],
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
        material_interface_ids: vec![material_interface_id.to_string()],
    }
}

pub(super) fn boundary_face(
    id: &str,
    node_ids: [&str; 3],
    source_face_id: &str,
) -> TetrahedronBoundaryFace {
    let node_ids = [
        entity(MeshingStage::ProtectedBoundaryComplex, node_ids[0]),
        entity(MeshingStage::ProtectedBoundaryComplex, node_ids[1]),
        entity(MeshingStage::ProtectedBoundaryComplex, node_ids[2]),
    ];
    TetrahedronBoundaryFace {
        face_id: entity(MeshingStage::ProtectedBoundaryComplex, id),
        source_edge_ids: source_edge_ids(node_ids.clone()),
        node_ids,
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
    }
}

pub(super) fn boundary_face_from_ids(
    id: &str,
    node_ids: [TopologyEntityId; 3],
    source_face_id: &str,
    source_edge_ids: [Option<TopologyEntityId>; 3],
) -> TetrahedronBoundaryFace {
    TetrahedronBoundaryFace {
        face_id: entity(MeshingStage::TetrahedronMesh, id),
        node_ids,
        source_face_id: entity(MeshingStage::SurfaceMesh, source_face_id),
        source_edge_ids,
    }
}

pub(super) fn tetrahedron_node(
    node_id: TopologyEntityId,
    coordinates_m: [f64; 3],
) -> TetrahedronMeshNode {
    TetrahedronMeshNode {
        node_id,
        coordinates_m,
    }
}

pub(super) fn sorted_face_ids(mut node_ids: [TopologyEntityId; 3]) -> [TopologyEntityId; 3] {
    node_ids.sort();
    node_ids
}

pub(super) fn source_edge_ids(node_ids: [TopologyEntityId; 3]) -> [Option<TopologyEntityId>; 3] {
    [
        source_edge_id_for_edge([node_ids[0].clone(), node_ids[1].clone()]),
        source_edge_id_for_edge([node_ids[1].clone(), node_ids[2].clone()]),
        source_edge_id_for_edge([node_ids[2].clone(), node_ids[0].clone()]),
    ]
}

pub(super) fn source_edge_id_for_edge(
    mut node_ids: [TopologyEntityId; 2],
) -> Option<TopologyEntityId> {
    node_ids.sort();
    (node_ids
        == [
            entity(MeshingStage::ProtectedBoundaryComplex, "0"),
            entity(MeshingStage::ProtectedBoundaryComplex, "1"),
        ])
    .then(|| entity(MeshingStage::CurveMesh, "edge_1"))
}

pub(super) fn entity(stage: MeshingStage, id: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage,
        id: id.to_string(),
    }
}
