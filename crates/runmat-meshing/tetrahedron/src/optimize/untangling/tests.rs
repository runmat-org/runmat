use super::*;
use runmat_meshing_core::contracts::{
    MeshingStage, StageEvidence, TetrahedronBoundaryFace, TetrahedronMeshNode,
};

#[test]
fn untangling_relocates_interior_near_singular_node() {
    let mut mesh = untangling_fixture([0.001, 0.001, 0.000001]);

    let report = untangle_tetrahedron_mesh_interior(
        &mut mesh,
        TetrahedronMeshUntanglingOptions {
            untangling: TetrahedronUntanglingOptions {
                near_singular_scaled_jacobian: 0.05,
                min_scaled_jacobian_improvement: 1.0e-12,
            },
            max_attempted_seeds: 4,
            max_relocated_seeds: 1,
            relaxation: 1.0,
        },
    );

    assert!(report.initial_near_singular_count > report.final_near_singular_count);
    assert_eq!(report.attempted_seed_count, 1);
    assert_eq!(report.relocated_seed_count, 1);
    assert_eq!(report.rejected_seed_count, 0);
    assert!(mesh.quality_optimized);
    assert_ne!(mesh.nodes[4].coordinates_m, [0.001, 0.001, 0.000001]);
    assert_eq!(
        mesh.evidence.entity_counts[TETRAHEDRON_UNTANGLING_PASS_COUNT],
        1
    );
    assert_eq!(
        mesh.evidence.entity_counts[TETRAHEDRON_UNTANGLING_RELOCATED_SEED_COUNT],
        1
    );
}

#[test]
fn untangling_rejects_boundary_only_near_singular_seed() {
    let mut mesh = boundary_only_near_singular_fixture();

    let report = untangle_tetrahedron_mesh_interior(
        &mut mesh,
        TetrahedronMeshUntanglingOptions {
            untangling: TetrahedronUntanglingOptions {
                near_singular_scaled_jacobian: 0.05,
                min_scaled_jacobian_improvement: 1.0e-12,
            },
            max_attempted_seeds: 4,
            max_relocated_seeds: 1,
            relaxation: 1.0,
        },
    );

    assert_eq!(report.attempted_seed_count, 1);
    assert_eq!(report.relocated_seed_count, 0);
    assert_eq!(report.rejected_seed_count, 1);
    assert_eq!(report.rejected_by_reason["no_accepted_relocation"], 1);
    assert!(mesh.quality_optimized);
}

#[test]
fn untangling_records_initial_and_final_near_singular_counts_when_budget_is_zero() {
    let mut mesh = untangling_fixture([0.001, 0.001, 0.000001]);

    let report = untangle_tetrahedron_mesh_interior(
        &mut mesh,
        TetrahedronMeshUntanglingOptions {
            max_attempted_seeds: 0,
            ..TetrahedronMeshUntanglingOptions::default()
        },
    );

    assert_eq!(report.attempted_seed_count, 0);
    assert_eq!(report.relocated_seed_count, 0);
    assert!(report.initial_near_singular_count > 0);
    assert_eq!(
        report.initial_near_singular_count,
        report.final_near_singular_count
    );
    assert!(!mesh.quality_optimized);
    assert_eq!(
        mesh.evidence.entity_counts[TETRAHEDRON_UNTANGLING_INITIAL_NEAR_SINGULAR_COUNT],
        report.initial_near_singular_count
    );
    assert_eq!(
        mesh.evidence.entity_counts[TETRAHEDRON_UNTANGLING_FINAL_NEAR_SINGULAR_COUNT],
        report.final_near_singular_count
    );
}

fn untangling_fixture(interior: Point3) -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "untangling_fixture".to_string(),
        tetrahedron_generation_family: "unknown".to_string(),
        nodes: vec![
            node("0", [0.0, 0.0, 0.0]),
            node("1", [1.0, 0.0, 0.0]),
            node("2", [0.0, 1.0, 0.0]),
            node("3", [0.0, 0.0, 1.0]),
            node("4", interior),
        ],
        elements: vec![
            element("0", ["4", "0", "1", "2"]),
            element("1", ["4", "0", "1", "3"]),
            element("2", ["4", "0", "2", "3"]),
            element("3", ["4", "1", "2", "3"]),
        ],
        boundary_faces: vec![
            boundary_face("boundary_0", ["0", "1", "2"]),
            boundary_face("boundary_1", ["0", "1", "3"]),
            boundary_face("boundary_2", ["0", "2", "3"]),
            boundary_face("boundary_3", ["1", "2", "3"]),
        ],
        recovery_complete: true,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

fn boundary_only_near_singular_fixture() -> TetrahedronMesh {
    TetrahedronMesh {
        mesh_id: "boundary_only_untangling_fixture".to_string(),
        tetrahedron_generation_family: "unknown".to_string(),
        nodes: vec![
            node("0", [0.0, 0.0, 0.0]),
            node("1", [1.0, 0.0, 0.0]),
            node("2", [0.0, 1.0, 0.0]),
            node("3", [0.001, 0.001, 0.000001]),
        ],
        elements: vec![element("0", ["3", "0", "1", "2"])],
        boundary_faces: vec![
            boundary_face("boundary_0", ["0", "1", "2"]),
            boundary_face("boundary_1", ["3", "0", "1"]),
            boundary_face("boundary_2", ["3", "0", "2"]),
            boundary_face("boundary_3", ["3", "1", "2"]),
        ],
        recovery_complete: true,
        quality_optimized: false,
        evidence: StageEvidence::complete(MeshingStage::TetrahedronMesh),
    }
}

fn element(id: &str, node_ids: [&str; 4]) -> Tetrahedron4Element {
    Tetrahedron4Element {
        element_id: entity(id),
        node_ids: node_ids.map(entity),
        material_region_id: "body".to_string(),
    }
}

fn boundary_face(id: &str, node_ids: [&str; 3]) -> TetrahedronBoundaryFace {
    TetrahedronBoundaryFace {
        face_id: entity(id),
        node_ids: node_ids.map(entity),
        source_face_id: entity(id),
        source_edge_ids: [None, None, None],
    }
}

fn node(id: &str, coordinates_m: Point3) -> TetrahedronMeshNode {
    TetrahedronMeshNode {
        node_id: entity(id),
        coordinates_m,
    }
}

fn entity(id: &str) -> TopologyEntityId {
    TopologyEntityId {
        stage: MeshingStage::TetrahedronMesh,
        id: id.to_string(),
    }
}
