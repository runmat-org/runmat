use super::*;
use crate::recover::{TetrahedronRecoveryKind, TetrahedronRecoveryStatus};

#[test]
fn generated_split_edge_box_recovers_without_split_refill_edits() {
    let plc = split_edge_box_plc();
    let mesh = generate_solver_tetrahedron_mesh_from_plc(&plc)
        .expect("split-edge box PLC should generate a solver Tetrahedron mesh");

    assert_eq!(
        mesh.mesh_id,
        "structured_box_boundary_conforming_tetrahedron_mesh"
    );
    assert_eq!(mesh.evidence.entity_counts["input_plc_protected_edges"], 2);

    let result = crate::recover::recover_tetrahedron_mesh_from_plc(&plc, mesh)
        .expect("generated split-edge box should recover all PLC constraints");

    assert!(result.tetrahedron_mesh.recovery_complete);
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["source_edge_items"],
        plc.protected_edges.len()
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["missing_source_edge_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["recovered_source_edge_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts
            ["post_repair_attempted_source_edge_split_refill_items"],
        0
    );
    assert_eq!(
        result.recovery_queue.evidence.entity_counts["applied_source_edge_split_refill_items"],
        0
    );
    assert!(result.recovery_queue.items.iter().all(|item| {
        item.kind != TetrahedronRecoveryKind::SourceEdge
            || item.status == TetrahedronRecoveryStatus::Recovered
    }));
}
