use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TetrahedronMesh};

use crate::protected_edges::sorted_edge;

use super::{
    boundary_diagonal::recover_boundary_diagonal_flip, TetrahedronProtectedEdgeTopology,
    TetrahedronRecoveryKind, TetrahedronRecoveryQueue, TetrahedronRecoveryStatus,
};

pub(super) struct AbsentSourceEdgeRecovery {
    pub attempted_source_edge_count: usize,
    pub source_edge_count: usize,
    pub boundary_face_count: usize,
    pub rejected_source_edge_count: usize,
    pub rejection_counts: BTreeMap<&'static str, usize>,
}

pub(super) fn recover_absent_protected_edges_by_boundary_diagonal_flip(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> AbsentSourceEdgeRecovery {
    let recoverable_source_edges = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceEdge
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.protected_edge_topology == Some(TetrahedronProtectedEdgeTopology::Absent)
        })
        .filter_map(|item| {
            Some((
                item.protected_edge_node_ids.clone()?,
                item.source_entity_id.clone()?,
            ))
        })
        .collect::<BTreeSet<_>>();

    let mut recovered = AbsentSourceEdgeRecovery {
        attempted_source_edge_count: 0,
        source_edge_count: 0,
        boundary_face_count: 0,
        rejected_source_edge_count: 0,
        rejection_counts: BTreeMap::new(),
    };
    for protected_edge in plc.protected_edges.iter().filter(|protected_edge| {
        recoverable_source_edges.contains(&(
            sorted_edge(protected_edge.node_ids.clone()),
            protected_edge.source_edge_id.clone(),
        ))
    }) {
        recovered.attempted_source_edge_count += 1;
        match recover_boundary_diagonal_flip(
            plc,
            tetrahedron_mesh,
            sorted_edge(protected_edge.node_ids.clone()),
        ) {
            Ok(boundary_face_count) => {
                recovered.source_edge_count += 1;
                recovered.boundary_face_count += boundary_face_count;
            }
            Err(rejection) => {
                recovered.rejected_source_edge_count += 1;
                *recovered
                    .rejection_counts
                    .entry(rejection.absent_source_edge_evidence_key())
                    .or_default() += 1;
            }
        }
    }

    recovered
}
