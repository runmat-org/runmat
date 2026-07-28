use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{
    PlcFacet, ProtectedBoundaryComplex, TetrahedronMesh, TopologyEntityId,
};

use crate::protected_edges::face_edges;

use super::{boundary_diagonal::recover_boundary_diagonal_flip, topology::sorted_topology_ids};
use super::{
    TetrahedronRecoveryKind, TetrahedronRecoveryQueue, TetrahedronRecoveryStatus,
    TetrahedronSourceFaceTopology,
};

pub(super) struct SourceFaceBoundaryDiagonalRecovery {
    pub attempted_source_face_pair_count: usize,
    pub source_face_pair_count: usize,
    pub source_face_count: usize,
    pub boundary_face_count: usize,
    pub rejected_source_face_pair_count: usize,
    pub rejected_source_face_count: usize,
    pub rejection_counts: BTreeMap<&'static str, usize>,
}

struct SourceFaceBoundaryDiagonalCandidates {
    diagonals: Vec<[TopologyEntityId; 2]>,
    unpaired_source_face_count: usize,
}

pub(super) fn recover_source_faces_by_boundary_diagonal_flip(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> SourceFaceBoundaryDiagonalRecovery {
    let mut recovery = SourceFaceBoundaryDiagonalRecovery {
        attempted_source_face_pair_count: 0,
        source_face_pair_count: 0,
        source_face_count: 0,
        boundary_face_count: 0,
        rejected_source_face_pair_count: 0,
        rejected_source_face_count: 0,
        rejection_counts: BTreeMap::new(),
    };
    let candidates =
        missing_absent_source_face_pair_diagonals(plc, initial_recovery_queue, tetrahedron_mesh);
    if candidates.unpaired_source_face_count > 0 {
        recovery.rejected_source_face_count += candidates.unpaired_source_face_count;
        *recovery
            .rejection_counts
            .entry("rejected_source_face_diagonal_recovery_unpaired_source_face")
            .or_default() += candidates.unpaired_source_face_count;
    }
    for recovered_diagonal in candidates.diagonals {
        recovery.attempted_source_face_pair_count += 1;
        match recover_boundary_diagonal_flip(plc, tetrahedron_mesh, recovered_diagonal) {
            Ok(boundary_face_count) => {
                recovery.source_face_pair_count += 1;
                recovery.source_face_count += boundary_face_count;
                recovery.boundary_face_count += boundary_face_count;
            }
            Err(rejection) => {
                recovery.rejected_source_face_pair_count += 1;
                *recovery
                    .rejection_counts
                    .entry(rejection.source_face_evidence_key())
                    .or_default() += 1;
            }
        }
    }
    recovery
}

fn missing_absent_source_face_pair_diagonals(
    plc: &ProtectedBoundaryComplex,
    initial_recovery_queue: &TetrahedronRecoveryQueue,
    tetrahedron_mesh: &TetrahedronMesh,
) -> SourceFaceBoundaryDiagonalCandidates {
    let recoverable_source_faces = initial_recovery_queue
        .items
        .iter()
        .filter(|item| {
            item.kind == TetrahedronRecoveryKind::SourceFace
                && item.status == TetrahedronRecoveryStatus::Missing
                && item.source_face_topology == Some(TetrahedronSourceFaceTopology::Absent)
        })
        .filter_map(|item| {
            Some((
                item.source_entity_id.clone()?,
                item.source_face_node_ids.clone()?,
            ))
        })
        .collect::<BTreeSet<_>>();
    let recovered_source_faces = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| {
            (
                face.source_face_id.clone(),
                sorted_topology_ids(face.node_ids.clone()),
            )
        })
        .collect::<BTreeSet<_>>();
    let missing_facets = plc
        .facets
        .iter()
        .filter(|facet| {
            let face_key = sorted_topology_ids(facet.node_ids.clone());
            recoverable_source_faces.contains(&(facet.source_face_id.clone(), face_key.clone()))
                && !recovered_source_faces.contains(&(facet.source_face_id.clone(), face_key))
        })
        .collect::<Vec<_>>();
    let missing_facets_by_edge = missing_facets
        .iter()
        .copied()
        .flat_map(|facet| {
            face_edges(facet.node_ids.clone())
                .into_iter()
                .map(move |edge| (edge, facet))
        })
        .fold(
            BTreeMap::<[TopologyEntityId; 2], Vec<&PlcFacet>>::new(),
            |mut facets_by_edge, (edge, facet)| {
                facets_by_edge.entry(edge).or_default().push(facet);
                facets_by_edge
            },
        );

    let mut paired_source_faces = BTreeSet::<(TopologyEntityId, [TopologyEntityId; 3])>::new();
    let diagonals = missing_facets_by_edge
        .into_iter()
        .filter_map(|(edge, facets)| {
            if facets.len() != 2 {
                return None;
            }
            for facet in facets {
                paired_source_faces.insert((
                    facet.source_face_id.clone(),
                    sorted_topology_ids(facet.node_ids.clone()),
                ));
            }
            Some(edge)
        })
        .collect::<Vec<_>>();
    let unpaired_source_face_count = missing_facets
        .iter()
        .filter(|facet| {
            !paired_source_faces.contains(&(
                facet.source_face_id.clone(),
                sorted_topology_ids(facet.node_ids.clone()),
            ))
        })
        .count();

    SourceFaceBoundaryDiagonalCandidates {
        diagonals,
        unpaired_source_face_count,
    }
}
