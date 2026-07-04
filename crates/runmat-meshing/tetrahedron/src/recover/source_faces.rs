use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{
    PlcFacet, ProtectedBoundaryComplex, TetrahedronMesh, TopologyEntityId,
};

use crate::protected_edges::face_edges;

use super::{boundary_diagonal::recover_boundary_diagonal_flip, topology::sorted_topology_ids};

pub(super) struct SourceFaceBoundaryDiagonalRecovery {
    pub attempted_source_face_pair_count: usize,
    pub source_face_pair_count: usize,
    pub source_face_count: usize,
    pub boundary_face_count: usize,
    pub rejected_source_face_pair_count: usize,
    pub rejection_counts: BTreeMap<&'static str, usize>,
}

pub(super) fn recover_source_faces_by_boundary_diagonal_flip(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &mut TetrahedronMesh,
) -> SourceFaceBoundaryDiagonalRecovery {
    let mut recovery = SourceFaceBoundaryDiagonalRecovery {
        attempted_source_face_pair_count: 0,
        source_face_pair_count: 0,
        source_face_count: 0,
        boundary_face_count: 0,
        rejected_source_face_pair_count: 0,
        rejection_counts: BTreeMap::new(),
    };
    for recovered_diagonal in missing_source_face_pair_diagonals(plc, tetrahedron_mesh) {
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

fn missing_source_face_pair_diagonals(
    plc: &ProtectedBoundaryComplex,
    tetrahedron_mesh: &TetrahedronMesh,
) -> Vec<[TopologyEntityId; 2]> {
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
    let missing_facets_by_edge = plc
        .facets
        .iter()
        .filter(|facet| {
            !recovered_source_faces.contains(&(
                facet.source_face_id.clone(),
                sorted_topology_ids(facet.node_ids.clone()),
            ))
        })
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

    missing_facets_by_edge
        .into_iter()
        .filter_map(|(edge, facets)| (facets.len() == 2).then_some(edge))
        .collect()
}
