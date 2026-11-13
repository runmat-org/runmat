//! Endpoint-local planar diagonal transactions over maintained face incidence.
//! Physical Delaunay admission remains the caller's separate responsibility.
use super::*;
use std::collections::BTreeMap;
#[derive(Clone, Copy, PartialEq)]
enum DiagonalKind {
    Boundary,
    Interior,
    AdmittedInterior,
}

type FaceUses = BTreeMap<[u32; 3], Vec<(u32, u32)>>;

#[cfg(test)]
mod tests;

pub(super) fn try_recover_boundary_diagonal(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    recover(
        topology,
        [left, right],
        protected_edges,
        protected_faces,
        DiagonalKind::Boundary,
        constraint_index,
        work,
    )
}

pub(super) fn try_recover_coplanar_patch_diagonal(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    recover(
        topology,
        [left, right],
        protected_edges,
        protected_faces,
        DiagonalKind::Interior,
        constraint_index,
        work,
    )
}

pub(in crate::cdt::segment_recovery) fn try_recover_admitted_planar_diagonal(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let edges = protected_edge_indices(topology, protected_edges, constraint_index)?;
    recover(
        topology,
        [left, right],
        &edges,
        protected_faces,
        DiagonalKind::AdmittedInterior,
        constraint_index,
        work,
    )
}

fn recover(
    topology: &mut TopologyMutation,
    endpoints: [u32; 2],
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    kind: DiagonalKind,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let [left, right] = endpoints;
    let interior = kind != DiagonalKind::Boundary;
    let left_faces = incident_faces(topology, left, right, interior, constraint_index, work)?;
    let right_faces = incident_faces(topology, right, left, interior, constraint_index, work)?;
    for (left_face, left_uses) in &left_faces {
        work.search_flip_step(constraint_index)?;
        let right_face = partner_face(*left_face, left, right);
        if let Some(right_uses) = right_faces.get(&right_face) {
            work.flip_attempt(constraint_index)?;
            let shared = left_face
                .iter()
                .copied()
                .filter(|vertex| right_face.contains(vertex))
                .collect::<Vec<_>>();
            if shared.len() != 2 {
                continue;
            }
            let mut edge = [shared[0], shared[1]];
            edge.sort_unstable();
            if protected_edges.contains(&edge)
                || protected_faces.contains(&stable_face(topology, *left_face))
                || protected_faces.contains(&stable_face(topology, right_face))
            {
                continue;
            }
            if orient_volume_nodes(
                [left, right, edge[0], edge[1]].map(|vertex| topology.nodes()[vertex as usize]),
            )
            .map_err(|failure| {
                invalid_topology(
                    constraint_index,
                    format!("planar diagonal predicate failed: {failure:?}"),
                )
            })? != PredicateSign::Zero
            {
                continue;
            }
            let apexes = left_uses
                .iter()
                .map(|(_, apex)| *apex)
                .collect::<BTreeSet<_>>();
            if apexes != right_uses.iter().map(|(_, apex)| *apex).collect() {
                continue;
            }
            let removed = left_uses
                .iter()
                .chain(right_uses)
                .map(|(index, _)| *index)
                .collect::<BTreeSet<_>>();
            if removed.len() != if interior { 4 } else { 2 } {
                continue;
            }
            if removed_internal_face_is_protected(
                topology,
                &removed.iter().copied().collect::<Vec<_>>(),
                protected_faces,
            ) {
                continue;
            }
            let mut created = Vec::new();
            for apex in apexes {
                let owner = |uses: &[(u32, u32)]| {
                    uses.iter()
                        .find(|(_, candidate)| *candidate == apex)
                        .expect("incident apex owner")
                        .0
                };
                let left_cell = topology
                    .tetrahedron(owner(left_uses))
                    .expect("live local owner");
                let right_cell = topology
                    .tetrahedron(owner(right_uses))
                    .expect("live local owner");
                if left_cell.region_id != right_cell.region_id {
                    break;
                }
                for vertex in edge {
                    created.push(([left, right, vertex, apex], left_cell.region_id.clone()));
                }
            }
            if created.len() != removed.len() {
                continue;
            }
            if kind == DiagonalKind::AdmittedInterior {
                use crate::cdt::insertion::local_admission::{
                    admits_replacement, LocalAdmissionError,
                };
                let vertices = created
                    .iter()
                    .map(|(vertices, _)| *vertices)
                    .collect::<Vec<_>>();
                let admitted = admits_replacement(topology, &removed, &vertices, &mut || {
                    work.search_flip_step(constraint_index)
                })
                .map_err(|failure| match failure {
                    LocalAdmissionError::Work(failure) => failure,
                    LocalAdmissionError::Predicate(failure) => invalid_topology(
                        constraint_index,
                        format!("local planar physical admission failed: {failure:?}"),
                    ),
                })?;
                if !admitted {
                    continue;
                }
            }
            let original = removed
                .iter()
                .map(|index| {
                    let cell = topology.tetrahedron(*index).expect("live local owner");
                    (cell.vertex_indices, cell.region_id.clone())
                })
                .collect();
            let created = match topology.replace_cavity(&removed, created) {
                Ok(created) => created,
                Err(failure)
                    if failure.kind
                        == crate::cdt::mutation::TopologyMutationErrorKind::InvalidTopology =>
                {
                    continue
                }
                Err(failure) => return Err(super::super::mutation_error(failure)),
            };
            let checked = edge_exists(
                topology,
                topology.nodes()[left as usize].identity,
                topology.nodes()[right as usize].identity,
                constraint_index,
                work,
            );
            if matches!(checked, Ok(true)) {
                return Ok(true);
            }
            topology
                .replace_cavity(&created.into_iter().collect(), original)
                .map_err(super::super::mutation_error)?;
            checked?;
        }
    }
    Ok(false)
}

// Each left face contains only the left endpoint; its partner must contain
// the same opposite edge and only the right endpoint. The maintained face map
// therefore enumerates the original Cartesian candidates by exact key lookup.
fn partner_face(mut face: [u32; 3], left: u32, right: u32) -> [u32; 3] {
    let endpoint = face
        .iter_mut()
        .find(|vertex| **vertex == left)
        .expect("left endpoint face");
    *endpoint = right;
    face.sort_unstable();
    face
}

fn incident_faces(
    topology: &TopologyMutation,
    endpoint: u32,
    exclude: u32,
    interior: bool,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<FaceUses, DelaunaySegmentRecoveryError> {
    let mut faces = BTreeMap::new();
    for index in topology.vertex_star(endpoint) {
        work.search_flip_step(constraint_index)?;
        let cell = topology.tetrahedron(index).expect("live endpoint star");
        for opposite in 0..4 {
            let face = face_opposite(cell.vertex_indices, opposite);
            if !face.contains(&endpoint) || face.contains(&exclude) || faces.contains_key(&face) {
                continue;
            }
            let owners = if interior {
                let Some(owners) = topology.face_adjacency(face) else {
                    continue;
                };
                owners.to_vec()
            } else {
                let Some(owner) = topology.boundary_face_owner(face) else {
                    continue;
                };
                vec![owner]
            };
            faces.insert(
                face,
                owners
                    .into_iter()
                    .map(|(owner, opposite)| {
                        (
                            owner,
                            topology
                                .tetrahedron(owner)
                                .expect("live face owner")
                                .vertex_indices[opposite],
                        )
                    })
                    .collect(),
            );
        }
    }
    Ok(faces)
}
