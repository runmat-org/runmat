//! Local three-to-two facet recovery. Trials preserve cavity coverage; exact
//! changed-neighbor checks precede commit and full verification remains at output.
use super::*;
use crate::cdt::{insertion::topology_access::TetrahedralTopology, mutation::TopologyMutation};

pub(in crate::cdt::facet_recovery) fn recover(
    topology: &mut TopologyMutation,
    facet: [StableDigest; 3],
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let indices = facet.map(|id| {
        topology
            .node_index(id)
            .ok_or_else(|| invalid_topology(constraint_index, "facet-flip node is missing"))
    });
    let [first, second, third] = indices;
    let mut facet_indices = [first?, second?, third?];
    facet_indices.sort_unstable();
    // Every admissible ring contains the requested facet vertices. Enumerate
    // only their incident opposite edges, then rederive complete edge stars.
    let mut candidates = BTreeMap::new();
    for vertex in facet_indices {
        for cell_index in topology.vertex_star(vertex) {
            work.search_step(constraint_index)?;
            let cell = topology
                .tetrahedron(cell_index)
                .expect("maintained facet star");
            let endpoints = cell
                .vertex_indices
                .into_iter()
                .filter(|v| !facet_indices.contains(v))
                .collect::<Vec<_>>();
            if endpoints.len() == 2 {
                let mut edge = [endpoints[0], endpoints[1]];
                edge.sort_unstable();
                let mut identities = edge.map(|v| topology.nodes()[v as usize].identity);
                identities.sort_unstable();
                candidates.insert(identities, edge);
            }
        }
    }
    for (identities, edge) in candidates {
        if protected_edges.binary_search(&identities).is_ok() {
            continue;
        }
        let star = topology.edge_tetrahedra(edge);
        if star.len() != 3 {
            continue;
        }
        let ring = star
            .iter()
            .flat_map(|index| {
                topology
                    .tetrahedron(*index)
                    .expect("live edge star")
                    .vertex_indices
            })
            .filter(|vertex| !edge.contains(vertex))
            .collect::<BTreeSet<_>>();
        if ring.into_iter().collect::<Vec<_>>() != facet_indices {
            continue;
        }
        work.flip_attempt(constraint_index)?;
        let region = topology
            .tetrahedron(star[0])
            .expect("live edge star")
            .region_id
            .clone();
        if star.iter().any(|index| {
            topology
                .tetrahedron(*index)
                .expect("live edge star")
                .region_id
                != region
        }) {
            continue;
        }
        let signs = edge.map(|apex| {
            topology.orient_nodes(
                [facet_indices[0], facet_indices[1], facet_indices[2], apex]
                    .map(|v| topology.nodes()[v as usize]),
            )
        });
        let [left, right] = signs;
        let predicate_error = |failure| {
            error(
                DelaunayFacetRecoveryErrorKind::InvalidTopology,
                Some(constraint_index),
                format!("facet flip predicate: {failure:?}"),
            )
        };
        if !matches!(
            (
                left.map_err(predicate_error)?,
                right.map_err(predicate_error)?
            ),
            (PredicateSign::Positive, PredicateSign::Negative)
                | (PredicateSign::Negative, PredicateSign::Positive)
        ) {
            continue;
        }
        if star.iter().any(|index| {
            let cell = topology.tetrahedron(*index).expect("live edge star");
            (0..4).any(|opposite| {
                let vertices = face(cell.vertex_indices, opposite);
                if !edge.iter().all(|v| vertices.contains(v)) {
                    return false;
                }
                let mut identities = vertices.map(|v| topology.nodes()[v as usize].identity);
                identities.sort_unstable();
                protected_faces.contains(&identities)
            })
        }) {
            continue;
        }
        let removed = star.into_iter().collect();
        let created = vec![
            (
                [
                    edge[0],
                    facet_indices[0],
                    facet_indices[1],
                    facet_indices[2],
                ],
                region.clone(),
            ),
            (
                [
                    edge[1],
                    facet_indices[0],
                    facet_indices[2],
                    facet_indices[1],
                ],
                region,
            ),
        ];
        let mut trial = topology.cavity_trial();
        let changed = match trial.replace_cavity(&removed, created) {
            Ok(changed) => changed,
            Err(_) => continue,
        };
        if !changed_adjacencies_admit(
            &trial,
            &changed,
            facet,
            protected_faces,
            constraint_index,
            work,
        )? {
            continue;
        }
        trial.commit();
        return Ok(true);
    }
    Ok(false)
}

fn face(vertices: [u32; 4], opposite: usize) -> [u32; 3] {
    let mut result = [0; 3];
    let mut cursor = 0;
    for (slot, vertex) in vertices.into_iter().enumerate() {
        if slot != opposite {
            result[cursor] = vertex;
            cursor += 1;
        }
    }
    result
}

fn changed_adjacencies_admit(
    topology: &TopologyMutation,
    changed: &[u32],
    recovered: [StableDigest; 3],
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<bool, DelaunayFacetRecoveryError> {
    let mut recovered = recovered;
    recovered.sort_unstable();
    for &index in changed {
        let cell = topology.tetrahedron(index).expect("created facet cell");
        for opposite in 0..4 {
            let Some(neighbor_index) = cell.neighbors[opposite] else {
                continue;
            };
            work.search_step(constraint_index)?;
            let vertices = face(cell.vertex_indices, opposite);
            let mut identities = vertices.map(|v| topology.nodes()[v as usize].identity);
            identities.sort_unstable();
            if identities == recovered || protected_faces.contains(&identities) {
                continue;
            }
            let neighbor = topology
                .tetrahedron(neighbor_index)
                .expect("maintained cavity neighbor");
            let query = *neighbor
                .vertex_indices
                .iter()
                .find(|v| !vertices.contains(v))
                .expect("neighbor apex");
            let [a, b, c, d] = cell.vertex_indices.map(|v| topology.nodes()[v as usize]);
            if topology
                .insphere_nodes([a, b, c, d, topology.nodes()[query as usize]])
                .map_err(|failure| {
                    error(
                        DelaunayFacetRecoveryErrorKind::InvalidTopology,
                        Some(constraint_index),
                        format!("facet adjacency predicate: {failure:?}"),
                    )
                })?
                == PredicateSign::Positive
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
