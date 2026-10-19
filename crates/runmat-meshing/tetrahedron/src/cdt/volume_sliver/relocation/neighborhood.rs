//! Bounded incidence traversal shared by local geometric and cavity moves.
use super::*;
use crate::cdt::insertion::topology_access::TetrahedralTopology;
use crate::cdt::volume_state::SearchState;
pub(super) fn vertices<'ctx>(
    state: &impl SearchState<'ctx>,
    source: &DelaunayTetrahedronQuality,
) -> Vec<u32> {
    let topology = state.topology();
    let eligible = |vertex| {
        let node = topology.node(vertex);
        !state.protected_node(node.identity) && !topology.vertex_is_boundary(vertex)
    };
    let mut vertices = source
        .node_identities
        .into_iter()
        .filter_map(|id| {
            let index = topology.node_index(id).expect("source corner");
            eligible(index).then_some(index)
        })
        .collect::<Vec<_>>();
    vertices.sort_unstable_by_key(|v| topology.node(*v).identity);
    let mut known = vertices.iter().copied().collect::<BTreeSet<_>>();
    let mut cursor = 0;
    while cursor < vertices.len() && vertices.len() < MAXIMUM_CAVITY_SEARCH_DIRECTIONS {
        let mut neighbors = BTreeSet::new();
        for cell in topology.ordered_vertex_star(vertices[cursor]) {
            for neighbor in topology.cell(cell).vertex_indices {
                if !known.contains(&neighbor) && eligible(neighbor) {
                    neighbors.insert((topology.node(neighbor).identity, neighbor));
                }
            }
        }
        for (_, neighbor) in neighbors {
            if known.insert(neighbor) {
                vertices.push(neighbor);
            }
            if vertices.len() == MAXIMUM_CAVITY_SEARCH_DIRECTIONS {
                break;
            }
        }
        cursor += 1;
    }
    vertices
}
