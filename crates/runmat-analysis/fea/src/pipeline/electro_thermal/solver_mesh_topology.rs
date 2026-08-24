use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::SolverMeshArtifact;

#[derive(Debug, Clone, Copy)]
pub(super) struct MeshConductanceEdge {
    pub(super) from: usize,
    pub(super) to: usize,
    pub(super) length_m: f64,
    pub(super) direction: [f64; 3],
}

#[derive(Debug)]
pub(super) struct MeshConductanceTopology {
    pub(super) edges: Vec<MeshConductanceEdge>,
    pub(super) source_node: usize,
    pub(super) ground_node: usize,
    pub(super) active_dimension_count: usize,
}

impl MeshConductanceTopology {
    pub(super) fn derive(mesh: &SolverMeshArtifact) -> Option<Self> {
        let node_indices = mesh
            .topology
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.node_id, index))
            .collect::<BTreeMap<_, _>>();
        let coordinates = mesh
            .topology
            .nodes
            .iter()
            .map(|node| node.coordinates_m)
            .collect::<Vec<_>>();
        if coordinates.len() < 2 {
            return None;
        }

        let mut edge_nodes = BTreeSet::new();
        for element in &mesh.topology.volume_elements {
            let corners = element.node_ids.get(..4)?;
            let indices = corners
                .iter()
                .map(|node_id| node_indices.get(node_id).copied())
                .collect::<Option<Vec<_>>>()?;
            for left in 0..indices.len() {
                for right in (left + 1)..indices.len() {
                    edge_nodes.insert([
                        indices[left].min(indices[right]),
                        indices[left].max(indices[right]),
                    ]);
                }
            }
        }
        let edges = edge_nodes
            .into_iter()
            .map(|[from, to]| {
                let delta = std::array::from_fn::<_, 3, _>(|axis| {
                    coordinates[to][axis] - coordinates[from][axis]
                });
                let length_m = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
                MeshConductanceEdge {
                    from,
                    to,
                    length_m: length_m.max(1.0e-12),
                    direction: delta.map(|value| value / length_m.max(1.0e-12)),
                }
            })
            .collect::<Vec<_>>();
        if edges.is_empty() {
            return None;
        }

        let bounds = std::array::from_fn::<_, 3, _>(|axis| {
            coordinates
                .iter()
                .fold([f64::INFINITY, f64::NEG_INFINITY], |[min, max], point| {
                    [min.min(point[axis]), max.max(point[axis])]
                })
        });
        let active_dimension_count = bounds
            .iter()
            .filter(|[min, max]| max - min > 1.0e-12)
            .count();
        let terminal_axis = (0..3).max_by(|left, right| {
            (bounds[*left][1] - bounds[*left][0])
                .total_cmp(&(bounds[*right][1] - bounds[*right][0]))
        })?;
        let source_node = (0..coordinates.len()).min_by(|left, right| {
            coordinates[*left][terminal_axis]
                .total_cmp(&coordinates[*right][terminal_axis])
                .then_with(|| left.cmp(right))
        })?;
        let ground_node = (0..coordinates.len()).max_by(|left, right| {
            coordinates[*left][terminal_axis]
                .total_cmp(&coordinates[*right][terminal_axis])
                .then_with(|| right.cmp(left))
        })?;
        (source_node != ground_node).then_some(Self {
            edges,
            source_node,
            ground_node,
            active_dimension_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{fixtures, ElementOrder};

    use super::MeshConductanceTopology;

    #[test]
    fn canonical_tetrahedron_projects_full_conductance_graph() {
        let mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let topology = MeshConductanceTopology::derive(&mesh).expect("conductance topology");

        assert_eq!(topology.edges.len(), 6);
        assert_eq!(topology.active_dimension_count, 3);
        assert_ne!(topology.source_node, topology.ground_node);
        assert!(topology.edges.iter().all(|edge| edge.length_m > 0.0));
    }
}
