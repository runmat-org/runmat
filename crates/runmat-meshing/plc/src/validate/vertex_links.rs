use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TopologyEntityId};

use super::PlcValidationError;

pub fn validate_vertex_links_are_manifold(
    plc: &ProtectedBoundaryComplex,
) -> Result<(), PlcValidationError> {
    let mut links = BTreeMap::<TopologyEntityId, VertexLink>::new();
    for facet in &plc.facets {
        for anchor_index in 0..3 {
            let anchor = facet.node_ids[anchor_index].clone();
            let left = facet.node_ids[(anchor_index + 1) % 3].clone();
            let right = facet.node_ids[(anchor_index + 2) % 3].clone();
            links
                .entry(anchor)
                .or_default()
                .insert_link_edge(left, right);
        }
    }

    for (node_id, link) in links {
        let link_component_count = link.component_count();
        if link_component_count != 1 || !link.neighbor_degrees_are_cyclic() {
            return Err(PlcValidationError::NonManifoldBoundaryVertex {
                node_id,
                incident_facet_count: link.incident_facet_count,
                link_component_count,
            });
        }
    }

    Ok(())
}

#[derive(Default)]
struct VertexLink {
    incident_facet_count: usize,
    adjacency: BTreeMap<TopologyEntityId, BTreeSet<TopologyEntityId>>,
}

impl VertexLink {
    fn insert_link_edge(&mut self, left: TopologyEntityId, right: TopologyEntityId) {
        self.incident_facet_count += 1;
        self.adjacency
            .entry(left.clone())
            .or_default()
            .insert(right.clone());
        self.adjacency.entry(right).or_default().insert(left);
    }

    fn neighbor_degrees_are_cyclic(&self) -> bool {
        !self.adjacency.is_empty()
            && self
                .adjacency
                .values()
                .all(|neighbors| neighbors.len() == 2)
    }

    fn component_count(&self) -> usize {
        let mut component_count = 0_usize;
        let mut visited = BTreeSet::<TopologyEntityId>::new();
        for start in self.adjacency.keys() {
            if visited.contains(start) {
                continue;
            }
            component_count += 1;
            let mut stack = vec![start.clone()];
            while let Some(node_id) = stack.pop() {
                if !visited.insert(node_id.clone()) {
                    continue;
                }
                if let Some(neighbors) = self.adjacency.get(&node_id) {
                    stack.extend(
                        neighbors
                            .iter()
                            .filter(|neighbor| !visited.contains(*neighbor))
                            .cloned(),
                    );
                }
            }
        }
        component_count
    }
}
