use runmat_meshing_core::contracts::{PlcProtectedEdge, TopologyEntityId};

pub(crate) fn face_edges(node_ids: [TopologyEntityId; 3]) -> [[TopologyEntityId; 2]; 3] {
    [
        sorted_edge([node_ids[0].clone(), node_ids[1].clone()]),
        sorted_edge([node_ids[1].clone(), node_ids[2].clone()]),
        sorted_edge([node_ids[2].clone(), node_ids[0].clone()]),
    ]
}

pub(crate) fn source_edge_ids_for_face_edges(
    protected_edges: &[PlcProtectedEdge],
    node_ids: [TopologyEntityId; 3],
) -> [Option<TopologyEntityId>; 3] {
    face_edges(node_ids).map(|face_edge| {
        protected_edges
            .iter()
            .find(|protected_edge| sorted_edge(protected_edge.node_ids.clone()) == face_edge)
            .map(|protected_edge| protected_edge.source_edge_id.clone())
    })
}

pub(crate) fn sorted_edge(mut node_ids: [TopologyEntityId; 2]) -> [TopologyEntityId; 2] {
    node_ids.sort();
    node_ids
}
