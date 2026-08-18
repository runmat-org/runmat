use std::collections::{BTreeMap, BTreeSet};

use super::{
    solver_midside_node_identity, BoundaryEdgeOrder, BoundaryTriangleOrder, ElementOrder,
    MeshingContractError, MeshingRequest, PersistentEntityKind, SolverMeshTopology,
    SolverNodeExactParameter, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

pub(super) fn validate_order_topology(
    topology: &SolverMeshTopology,
    request: &MeshingRequest,
    node_ids: &BTreeSet<u64>,
) -> Result<(), MeshingContractError> {
    match request.element_order {
        ElementOrder::Tet4 => validate_linear(topology, node_ids),
        ElementOrder::Tet10 => validate_quadratic(topology, node_ids),
    }
}

fn validate_linear(
    topology: &SolverMeshTopology,
    node_ids: &BTreeSet<u64>,
) -> Result<(), MeshingContractError> {
    if topology
        .boundary_faces
        .iter()
        .any(|face| face.order != BoundaryTriangleOrder::Tri3)
        || topology
            .boundary_edges
            .iter()
            .any(|edge| edge.order != BoundaryEdgeOrder::Line2)
    {
        return Err(invalid(
            "Tet4 topology requires Tri3 faces and Line2 boundary edges",
        ));
    }
    let referenced = topology
        .volume_elements
        .iter()
        .flat_map(|element| element.node_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    if &referenced != node_ids {
        return Err(invalid(
            "every Tet4 node must be referenced as a volume corner",
        ));
    }
    validate_boundary_faces(topology, &BTreeMap::new())
}

fn validate_quadratic(
    topology: &SolverMeshTopology,
    node_ids: &BTreeSet<u64>,
) -> Result<(), MeshingContractError> {
    if topology
        .boundary_faces
        .iter()
        .any(|face| face.order != BoundaryTriangleOrder::Tri6)
        || topology
            .boundary_edges
            .iter()
            .any(|edge| edge.order != BoundaryEdgeOrder::Line3)
    {
        return Err(invalid(
            "Tet10 topology requires Tri6 faces and Line3 boundary edges",
        ));
    }
    let mut corners = BTreeSet::new();
    let mut midside_nodes = BTreeSet::new();
    let mut midpoint_by_edge = BTreeMap::new();
    let stable_identity_by_node = topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.stable_identity))
        .collect::<BTreeMap<_, _>>();
    for element in &topology.volume_elements {
        corners.extend(element.node_ids[..4].iter().copied());
        for (local_edge, endpoints) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.iter().enumerate() {
            let edge = sorted_pair([
                element.node_ids[endpoints[0]],
                element.node_ids[endpoints[1]],
            ]);
            let midpoint = element.node_ids[4 + local_edge];
            let endpoint_identities = edge.map(|node| stable_identity_by_node[&node]);
            if stable_identity_by_node[&midpoint]
                != solver_midside_node_identity(endpoint_identities)
            {
                return Err(invalid(
                    "Tet10 midside identity must derive from its stable endpoint identities",
                ));
            }
            midside_nodes.insert(midpoint);
            if midpoint_by_edge
                .insert(edge, midpoint)
                .is_some_and(|existing| existing != midpoint)
            {
                return Err(invalid(
                    "adjacent Tet10 elements disagree on a shared midside node",
                ));
            }
        }
    }
    if !corners.is_disjoint(&midside_nodes)
        || &corners
            .union(&midside_nodes)
            .copied()
            .collect::<BTreeSet<_>>()
            != node_ids
    {
        return Err(invalid(
            "Tet10 corner and midside nodes must be disjoint and complete",
        ));
    }
    validate_boundary_faces(topology, &midpoint_by_edge)?;
    let nodes = topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node))
        .collect::<BTreeMap<_, _>>();
    for face in &topology.boundary_faces {
        let source_faces = face
            .provenance
            .iter()
            .filter(|entity| entity.kind == PersistentEntityKind::Face)
            .collect::<BTreeSet<_>>();
        if source_faces.is_empty()
            || face.node_ids[3..].iter().any(|node_id| {
                !nodes[node_id].exact_parameters.iter().any(|parameter| {
                    matches!(
                        parameter,
                        SolverNodeExactParameter::Surface { source_face_id, .. }
                            if source_faces.contains(source_face_id)
                    )
                })
            })
        {
            return Err(invalid(
                "Tri6 midside nodes must retain exact parameters for their source face",
            ));
        }
    }
    for edge in &topology.boundary_edges {
        let key = sorted_pair([edge.node_ids[0], edge.node_ids[1]]);
        if midpoint_by_edge.get(&key) != edge.node_ids.get(2) {
            return Err(invalid(
                "Line3 boundary edge does not use its volume-edge midpoint",
            ));
        }
        let source_edges = edge
            .provenance
            .iter()
            .filter(|entity| entity.kind == PersistentEntityKind::Edge)
            .collect::<BTreeSet<_>>();
        if !source_edges.is_empty()
            && !nodes[&edge.node_ids[2]]
                .exact_parameters
                .iter()
                .any(|parameter| {
                    matches!(
                        parameter,
                        SolverNodeExactParameter::Curve { source_edge_id, .. }
                            if source_edges.contains(source_edge_id)
                    )
                })
        {
            return Err(invalid(
                "Line3 exact-edge midpoint must retain its exact curve parameter",
            ));
        }
    }
    Ok(())
}

fn validate_boundary_faces(
    topology: &SolverMeshTopology,
    midpoint_by_edge: &BTreeMap<[u64; 2], u64>,
) -> Result<(), MeshingContractError> {
    let elements = topology
        .volume_elements
        .iter()
        .map(|element| (element.element_id, &element.node_ids[..4]))
        .collect::<BTreeMap<_, _>>();
    for face in &topology.boundary_faces {
        let corners = &face.node_ids[..3];
        if face.adjacent_volume_element_ids.iter().any(|element_id| {
            elements
                .get(element_id)
                .is_none_or(|element| !is_tetrahedron_face(corners, element))
        }) {
            return Err(invalid(
                "boundary face corners do not form a face of every adjacent tetrahedron",
            ));
        }
        if face.order == BoundaryTriangleOrder::Tri6 {
            let pairs = [[0, 1], [1, 2], [2, 0]];
            if pairs.iter().enumerate().any(|(index, pair)| {
                midpoint_by_edge.get(&sorted_pair([corners[pair[0]], corners[pair[1]]]))
                    != face.node_ids.get(3 + index)
            }) {
                return Err(invalid(
                    "Tri6 boundary face does not use its volume-edge midside nodes",
                ));
            }
        }
    }
    Ok(())
}

fn is_tetrahedron_face(face: &[u64], element: &[u64]) -> bool {
    face.iter().all(|node| element.contains(node))
}

fn sorted_pair(mut nodes: [u64; 2]) -> [u64; 2] {
    nodes.sort_unstable();
    nodes
}

fn invalid(reason: impl Into<String>) -> MeshingContractError {
    MeshingContractError::invalid("mesh element order topology", reason)
}
