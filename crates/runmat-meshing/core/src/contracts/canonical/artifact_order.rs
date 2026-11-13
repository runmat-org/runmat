use super::{
    artifact::validation_checkpoint,
    artifact_workspace::{flags, reserve},
    solver_midside_node_identity, BoundaryEdgeOrder, BoundaryTriangleOrder, ElementOrder,
    MeshDimension, MeshingCancellationSignal, MeshingContractError, MeshingRequest,
    PersistentEntityKind, SolverMeshTopology, SolverNodeExactParameter,
    TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

pub(super) fn validate_order_topology(
    topology: &SolverMeshTopology,
    request: &MeshingRequest,
    node_ids: &[u64],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    match request.element_order {
        ElementOrder::Tet4 => validate_linear(topology, request.dimension, node_ids, cancellation),
        ElementOrder::Tet10 => validate_quadratic(topology, node_ids, cancellation),
    }
}

fn validate_linear(
    topology: &SolverMeshTopology,
    dimension: MeshDimension,
    node_ids: &[u64],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    validation_checkpoint(cancellation)?;
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
    let mut referenced = flags(node_ids.len(), cancellation)?;
    // Every node belongs to an element of the requested dimension: tetrahedra for solids,
    // triangles for sheets, and segments for curve-only beam meshes.
    let connectivity: Box<dyn Iterator<Item = &Vec<u64>>> = match dimension {
        MeshDimension::Solid => Box::new(topology.volume_elements.iter().map(|e| &e.node_ids)),
        MeshDimension::Surface => Box::new(topology.boundary_faces.iter().map(|f| &f.node_ids)),
        MeshDimension::Curve => Box::new(topology.boundary_edges.iter().map(|e| &e.node_ids)),
    };
    for node_ids_of_element in connectivity {
        validation_checkpoint(cancellation)?;
        for id in node_ids_of_element {
            referenced[node_ids.binary_search(id).expect("validated node")] = 1;
        }
    }
    if referenced.contains(&0) {
        return Err(invalid(
            "every linear node must be referenced as an element corner",
        ));
    }
    validate_boundary_faces(topology, &[], cancellation)
}

fn validate_quadratic(
    topology: &SolverMeshTopology,
    node_ids: &[u64],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    validation_checkpoint(cancellation)?;
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
    let mut corners = flags(node_ids.len(), cancellation)?;
    let mut midside_nodes = flags(node_ids.len(), cancellation)?;
    let edge_count =
        super::artifact_resources::volume_edge_occurrences(topology.volume_elements.len())?;
    let mut midpoint_by_edge = reserve(edge_count, cancellation)?;
    let node = |id: &u64| &topology.nodes[node_ids.binary_search(id).expect("validated node")];
    for element in &topology.volume_elements {
        validation_checkpoint(cancellation)?;
        for id in &element.node_ids[..4] {
            corners[node_ids.binary_search(id).expect("validated node")] = 1;
        }
        for (local_edge, endpoints) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.iter().enumerate() {
            let edge = sorted_pair([
                element.node_ids[endpoints[0]],
                element.node_ids[endpoints[1]],
            ]);
            let midpoint = element.node_ids[4 + local_edge];
            let endpoint_identities = edge.map(|id| node(&id).stable_identity);
            if node(&midpoint).stable_identity != solver_midside_node_identity(endpoint_identities)
            {
                return Err(invalid(
                    "Tet10 midside identity must derive from its stable endpoint identities",
                ));
            }
            midside_nodes[node_ids.binary_search(&midpoint).expect("validated node")] = 1;
            midpoint_by_edge.push((edge, midpoint));
        }
    }
    validation_checkpoint(cancellation)?;
    midpoint_by_edge.sort_unstable();
    validation_checkpoint(cancellation)?;
    if midpoint_by_edge
        .windows(2)
        .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1)
    {
        return Err(invalid(
            "adjacent Tet10 elements disagree on a shared midside node",
        ));
    }
    midpoint_by_edge.dedup();
    if corners
        .iter()
        .zip(&midside_nodes)
        .any(|(corner, midpoint)| *corner + *midpoint != 1)
    {
        return Err(invalid(
            "Tet10 corner and midside nodes must be disjoint and complete",
        ));
    }
    validate_boundary_faces(topology, &midpoint_by_edge, cancellation)?;
    for face in &topology.boundary_faces {
        validation_checkpoint(cancellation)?;
        if !face.provenance.iter().any(|entity| entity.kind == PersistentEntityKind::Face)
            || face.node_ids[3..].iter().any(|node_id| {
                !node(node_id).exact_parameters.iter().any(|parameter| {
                    matches!(parameter,
                        SolverNodeExactParameter::Surface { source_face_id, .. }
                            if face.provenance.iter().any(|entity| entity.kind == PersistentEntityKind::Face && entity == source_face_id)
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
        validation_checkpoint(cancellation)?;
        let key = sorted_pair([edge.node_ids[0], edge.node_ids[1]]);
        if midpoint(&midpoint_by_edge, &key) != edge.node_ids.get(2) {
            return Err(invalid(
                "Line3 boundary edge does not use its volume-edge midpoint",
            ));
        }
        if edge.provenance.iter().any(|entity| entity.kind == PersistentEntityKind::Edge)
            && !node(&edge.node_ids[2]).exact_parameters.iter().any(|parameter| {
                matches!(parameter,
                    SolverNodeExactParameter::Curve { source_edge_id, .. }
                        if edge.provenance.iter().any(|entity| entity.kind == PersistentEntityKind::Edge && entity == source_edge_id)
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
    midpoint_by_edge: &[([u64; 2], u64)],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    for face in &topology.boundary_faces {
        validation_checkpoint(cancellation)?;
        let corners = &face.node_ids[..3];
        if face.adjacent_volume_element_ids.iter().any(|element_id| {
            topology
                .volume_elements
                .binary_search_by_key(element_id, |element| element.element_id)
                .ok()
                .is_none_or(|index| {
                    !is_tetrahedron_face(corners, &topology.volume_elements[index].node_ids[..4])
                })
        }) {
            return Err(invalid(
                "boundary face corners do not form a face of every adjacent tetrahedron",
            ));
        }
        if face.order == BoundaryTriangleOrder::Tri6 {
            let pairs = [[0, 1], [1, 2], [2, 0]];
            if pairs.iter().enumerate().any(|(index, pair)| {
                midpoint(
                    midpoint_by_edge,
                    &sorted_pair([corners[pair[0]], corners[pair[1]]]),
                ) != face.node_ids.get(3 + index)
            }) {
                return Err(invalid(
                    "Tri6 boundary face does not use its volume-edge midside nodes",
                ));
            }
        }
    }
    Ok(())
}

fn midpoint<'a>(edges: &'a [([u64; 2], u64)], key: &[u64; 2]) -> Option<&'a u64> {
    edges
        .binary_search_by_key(key, |entry| entry.0)
        .ok()
        .map(|index| &edges[index].1)
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
