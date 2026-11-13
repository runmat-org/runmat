use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::SolverMeshArtifact;

use super::{components::connected_components, NullspaceEvidenceError};

pub(super) fn solid_face_connected_components(
    mesh: &SolverMeshArtifact,
    node_by_id: &BTreeMap<u64, usize>,
) -> Result<Vec<Vec<usize>>, NullspaceEvidenceError> {
    const CORNER_FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

    let element_count = mesh.topology.volume_elements.len();
    if element_count == 0 {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    let mut faces = BTreeMap::<[u64; 3], Vec<usize>>::new();
    let mut element_nodes = Vec::with_capacity(element_count);
    for (element_index, element) in mesh.topology.volume_elements.iter().enumerate() {
        if element.node_ids.len() < 4 {
            return Err(NullspaceEvidenceError::DimensionMismatch);
        }
        let nodes = element
            .node_ids
            .iter()
            .map(|node| node_by_id.get(node).copied())
            .collect::<Option<BTreeSet<_>>>()
            .ok_or(NullspaceEvidenceError::DimensionMismatch)?;
        if nodes.len() != element.node_ids.len() {
            return Err(NullspaceEvidenceError::DimensionMismatch);
        }
        element_nodes.push(nodes);
        for face in CORNER_FACES {
            let mut key = [
                element.node_ids[face[0]],
                element.node_ids[face[1]],
                element.node_ids[face[2]],
            ];
            key.sort_unstable();
            faces.entry(key).or_default().push(element_index);
        }
    }

    let mut neighbors = vec![Vec::new(); element_count];
    for owners in faces.values() {
        match owners.as_slice() {
            [_] => {}
            [left, right] => {
                neighbors[*left].push(*right);
                neighbors[*right].push(*left);
            }
            _ => return Err(NullspaceEvidenceError::AmbiguousStructuralConnectivity),
        }
    }
    let element_components = connected_components(&neighbors)?;
    let mut node_owner = BTreeMap::<usize, usize>::new();
    let mut components = Vec::with_capacity(element_components.len());
    for (component_index, elements) in element_components.into_iter().enumerate() {
        let nodes = elements
            .iter()
            .flat_map(|element| element_nodes[*element].iter().copied())
            .collect::<BTreeSet<_>>();
        for node in &nodes {
            if node_owner
                .insert(*node, component_index)
                .is_some_and(|owner| owner != component_index)
            {
                return Err(NullspaceEvidenceError::AmbiguousStructuralConnectivity);
            }
        }
        components.push(nodes.into_iter().collect());
    }
    if node_owner.len() != mesh.topology.nodes.len() {
        return Err(NullspaceEvidenceError::DimensionMismatch);
    }
    Ok(components)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::{
        fixtures, ElementOrder, SolverMeshNode, SolverVolumeElement, StableDigest,
    };

    fn append_node(mesh: &mut SolverMeshArtifact, node_id: u64, coordinates_m: [f64; 3]) {
        let topo = &mut *mesh.topology;
        topo.nodes.push(SolverMeshNode {
            node_id,
            stable_identity: StableDigest([node_id as u8; 32]),
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(coordinates_m),
            provenance: topo.nodes[0].provenance.clone(),
            exact_parameters: Vec::new(),
        });
    }

    fn append_tet(mesh: &mut SolverMeshArtifact, element_id: u64, node_ids: [u64; 4]) {
        let topo = &mut *mesh.topology;
        topo.volume_elements.push(SolverVolumeElement {
            element_id,
            stable_identity: StableDigest([element_id as u8; 32]),
            order: ElementOrder::Tet4,
            node_ids: node_ids.into(),
            region_id: topo.volume_elements[0].region_id.clone(),
            provenance: topo.volume_elements[0].provenance.clone(),
        });
    }

    fn node_index(mesh: &SolverMeshArtifact) -> BTreeMap<u64, usize> {
        mesh.topology
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.node_id, index))
            .collect()
    }

    #[test]
    fn joins_only_across_complete_faces() {
        let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        append_node(&mut mesh, 5, [0.0, 0.0, -1.0]);
        append_tet(&mut mesh, 2, [1, 2, 3, 5]);

        let components = solid_face_connected_components(&mesh, &node_index(&mesh)).unwrap();
        assert_eq!(components, vec![vec![0, 1, 2, 3, 4]]);
    }

    #[test]
    fn rejects_node_or_edge_only_contact() {
        for shared_nodes in 1..=2 {
            let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
            append_node(&mut mesh, 5, [2.0, 0.0, 0.0]);
            append_node(&mut mesh, 6, [2.0, 1.0, 0.0]);
            append_node(&mut mesh, 7, [2.0, 0.0, 1.0]);
            let nodes = if shared_nodes == 1 {
                [1, 5, 6, 7]
            } else {
                [1, 2, 5, 6]
            };
            append_tet(&mut mesh, 2, nodes);

            assert_eq!(
                solid_face_connected_components(&mesh, &node_index(&mesh)),
                Err(NullspaceEvidenceError::AmbiguousStructuralConnectivity)
            );
        }
    }
}
