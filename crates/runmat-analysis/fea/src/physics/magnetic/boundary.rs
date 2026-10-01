use super::MagnetostaticError;
use crate::topology::{AnalysisProjection, TetrahedralEdgeSpace};
use runmat_meshing_core::BoundaryFaceRole;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct MagneticBoundarySpace {
    pub free_edges: Vec<usize>,
    pub gauge_nodes: Vec<usize>,
}

const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

impl MagneticBoundarySpace {
    pub fn derive(
        projection: &AnalysisProjection,
        edges: &TetrahedralEdgeSpace,
    ) -> Result<Self, MagnetostaticError> {
        // The initial full-grounded envelope is a connected tetrahedral ball.
        // Cavity handles/harmonic cohomology require a different nullspace contract.
        let mut faces = BTreeMap::<[usize; 3], Vec<usize>>::new();
        let mut corner_nodes = BTreeSet::new();
        for (index, element) in projection.elements.iter().enumerate() {
            corner_nodes.extend(element.node_indices[..4].iter().copied());
            for face in FACES {
                let mut nodes = face.map(|i| element.node_indices[i]);
                nodes.sort_unstable();
                faces.entry(nodes).or_default().push(index);
            }
        }
        if faces.values().any(|owners| owners.len() > 2) || projection.elements.is_empty() {
            return Err(MagnetostaticError::Unsupported(
                "nonmanifold or empty magnetic volume".into(),
            ));
        }
        let mut adjacency = vec![Vec::new(); projection.elements.len()];
        for owners in faces.values().filter(|owners| owners.len() == 2) {
            adjacency[owners[0]].push(owners[1]);
            adjacency[owners[1]].push(owners[0]);
        }
        let mut visited = BTreeSet::new();
        let mut queue = vec![0];
        while let Some(index) = queue.pop() {
            if visited.insert(index) {
                queue.extend(&adjacency[index]);
            }
        }
        if visited.len() != projection.elements.len() {
            return Err(MagnetostaticError::Unsupported(
                "disconnected magnetic volume".into(),
            ));
        }
        let exterior: BTreeSet<_> = faces
            .iter()
            .filter(|(_, owners)| owners.len() == 1)
            .map(|(&nodes, _)| nodes)
            .collect();
        let mut declared = BTreeSet::new();
        for face in &projection.boundary_faces {
            if face.role != BoundaryFaceRole::Exterior {
                continue;
            }
            let mut nodes = [
                face.node_indices[0],
                face.node_indices[1],
                face.node_indices[2],
            ];
            nodes.sort_unstable();
            if !declared.insert(nodes) {
                return Err(MagnetostaticError::Topology(
                    "duplicate exterior face".into(),
                ));
            }
        }
        if declared != exterior {
            return Err(MagnetostaticError::Topology(
                "incomplete exact exterior inventory".into(),
            ));
        }
        let mut boundary_edges = BTreeMap::<[usize; 2], usize>::new();
        let mut boundary_nodes = BTreeSet::new();
        let mut boundary_neighbors = BTreeMap::<usize, BTreeSet<usize>>::new();
        for [a, b, c] in exterior.iter().copied() {
            boundary_nodes.extend([a, b, c]);
            for [u, v] in [[a, b], [a, c], [b, c]] {
                *boundary_edges.entry([u, v]).or_default() += 1;
                boundary_neighbors.entry(u).or_default().insert(v);
                boundary_neighbors.entry(v).or_default().insert(u);
            }
        }
        if boundary_edges.values().any(|&owners| owners != 2) || boundary_nodes.is_empty() {
            return Err(MagnetostaticError::Unsupported(
                "magnetic boundary is not a closed manifold".into(),
            ));
        }
        // Edge incidence alone permits pinched vertices. Each boundary vertex
        // link must be one connected cycle, not two surfaces meeting at a node.
        for &node in &boundary_nodes {
            let mut link = BTreeMap::<usize, Vec<usize>>::new();
            for face in exterior.iter().filter(|face| face.contains(&node)) {
                let others: Vec<_> = face
                    .iter()
                    .copied()
                    .filter(|&other| other != node)
                    .collect();
                link.entry(others[0]).or_default().push(others[1]);
                link.entry(others[1]).or_default().push(others[0]);
            }
            let mut link_visited = BTreeSet::new();
            let mut link_queue = vec![*link.keys().next().unwrap()];
            while let Some(vertex) = link_queue.pop() {
                if link_visited.insert(vertex) {
                    link_queue.extend(&link[&vertex]);
                }
            }
            if link.values().any(|neighbors| neighbors.len() != 2)
                || link_visited.len() != link.len()
            {
                return Err(MagnetostaticError::Unsupported(
                    "nonmanifold magnetic boundary vertex".into(),
                ));
            }
        }
        let mut visited = BTreeSet::new();
        let mut queue = vec![*boundary_nodes.first().unwrap()];
        while let Some(node) = queue.pop() {
            if visited.insert(node) {
                queue.extend(&boundary_neighbors[&node]);
            }
        }
        // Embedded connected closed spherical boundary, with the volume's
        // canonical manifold connectivity, excludes cavity/handle nullspaces.
        if visited.len() != boundary_nodes.len()
            || boundary_nodes.len() as i128 - boundary_edges.len() as i128 + exterior.len() as i128
                != 2
            || corner_nodes.len() as i128 - edges.edges().len() as i128 + faces.len() as i128
                - projection.elements.len() as i128
                != 1
        {
            return Err(MagnetostaticError::Unsupported(
                "magnetic cohomology is not a tetrahedral ball".into(),
            ));
        }
        let free_edges = edges
            .edges()
            .iter()
            .enumerate()
            .filter_map(|(i, edge)| {
                let mut nodes = edge.node_indices;
                nodes.sort_unstable();
                (!boundary_edges.contains_key(&nodes)).then_some(i)
            })
            .collect();
        let gauge_nodes = corner_nodes.difference(&boundary_nodes).copied().collect();
        Ok(Self {
            free_edges,
            gauge_nodes,
        })
    }
}
