use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::SolverMeshArtifact;

pub(super) struct MaxwellMeshTopology {
    pub(super) edges: Vec<MaxwellMeshEdge>,
    pub(super) elements: Vec<MaxwellMeshElement>,
    pub(super) reference_element_area_m2: f64,
}

pub(super) struct MaxwellMeshEdge {
    pub(super) from_node: usize,
    pub(super) to_node: usize,
    pub(super) length_m: f64,
}

pub(super) struct MaxwellMeshElement {
    pub(super) edge_indices: Vec<usize>,
    pub(super) orientations: Vec<f64>,
    pub(super) area_m2: f64,
}

impl MaxwellMeshTopology {
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
        let mut edge_indices = BTreeMap::<[usize; 2], usize>::new();
        let mut elements = Vec::with_capacity(mesh.topology.volume_elements.len());

        for element in &mesh.topology.volume_elements {
            let corners = element.node_ids.get(..4)?;
            let nodes = corners
                .iter()
                .map(|node_id| node_indices.get(node_id).copied())
                .collect::<Option<Vec<_>>>()?;
            let mut local_indices = Vec::with_capacity(6);
            let mut orientations = Vec::with_capacity(6);
            for (left, right) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                let directed = [nodes[left], nodes[right]];
                let canonical = [directed[0].min(directed[1]), directed[0].max(directed[1])];
                let next_index = edge_indices.len();
                local_indices.push(*edge_indices.entry(canonical).or_insert(next_index));
                orientations.push(if directed == canonical { 1.0 } else { -1.0 });
            }
            elements.push(MaxwellMeshElement {
                edge_indices: local_indices,
                orientations,
                area_m2: triangle_area(
                    coordinates[nodes[0]],
                    coordinates[nodes[1]],
                    coordinates[nodes[2]],
                )
                .max(1.0e-12),
            });
        }

        let edges = edge_indices
            .into_keys()
            .map(|[from_node, to_node]| MaxwellMeshEdge {
                from_node,
                to_node,
                length_m: distance(coordinates[from_node], coordinates[to_node]).max(1.0e-12),
            })
            .collect::<Vec<_>>();
        let reference_element_area_m2 = elements.iter().map(|element| element.area_m2).sum::<f64>()
            / elements.len().max(1) as f64;
        (!edges.is_empty() && !elements.is_empty()).then_some(Self {
            edges,
            elements,
            reference_element_area_m2,
        })
    }
}

pub(super) fn boundary_node_indices<'a>(
    mesh: &SolverMeshArtifact,
    selector_ids: impl IntoIterator<Item = &'a str>,
) -> Vec<usize> {
    let selector_ids = selector_ids.into_iter().collect::<BTreeSet<_>>();
    let node_indices = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    mesh.topology
        .boundary_faces
        .iter()
        .filter(|face| {
            face.provenance
                .iter()
                .any(|entity| selector_ids.contains(entity.source_topology_id.as_str()))
        })
        .flat_map(|face| face.node_ids.iter())
        .filter_map(|node_id| node_indices.get(node_id).copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = subtract(b, a);
    let ac = subtract(c, a);
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * norm(cross)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(subtract(a, b))
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn norm(vector: [f64; 3]) -> f64 {
    vector
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{fixtures, ElementOrder};

    use super::boundary_node_indices;

    #[test]
    fn boundary_selector_resolves_canonical_node_indices() {
        let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        mesh.topology.boundary_faces[0].provenance[0].source_topology_id = "ground".to_owned();

        assert_eq!(boundary_node_indices(&mesh, ["ground"]), vec![0, 1, 2]);
        assert!(boundary_node_indices(&mesh, ["missing"]).is_empty());
    }
}
