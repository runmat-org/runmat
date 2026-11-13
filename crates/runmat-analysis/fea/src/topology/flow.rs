use std::collections::BTreeMap;

use runmat_meshing_core::SolverMeshArtifact;

/// Deterministic finite-volume-oriented projection of a canonical solver mesh.
///
/// This is geometry and connectivity derived from the admitted mesh. It contains no fluid
/// constitutive law, boundary policy, solver choice, or placement state.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowTopologyProjection {
    pub node_count: usize,
    pub control_volume_count: usize,
    pub face_count: usize,
    pub internal_face_count: usize,
    pub boundary_face_count: usize,
    pub domain_length_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Length, f64>,
    pub hydraulic_diameter_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Length, f64>,
    pub mean_boundary_face_area_m2: f64,
    pub dx_m: runmat_types::physical::Quantity<runmat_types::physical::kind::Length, f64>,
    pub active_dimension_count: usize,
    pub edge_nodes: Vec<[u32; 2]>,
    pub boundary_face_edges: Vec<[u32; 3]>,
}

impl FlowTopologyProjection {
    pub fn derive(mesh: &SolverMeshArtifact) -> Self {
        let node_positions = mesh
            .topology
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| {
                (
                    node.node_id,
                    (index as u32, node.coordinates_m.canonical_value()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut edge_indices = BTreeMap::<[u32; 2], u32>::new();
        let mut boundary_face_edges = Vec::new();
        let mut total_boundary_area_m2 = 0.0;

        for face in &mesh.topology.boundary_faces {
            let Some(corners) = face.node_ids.get(..3) else {
                continue;
            };
            let Some(indexed) = corners
                .iter()
                .map(|node_id| node_positions.get(node_id).copied())
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };

            let mut local_edges = [0_u32; 3];
            for (local, (left, right)) in [(0, 1), (1, 2), (2, 0)].into_iter().enumerate() {
                let edge = [
                    indexed[left].0.min(indexed[right].0),
                    indexed[left].0.max(indexed[right].0),
                ];
                let next_index = edge_indices.len() as u32;
                local_edges[local] = *edge_indices.entry(edge).or_insert(next_index);
            }
            total_boundary_area_m2 += triangle_area(indexed[0].1, indexed[1].1, indexed[2].1);
            boundary_face_edges.push(local_edges);
        }

        let edge_nodes = edge_indices.into_keys().collect::<Vec<_>>();
        let spans = coordinate_spans(mesh);
        let domain_length_m = spans.into_iter().fold(0.0_f64, f64::max).max(1.0e-12);
        let active_dimension_count = spans
            .into_iter()
            .filter(|span| *span > domain_length_m * 1.0e-12)
            .count();
        let mut positive_spans = spans.into_iter().filter(|span| *span > 0.0);
        let hydraulic_diameter_m = positive_spans
            .next()
            .map(|first| positive_spans.fold(first, f64::min))
            .unwrap_or(domain_length_m)
            .max(1.0e-12);
        let control_volume_count = mesh.topology.volume_elements.len().max(1);
        let boundary_face_count = boundary_face_edges.len();
        let internal_face_count = mesh
            .topology
            .neighbors
            .iter()
            .filter(|neighbor| neighbor.adjacent_element_id.is_some())
            .count()
            / 2;

        Self {
            node_count: mesh.topology.nodes.len().max(2),
            control_volume_count,
            face_count: boundary_face_count + internal_face_count,
            internal_face_count,
            boundary_face_count,
            domain_length_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(domain_length_m),
            hydraulic_diameter_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(hydraulic_diameter_m),
            mean_boundary_face_area_m2: (total_boundary_area_m2
                / boundary_face_count.max(1) as f64)
                .max(1.0e-12),
            dx_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Length, _>::new(
                domain_length_m / control_volume_count as f64,
            ),
            active_dimension_count,
            edge_nodes,
            boundary_face_edges,
        }
    }
}

fn coordinate_spans(mesh: &SolverMeshArtifact) -> [f64; 3] {
    let bounds = mesh
        .topology
        .nodes
        .iter()
        .fold(None::<[[f64; 3]; 2]>, |bounds, node| {
            Some(match bounds {
                None => [
                    node.coordinates_m.canonical_value(),
                    node.coordinates_m.canonical_value(),
                ],
                Some(mut bounds) => {
                    for axis in 0..3 {
                        bounds[0][axis] =
                            bounds[0][axis].min(node.coordinates_m.canonical_value()[axis]);
                        bounds[1][axis] =
                            bounds[1][axis].max(node.coordinates_m.canonical_value()[axis]);
                    }
                    bounds
                }
            })
        });
    bounds
        .map(|bounds| {
            [
                bounds[1][0] - bounds[0][0],
                bounds[1][1] - bounds[0][1],
                bounds[1][2] - bounds[0][2],
            ]
        })
        .unwrap_or([1.0; 3])
}

fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    0.5 * norm(cross(subtract(b, a), subtract(c, a)))
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
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
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    use super::FlowTopologyProjection;

    #[test]
    fn derives_flow_connectivity_and_scales_from_canonical_tetrahedron() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);

        let topology = FlowTopologyProjection::derive(&mesh);

        assert_eq!(topology.node_count, 4);
        assert_eq!(topology.control_volume_count, 1);
        assert_eq!(topology.boundary_face_count, 4);
        assert_eq!(topology.internal_face_count, 0);
        assert_eq!(topology.face_count, 4);
        assert_eq!(topology.edge_nodes.len(), 6);
        assert_eq!(topology.boundary_face_edges.len(), 4);
        assert_eq!(topology.active_dimension_count, 3);
        assert!(topology.domain_length_m.canonical_value() > 0.0);
        assert!(topology.hydraulic_diameter_m.canonical_value() > 0.0);
        assert!(topology.mean_boundary_face_area_m2 > 0.0);
        assert!(topology.dx_m.canonical_value() > 0.0);
    }

    #[test]
    fn derivation_is_independent_of_boundary_face_order() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let mut reordered = mesh.clone();
        reordered.topology.boundary_faces.reverse();

        let canonical = FlowTopologyProjection::derive(&mesh);
        let permuted = FlowTopologyProjection::derive(&reordered);

        assert_eq!(canonical.node_count, permuted.node_count);
        assert_eq!(
            canonical.control_volume_count,
            permuted.control_volume_count
        );
        assert_eq!(canonical.face_count, permuted.face_count);
        assert_eq!(canonical.edge_nodes, permuted.edge_nodes);
        assert_eq!(
            canonical.mean_boundary_face_area_m2,
            permuted.mean_boundary_face_area_m2
        );
    }
}
