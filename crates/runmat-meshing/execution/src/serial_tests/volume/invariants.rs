//! Independent physical and connectivity checks permit legitimate adaptive refinements.
use runmat_meshing_core::{
    ElementOrder, MeshingRequest, SolverMeshTopology, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn assert_unit_tetrahedron(topology: &SolverMeshTopology, request: &MeshingRequest) {
    assert!(!topology.volume_elements.is_empty());
    assert!(topology.nodes.len() as u64 <= request.limits.maximum_nodes);
    assert!(topology.volume_elements.len() as u64 <= request.limits.maximum_elements);
    let nodes = topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(nodes.len(), topology.nodes.len());
    for point in nodes.values() {
        assert!(point
            .iter()
            .all(|value| value.is_finite() && *value >= -1.0e-12));
        assert!(point.iter().sum::<f64>() <= 1.0 + 1.0e-12);
    }
    let mut corners = BTreeSet::new();
    let mut edge_nodes = BTreeMap::new();
    let mut volume = 0.0;
    for element in &topology.volume_elements {
        assert_eq!(element.order, ElementOrder::Tet10);
        assert_eq!(element.node_ids.len(), 10);
        let points: [[f64; 3]; 4] = std::array::from_fn(|index| nodes[&element.node_ids[index]]);
        let [first, second, third]: [[f64; 3]; 3] = std::array::from_fn(|edge| {
            std::array::from_fn(|axis| points[edge + 1][axis] - points[0][axis])
        });
        let determinant = first[0] * (second[1] * third[2] - second[2] * third[1])
            - first[1] * (second[0] * third[2] - second[2] * third[0])
            + first[2] * (second[0] * third[1] - second[1] * third[0]);
        assert!(determinant.is_finite() && determinant > 0.0);
        volume += determinant / 6.0;
        corners.extend(element.node_ids[..4].iter().copied());
        for (index, [left, right]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
            let mut edge = [element.node_ids[left], element.node_ids[right]];
            edge.sort_unstable();
            let midpoint = element.node_ids[4 + index];
            assert_eq!(*edge_nodes.entry(edge).or_insert(midpoint), midpoint);
        }
    }
    assert!(
        (volume - 1.0 / 6.0).abs() < 1.0e-12,
        "independent volume {volume}"
    );
    assert!(edge_nodes.values().all(|node| !corners.contains(node)));
    assert_eq!(topology.nodes.len(), corners.len() + edge_nodes.len());
}

pub(super) fn assert_boundary_conservation(
    volume: &runmat_meshing_tetrahedron::cdt::DelaunayVolumeMesh,
    surface: &runmat_meshing_surface::ExactSurfaceMesh,
) {
    let source_nodes = surface
        .nodes
        .iter()
        .map(|node| (node.node_id, node.point_m))
        .collect::<BTreeMap<_, _>>();
    let recovered_nodes = volume
        .topology
        .nodes
        .iter()
        .map(|node| (node.identity, node.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();
    let mut expected = BTreeMap::new();
    for triangle in &surface.triangles {
        *expected.entry(&triangle.source_face_id).or_insert(0.0) +=
            triangle_area(triangle.node_ids.map(|id| source_nodes[&id]));
    }
    let mut recovered = BTreeMap::new();
    for facet in &volume.provenance.facets {
        let owners = facet
            .entity_ids
            .iter()
            .filter(|id| expected.contains_key(id))
            .collect::<Vec<_>>();
        assert_eq!(owners.len(), 1);
        *recovered.entry(owners[0]).or_insert(0.0) +=
            triangle_area(facet.node_identities.map(|id| recovered_nodes[&id]));
    }
    assert_eq!(expected.len(), recovered.len());
    for (face, area) in &expected {
        assert!(
            (recovered[face] - area).abs() < 1.0e-12,
            "boundary area differs for {face:?}"
        );
    }
    let area: f64 = recovered.values().sum();
    assert!((area - (1.5 + 3.0_f64.sqrt() * 0.5)).abs() < 1.0e-12);
}

fn triangle_area(points: [[f64; 3]; 3]) -> f64 {
    let first: [f64; 3] = std::array::from_fn(|axis| points[1][axis] - points[0][axis]);
    let second: [f64; 3] = std::array::from_fn(|axis| points[2][axis] - points[0][axis]);
    let cross = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ];
    0.5 * cross[0].hypot(cross[1]).hypot(cross[2])
}
