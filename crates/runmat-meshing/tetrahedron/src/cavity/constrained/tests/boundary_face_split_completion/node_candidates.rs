use super::super::*;

#[test]
fn boundary_face_split_candidates_include_bounded_interior_lattice() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");

    let candidates = boundary_face_split_node_candidates([0, 1, 2], &boundary_nodes);

    assert!(candidates.len() >= 40);
    assert!(candidates.len() <= 64);
    assert!(candidates.iter().all(|node| node.node_id == 4));
    assert!(candidates.iter().all(|node| {
        node.coordinates_m[0] > 0.0
            && node.coordinates_m[1] > 0.0
            && node.coordinates_m[2] == 0.0
            && node.coordinates_m[0] + node.coordinates_m[1] < 1.0
    }));
    assert!(candidates.iter().any(|node| {
        (node.coordinates_m[0] - 0.1).abs() <= 1.0e-12
            && (node.coordinates_m[1] - 0.1).abs() <= 1.0e-12
    }));
}
