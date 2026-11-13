use super::*;
use runmat_meshing_core::{fixtures, ElementOrder};

fn projection(order: ElementOrder) -> AnalysisProjection {
    AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order)).unwrap()
}

#[test]
fn both_geometry_orders_select_exact_canonical_exterior_rows() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection = projection(order);
        let space = TetrahedralEdgeSpace::derive(&projection).unwrap();
        let rows = space.exterior_edge_indices(&projection).unwrap();
        assert_eq!(rows, (0..space.edges().len()).collect::<Vec<_>>());
        assert_eq!(rows.len(), 6);
        let mut reordered = projection.clone();
        reordered.boundary_faces.reverse();
        assert_eq!(space.exterior_edge_indices(&reordered).unwrap(), rows);
    }
}

#[test]
fn missing_duplicate_foreign_and_wrong_owner_faces_fail_closed() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection = projection(order);
        let space = TetrahedralEdgeSpace::derive(&projection).unwrap();
        let mut missing = projection.clone();
        missing.boundary_faces.pop();
        assert!(space.exterior_edge_indices(&missing).is_err());
        let mut duplicate = projection.clone();
        duplicate
            .boundary_faces
            .push(duplicate.boundary_faces[0].clone());
        assert!(space.exterior_edge_indices(&duplicate).is_err());
        let mut foreign = projection.clone();
        foreign.boundary_faces[0].node_indices[0] = projection.nodes.len();
        assert!(space.exterior_edge_indices(&foreign).is_err());
        let mut owner = projection.clone();
        owner.boundary_faces[0].adjacent_element_indices.clear();
        assert!(space.exterior_edge_indices(&owner).is_err());
    }
}

#[test]
fn foreign_projection_and_mutated_bound_element_or_node_fail_closed() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let projection = projection(order);
        let space = TetrahedralEdgeSpace::derive(&projection).unwrap();
        let mut foreign = projection.clone();
        foreign.projection_digest = runmat_execution::Digest::sha256(b"foreign");
        assert!(space.exterior_edge_indices(&foreign).is_err());
        let mut element = projection.clone();
        element.elements[0].node_indices.swap(0, 1);
        assert!(space.exterior_edge_indices(&element).is_err());
        let mut node = projection.clone();
        node.nodes.swap(0, 1);
        assert!(space.exterior_edge_indices(&node).is_err());
    }
}

#[test]
fn nonmanifold_volume_incidence_fails_before_selection() {
    // Derive the edge space from the same malformed volume inventory so this
    // witnesses incidence validation rather than only a projection binding.
    let mut projection = projection(ElementOrder::Tet4);
    for value in [1u8, 2] {
        let mut element = projection.elements[0].clone();
        element.stable_identity = runmat_meshing_core::StableDigest([value; 32]);
        projection.elements.push(element);
    }
    let space = TetrahedralEdgeSpace::derive(&projection).unwrap();
    assert!(space.exterior_edge_indices(&projection).is_err());
}
