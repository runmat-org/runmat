use super::*;
use crate::topology::AnalysisProjection;
use runmat_meshing_core::fixtures::{
    connected_quadratic_tetrahedron_solver_mesh, connected_tetrahedron_solver_mesh,
    quadratic_slab_mesh, slab_mesh,
};
use std::collections::BTreeSet;

#[test]
fn counted_adjacency_matches_independent_element_incidence_and_preserves_order() {
    for mesh in [
        slab_mesh(2),
        quadratic_slab_mesh(2),
        connected_tetrahedron_solver_mesh(),
        connected_quadratic_tetrahedron_solver_mesh(),
    ] {
        let original = mesh.clone();
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let mut expected = vec![BTreeSet::new(); mesh.topology.nodes.len()];
        for element in &mesh.topology.volume_elements {
            for left in &element.node_ids {
                for right in &element.node_ids {
                    if left != right {
                        let source = mesh
                            .topology
                            .nodes
                            .iter()
                            .position(|node| node.node_id == *left)
                            .unwrap();
                        let neighbor = mesh
                            .topology
                            .nodes
                            .iter()
                            .position(|node| node.node_id == *right)
                            .unwrap();
                        expected[source].insert(neighbor);
                    }
                }
            }
        }
        let expected = expected
            .into_iter()
            .map(|set| set.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(projection.node_neighbors, expected);
        let cold = projection.clone();
        assert_eq!(cold.node_neighbors, expected);
        assert_eq!(cold.projection_digest, projection.projection_digest);
        for (index, node) in mesh.topology.nodes.iter().enumerate() {
            assert_eq!(projection.node_index(node.node_id), Some(index));
            for &neighbor in &expected[index] {
                assert!(expected[neighbor].contains(&index));
            }
        }
        assert_eq!(mesh, original);
    }
}

#[test]
fn empty_and_overflowing_occurrences_or_cancelled_construction_fail_at_the_owner() {
    assert_eq!(occurrence_count([0, 1, 4, 10].into_iter()).unwrap(), 102);
    assert!(occurrence_count([usize::MAX].into_iter()).is_err());
    assert_eq!(derive(&[], 3).unwrap(), vec![Vec::<usize>::new(); 3]);
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        derive(&[], 3).unwrap_err(),
        AnalysisProjectionError::Cancelled
    );
    assert_eq!(
        AnalysisProjection::derive(&slab_mesh(1)).unwrap_err(),
        AnalysisProjectionError::Cancelled
    );
}
