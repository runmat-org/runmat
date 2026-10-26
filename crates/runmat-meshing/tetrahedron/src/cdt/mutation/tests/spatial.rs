use super::*;
use runmat_meshing_core::Aabb3;

#[test]
fn lazy_query_matches_inventory_after_append_and_cancelled_synchronization() {
    let mut topology = triangular_bipyramid();
    let query = Aabb3 {
        min_m: [-0.1; 3],
        max_m: [0.75; 3],
    };
    let expected = |topology: &TopologyMutation| {
        topology
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| query.contains_point(node.coordinates_m.canonical_value()))
            .map(|(index, _)| index as u32)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        topology
            .spatial_node_candidates(Some(query), &mut || Ok::<_, ()>(()))
            .unwrap(),
        expected(&topology)
    );
    topology.append_node(node(60, [0.25; 3])).unwrap();
    topology.append_node(node(70, [8.0; 3])).unwrap();
    assert!(topology
        .spatial_node_candidates(Some(query), &mut || Err(()))
        .is_err());
    assert_eq!(
        topology
            .spatial_node_candidates(Some(query), &mut || Ok::<_, ()>(()))
            .unwrap(),
        expected(&topology)
    );
    let outside = Aabb3 {
        min_m: [7.9; 3],
        max_m: [8.1; 3],
    };
    assert_eq!(
        topology
            .spatial_node_candidates(Some(outside), &mut || Ok::<_, ()>(()))
            .unwrap(),
        vec![6]
    );
    assert_eq!(topology.node_coordinate_scale_m(), 8.0);
}

fn reordered_topology() -> TopologyMutation {
    let mut topology = triangular_bipyramid();
    let interior = topology.append_node(node(5, [0.25, 0.25, 0.25])).unwrap();
    topology
        .replace_cavity(
            &BTreeSet::from([0]),
            vec![
                ([0, 1, 2, interior], None),
                ([0, 1, 3, interior], None),
                ([0, 2, 3, interior], None),
                ([1, 2, 3, interior], None),
            ],
        )
        .unwrap();
    topology
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A small admitted cavity introduces a node whose identity changes canonical ordering.
//   The original point index storage is retained and queries match an independent exhaustive
//   inventory before further append. No original heat acceptance, size-series or whole-task peak
//   claim.
#[test]
fn canonical_reordering_reuses_original_point_inventory_and_matches_exhaustive_queries() {
    let mut topology = reordered_topology();
    let bounds = Aabb3 {
        min_m: [-0.1; 3],
        max_m: [0.75; 3],
    };
    topology
        .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
        .unwrap();
    let original_storage = topology.spatial_index.borrow().as_ref().unwrap().clone();
    let canonical = topology
        .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    assert_ne!(topology.nodes, canonical.nodes);
    topology
        .adopt_canonical_with_spatial_index(canonical, &mut || Ok::<_, ()>(()))
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(
        topology.spatial_index.borrow().as_ref().unwrap(),
        &original_storage,
    ));
    let expected = topology
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| bounds.contains_point(node.coordinates_m.canonical_value()))
        .map(|(index, _)| index as u32)
        .collect::<Vec<_>>();
    assert_eq!(
        topology
            .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
            .unwrap(),
        expected
    );
    topology.append_node(node(60, [0.3; 3])).unwrap();
    assert_eq!(
        topology
            .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
            .unwrap()
            .last(),
        Some(&6)
    );
}

#[test]
fn cancelled_rebase_preserves_topology_and_index_and_changed_coordinates_rebuild() {
    let mut topology = reordered_topology();
    let bounds = Aabb3 {
        min_m: [-0.1; 3],
        max_m: [0.75; 3],
    };
    let before = topology
        .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
        .unwrap();
    let canonical = topology
        .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    let nodes = topology.nodes.clone();
    let mut remaining = 3;
    assert!(topology
        .adopt_canonical_with_spatial_index(canonical.clone(), &mut || {
            remaining -= 1;
            if remaining == 0 {
                Err(())
            } else {
                Ok(())
            }
        })
        .is_err());
    assert_eq!(topology.nodes, nodes);
    assert_eq!(
        topology
            .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
            .unwrap(),
        before
    );
    let mut translated = canonical;
    for node in &mut translated.nodes {
        node.coordinates_m =
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                node.coordinates_m
                    .canonical_value()
                    .map(|value| value + 10.0),
            );
    }
    topology
        .adopt_canonical_with_spatial_index(translated, &mut || Ok::<_, ()>(()))
        .unwrap();
    assert!(topology.spatial_index.borrow().is_none());
    assert!(topology
        .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
        .unwrap()
        .is_empty());
}

#[test]
fn cavity_trial_shares_geometry_but_append_keeps_original_query_inventory_unchanged() {
    let topology = triangular_bipyramid();
    let bounds = Aabb3 {
        min_m: [-0.1; 3],
        max_m: [0.75; 3],
    };
    let original = topology
        .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
        .unwrap();
    let mut trial = topology.clone();
    assert!(std::sync::Arc::ptr_eq(
        topology.spatial_index.borrow().as_ref().unwrap(),
        trial.spatial_index.borrow().as_ref().unwrap(),
    ));
    trial.append_node(node(60, [0.25; 3])).unwrap();
    assert!(trial
        .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
        .unwrap()
        .contains(&5));
    assert_eq!(
        topology
            .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
            .unwrap(),
        original
    );
    assert!(!std::sync::Arc::ptr_eq(
        topology.spatial_index.borrow().as_ref().unwrap(),
        trial.spatial_index.borrow().as_ref().unwrap(),
    ));
}

#[test]
fn canonical_handoff_completes_a_retained_append_prefix_before_reordering() {
    let mut topology = triangular_bipyramid();
    let bounds = Aabb3 {
        min_m: [-0.1; 3],
        max_m: [0.75; 3],
    };
    topology
        .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
        .unwrap();
    let original_index = std::sync::Arc::as_ptr(topology.spatial_index.borrow().as_ref().unwrap());
    let interior = topology.append_node(node(5, [0.25; 3])).unwrap();
    topology
        .replace_cavity(
            &BTreeSet::from([0]),
            vec![
                ([0, 1, 2, interior], None),
                ([0, 1, 3, interior], None),
                ([0, 2, 3, interior], None),
                ([1, 2, 3, interior], None),
            ],
        )
        .unwrap();
    let canonical = topology
        .snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    topology
        .adopt_canonical_with_spatial_index(canonical, &mut || Ok::<_, ()>(()))
        .unwrap();
    assert_eq!(
        std::sync::Arc::as_ptr(topology.spatial_index.borrow().as_ref().unwrap()),
        original_index
    );
    let expected = topology
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| bounds.contains_point(node.coordinates_m.canonical_value()))
        .map(|(i, _)| i as u32)
        .collect::<Vec<_>>();
    assert_eq!(
        topology
            .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
            .unwrap(),
        expected
    );
    assert_eq!(
        topology.spatial_index.borrow().as_ref().unwrap().len(),
        topology.nodes.len()
    );
}

#[test]
fn cancelled_prefix_handoff_preserves_original_topology_and_resumable_query() {
    let bounds = Aabb3 {
        min_m: [-0.1; 3],
        max_m: [0.75; 3],
    };
    let mut original = triangular_bipyramid();
    original
        .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
        .unwrap();
    let interior = original.append_node(node(5, [0.25; 3])).unwrap();
    original
        .replace_cavity(
            &BTreeSet::from([0]),
            vec![
                ([0, 1, 2, interior], None),
                ([0, 1, 3, interior], None),
                ([0, 2, 3, interior], None),
                ([1, 2, 3, interior], None),
            ],
        )
        .unwrap();
    let canonical = original
        .validated_snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    let mut completed = false;
    for allowance in 0..100 {
        let mut candidate = original.clone();
        let mut remaining = allowance;
        let result = candidate.adopt_canonical_with_spatial_index(canonical.clone(), &mut || {
            if remaining == 0 {
                Err(())
            } else {
                remaining -= 1;
                Ok(())
            }
        });
        if result.is_ok() {
            completed = true;
            break;
        }
        assert_eq!(candidate.nodes, original.nodes);
        assert_eq!(
            candidate
                .validated_snapshot(DelaunayTopologyOptions::default(), &NeverCancelled)
                .unwrap(),
            canonical
        );
        let expected = candidate
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| bounds.contains_point(node.coordinates_m.canonical_value()))
            .map(|(index, _)| index as u32)
            .collect::<Vec<_>>();
        assert_eq!(
            candidate
                .spatial_node_candidates(Some(bounds), &mut || Ok::<_, ()>(()))
                .unwrap(),
            expected
        );
    }
    assert!(completed);
}
