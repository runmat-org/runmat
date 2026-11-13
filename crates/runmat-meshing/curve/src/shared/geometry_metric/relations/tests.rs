use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Control;
impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The 257-owner, 32768-use incidence fixture performs exactly two owner resolutions per
//   use, and borrowed row queries preserve original use order without further owner resolution.
//   This is component work-count evidence, not arbitrary native geometry throughput.
#[test]
fn incidence_construction_is_linear_and_queries_preserve_original_use_order() {
    let (owners, uses) = (257, 32768);
    let calls = AtomicUsize::new(0);
    let owner = |index| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(index * 17 % owners)
    };
    let index = UsesByOwner::build(owners, uses, owner, &Control).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2 * uses);
    for target in 0..owners {
        assert_eq!(
            index.row(target),
            (0..uses)
                .filter(|index| index * 17 % owners == target)
                .collect::<Vec<_>>()
                .as_slice()
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 2 * uses);
    assert!(UsesByOwner::build(1, 1, |_| Ok(1), &Control).is_err());
}

#[test]
fn borrowed_relation_queries_match_admitted_topology_without_copying_entities() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let index = TopologyRelations::new(&topology, &Control).unwrap();
    for edge in &topology.edges {
        let actual = index.edge_coedges(&edge.id).collect::<Vec<_>>();
        let expected = topology
            .coedges
            .iter()
            .filter(|coedge| coedge.edge_id == edge.id)
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| std::ptr::eq(*a, b)));
        assert_eq!(
            index
                .world_transform_for(&edge.id)
                .unwrap()
                .0
                .map(f64::to_bits),
            topology
                .world_transform_for(&edge.id)
                .unwrap()
                .0
                .map(f64::to_bits)
        );
    }
    for face in &topology.faces {
        let actual = index.face_coedges(&face.id).collect::<Vec<_>>();
        let expected = topology
            .coedges
            .iter()
            .filter(|coedge| coedge.face_id == face.id)
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| std::ptr::eq(*a, b)));
    }
}
