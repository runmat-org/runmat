use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
struct Probe {
    calls: AtomicUsize,
    cancel_at: usize,
}
impl MeshingCancellationSignal for Probe {
    fn is_cancelled(&self) -> bool {
        self.calls.fetch_add(1, Ordering::Relaxed) >= self.cancel_at
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Prepared point lookup matches exhaustive coordinate equality, including signed zero
//   and adjacent floating-point coordinates, through4096 unrelated cells with fewer than128 query
//   probes. Preparation and complete native refinement need separate evidence.
#[test]
fn prepared_candidate_locations_match_exhaustive_ieee_equality_with_bounded_queries() {
    for cells in [0, 64, 4096] {
        let topology = crate::cdt::topology::queries::tests::topology(cells);
        let probe = Probe {
            calls: AtomicUsize::new(0),
            cancel_at: usize::MAX,
        };
        let index = CandidateQueries::new(&topology, &probe).unwrap();
        for point in [
            [0.0; 3],
            [-0.0, 0.0, 0.0],
            [0.25; 3],
            [0.25f64.next_up(), 0.25, 0.25],
            [-2.0; 3],
            topology
                .nodes
                .last()
                .unwrap()
                .coordinates_m
                .canonical_value(),
        ] {
            let reference = topology
                .nodes
                .iter()
                .any(|node| node.coordinates_m.canonical_value() == point);
            probe.calls.store(0, Ordering::Relaxed);
            assert_eq!(index.contains_location(point, &probe).unwrap(), reference);
            assert!(
                probe.calls.load(Ordering::Relaxed) < 128,
                "point lookup traversed unrelated inventory"
            );
        }
    }
}

#[test]
fn candidate_index_construction_and_queries_preserve_cancellation_and_original_nodes() {
    let topology = crate::cdt::topology::queries::tests::topology(64);
    let original = topology.clone();
    let probe = Probe {
        calls: AtomicUsize::new(0),
        cancel_at: 8,
    };
    assert!(matches!(CandidateQueries::new(&topology, &probe),
        Err(error) if error.kind==DelaunayVolumeRefinementCandidateErrorKind::Cancelled));
    assert_eq!(topology, original);
    let probe = Probe {
        calls: AtomicUsize::new(0),
        cancel_at: usize::MAX,
    };
    let index = CandidateQueries::new(&topology, &probe).unwrap();
    probe.calls.store(usize::MAX, Ordering::Relaxed);
    assert!(matches!(index.contains_location([0.25;3], &probe),
        Err(error) if error.kind==DelaunayVolumeRefinementCandidateErrorKind::Cancelled));
    assert_eq!(topology, original);
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An interrupted multi-point index update preserves the entire original lookup;
//   successful append matches the complete old-plus-new coordinate inventory. This checks
//   transactional point-state reuse, not measured allocator demand or physical insertion
//   acceptance.
#[test]
fn accepted_point_append_is_transactional_and_preserves_original_coordinate_lookup() {
    let topology = crate::cdt::topology::queries::tests::topology(64);
    let first = crate::cdt::DelaunayVolumeNode::explicit(
        runmat_meshing_core::StableDigest::from_bytes([240; 32]),
        [0.125, 0.15, 0.125],
    );
    let second = crate::cdt::DelaunayVolumeNode::explicit(
        runmat_meshing_core::StableDigest::from_bytes([241; 32]),
        [0.15, 0.125, 0.15],
    );
    let unlimited = Probe {
        calls: AtomicUsize::new(0),
        cancel_at: usize::MAX,
    };
    let mut dry_run = CandidateQueries::new(&topology, &unlimited).unwrap();
    unlimited.calls.store(0, Ordering::Relaxed);
    dry_run.append(&[first], &unlimited).unwrap();
    let after_first = unlimited.calls.load(Ordering::Relaxed);
    let mut index = CandidateQueries::new(&topology, &unlimited).unwrap();
    let cancelled = Probe {
        calls: AtomicUsize::new(0),
        cancel_at: after_first + 1,
    };
    assert!(matches!(index.append(&[first,second], &cancelled),
        Err(error) if error.kind==DelaunayVolumeRefinementCandidateErrorKind::Cancelled));
    assert_eq!(index.locations.len(), topology.nodes.len());
    assert!(!index
        .contains_location(first.coordinates_m.canonical_value(), &unlimited)
        .unwrap());
    index.append(&[first, second], &unlimited).unwrap();
    assert_eq!(index.locations.len(), topology.nodes.len() + 2);
    for point in [
        first.coordinates_m.canonical_value(),
        second.coordinates_m.canonical_value(),
        [-0.0, 0.0, 0.0],
        [5.0; 3],
    ] {
        let reference = topology
            .nodes
            .iter()
            .chain([&first, &second])
            .any(|node| node.coordinates_m.canonical_value() == point);
        assert_eq!(
            index.contains_location(point, &unlimited).unwrap(),
            reference
        );
    }
}

#[test]
fn insertion_view_retains_canonical_authority_and_coordinate_uniqueness() {
    use crate::cdt::insertion::topology_access::TetrahedralTopology;
    let topology = crate::cdt::topology::queries::tests::topology(4096);
    let queries = CandidateQueries::new(&topology, &runmat_meshing_core::NeverCancelled).unwrap();
    let view = queries.topology(&topology);
    assert!(std::ptr::eq(view.nodes(), topology.nodes.as_slice()));
    for node in topology.nodes.iter().step_by(41) {
        assert_eq!(
            view.node_index(node.identity),
            topology.node_index(node.identity)
        );
        assert!(view.contains_position(node.coordinates_m.canonical_value()));
    }
    for point in [[-0.0; 3], [0.25; 3], [0.125; 3], [f64::MAX; 3]] {
        assert_eq!(
            view.contains_position(point),
            topology.contains_position(point)
        );
    }
    for index in [0, 2, topology.tetrahedra.len() - 1] {
        let cell = view.tetrahedron(index as u32).unwrap();
        assert_eq!(
            cell.vertex_indices,
            topology.tetrahedra[index].vertex_indices
        );
        assert_eq!(
            view.tetrahedron_index(cell.vertex_indices),
            Some(index as u32)
        );
    }
}
