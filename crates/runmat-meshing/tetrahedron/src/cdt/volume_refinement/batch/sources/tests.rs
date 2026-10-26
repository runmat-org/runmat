use super::*;
use crate::cdt::volume_refinement::tests::{context, quality, region};
use crate::cdt::{assign_delaunay_volume_regions, DelaunayTopologyOptions};
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A batch-sized frontier matches the complete priority-sort prefix through4096
//   unrelated cells, including ties, empty selection and cancellation. It bounds retained source
//   references; preparation still reads the quality inventory and does not establish complete
//   localized refinement throughput.
#[test]
fn bounded_frontier_matches_the_full_priority_order_across_unrelated_inventory() {
    for unrelated in [0, 64, 4096] {
        let topology = crate::cdt::topology::queries::tests::topology(unrelated);
        let count = topology.tetrahedra.len();
        let topology = assign_delaunay_volume_regions(
            topology,
            vec![region(); count],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let provenance = context(&topology);
        let (mut quality, quality_options) = quality(&topology, &provenance, 0.5);
        for (index, cell) in quality.tetrahedra.iter_mut().enumerate() {
            cell.refinement_violation_ratio = 1.5 + (index % 7) as f64;
        }
        let metric_request = crate::cdt::volume_refinement::tests::metric();
        let input = DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: &metric_request,
            provenance: &provenance,
            quality: &quality,
            quality_options,
            fixed_node_identities: &[],
        };
        let mut reference = quality.tetrahedra.iter().collect::<Vec<_>>();
        reference.sort_by(|left, right| {
            compare_violation_priority(
                right.refinement_violation_ratio,
                tetrahedron_geometry_key(&topology, right.node_identities),
                right.node_identities,
                left.refinement_violation_ratio,
                tetrahedron_geometry_key(&topology, left.node_identities),
                left.node_identities,
            )
        });
        for maximum in [0, 1, 8, 128, count + 1] {
            let selected = select(input, maximum, &NeverCancelled).unwrap();
            assert_eq!(selected, reference[..maximum.min(reference.len())]);
        }
        struct Cancelled;
        impl MeshingCancellationSignal for Cancelled {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        assert!(
            matches!(select(input,8,&Cancelled),Err(failure) if failure.kind==DelaunayVolumeRefinementStepErrorKind::Cancelled)
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Bounded windows visit every source before wrapping on an unchanged inventory,
//   including a blocked leading priority band. Priorities are synthetic scheduling inputs; this
//   does not establish cavity feasibility, physical convergence or localized whole-mesh work.
#[test]
fn bounded_windows_visit_blocked_leaders_and_the_entire_tail_before_wrapping() {
    let topology = crate::cdt::topology::queries::tests::topology(256);
    let count = topology.tetrahedra.len();
    let topology = assign_delaunay_volume_regions(
        topology,
        vec![region(); count],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let provenance = context(&topology);
    let (mut evidence, options) = quality(&topology, &provenance, 0.5);
    for (index, cell) in evidence.tetrahedra.iter_mut().enumerate() {
        // Scheduling evidence only: the first priority band represents sources
        // whose candidate cavities failed. This test makes no mesh admission claim.
        cell.refinement_violation_ratio = 2.0 + (index % 5) as f64;
    }
    let request = crate::cdt::volume_refinement::tests::metric();
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &request,
        provenance: &provenance,
        quality: &evidence,
        quality_options: options,
        fixed_node_identities: &[],
    };
    for width in [1, 7, 128] {
        let mut cursor = Frontier::default();
        let mut visited = BTreeSet::new();
        let first =
            select_window(input, width, &mut cursor, &NeverCancelled).unwrap()[0].node_identities;
        while visited.len() < count {
            let window = select_window(input, width, &mut cursor, &NeverCancelled).unwrap();
            assert!(!window.is_empty());
            assert!(window.len() <= width);
            for source in window {
                assert!(
                    visited.insert(source.node_identities),
                    "a blocked leader was retried before the tail"
                );
                cursor.advance(input, source);
            }
        }
        assert!(!cursor.has_more());
        assert_eq!(
            select_window(input, width, &mut cursor, &NeverCancelled).unwrap()[0].node_identities,
            first,
            "deferred sources must become eligible on the next pass"
        );
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A partial window resumes after its last examined source rather than skipping
//   unexamined entries or restarting the leading source. The cursor is scheduling state, not
//   reusable rejection evidence.
#[test]
fn interrupted_window_resumes_at_the_last_examined_source() {
    let topology = crate::cdt::topology::queries::tests::topology(1);
    let count = topology.tetrahedra.len();
    let topology = assign_delaunay_volume_regions(
        topology,
        vec![region(); count],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let provenance = context(&topology);
    let (evidence, options) = quality(&topology, &provenance, 0.5);
    let request = crate::cdt::volume_refinement::tests::metric();
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &request,
        provenance: &provenance,
        quality: &evidence,
        quality_options: options,
        fixed_node_identities: &[],
    };
    let mut cursor = Frontier::default();
    let window = select_window(input, 128, &mut cursor, &NeverCancelled).unwrap();
    assert!(window.len() >= 2);
    cursor.advance(input, window[0]);
    let remaining = select_window(input, 128, &mut cursor, &NeverCancelled).unwrap();
    assert_eq!(
        remaining
            .iter()
            .map(|source| source.node_identities)
            .collect::<Vec<_>>(),
        window[1..]
            .iter()
            .map(|source| source.node_identities)
            .collect::<Vec<_>>()
    );
}
