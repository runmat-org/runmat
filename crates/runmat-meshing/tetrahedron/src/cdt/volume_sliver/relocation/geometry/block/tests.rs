use super::*;
use crate::cdt::test_fixtures::coupled_quality;
use runmat_meshing_core::NeverCancelled;
use runmat_meshing_size::metric::MetricTensor3;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A symmetric two-point block improves the minimum incident shape quality while
//   preserving exact positive orientation. Metric resolution occurs once per local cell, and
//   one/four reusable CPU grants produce identical proposals. Numerical probes use prepared local
//   geometry; this does not establish original-study acceptance or whole-task speedup.
#[test]
fn prepared_neighboring_blocks_escape_the_symmetric_minimax_trap_and_preserve_cpu_order() {
    let (topology, _) = coupled_quality::fixture(false, 0);
    let cells = (0..topology.tetrahedra.len() as u32).collect();
    let calls = std::cell::Cell::new(0);
    let prepared = PreparedBlock::prepare(&topology, &[6, 7], &cells, &|_| {
        calls.set(calls.get() + 1);
        MetricTensor3::isotropic_length_m(1.).unwrap()
    })
    .unwrap();
    let initial = prepared.score(&prepared.start).unwrap();
    let mut previous = None;
    for threads in [1, 4] {
        let points =
            runmat_cpu_execution::with_cpu_pool(threads, || prepared.optimize(&NeverCancelled))
                .unwrap()
                .unwrap()
                .unwrap();
        assert!(prepared.score(&points).unwrap() > 1.5 * initial);
        if let Some(previous) = previous {
            assert_eq!(points, previous);
        }
        previous = Some(points);
    }
    assert_eq!(calls.get(), topology.tetrahedra.len());
}

#[test]
fn single_point_polls_equal_complete_block_scores_including_inversions() {
    let (topology, _) = coupled_quality::fixture(false, 0);
    let cells = (0..topology.tetrahedra.len() as u32).collect();
    let prepared = PreparedBlock::prepare(&topology, &[6, 7], &cells, &|_| {
        MetricTensor3::isotropic_length_m(1.).unwrap()
    })
    .unwrap();
    let accepted = prepared.accept(&prepared.start).unwrap();
    assert_eq!(
        aggregate(&accepted.values),
        prepared.score(&prepared.start).unwrap()
    );
    let mut inverted = 0;
    for slot in 0..prepared.start.len() {
        for axis in 0..3 {
            for step in [-2., -0.5, -1e-3, 1e-3, 0.5, 2.] {
                let mut points = prepared.start.clone();
                points[slot][axis] += step * prepared.scale;
                let complete = prepared.score(&points).unwrap();
                inverted += usize::from(complete == 0.);
                assert_eq!(
                    prepared.poll(&accepted, slot, &points).unwrap().to_bits(),
                    complete.to_bits(),
                    "slot {slot}, axis {axis}, step {step}"
                );
            }
        }
    }
    assert!(inverted > 0, "fixture never exercised an inverted poll");
}
