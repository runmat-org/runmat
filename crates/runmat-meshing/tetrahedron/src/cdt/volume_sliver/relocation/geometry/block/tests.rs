use super::*;
use crate::cdt::test_fixtures::coupled_quality;
use runmat_meshing_core::NeverCancelled;
use runmat_meshing_size::metric::MetricTensor3;

#[test]
fn prepared_neighboring_blocks_escape_the_symmetric_minimax_trap_and_preserve_cpu_order() {
    let (topology, _) = coupled_quality::fixture(false, 0);
    let cells = (0..topology.tetrahedra.len() as u32).collect();
    let calls = std::cell::Cell::new(0);
    let prepared = PreparedBlock::prepare(&topology, &[6, 7], &cells, &|_| {
        calls.set(calls.get() + 1);
        MetricTensor3::isotropic_length_m(1.).unwrap()
    });
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
