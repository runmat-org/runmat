use super::*;
use crate::cdt::test_fixtures::coupled_quality::fixture;
use std::cell::Cell;

#[test]
fn fixed_star_probes_reuse_the_original_local_metric_inventory() {
    use crate::cdt::topology::VolumeTopologyRead;
    let (topology, _) = fixture(false, 0);
    let index = 6;
    let star = topology.ordered_vertex_star(index as u32);
    let calls = Cell::new(0);
    let prepared =
        SmoothingObjective::Coupled(Default::default()).prepare(&topology, &star, index, &|cell| {
            calls.set(calls.get() + 1);
            // Distinct incident metrics must remain distinct in the prepared model.
            runmat_meshing_size::metric::MetricTensor3::isotropic_length_m(
                1.0 + f64::from(cell) / 10.,
            )
            .unwrap()
        });
    assert_eq!(calls.get(), star.len());
    let point = topology.nodes[index].coordinates_m.canonical_value();
    let before = prepared.score(point);
    for _ in 0..128 {
        assert!(prepared.score(point) == before);
    }
    assert_eq!(
        calls.get(),
        star.len(),
        "numerical probes must not requery metrics"
    );
    assert_eq!(prepared.cells.len(), star.len());
    assert!(prepared
        .cells
        .windows(2)
        .any(|pair| pair[0].metric != pair[1].metric));
}
