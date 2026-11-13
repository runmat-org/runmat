use super::*;
use runmat_meshing_size::metric::MetricTensor3;
use std::cell::Cell;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Prepared edge-star children match separately evaluated child geometry and resolve
//   each child scope once across 128 repeated probes. The complete trial still rederives graded
//   size and quality; this local-work check is not a whole-task memory or speedup claim.
#[test]
fn split_probes_resolve_each_child_scope_once() {
    let (topology, _) = crate::cdt::volume_sliver::tests::interior_sliver_fixture();
    let edge = (0..topology.nodes.len() as u32)
        .flat_map(|a| ((a + 1)..topology.nodes.len() as u32).map(move |b| [a, b]))
        .find(|edge| topology.edge_star(*edge).len() >= 3 && !topology.edge_is_boundary(*edge))
        .unwrap();
    let star = topology.edge_star(edge);
    let metric = |face: [u32; 3]| MetricTensor3 {
        xx: 1. + f64::from(face.into_iter().sum::<u32>()),
        yy: 1.,
        zz: 0.25,
        xy: 0.,
        xz: 0.,
        yz: 0.,
    };
    let calls = Cell::new(0);
    let prepared = PreparedEdgeStar::prepare(&topology, &star, edge, &|face| {
        calls.set(calls.get() + 1);
        Ok(metric(face))
    })
    .unwrap();
    let point = std::array::from_fn(|axis| {
        (topology.nodes[edge[0] as usize]
            .coordinates_m
            .canonical_value()[axis]
            + topology.nodes[edge[1] as usize]
                .coordinates_m
                .canonical_value()[axis])
            * 0.5
    });
    let mut expected = f64::INFINITY;
    for cell in &star {
        let opposite_vertices = topology
            .cell(*cell)
            .vertex_indices
            .into_iter()
            .filter(|vertex| !edge.contains(vertex))
            .collect::<Vec<_>>();
        let opposite = topology
            .cell(*cell)
            .vertex_indices
            .into_iter()
            .filter(|vertex| !edge.contains(vertex))
            .map(|vertex| {
                topology.nodes[vertex as usize]
                    .coordinates_m
                    .canonical_value()
            })
            .collect::<Vec<_>>();
        for endpoint in edge {
            let points = [
                topology.nodes[endpoint as usize]
                    .coordinates_m
                    .canonical_value(),
                point,
                opposite[0],
                opposite[1],
            ];
            let value = crate::cdt::volume_quality::evaluate_tetrahedron_geometry(
                points,
                metric([endpoint, opposite_vertices[0], opposite_vertices[1]]),
            )
            .unwrap()
            .unwrap();
            expected = expected.min(value.metric_scaled_jacobian);
        }
    }
    assert!(expected > 0.);
    for _ in 0..128 {
        assert_eq!(prepared.score(point).unwrap(), expected);
    }
    assert_eq!(
        calls.get(),
        2 * star.len(),
        "each child scope is resolved once, never per probe"
    );
}
