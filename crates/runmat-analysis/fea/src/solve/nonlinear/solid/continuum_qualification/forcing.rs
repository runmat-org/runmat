//! Independent polynomial work moments for the original fixed dead-load fixture.
//! Constant body/face intensities integrate analytically over the unit cube.
use super::*;

#[test]
fn original_dead_load_projection_preserves_all_representable_polynomial_work() {
    for quadratic in [false, true] {
        for cells in [1, 2, 4] {
            let (_, mesh, summary) = fixture::prepare(cells, quadratic);
            let degree = if quadratic { 2 } else { 1 };
            let mut maximum_error = 0.0_f64;
            for a in 0..=degree {
                for b in 0..=degree - a {
                    for c in 0..=degree - a - b {
                        // Test physical loads before constraint elimination; the
                        // virtual field is an operator probe, not a solved state.
                        let work = mesh
                            .topology
                            .nodes
                            .iter()
                            .enumerate()
                            .map(|(i, node)| {
                                let [x, y, z] = node.coordinates_m.canonical_value();
                                let value = x.powi(a) * y.powi(b) * z.powi(c);
                                value
                                    * summary.physical_load[summary
                                        .structural_dof_layout
                                        .index(i, StructuralDofKind::Ux)
                                        .unwrap()]
                            })
                            .sum::<f64>();
                        let body = -tangent() * B;
                        let expected = body / ((a + 1) * (b + 1) * (c + 1)) as f64
                            + stress(1.0)[0] / ((b + 1) * (c + 1)) as f64;
                        let error = (work - expected).abs();
                        maximum_error = maximum_error.max(error);
                        assert!(error < 1e-10,
                            "original load work cells={cells} quadratic={quadratic} monomial=({a},{b},{c}): actual={work}, expected={expected}, error={error}");
                    }
                }
            }
            eprintln!("original dead-load work cells={cells} quadratic={quadratic} maximumerror={maximum_error}");
        }
    }
}
