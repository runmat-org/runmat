use super::*;

#[test]
fn gradient_owner_settles_valid_and_invalid_norm_prefixes() {
    use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};
    use std::sync::Arc;

    let metric = MetricTensor3::isotropic_length_m(1.0).unwrap();
    let prepared = PreparedJacobian::new(metric).unwrap().unwrap();
    // Each kernel reserves its full bound before it runs, so the ledger needs room for the
    // second kernel's 760-operation reservation; settlement keeps only its 75-operation prefix.
    let budget = OperationCount::from_operations(760 + 760);
    let settled = OperationCount::from_operations(760 + 75);
    let ledger = Arc::new(UsageLedger::new(budget, budget).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    context.run_sync(|| {
        assert!(prepared
            .corner_gradients([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ])
            .unwrap()
            .is_some());
        assert!(prepared.corner_gradients([[0.0; 3]; 4]).unwrap().is_none());
    });
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.completed.meshing, settled);
    assert_eq!(snapshot.reserved, OperationCount::ZERO);
}
use runmat_meshing_size::metric::MetricTensor3;

// Independent physical-coordinate metric formula, without Cholesky or derivatives.
fn corner_value(points: [[f64; 3]; 4], corner: usize, m: MetricTensor3) -> f64 {
    let edges = (0..4)
        .filter(|i| *i != corner)
        .map(|i| std::array::from_fn(|a| points[i][a] - points[corner][a]))
        .collect::<Vec<[f64; 3]>>();
    let determinant = dot(cross(edges[0], edges[1]), edges[2]).abs();
    let metric_det = m.xx * (m.yy * m.zz - m.yz * m.yz) - m.xy * (m.xy * m.zz - m.xz * m.yz)
        + m.xz * (m.xy * m.yz - m.xz * m.yy);
    let product = edges
        .into_iter()
        .map(|[x, y, z]| {
            m.xx * x * x
                + m.yy * y * y
                + m.zz * z * z
                + 2. * (m.xy * x * y + m.xz * x * z + m.yz * y * z)
        })
        .product::<f64>();
    2_f64.sqrt() * determinant * metric_det.sqrt() / product.sqrt()
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Analytic per-corner derivatives agree with independent physical-coordinate metric
//   finite differences across point ordering, translation, scale and off-diagonal anisotropy.
//   Degenerate and nonfinite inputs reject derivative proposals. Exact orientation and full quality
//   remain separate acceptance authorities; these arithmetic checks establish no whole-model
//   convergence or throughput.
#[test]
fn corner_derivatives_match_independent_metric_differences_across_orders_and_scales() {
    let metric = MetricTensor3 {
        xx: 4.,
        yy: 9.,
        zz: 16.,
        xy: 0.5,
        xz: 0.25,
        yz: 0.75,
    };
    let prepared = PreparedJacobian::new(metric).unwrap().unwrap();
    for scale in [1e-3, 1., 1e3] {
        for reverse in [false, true] {
            let mut points = [
                [0.1, 0.2, 0.3],
                [1.3, 0.1, 0.2],
                [0.2, 1.1, 0.4],
                [0.3, 0.4, 0.9],
            ]
            .map(|p| p.map(|v| scale * (v + 8.)));
            if reverse {
                points.swap(1, 2);
            }
            let samples = prepared.corner_gradients(points).unwrap().unwrap();
            for (corner, sample) in samples.into_iter().enumerate() {
                assert!((sample.value - corner_value(points, corner, metric)).abs() < 1e-12);
                for vertex in 0..4 {
                    for axis in 0..3 {
                        let h = scale * 1e-6;
                        let mut plus = points;
                        plus[vertex][axis] += h;
                        let mut minus = points;
                        minus[vertex][axis] -= h;
                        let numerical = (corner_value(plus, corner, metric)
                            - corner_value(minus, corner, metric))
                            / (2. * h);
                        assert!(
                            (sample.gradients[vertex][axis] - numerical).abs() * scale < 2e-8,
                            "corner={corner} vertex={vertex} axis={axis}"
                        );
                    }
                }
            }
        }
    }
    assert!(prepared.corner_gradients([[0.; 3]; 4]).unwrap().is_none());
    assert!(prepared
        .corner_gradients([[f64::NAN; 3]; 4])
        .unwrap()
        .is_none());
}
