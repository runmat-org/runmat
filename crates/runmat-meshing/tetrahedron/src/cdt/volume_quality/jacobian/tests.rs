//! Differential arithmetic coverage includes rejection, ordering, translation,
//! extreme scales and off-diagonal metrics; full reports remain the reference.
use super::*;

#[test]
fn prepared_score_settles_one_owned_batch_and_invalid_edge_prefix() {
    use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};
    use std::sync::Arc;

    let metric = MetricTensor3::isotropic_length_m(1.0).unwrap();
    let prepared = PreparedJacobian::new(metric).unwrap().unwrap();
    let budget = OperationCount::from_operations(382 + 57);
    let ledger = Arc::new(UsageLedger::new(budget, budget).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    context.run_sync(|| {
        assert!(prepared
            .score([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ])
            .unwrap()
            .is_some());
        assert!(prepared.score([[0.0; 3]; 4]).unwrap().is_none());
    });
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.completed.meshing, budget);
    assert_eq!(snapshot.reserved, OperationCount::ZERO);
}

#[test]
fn search_scores_match_full_quality_reports_bit_for_bit() {
    let metrics = [
        MetricTensor3::isotropic_length_m(1.).unwrap(),
        MetricTensor3 {
            xx: 4.,
            yy: 9.,
            zz: 16.,
            xy: 0.5,
            xz: 0.25,
            yz: 0.75,
        },
        MetricTensor3 {
            xx: -1.,
            yy: 1.,
            zz: 1.,
            xy: 0.,
            xz: 0.,
            yz: 0.,
        },
    ];
    for metric in metrics {
        for exponent in [-600, -100, 0, 100, 600] {
            let scale = 2_f64.powi(exponent);
            for height in [0., 1.0e-12, 0.25, 1.] {
                let points = [
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [0.25, 0.25, height],
                ]
                .map(|point| point.map(|value| value * scale));
                for ordered in [
                    points,
                    [points[3], points[1], points[0], points[2]],
                    points.map(|point| point.map(|value| value + scale * 1024.)),
                ] {
                    let report =
                        crate::cdt::volume_quality::evaluate_tetrahedron_geometry(ordered, metric)
                            .unwrap()
                            .map(|quality| quality.metric_scaled_jacobian.to_bits());
                    assert_eq!(
                        evaluate_tetrahedron_jacobian(ordered, metric)
                            .unwrap()
                            .map(f64::to_bits),
                        report
                    );
                }
            }
        }
        for value in [f64::NAN, f64::INFINITY, f64::MAX] {
            let points = [
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [value, value, value],
            ];
            assert_eq!(
                evaluate_tetrahedron_jacobian(points, metric)
                    .unwrap()
                    .map(f64::to_bits),
                crate::cdt::volume_quality::evaluate_tetrahedron_geometry(points, metric)
                    .unwrap()
                    .map(|quality| quality.metric_scaled_jacobian.to_bits())
            );
        }
    }
}
