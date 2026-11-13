use super::super::{
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricSourceKind,
    ResolvedMetricField,
};
use super::*;
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use std::collections::BTreeSet;

fn quadratic(m: MetricTensor3, [x, y, z]: [f64; 3]) -> f64 {
    m.xx * x * x + m.yy * y * y + m.zz * z * z + 2. * (m.xy * x * y + m.xz * x * z + m.yz * y * z)
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Nine,33 and4097 noncommuting constraints share a source-count-independent
//   positive-subspace envelope; independent directional inequalities check each input and the
//   common normal. Dominating original tensors remain exact. This is numerical sizing construction
//   evidence, not an interval certificate, spatial localization or native CAD acceptance.
#[test]
fn thousands_of_noncommuting_sources_do_not_multiply_density_or_refine_the_common_normal() {
    let base = MetricTensor3::isotropic_length_m(1.).unwrap();
    for count in [9, 33, 4097] {
        let mut contributions = Vec::new();
        let mut incident = BTreeSet::new();
        for index in 0..count {
            let a = std::f64::consts::TAU * index as f64 / count as f64;
            let [x, y] = [a.cos(), a.sin()];
            let id = PersistentEntityId::new(
                PersistentEntityKind::Edge,
                format!("source-{index:05}"),
                vec![],
            );
            incident.insert(id.clone());
            contributions.push(MetricContribution {
                source: MetricSourceKind::Curve,
                scope: MetricContributionScope::Entity { entity_id: id },
                metric: MetricTensor3 {
                    xx: 1. + 99. * x * x,
                    yy: 1. + 99. * y * y,
                    zz: 1.,
                    xy: 99. * x * y,
                    xz: 0.,
                    yz: 0.,
                },
            });
        }
        let request = MetricFieldRequest {
            entity_supports: Vec::new(),
            located_sources: Vec::new(),
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: base,
            maximum_grading_ratio: 1.3,
            contributions,
            cells_across_gap: None,
        };
        let resolved = ResolvedMetricField::new(&request)
            .unwrap()
            .resolve(&incident)
            .unwrap();
        assert_eq!(resolved.applied_contribution_count, count);
        assert!((resolved.metric.zz - 1.).abs() < 1e-8);
        // Independent directional inequalities test every original tensor,
        // including oblique probes; the bound is independent of source count.
        for (i, c) in request.contributions.iter().enumerate() {
            let a = i as f64 * 0.371;
            for v in [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [a.cos(), a.sin(), 0.3],
            ] {
                let norm = v.iter().map(|x| x * x).sum::<f64>();
                assert!(quadratic(resolved.metric, v) + 1e-8 * norm >= quadratic(c.metric, v));
                assert!(quadratic(resolved.metric, v) <= 100.00000001 * norm);
            }
        }
    }
}
#[test]
fn indefinite_differences_and_rotated_anisotropic_baselines_remain_dominated() {
    let a = MetricTensor3 {
        xx: 9.,
        yy: 4.,
        zz: 2.,
        xy: 1.,
        xz: 0.2,
        yz: -0.1,
    };
    let b = MetricTensor3 {
        xx: 3.,
        yy: 12.,
        zz: 5.,
        xy: -0.8,
        xz: 0.3,
        yz: 0.4,
    };
    let c = intersect(a, b).unwrap();
    for i in 0..2048 {
        let t = i as f64 * std::f64::consts::FRAC_1_PI;
        let v = [t.sin(), (t * 1.37).cos(), (t * 0.73).sin()];
        assert!(quadratic(c, v) + 1e-10 >= quadratic(a, v));
        assert!(quadratic(c, v) + 1e-10 >= quadratic(b, v));
    }
}
