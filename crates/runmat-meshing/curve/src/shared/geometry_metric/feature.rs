//! Feature scale comes from geometric extent, not numerical probe spacing.
//!
//! The farthest pair of exact world-space witnesses gives a lower bound on an
//! edge's diameter, including closed edges with coincident endpoints. Curvature
//! and proximity contribute their own restrictions; constructive chord/tangent
//! checks remain authoritative. Adding probes to a straight edge cannot impose
//! artificial refinement. The caller uses a fixed, bounded witness inventory.
pub(super) fn witness_span_m(points: &[[f64; 3]]) -> Option<f64> {
    let mut span = 0.0_f64;
    for (index, left) in points.iter().enumerate() {
        for right in &points[index + 1..] {
            let distance = (left[0] - right[0])
                .hypot(left[1] - right[1])
                .hypot(left[2] - right[2]);
            span = span.max(distance);
        }
    }
    (span.is_finite() && span > 0.0).then_some(span)
}

/// Preserve feature extent along its sampled directions. A short transverse
/// edge does not impose that scale along an unrelated extrusion direction.
pub(super) fn directional_metric(
    metric: runmat_meshing_core::MetricTensor3,
    points: &[[f64; 3]],
    span_m: f64,
) -> Result<runmat_meshing_core::MetricTensor3, runmat_meshing_size::metric::MetricContractError> {
    let mut envelope = runmat_meshing_size::metric::DirectionalMetricEnvelope::new(metric)?;
    for pair in points.windows(2) {
        let direction = std::array::from_fn(|axis| pair[1][axis] - pair[0][axis]);
        if direction.iter().any(|value| *value != 0.0) {
            envelope.include(direction, span_m)?;
        }
    }
    envelope.metric()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_transverse_feature_does_not_refine_the_extrusion_axis() {
        let base = runmat_meshing_core::MetricTensor3::isotropic_length_m(0.01).unwrap();
        let points = [[2.0, -3.0, 4.0], [2.0, -2.99975, 4.0]];
        let span = witness_span_m(&points).unwrap();
        let metric = directional_metric(base, &points, span).unwrap();
        assert_eq!(metric.xx, base.xx);
        assert_eq!(metric.zz, base.zz);
        assert!(metric.yy >= span.powi(-2));
    }

    #[test]
    fn straight_feature_size_is_independent_of_probe_density_and_spacing() {
        for count in [2, 9, 33, 257] {
            for power in [1, 3] {
                // A one-millimetre feature under a translated, anisotropic placement.
                let points = (0..count)
                    .map(|index| {
                        let fraction = (index as f64 / (count - 1) as f64).powi(power);
                        [2.0, -3.0, 0.001 * fraction]
                    })
                    .collect::<Vec<_>>();
                assert_eq!(witness_span_m(&points), Some(0.001));
            }
        }
    }

    #[test]
    fn closed_feature_uses_its_extent_and_degenerate_witnesses_do_not_invent_size() {
        let points = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        assert_eq!(witness_span_m(&points), Some(2.0));
        assert_eq!(witness_span_m(&[]), None);
        assert_eq!(witness_span_m(&[[0.0; 3]; 9]), None);
    }
}
