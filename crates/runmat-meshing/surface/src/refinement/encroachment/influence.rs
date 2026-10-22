//! Conservative chart-space enclosure of the existing rounded diametral predicate.
//! Exact encroachment remains the authority; inconclusive bounds retain the plane.
use crate::ParametricMetricTensor;
use runmat_meshing_core::quality::spatial_index::Aabb3;

pub(super) fn bounds(
    first: [f64; 2],
    second: [f64; 2],
    metric: ParametricMetricTensor,
    extent: f64,
) -> Aabb3 {
    let plane = Aabb3 {
        min_m: [-f64::MAX, -f64::MAX, 0.0],
        max_m: [f64::MAX, f64::MAX, 0.0],
    };
    let scale = metric.uu.abs().max(metric.vv.abs()).max(metric.uv.abs());
    let allowance = 1024.0 * f64::EPSILON * scale;
    let lower = metric.uu.min(metric.vv) - metric.uv.abs() - allowance;
    if lower <= 0.0 || !lower.is_finite() || extent > (f64::MAX / (16.0 * scale.max(1.0))).sqrt() {
        return plane;
    }
    let midpoint = [
        first[0] * 0.5 + second[0] * 0.5,
        first[1] * 0.5 + second[1] * 0.5,
    ];
    let half_length = ((first[0] - second[0]) * 0.5).hypot((first[1] - second[1]) * 0.5);
    // With q = p-midpoint and h = (b-a)/2, the exact product is
    // q^T M q - h^T M h. The lower/upper spectral bounds below include
    // an allowance exceeding both the predicate's 32ε product-sum band
    // and floating midpoint/product error. The factor two only widens the
    // broad phase; it does not change the predicate or quality target.
    let radius = (2.0 * half_length * ((2.0 * scale + allowance) / lower).sqrt()
        + 64.0 * f64::EPSILON * extent
        + f64::MIN_POSITIVE.sqrt())
    .next_up();
    if !radius.is_finite() {
        return plane;
    }
    Aabb3 {
        min_m: [
            (midpoint[0] - radius).next_down(),
            (midpoint[1] - radius).next_down(),
            0.0,
        ],
        max_m: [
            (midpoint[0] + radius).next_up(),
            (midpoint[1] + radius).next_up(),
            0.0,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Conservative segment-influence boxes retain direct rounded encroachment decisions
    //   across the tested anisotropy and translations. This enclosure witness does not replace complete
    //   surface quality acceptance.
    #[test]
    fn influence_enclosure_never_discards_direct_encroachment() {
        for metric in [
            ParametricMetricTensor {
                uu: 1.0,
                uv: 0.0,
                vv: 1.0,
            },
            ParametricMetricTensor {
                uu: 100.0,
                uv: 0.1,
                vv: 0.01,
            },
            ParametricMetricTensor {
                uu: 1.0,
                uv: 0.99999999999,
                vv: 1.0,
            },
        ] {
            for offset in [0.0, 1e12] {
                let first = [offset - 1.0, offset];
                let second = [offset + 1.0, offset];
                let enclosure = bounds(first, second, metric, offset.abs() + 100.0);
                for x in -80..=80 {
                    for y in -80..=80 {
                        let point = [offset + x as f64 * 0.025, offset + y as f64 * 0.025];
                        if crate::refinement::encroachment::metric_diametral_disk_contains(
                            point, first, second, metric,
                        ) == Some(true)
                        {
                            assert!(enclosure.contains_point([point[0], point[1], 0.0]));
                        }
                    }
                }
            }
        }
    }
}
