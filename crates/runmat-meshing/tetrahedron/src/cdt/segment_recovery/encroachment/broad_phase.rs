//! Outward sphere enclosure; exact diametral predicates remain authoritative.
use runmat_meshing_core::Aabb3;

pub(super) struct DiametralRegion {
    pub bounds: Aabb3,
    center: [f64; 3],
    radius_squared: f64,
}
impl DiametralRegion {
    pub fn new(endpoints: [[f64; 3]; 2]) -> Option<Self> {
        let center =
            std::array::from_fn(|axis| 0.5 * endpoints[0][axis] + 0.5 * endpoints[1][axis]);
        let differences: [f64; 3] =
            std::array::from_fn(|axis| (endpoints[1][axis] - endpoints[0][axis]).abs().next_up());
        let scale = differences.into_iter().fold(0.0_f64, f64::max);
        if !scale.is_finite() {
            return None;
        }
        // Each positive operation is rounded outward. Scaling avoids overflow
        // in the norm and preserves an upper bound for subnormal differences.
        let sum = differences.into_iter().fold(0.0_f64, |sum, difference| {
            let normalized = (difference / scale).next_up();
            (sum + (normalized * normalized).next_up()).next_up()
        });
        let radius = (0.5 * (sum.sqrt().next_up() * scale).next_up()).next_up();
        // Enclose the rounding of the midpoint as well. A conservative lower
        // distance to this enlarged sphere can reject only exterior boxes.
        let magnitude = endpoints
            .into_iter()
            .flatten()
            .map(f64::abs)
            .fold(radius.max(1.0), f64::max);
        let radius = (radius + 16.0 * f64::EPSILON * magnitude).next_up();
        let bounds = Aabb3 {
            min_m: center.map(|coordinate| (coordinate - radius).next_down()),
            max_m: center.map(|coordinate| (coordinate + radius).next_up()),
        };
        if !radius.is_finite()
            || !bounds
                .min_m
                .into_iter()
                .chain(bounds.max_m)
                .all(f64::is_finite)
        {
            return None;
        }
        Some(Self {
            bounds,
            center,
            radius_squared: (radius * radius).next_up(),
        })
    }
    pub fn intersects(&self, bounds: Aabb3) -> bool {
        let lower_squared = (0..3).fold(0.0_f64, |sum, axis| {
            let gap = if self.center[axis] < bounds.min_m[axis] {
                bounds.min_m[axis] - self.center[axis]
            } else if self.center[axis] > bounds.max_m[axis] {
                self.center[axis] - bounds.max_m[axis]
            } else {
                0.0
            };
            let lower = gap.next_down().max(0.0);
            (sum + (lower * lower).next_down().max(0.0))
                .next_down()
                .max(0.0)
        });
        lower_squared <= self.radius_squared
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::quality::predicate::{
        diametral_sphere_position_constructed, Point3Construction, PredicateSign,
        SpatialPredicatePoint,
    };

    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Outward sphere enclosure retains sampled exact-predicate interior and boundary points
    //   across scales. Exact recovery predicates retain authority. Component enclosure evidence does
    //   not qualify a complete mesh or original heat study.
    #[test]
    fn sphere_pruning_preserves_exact_boundary_and_interior_across_scales() {
        for scale in [1e-100, 1e-8, 1.0, 1e100] {
            let endpoints = [[-scale, -scale, -scale], [scale, scale, scale]];
            let region = DiametralRegion::new(endpoints).unwrap();
            for x in -4..=4 {
                for y in -4..=4 {
                    for z in -4..=4 {
                        let point = [x, y, z].map(|value| f64::from(value) * scale / 2.0);
                        let explicit = |identity, coordinates| SpatialPredicatePoint {
                            identity: runmat_meshing_core::StableDigest::from_bytes([identity; 32]),
                            coordinates,
                            construction: Point3Construction::Explicit,
                        };
                        let relation = diametral_sphere_position_constructed(
                            [explicit(1, endpoints[0]), explicit(2, endpoints[1])],
                            explicit(3, point),
                        )
                        .unwrap();
                        if relation != PredicateSign::Positive {
                            assert!(region.bounds.contains_point(point));
                            assert!(region.intersects(Aabb3 {
                                min_m: point,
                                max_m: point
                            }));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn corners_are_pruned_but_overflow_keeps_exhaustive_fallback() {
        let region = DiametralRegion::new([[-1.0; 3], [1.0; 3]]).unwrap();
        assert!(!region.intersects(Aabb3 {
            min_m: [1.6; 3],
            max_m: [1.7; 3]
        }));
        assert!(region.intersects(Aabb3 {
            min_m: [-2.0; 3],
            max_m: [2.0; 3]
        }));
        assert!(DiametralRegion::new([[-f64::MAX; 3], [f64::MAX; 3]]).is_none());
        let region = DiametralRegion::new([[0.0; 3]; 2]).unwrap();
        assert!(region.intersects(Aabb3 {
            min_m: [0.0; 3],
            max_m: [0.0; 3]
        }));
    }
}
