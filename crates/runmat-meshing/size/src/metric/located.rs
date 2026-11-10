//! Sizing that applies only where its cause is: a gap that narrows along part of an edge.
//!
//! An entity-local source with an entity support applies its tensor at full strength along the
//! whole entity. That is right for curvature or a feature edge, whose size holds along the edge.
//! It is wrong for a gap, whose width varies along the edge: two circles that come within
//! 0.28 mm of each other at one point and are 5 mm apart elsewhere would size the whole circle
//! to the narrowest point, and every consumer would then grade the volume from that whole
//! circle.
//!
//! A located source carries its own support samples, each with the size that applies there as a
//! multiple of the source's narrowest size. Between consecutive samples the size varies
//! linearly. At a point `x` the source's size is the smallest graded size over its support:
//!
//! `h(x) = min_t [ c(t) + (r - 1) * max(0, |x - p(t)| - c(t)) ]`
//!
//! where `p(t)` runs along the support, `c(t)` is the interpolated size and `r` the request's
//! maximum grading ratio. With constant `c` this is the entity-support law of `local`. The
//! minimum over each segment is exact: the graded size is convex along the segment, so it is
//! reached at a segment end, at the stationary point of its graded branch, at the foot of the
//! perpendicular, or where the two branches meet.
use super::*;

/// One sample of a located source's support.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocatedMetricSample {
    pub point_m: [f64; 3],
    /// The size here as a multiple of the source tensor's size; at least 1.
    pub size_scale: f64,
}

/// Entity-local sizing whose strength varies along its own support (see the module notes).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocatedMetricSource {
    pub source: MetricSourceKind,
    /// The entity whose geometry causes the sizing (provenance).
    pub entity_id: PersistentEntityId,
    /// The tensor at the support's narrowest sample (`size_scale = 1`).
    pub metric: MetricTensor3,
    /// Ordered samples; consecutive samples form segments. One sample is a point source.
    pub samples: Vec<LocatedMetricSample>,
}

pub(super) const MAXIMUM_LOCATED_SOURCES: usize = 1_000_000;

impl LocatedMetricSource {
    pub fn validate(&self) -> Result<(), MetricContractError> {
        self.metric.validate()?;
        if matches!(
            self.source,
            MetricSourceKind::Global | MetricSourceKind::Region
        ) {
            return Err(MetricContractError::invalid(
                "located metric source",
                "global and region sizing are not located",
            ));
        }
        self.entity_id
            .validate()
            .map_err(|error| MetricContractError::invalid(error.field, error.reason))?;
        if self.samples.is_empty() || self.samples.len() > local::MAXIMUM_SUPPORT_POINTS {
            return Err(MetricContractError::invalid(
                "located metric source",
                "must contain between 1 and 4096 samples",
            ));
        }
        if self.samples.iter().any(|sample| {
            sample.point_m.iter().any(|value| !value.is_finite())
                || !sample.size_scale.is_finite()
                || sample.size_scale < 1.0
        }) {
            return Err(MetricContractError::invalid(
                "located metric source",
                "samples must be finite with a size scale of at least one",
            ));
        }
        Ok(())
    }
}

pub(super) fn validate_located(sources: &[LocatedMetricSource]) -> Result<(), MetricContractError> {
    if sources.len() > MAXIMUM_LOCATED_SOURCES {
        return Err(MetricContractError::invalid(
            "located metric sources",
            "must contain at most 1000000 entries",
        ));
    }
    for source in sources {
        source.validate()?;
    }
    if sources
        .windows(2)
        .any(|pair| compare_located(&pair[0], &pair[1]) != Ordering::Less)
    {
        return Err(MetricContractError::invalid(
            "located metric sources",
            "must be unique and canonically ordered by source, entity, tensor and samples",
        ));
    }
    Ok(())
}

/// Canonical order: source rank, entity, tensor, then samples.
pub fn compare_located(left: &LocatedMetricSource, right: &LocatedMetricSource) -> Ordering {
    left.source
        .canonical_rank()
        .cmp(&right.source.canonical_rank())
        .then_with(|| left.entity_id.cmp(&right.entity_id))
        .then_with(|| compare_metrics(left.metric, right.metric))
        .then_with(|| left.samples.len().cmp(&right.samples.len()))
        .then_with(|| {
            left.samples
                .iter()
                .zip(&right.samples)
                .map(|(left, right)| {
                    left.point_m
                        .iter()
                        .chain([&left.size_scale])
                        .zip(right.point_m.iter().chain([&right.size_scale]))
                        .map(|(left, right)| left.total_cmp(right))
                        .find(|ordering| *ordering != Ordering::Equal)
                        .unwrap_or(Ordering::Equal)
                })
                .find(|ordering| *ordering != Ordering::Equal)
                .unwrap_or(Ordering::Equal)
        })
}

/// Sorts into canonical order and drops exact duplicates.
pub fn canonicalize_located(sources: &mut Vec<LocatedMetricSource>) {
    sources.sort_by(compare_located);
    sources.dedup_by(|left, right| compare_located(left, right) == Ordering::Equal);
}

/// The smallest graded size of a located support at `point`, for a source whose narrowest
/// size is `size_m` and a grading slope `slope = r - 1 > 0`.
pub(super) fn located_size(
    samples: &[LocatedMetricSample],
    size_m: f64,
    slope: f64,
    point: [f64; 3],
) -> f64 {
    let graded = |size: f64, distance: f64| size + slope * (distance - size).max(0.0);
    if samples.len() == 1 {
        let sample = samples[0];
        return graded(size_m * sample.size_scale, distance(point, sample.point_m));
    }
    let mut best = f64::INFINITY;
    for pair in samples.windows(2) {
        best = best.min(segment_size(pair[0], pair[1], size_m, slope, point));
    }
    best
}

/// Lower bound on [`located_size`] at a point whose distance to the support's bounds is at
/// least `distance_m`.
pub(super) fn located_size_lower_bound(
    minimum_size_m: f64,
    maximum_size_m: f64,
    slope: f64,
    distance_m: f64,
) -> f64 {
    // h = max(c, (1 - slope) c + slope d) over c in [minimum, maximum] and d >= distance_m.
    let c = if slope <= 1.0 {
        minimum_size_m
    } else {
        maximum_size_m
    };
    minimum_size_m.max((1.0 - slope) * c + slope * distance_m)
}

fn segment_size(
    first: LocatedMetricSample,
    second: LocatedMetricSample,
    size_m: f64,
    slope: f64,
    point: [f64; 3],
) -> f64 {
    let start = size_m * first.size_scale;
    let change = size_m * second.size_scale - start;
    let along = difference(second.point_m, first.point_m);
    let length = dot(along, along).sqrt();
    let at = |t: f64| {
        let t = t.clamp(0.0, 1.0);
        let location = std::array::from_fn(|axis| first.point_m[axis] + t * along[axis]);
        let size = start + t * change;
        size + slope * (distance(point, location) - size).max(0.0)
    };
    let mut best = at(0.0).min(at(1.0));
    if !(length > 0.0) {
        return best;
    }
    let offset = difference(point, first.point_m);
    let foot = dot(offset, along) / length;
    let perpendicular = (dot(offset, offset) - foot * foot).max(0.0).sqrt();
    best = best.min(at(foot / length));
    // Stationary point of (1 - slope) c(t) + slope d(t).
    let ratio = -(1.0 - slope) * change / (slope * length);
    if perpendicular > 0.0 && ratio.abs() < 1.0 {
        let shift = ratio * perpendicular / (1.0 - ratio * ratio).sqrt();
        best = best.min(at((foot + shift) / length));
    }
    // Where d(t) = c(t): (L^2 - D^2) t^2 - 2 (L f + c0 D) t + (|o|^2 - c0^2) = 0.
    let a = length * length - change * change;
    let b = -2.0 * (length * foot + start * change);
    let c = dot(offset, offset) - start * start;
    if a.abs() > f64::EPSILON * length * length {
        let discriminant = b * b - 4.0 * a * c;
        if discriminant >= 0.0 {
            let root = discriminant.sqrt();
            best = best.min(at((-b - root) / (2.0 * a)));
            best = best.min(at((-b + root) / (2.0 * a)));
        }
    } else if b != 0.0 {
        best = best.min(at(-c / b));
    }
    best
}

fn difference(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    let offset = difference(left, right);
    dot(offset, offset).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(x: f64, size_scale: f64) -> LocatedMetricSample {
        LocatedMetricSample {
            point_m: [x, 0.0, 0.0],
            size_scale,
        }
    }

    /// Brute-force minimum over a fine parameter grid.
    fn sampled(samples: &[LocatedMetricSample], size: f64, slope: f64, point: [f64; 3]) -> f64 {
        let mut best = f64::INFINITY;
        for pair in samples.windows(2) {
            for step in 0..=20_000 {
                let t = f64::from(step) / 20_000.0;
                let location: [f64; 3] = std::array::from_fn(|axis| {
                    pair[0].point_m[axis] + t * (pair[1].point_m[axis] - pair[0].point_m[axis])
                });
                let c = size * (pair[0].size_scale + t * (pair[1].size_scale - pair[0].size_scale));
                best = best.min(c + slope * (distance(point, location) - c).max(0.0));
            }
        }
        best
    }

    #[test]
    fn segment_minimum_matches_a_dense_search() {
        let samples = [sample(0.0, 1.0), sample(5.0, 8.0), sample(9.0, 2.0)];
        for slope in [0.1, 0.3, 1.0, 2.5] {
            for point in [
                [0.0, 0.0, 0.0],
                [2.5, 0.3, 0.0],
                [2.5, 4.0, 1.0],
                [7.0, 0.05, 0.0],
                [-3.0, 1.0, 0.0],
                [12.0, -2.0, 0.5],
                [4.9, 0.0, 0.0],
            ] {
                let exact = located_size(&samples, 0.1, slope, point);
                let dense = sampled(&samples, 0.1, slope, point);
                assert!(
                    exact <= dense + 1e-12 && dense - exact < 1e-4,
                    "slope {slope} point {point:?}: exact {exact} dense {dense}"
                );
            }
        }
    }

    #[test]
    fn a_constant_located_support_matches_the_entity_support_law() {
        let samples = [sample(0.0, 1.0), sample(10.0, 1.0)];
        let point = [4.0, 3.0, 0.0];
        let size = 0.2;
        let expected = size + 0.3 * (3.0 - size);
        assert!((located_size(&samples, size, 0.3, point) - expected).abs() < 1e-12);
    }

    #[test]
    fn a_gap_that_narrows_at_one_sample_refines_only_near_that_sample() {
        // A 0.1 mm size at x = 0, rising to 2 mm at x = 10 mm and beyond.
        let samples = [sample(0.0, 1.0), sample(0.01, 20.0), sample(0.04, 20.0)];
        let near = located_size(&samples, 0.0001, 0.3, [0.0, 0.0005, 0.0]);
        let far = located_size(&samples, 0.0001, 0.3, [0.04, 0.0005, 0.0]);
        assert!(near < 0.00025, "{near}");
        // At the far end the graded law from the narrow sample (0.1 + 0.3 * 39.9 mm) and the
        // sample's own 2 mm size bound the size, not the narrowest 0.1 mm.
        assert!(far > 0.0019, "{far}");
    }

    #[test]
    fn lower_bound_never_exceeds_the_size() {
        let samples = [sample(0.0, 1.0), sample(5.0, 8.0)];
        for point in [[1.0, 2.0, 0.0], [7.0, 0.0, 0.0], [-4.0, 1.0, 1.0]] {
            for slope in [0.3, 2.0] {
                let size = located_size(&samples, 0.1, slope, point);
                let bound_distance = point[1]
                    .hypot(point[2])
                    .max((point[0] - 5.0).max(0.0).max(-point[0]));
                let bound = located_size_lower_bound(0.1, 0.8, slope, bound_distance);
                assert!(bound <= size + 1e-12, "{bound} > {size}");
            }
        }
    }
}
