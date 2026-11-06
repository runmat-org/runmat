//! Entity-local sizing graded by distance from its exact support.
//!
//! An edge-scoped contribution (curvature, feature, proximity, contact) constrains the mesh near
//! that edge, not across every face the edge bounds. When the request carries the edge's sampled
//! world-space support, consumers evaluate it at a point `x` as the original tensor relaxed to
//! the characteristic size `h + (r - 1) * max(0, d(x, support) - h)`, where `r` is the request's
//! maximum grading ratio: full strength across the element layer touching the support, then
//! graded. Scaling preserves anisotropy and the result is intersected with the constant
//! part of the field, so it is never coarser than the entity-free sizing. An entity without a
//! support keeps the constant semantics: its contribution applies across every incident face.
use super::*;

/// Sampled world-space support of one entity-local contribution scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricEntitySupport {
    pub entity_id: PersistentEntityId,
    /// Ordered samples along the entity; consecutive samples form the support polyline.
    pub points_m: Vec<[f64; 3]>,
}

pub(super) const MAXIMUM_SUPPORT_POINTS: usize = 4096;

pub(super) fn validate_supports(
    supports: &[MetricEntitySupport],
) -> Result<(), MetricContractError> {
    if supports.len() > 1_000_000 {
        return Err(MetricContractError::invalid(
            "metric entity supports",
            "must contain at most 1000000 entries",
        ));
    }
    if supports
        .windows(2)
        .any(|pair| pair[0].entity_id >= pair[1].entity_id)
    {
        return Err(MetricContractError::invalid(
            "metric entity supports",
            "must be unique and ordered by entity identity",
        ));
    }
    for support in supports {
        support
            .entity_id
            .validate()
            .map_err(|error| MetricContractError::invalid(error.field, error.reason))?;
        if !matches!(
            support.entity_id.kind,
            PersistentEntityKind::Edge | PersistentEntityKind::Vertex
        ) {
            return Err(MetricContractError::invalid(
                "metric entity support",
                "only edges and vertices carry local supports",
            ));
        }
        if support.points_m.is_empty() || support.points_m.len() > MAXIMUM_SUPPORT_POINTS {
            return Err(MetricContractError::invalid(
                "metric entity support",
                "must contain between 1 and 4096 samples",
            ));
        }
        if support
            .points_m
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(MetricContractError::invalid(
                "metric entity support",
                "samples must be finite",
            ));
        }
    }
    Ok(())
}

/// One entity-local tensor applicable to a constant query scope, with its support bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalMetricSource {
    support: usize,
    metric: MetricTensor3,
    characteristic_length_m: f64,
    bounds: ([f64; 3], [f64; 3]),
}

impl ResolvedMetricField {
    fn support_index(&self, entity: &PersistentEntityId) -> Option<usize> {
        self.request
            .entity_supports
            .binary_search_by(|support| support.entity_id.cmp(entity))
            .ok()
    }

    /// True when `entity` is graded from its support instead of applying uniformly.
    pub fn is_local(&self, entity: &PersistentEntityId) -> bool {
        self.support_index(entity).is_some()
    }

    /// The part of the field that applies uniformly over a scope: every applicable contribution
    /// except entity-local ones with a support. Provenance counts only those contributions.
    pub fn resolve_constant(
        &self,
        incident_entities: &BTreeSet<PersistentEntityId>,
    ) -> Result<ResolvedMetricEvaluation, MetricContractError> {
        if self.request.entity_supports.is_empty() {
            return self.resolve(incident_entities);
        }
        let constant = incident_entities
            .iter()
            .filter(|entity| !self.is_local(entity))
            .cloned()
            .collect();
        self.resolve(&constant)
    }

    /// Entity-local sources among `incident_entities`, in canonical entity order.
    pub fn local_sources(
        &self,
        incident_entities: &BTreeSet<PersistentEntityId>,
    ) -> Result<Vec<LocalMetricSource>, MetricContractError> {
        let mut sources = Vec::new();
        for entity in incident_entities {
            let Some(support) = self.support_index(entity) else {
                continue;
            };
            let mut applicable = self
                .index
                .applicable(&self.request, &BTreeSet::from([entity.clone()]));
            if applicable.is_empty() {
                continue;
            }
            intersection::deduplicate(&self.request, &mut applicable);
            let metric = intersection::combine(&self.request, &applicable)?;
            sources.push(self.supported_source(support, metric)?);
        }
        Ok(sources)
    }

    /// A tensor graded from the support of `entity`, if that entity has one.
    pub(crate) fn source_on(
        &self,
        entity: &PersistentEntityId,
        metric: MetricTensor3,
    ) -> Result<Option<LocalMetricSource>, MetricContractError> {
        self.support_index(entity)
            .map(|support| self.supported_source(support, metric))
            .transpose()
    }

    pub(crate) fn has_supports(&self) -> bool {
        !self.request.entity_supports.is_empty()
    }

    fn supported_source(
        &self,
        support: usize,
        metric: MetricTensor3,
    ) -> Result<LocalMetricSource, MetricContractError> {
        let characteristic_length_m = metric.conservative_minimum_length_m()?;
        let points = &self.request.entity_supports[support].points_m;
        let mut minimum = [f64::INFINITY; 3];
        let mut maximum = [f64::NEG_INFINITY; 3];
        for point in points {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(point[axis]);
                maximum[axis] = maximum[axis].max(point[axis]);
            }
        }
        Ok(LocalMetricSource {
            support,
            metric,
            characteristic_length_m,
            bounds: (minimum, maximum),
        })
    }

    /// Evaluate `constant` intersected with every local source graded to `point_m`.
    pub fn evaluate_graded(
        &self,
        constant: MetricTensor3,
        sources: &[LocalMetricSource],
        point_m: [f64; 3],
    ) -> Result<MetricTensor3, MetricContractError> {
        if sources.is_empty() {
            return Ok(constant);
        }
        if point_m.iter().any(|value| !value.is_finite()) {
            return Err(MetricContractError::invalid(
                "graded metric query",
                "point must be finite",
            ));
        }
        let slope = self.request.maximum_grading_ratio - 1.0;
        if slope <= 0.0 {
            // Without grading every local source applies at full strength everywhere.
            let mut metric = constant;
            for source in sources {
                metric = intersection::intersect(metric, source.metric)?;
            }
            metric.validate()?;
            return Ok(metric);
        }
        // Nearest sources first (ties keep canonical order): each one that tightens the metric
        // shrinks the distance beyond which a later source is isotropically coarser than the
        // metric's coarsest direction, so it is dominated and skipped. Every consumer, including
        // the independent validator, evaluates through this function.
        let mut order = sources
            .iter()
            .map(|source| (bounds_distance(source.bounds, point_m), source))
            .collect::<Vec<_>>();
        order.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut metric = constant;
        let mut reach_length = coarsest_length_bound(metric)?;
        for (lower_bound, source) in order {
            let reach = source.characteristic_length_m
                + (reach_length - source.characteristic_length_m) / slope;
            if lower_bound > reach {
                continue;
            }
            let distance = polyline_distance(
                &self.request.entity_supports[source.support].points_m,
                point_m,
            );
            if distance > reach {
                continue;
            }
            // Full strength across the element layer touching the support: the volume applies
            // an entity's sizing to every cell with a corner on it, so boundary facets touching
            // the entity must already satisfy it. Grading starts one characteristic length out.
            let graded_distance = (distance - source.characteristic_length_m).max(0.0);
            let scale = 1.0 + slope * graded_distance / source.characteristic_length_m;
            let graded = scaled(source.metric, scale)?;
            let tightened = intersection::intersect(metric, graded)?;
            if tightened != metric {
                metric = tightened;
                reach_length = coarsest_length_bound(metric)?;
            }
        }
        metric.validate()?;
        Ok(metric)
    }
}

/// Upper bound on the coarsest directional length of a tensor: its smallest eigenvalue is at
/// least `det / lambda_max^2`, and a row-sum bound dominates `lambda_max`.
pub(super) fn coarsest_length_bound(metric: MetricTensor3) -> Result<f64, MetricContractError> {
    metric.validate()?;
    let maximum = metric.maximum_row_density();
    let determinant = metric.xx * metric.yy * metric.zz + 2.0 * metric.xy * metric.xz * metric.yz
        - metric.xx * metric.yz * metric.yz
        - metric.yy * metric.xz * metric.xz
        - metric.zz * metric.xy * metric.xy;
    let minimum_eigenvalue = determinant / (maximum * maximum);
    let length = minimum_eigenvalue.sqrt().recip();
    if length.is_finite() && length > 0.0 {
        Ok(length)
    } else {
        Ok(f64::INFINITY)
    }
}

pub(super) fn scaled(metric: MetricTensor3, size_scale: f64) -> Result<MetricTensor3, MetricContractError> {
    let factor = (size_scale * size_scale).recip();
    let result = MetricTensor3 {
        xx: metric.xx * factor,
        yy: metric.yy * factor,
        zz: metric.zz * factor,
        xy: metric.xy * factor,
        xz: metric.xz * factor,
        yz: metric.yz * factor,
    };
    result.validate()?;
    Ok(result)
}

pub(super) fn bounds_distance(bounds: ([f64; 3], [f64; 3]), point: [f64; 3]) -> f64 {
    (0..3)
        .map(|axis| {
            let below = bounds.0[axis] - point[axis];
            let above = point[axis] - bounds.1[axis];
            below.max(above).max(0.0)
        })
        .map(|distance| distance * distance)
        .sum::<f64>()
        .sqrt()
}

pub(super) fn polyline_distance(points: &[[f64; 3]], point: [f64; 3]) -> f64 {
    let squared = |a: [f64; 3]| a.iter().map(|value| value * value).sum::<f64>();
    let difference = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    if points.len() == 1 {
        return squared(difference(point, points[0])).sqrt();
    }
    let mut best = f64::INFINITY;
    for pair in points.windows(2) {
        let segment = difference(pair[1], pair[0]);
        let offset = difference(point, pair[0]);
        let length = squared(segment);
        let fraction = if length > 0.0 {
            ((offset[0] * segment[0] + offset[1] * segment[1] + offset[2] * segment[2]) / length)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let nearest = [
            pair[0][0] + fraction * segment[0],
            pair[0][1] + fraction * segment[1],
            pair[0][2] + fraction * segment[2],
        ];
        best = best.min(squared(difference(point, nearest)));
    }
    best.sqrt()
}
