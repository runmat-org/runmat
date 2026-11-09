//! Point evaluation of the metric field inside a volume, and edge length under a varying metric.
//!
//! The curve and surface stages evaluate the field at points: a face's constant part intersected
//! with its incident entity-local sources, each graded by distance from its support (see
//! `local`). A volume stage continues the same field into the interior. Interior points also feel
//! the constant sizing of nearby faces (curvature, face controls): a face-supported source grades
//! that tensor by the distance from the face's realized triangles, with the law used for edge
//! supports, so the volume coarsens away from a fine face at the requested grading ratio.
//!
//! Edge length under a varying metric is the standard remeshing integral
//! `l(e) = integral_0^1 sqrt(e^T M(a + t e) e) dt`. With the size interpolated geometrically
//! between the endpoint metrics, the integral is the logarithmic mean of the endpoint lengths.
use super::local::{bounds_distance, coarsest_length_bound, scaled};
use super::*;

/// A graded source whose support is a triangulated face.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceMetricSource {
    metric: MetricTensor3,
    characteristic_length_m: f64,
    tree: TriangleTree,
}

impl ResolvedMetricField {
    /// Every entity-local source carried by the request, in canonical entity order.
    pub fn all_local_sources(&self) -> Result<Vec<LocalMetricSource>, MetricContractError> {
        Ok(self.entity_local.clone())
    }

    /// The constant sizing of a face scope, graded from the face's triangles. `None` when the
    /// scope is nowhere finer than `baseline` (the sizing every point already receives) or has no
    /// triangles.
    pub fn surface_source(
        &self,
        incident_entities: &BTreeSet<PersistentEntityId>,
        baseline: MetricTensor3,
        triangles: Vec<[[f64; 3]; 3]>,
    ) -> Result<Option<SurfaceMetricSource>, MetricContractError> {
        let metric = self.resolve_constant(incident_entities)?.metric;
        if triangles.is_empty() || intersection::dominates(baseline, metric) {
            return Ok(None);
        }
        if triangles.iter().flatten().flatten().any(|v| !v.is_finite()) {
            return Err(MetricContractError::invalid(
                "surface metric support",
                "triangle coordinates must be finite",
            ));
        }
        Ok(Some(SurfaceMetricSource {
            metric,
            characteristic_length_m: metric.conservative_minimum_length_m()?,
            tree: TriangleTree::new(triangles),
        }))
    }

    /// `constant` intersected with every entity-local source and every face-supported source,
    /// each graded to `point_m`.
    pub fn evaluate_graded_with_surfaces(
        &self,
        constant: MetricTensor3,
        local: &[LocalMetricSource],
        surfaces: &[SurfaceMetricSource],
        point_m: [f64; 3],
    ) -> Result<MetricTensor3, MetricContractError> {
        let mut metric = self.evaluate_graded(constant, local, point_m)?;
        if surfaces.is_empty() {
            return Ok(metric);
        }
        let slope = self.request.maximum_grading_ratio - 1.0;
        if slope <= 0.0 {
            for source in surfaces {
                metric = intersection::intersect(metric, source.metric)?;
            }
            metric.validate()?;
            return Ok(metric);
        }
        let mut order = surfaces
            .iter()
            .map(|source| (bounds_distance(source.tree.bounds(), point_m), source))
            .collect::<Vec<_>>();
        order.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut reach_length = coarsest_length_bound(metric)?;
        for (lower_bound, source) in order {
            // Beyond this distance the graded source is coarser than the current metric's
            // coarsest direction, so it cannot tighten it.
            let reach = source.characteristic_length_m
                + (reach_length - source.characteristic_length_m) / slope;
            if lower_bound > reach {
                continue;
            }
            let Some(distance) = source.tree.distance_within(point_m, reach) else {
                continue;
            };
            let graded_distance = (distance - source.characteristic_length_m).max(0.0);
            let scale = 1.0 + slope * graded_distance / source.characteristic_length_m;
            let tightened = intersection::intersect(metric, scaled(source.metric, scale)?)?;
            if tightened != metric {
                metric = tightened;
                reach_length = coarsest_length_bound(metric)?;
            }
        }
        metric.validate()?;
        Ok(metric)
    }
}

/// The most restrictive metric that satisfies both operands in every direction.
pub fn intersect_metrics(
    left: MetricTensor3,
    right: MetricTensor3,
) -> Result<MetricTensor3, MetricContractError> {
    intersection::intersect(left, right)
}

/// Length of the edge `delta` when the metric varies from `start` to `end` along it, with the
/// size interpolated geometrically: the logarithmic mean of the endpoint lengths. Equal endpoint
/// metrics give exactly the constant-metric length.
pub fn integrated_edge_length(delta: [f64; 3], start: MetricTensor3, end: MetricTensor3) -> f64 {
    let length = |metric: MetricTensor3| {
        let [x, y, z] = delta;
        (metric.xx * x * x
            + metric.yy * y * y
            + metric.zz * z * z
            + 2.0 * (metric.xy * x * y + metric.xz * x * z + metric.yz * y * z))
            .max(0.0)
            .sqrt()
    };
    let (first, second) = (length(start), length(end));
    if first == second {
        return first;
    }
    if !first.is_finite() || !second.is_finite() {
        return f64::INFINITY;
    }
    if first <= 0.0 || second <= 0.0 {
        return first.max(second);
    }
    // Order the endpoint lengths so an edge measures the same in both directions, bit for bit.
    let (short, long) = if first < second {
        (first, second)
    } else {
        (second, first)
    };
    let logarithm = (long / short).ln();
    if logarithm < 1.0e-9 {
        return 0.5 * (short + long);
    }
    (long - short) / logarithm
}

/// Bounding-volume hierarchy over a face's triangles for bounded nearest-distance queries.
#[derive(Debug, Clone, PartialEq)]
struct TriangleTree {
    triangles: Vec<[[f64; 3]; 3]>,
    /// Nodes: bounds and either children (internal) or a triangle range (leaf).
    nodes: Vec<TreeNode>,
}

#[derive(Debug, Clone, PartialEq)]
struct TreeNode {
    bounds: ([f64; 3], [f64; 3]),
    kind: TreeNodeKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TreeNodeKind {
    Leaf { start: usize, end: usize },
    Internal { left: usize, right: usize },
}

const LEAF_TRIANGLES: usize = 8;

impl TriangleTree {
    fn new(mut triangles: Vec<[[f64; 3]; 3]>) -> Self {
        let mut nodes = Vec::with_capacity(2 * triangles.len() / LEAF_TRIANGLES + 1);
        let count = triangles.len();
        Self::build(&mut triangles, 0, count, &mut nodes);
        Self { triangles, nodes }
    }

    fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        self.nodes[0].bounds
    }

    fn build(
        triangles: &mut [[[f64; 3]; 3]],
        start: usize,
        end: usize,
        nodes: &mut Vec<TreeNode>,
    ) -> usize {
        let bounds = triangle_bounds(&triangles[start..end]);
        let index = nodes.len();
        nodes.push(TreeNode {
            bounds,
            kind: TreeNodeKind::Leaf { start, end },
        });
        if end - start > LEAF_TRIANGLES {
            let extent = std::array::from_fn::<f64, 3, _>(|axis| bounds.1[axis] - bounds.0[axis]);
            let axis = (0..3)
                .max_by(|left, right| extent[*left].total_cmp(&extent[*right]))
                .unwrap_or(0);
            let centroid = |triangle: &[[f64; 3]; 3]| {
                triangle[0][axis] + triangle[1][axis] + triangle[2][axis]
            };
            triangles[start..end].sort_by(|left, right| centroid(left).total_cmp(&centroid(right)));
            let middle = start + (end - start) / 2;
            let left = Self::build(triangles, start, middle, nodes);
            let right = Self::build(triangles, middle, end, nodes);
            nodes[index].kind = TreeNodeKind::Internal { left, right };
        }
        index
    }

    /// Distance from `point` to the nearest triangle, when it is at most `limit`.
    fn distance_within(&self, point: [f64; 3], limit: f64) -> Option<f64> {
        let mut best = f64::INFINITY;
        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];
            let lower = bounds_distance(node.bounds, point);
            if lower > limit || lower >= best {
                continue;
            }
            match node.kind {
                TreeNodeKind::Leaf { start, end } => {
                    for triangle in &self.triangles[start..end] {
                        best = best.min(point_triangle_distance(point, triangle));
                    }
                }
                TreeNodeKind::Internal { left, right } => {
                    let (near, far) = if bounds_distance(self.nodes[left].bounds, point)
                        <= bounds_distance(self.nodes[right].bounds, point)
                    {
                        (left, right)
                    } else {
                        (right, left)
                    };
                    stack.push(far);
                    stack.push(near);
                }
            }
        }
        (best <= limit).then_some(best)
    }
}

fn triangle_bounds(triangles: &[[[f64; 3]; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for point in triangles.iter().flatten() {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    (minimum, maximum)
}

/// Euclidean distance from a point to a closed triangle (closest-point regions of Ericson,
/// Real-Time Collision Detection, 5.1.5).
fn point_triangle_distance(p: [f64; 3], triangle: &[[f64; 3]; 3]) -> f64 {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let along = |origin: [f64; 3], direction: [f64; 3], t: f64| {
        [
            origin[0] + t * direction[0],
            origin[1] + t * direction[1],
            origin[2] + t * direction[2],
        ]
    };
    let [a, b, c] = *triangle;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    let closest = if d1 <= 0.0 && d2 <= 0.0 {
        a
    } else {
        let bp = sub(p, b);
        let d3 = dot(ab, bp);
        let d4 = dot(ac, bp);
        if d3 >= 0.0 && d4 <= d3 {
            b
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                along(a, ab, d1 / (d1 - d3))
            } else {
                let cp = sub(p, c);
                let d5 = dot(ab, cp);
                let d6 = dot(ac, cp);
                if d6 >= 0.0 && d5 <= d6 {
                    c
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                        along(a, ac, d2 / (d2 - d6))
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                            along(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)))
                        } else {
                            let denominator = 1.0 / (va + vb + vc);
                            let v = vb * denominator;
                            let w = vc * denominator;
                            [
                                a[0] + ab[0] * v + ac[0] * w,
                                a[1] + ab[1] * v + ac[1] * w,
                                a[2] + ab[2] * v + ac[2] * w,
                            ]
                        }
                    }
                }
            }
        }
    };
    let offset = sub(p, closest);
    dot(offset, offset).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isotropic(length: f64) -> MetricTensor3 {
        MetricTensor3::isotropic_length_m(length).unwrap()
    }

    #[test]
    fn integrated_length_is_the_logarithmic_mean_of_the_endpoint_lengths() {
        let delta = [1.0, 0.0, 0.0];
        assert_eq!(
            integrated_edge_length(delta, isotropic(0.5), isotropic(0.5)),
            2.0
        );
        // Size growing geometrically from 0.1 to 1.0 over a unit edge, h(t) = 0.1 * 10^t:
        // integral of dt / h(t) = (10 - 1) / ln(10).
        let length = integrated_edge_length(delta, isotropic(0.1), isotropic(1.0));
        assert!((length - 9.0 / 10.0_f64.ln()).abs() < 1.0e-12, "{length}");
        assert_eq!(
            length,
            integrated_edge_length(delta, isotropic(1.0), isotropic(0.1))
        );
        // Between the endpoint lengths, below their arithmetic mean.
        assert!(length > 1.0 && length < 5.5);
    }

    #[test]
    fn point_triangle_distance_covers_every_closest_feature() {
        let triangle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let cases = [
            ([0.25, 0.25, 2.0], 2.0),
            ([-1.0, -1.0, 0.0], 2.0_f64.sqrt()),
            ([2.0, 0.0, 0.0], 1.0),
            ([0.0, 3.0, 0.0], 2.0),
            ([0.5, -1.0, 0.0], 1.0),
            ([-1.0, 0.5, 0.0], 1.0),
            ([1.0, 1.0, 0.0], 0.5_f64.sqrt()),
        ];
        for (point, expected) in cases {
            let distance = point_triangle_distance(point, &triangle);
            assert!(
                (distance - expected).abs() < 1.0e-12,
                "{point:?}: {distance}"
            );
        }
    }

    #[test]
    fn triangle_tree_matches_brute_force_nearest_distance() {
        let mut triangles = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                let (x, y) = (i as f64 * 0.1, j as f64 * 0.1);
                let z = 0.05 * (x * 3.0).sin();
                triangles.push([[x, y, z], [x + 0.1, y, z], [x, y + 0.1, z]]);
            }
        }
        let tree = TriangleTree::new(triangles.clone());
        for point in [
            [0.37, 1.21, 0.4],
            [-0.5, 0.2, 0.0],
            [2.5, 2.5, -1.0],
            [1.0, 1.0, 0.01],
        ] {
            let brute = triangles
                .iter()
                .map(|triangle| point_triangle_distance(point, triangle))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(tree.distance_within(point, f64::INFINITY), Some(brute));
            assert_eq!(tree.distance_within(point, brute * 0.5), None);
        }
    }

    #[test]
    fn face_source_grades_its_constant_sizing_from_the_triangles() {
        let face =
            PersistentEntityId::new(PersistentEntityKind::Face, "face-1".to_owned(), Vec::new());
        let request = MetricFieldRequest {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: isotropic(1.0),
            maximum_grading_ratio: 1.5,
            contributions: vec![MetricContribution {
                source: MetricSourceKind::Face,
                scope: MetricContributionScope::Entity {
                    entity_id: face.clone(),
                },
                metric: isotropic(0.1),
            }],
            entity_supports: Vec::new(),
            cells_across_gap: None,
        };
        let field = ResolvedMetricField::new(&request).unwrap();
        let baseline = field.resolve(&BTreeSet::new()).unwrap().metric;
        let triangles = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
        let source = field
            .surface_source(&BTreeSet::from([face.clone()]), baseline, triangles)
            .unwrap()
            .unwrap();
        let at = |z: f64| {
            field
                .evaluate_graded_with_surfaces(
                    baseline,
                    &[],
                    std::slice::from_ref(&source),
                    [0.2, 0.2, z],
                )
                .unwrap()
                .conservative_minimum_length_m()
                .unwrap()
        };
        // Full strength across the first layer, then growing by the grading slope.
        assert!((at(0.05) - 0.1).abs() < 1.0e-12);
        assert!((at(0.3) - 0.1 * (1.0 + 0.5 * 2.0)).abs() < 1.0e-12);
        // Far away the global sizing governs.
        assert!((at(10.0) - 1.0).abs() < 1.0e-12);
        // A scope no finer than the baseline contributes no source.
        assert!(field
            .surface_source(&BTreeSet::new(), baseline, vec![[[0.0; 3]; 3]])
            .unwrap()
            .is_none());
    }
}
