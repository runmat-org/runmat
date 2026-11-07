use std::collections::BTreeSet;

use runmat_geometry_core::{
    GeometryAllocationReservation, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::{
    MetricContribution, MetricContributionScope, MetricSourceKind, MetricTensor3,
};

use super::{SharedCurveError, SharedCurveErrorKind};
use runmat_meshing_size::metric::DirectionalMetricEnvelope;

#[derive(Clone, Debug)]
pub(super) struct EdgeWitness {
    pub edge_id: PersistentEntityId,
    /// The exact edge's geometric tolerance; separations within the pair's combined
    /// tolerance are the same location.
    pub tolerance_m: f64,
    pub endpoint_ids: BTreeSet<PersistentEntityId>,
    pub face_ids: BTreeSet<PersistentEntityId>,
    /// Solids bounded by the edge's faces; empty for sheet geometry.
    pub solid_ids: BTreeSet<PersistentEntityId>,
    pub points_m: Vec<[f64; 3]>,
}

/// Whether the space between features bounding the solids `left` and `right` lies outside
/// every meshed region the features bound. The mesh fills solids only, and a face bounds the
/// solids on both of its sides (a face shared by two regions bounds both). Features of one
/// solid, or of solids that share a bounding region, face each other across material of that
/// region: a thin wall, or a thin layer of a region bonded between them. Features with no
/// bounded solid in common belong to bodies that meet only where an authored contact joins
/// them; elsewhere the space between them is either the exterior of both, which is never
/// meshed, or a third region, whose own bounding features measure its thickness. Sheet
/// geometry bounds no solid, so it keeps every gap.
fn gap_is_unmeshed_void(
    left: &BTreeSet<PersistentEntityId>,
    right: &BTreeSet<PersistentEntityId>,
) -> bool {
    !left.is_empty() && !right.is_empty() && left.is_disjoint(right)
}

#[derive(Clone, Debug, Default)]
pub(super) struct ContactProximityPairs {
    pub faces: BTreeSet<(PersistentEntityId, PersistentEntityId)>,
    pub vertices: BTreeSet<(PersistentEntityId, PersistentEntityId)>,
}

pub(super) fn derive_proximity_contributions(
    mut witnesses: Vec<EdgeWitness>,
    contact_pairs: &ContactProximityPairs,
    base_metric: MetricTensor3,
    search_radius_m: f64,
    cells_across_gap: u32,
    control: &dyn GeometryEvaluationControl,
) -> Result<Vec<MetricContribution>, SharedCurveError> {
    if !search_radius_m.is_finite() || search_radius_m <= 0.0 {
        return Err(SharedCurveError::invalid_request(
            "curve proximity radius",
            "must be finite and greater than zero",
        ));
    }
    for witness in &witnesses {
        if witness.points_m.is_empty()
            || witness
                .points_m
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
        {
            return Err(SharedCurveError::new(
                SharedCurveErrorKind::GeometryEvaluation(
                    runmat_geometry_core::GeometryEvaluationErrorKind::InvalidResult,
                ),
                "curve proximity witnesses",
                "each exact edge must provide finite witness points",
            )
            .for_edge(&witness.edge_id));
        }
    }
    witnesses.sort_by(|left, right| {
        bounds(left).0[0]
            .total_cmp(&bounds(right).0[0])
            .then_with(|| left.edge_id.cmp(&right.edge_id))
    });
    let bytes = witnesses
        .len()
        .checked_mul(std::mem::size_of::<[Option<DirectionalMetricEnvelope>; 2]>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            SharedCurveError::invalid_request("proximity metric inventory", "storage overflow")
        })?;
    let _storage = GeometryAllocationReservation::new(control, bytes).map_err(|error| {
        SharedCurveError::new(
            SharedCurveErrorKind::GeometryEvaluation(error.kind),
            "proximity metric inventory",
            error.reason,
        )
    })?;
    let base = DirectionalMetricEnvelope::new(base_metric).map_err(|error| {
        SharedCurveError::invalid_request("proximity metric baseline", error.to_string())
    })?;
    let mut metrics = vec![[None; 2]; witnesses.len()];
    for left_index in 0..witnesses.len() {
        let left = &witnesses[left_index];
        let left_bounds = bounds(left);
        for (right_index, right) in witnesses.iter().enumerate().skip(left_index + 1) {
            let right_bounds = bounds(right);
            if right_bounds.0[0] - left_bounds.1[0] > search_radius_m {
                break;
            }
            if !left.endpoint_ids.is_disjoint(&right.endpoint_ids)
                || bounds_distance(left_bounds, right_bounds) > search_radius_m
            {
                continue;
            }
            let source = if witnesses_are_contact(left, right, contact_pairs) {
                MetricSourceKind::Contact
            } else {
                MetricSourceKind::Proximity
            };
            if source == MetricSourceKind::Proximity
                && gap_is_unmeshed_void(&left.solid_ids, &right.solid_ids)
            {
                // Separate bodies across a void (or seated together, coincident): no meshed
                // gap to resolve, so neither body is refined by the other.
                continue;
            }
            let mut minimum = f64::INFINITY;
            let mut separation_direction = [0.0; 3];
            for left_point in &left.points_m {
                // The pair search is quadratic in the sample counts: probe responsiveness per
                // row so the work between checkpoints is one row, not one whole pair.
                control
                    .checkpoint()
                    .map_err(|error| geometry_error(&left.edge_id, error))?;
                for right_point in &right.points_m {
                    control
                        .consume_search_work(1)
                        .map_err(|error| geometry_error(&left.edge_id, error))?;
                    let separation = distance(*left_point, *right_point);
                    if separation < minimum {
                        minimum = separation;
                        separation_direction =
                            std::array::from_fn(|axis| right_point[axis] - left_point[axis]);
                    }
                }
            }
            if minimum <= coincidence_tolerance_m(left, right) {
                // Authored contact edges that coincide have no gap to resolve. Two separately
                // evaluated coincident edges agree only to their geometric tolerance and
                // coordinate roundoff, so a separation inside that is coincidence, not a gap
                // to size for. Edges of separate bodies never reach here. Within one body (or
                // bodies sharing a region), coincident nonincident edges are a topology defect.
                if source == MetricSourceKind::Contact {
                    continue;
                }
                return Err(SharedCurveError::new(
                    SharedCurveErrorKind::GeometricMismatch,
                    "curve proximity",
                    "nonincident exact edge witnesses coincide",
                )
                .for_edge(&left.edge_id));
            }
            if minimum < search_radius_m {
                let source_index = usize::from(source == MetricSourceKind::Contact);
                for index in [left_index, right_index] {
                    let current = metrics[index][source_index].get_or_insert(base);
                    current
                        .include(separation_direction, minimum / f64::from(cells_across_gap))
                        .map_err(|error| {
                            SharedCurveError::invalid_request(
                                "directional proximity metric",
                                error.to_string(),
                            )
                            .for_edge(&witnesses[index].edge_id)
                        })?;
                }
            }
        }
    }
    let mut contributions = Vec::new();
    for (witness, metrics) in witnesses.into_iter().zip(metrics) {
        for (metric, source) in metrics
            .into_iter()
            .zip([MetricSourceKind::Proximity, MetricSourceKind::Contact])
        {
            if let Some(metric) = metric {
                contributions.push(MetricContribution {
                    source,
                    scope: MetricContributionScope::Entity {
                        entity_id: witness.edge_id.clone(),
                    },
                    metric: metric.metric().map_err(|error| {
                        SharedCurveError::invalid_request(
                            "proximity metric envelope",
                            error.to_string(),
                        )
                        .for_edge(&witness.edge_id)
                    })?,
                });
            }
        }
    }
    // The parent request performs the one authoritative canonical ordering
    // together with curvature and feature contributions.
    Ok(contributions)
}

/// Boundary samples of one face, for measuring gaps between facing surfaces.
#[derive(Clone, Debug)]
pub(super) struct FaceWitness {
    pub face_id: PersistentEntityId,
    /// The largest geometric tolerance of the face's edges.
    pub tolerance_m: f64,
    pub vertex_ids: BTreeSet<PersistentEntityId>,
    /// Solids the face bounds (both sides of a shared face); empty for sheet geometry.
    pub solid_ids: BTreeSet<PersistentEntityId>,
    /// One polyline per bounding edge: world-space points where that edge lands on the face, in
    /// edge parameter order, with the unit surface normal there.
    pub boundaries: Vec<Vec<([f64; 3], [f64; 3])>>,
}

impl FaceWitness {
    pub fn new(face_id: PersistentEntityId) -> Self {
        Self {
            face_id,
            tolerance_m: 0.0,
            vertex_ids: BTreeSet::new(),
            solid_ids: BTreeSet::new(),
            boundaries: Vec::new(),
        }
    }
}

/// Two surfaces face each other across a gap when the separation between their boundaries lies
/// within this angle (about 25 degrees) of both surface normals, up to sign: the two sides of a
/// thin wall, the inner and outer surfaces of a tube, the two walls of a slot. Surfaces meeting
/// through a blend or a chamfer separate obliquely and are not a gap.
const FACING_COSINE: f64 = 0.9;

/// Gaps between facing surfaces: each pair of faces that share no vertex and whose boundaries
/// face each other across a separation `g` below the search radius receives an isotropic size
/// `g / cells_across_gap` over both faces, so the surface mesh and every volume cell touching
/// either face resolve the gap with that many cells (the volume grades from there). Edges alone
/// measure the gap only along the faces' boundaries, and only in the direction across it; this
/// is the same proximity source applied to the surfaces between them, in every direction. The
/// separation runs from each boundary sample of one face to the closest point of the other
/// face's boundary polylines, so the two faces' edges need not be sampled at matching points:
/// the top and bottom faces of a thin layer meet its side faces along the same outline, and each
/// sample on one outline finds the point straight across on the other. Contact pairs, coincident
/// faces, and faces of separate bodies with no region in common (the space between them is
/// unmeshed) have no gap to resolve.
pub(super) fn derive_face_proximity_contributions(
    mut witnesses: Vec<FaceWitness>,
    contact_pairs: &ContactProximityPairs,
    search_radius_m: f64,
    cells_across_gap: u32,
    control: &dyn GeometryEvaluationControl,
) -> Result<Vec<MetricContribution>, SharedCurveError> {
    if !search_radius_m.is_finite() || search_radius_m <= 0.0 {
        return Err(SharedCurveError::invalid_request(
            "surface proximity radius",
            "must be finite and greater than zero",
        ));
    }
    witnesses.retain(|witness| witness.boundaries.iter().any(|boundary| !boundary.is_empty()));
    let face_bounds = |witness: &FaceWitness| {
        let mut minimum = [f64::INFINITY; 3];
        let mut maximum = [f64::NEG_INFINITY; 3];
        for (point, _) in witness.boundaries.iter().flatten() {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(point[axis]);
                maximum[axis] = maximum[axis].max(point[axis]);
            }
        }
        (minimum, maximum)
    };
    witnesses.sort_by(|left, right| {
        face_bounds(left).0[0]
            .total_cmp(&face_bounds(right).0[0])
            .then_with(|| left.face_id.cmp(&right.face_id))
    });
    let mut gaps = vec![f64::INFINITY; witnesses.len()];
    for left_index in 0..witnesses.len() {
        let left = &witnesses[left_index];
        let left_bounds = face_bounds(left);
        for (right_index, right) in witnesses.iter().enumerate().skip(left_index + 1) {
            let right_bounds = face_bounds(right);
            if right_bounds.0[0] - left_bounds.1[0] > search_radius_m {
                break;
            }
            if !left.vertex_ids.is_disjoint(&right.vertex_ids)
                || gap_is_unmeshed_void(&left.solid_ids, &right.solid_ids)
                || bounds_distance(left_bounds, right_bounds) > search_radius_m
                || contact_pairs
                    .faces
                    .contains(&ordered_pair(left.face_id.clone(), right.face_id.clone()))
            {
                continue;
            }
            let tolerance = (left.tolerance_m + right.tolerance_m).max(
                64.0 * f64::EPSILON
                    * left_bounds
                        .0
                        .iter()
                        .chain(&left_bounds.1)
                        .chain(&right_bounds.0)
                        .chain(&right_bounds.1)
                        .fold(0.0_f64, |largest, value| largest.max(value.abs())),
            );
            let mut minimum = f64::INFINITY;
            for (points, polylines) in [(left, right), (right, left)] {
                for (point, point_normal) in points.boundaries.iter().flatten() {
                    control.checkpoint().map_err(|error| {
                        SharedCurveError::new(
                            SharedCurveErrorKind::GeometryEvaluation(error.kind),
                            "surface proximity search",
                            error.reason,
                        )
                    })?;
                    for (start, end) in polylines.boundaries.iter().flat_map(|boundary| {
                        // A one-sample boundary is a point; otherwise consecutive samples.
                        boundary
                            .iter()
                            .zip(boundary.iter().skip(1))
                            .chain((boundary.len() == 1).then(|| (&boundary[0], &boundary[0])))
                    }) {
                        control.consume_search_work(1).map_err(|error| {
                            SharedCurveError::new(
                                SharedCurveErrorKind::GeometryEvaluation(error.kind),
                                "surface proximity search",
                                error.reason,
                            )
                        })?;
                        let (closest, closest_normal) = closest_boundary_point(*point, start, end);
                        let separation = distance(*point, closest);
                        if separation >= minimum || separation >= search_radius_m {
                            continue;
                        }
                        if separation <= tolerance {
                            // Touching faces (separate bodies seated together) have no gap
                            // here; edge proximity reports coincidence defects within one body.
                            continue;
                        }
                        let direction: [f64; 3] = std::array::from_fn(|axis| {
                            (closest[axis] - point[axis]) / separation
                        });
                        let facing = |normal: &[f64; 3]| {
                            (0..3)
                                .map(|axis| normal[axis] * direction[axis])
                                .sum::<f64>()
                                .abs()
                                >= FACING_COSINE
                        };
                        if facing(point_normal) && facing(&closest_normal) {
                            minimum = separation;
                        }
                    }
                }
            }
            if minimum.is_finite() {
                gaps[left_index] = gaps[left_index].min(minimum);
                gaps[right_index] = gaps[right_index].min(minimum);
            }
        }
    }
    let mut contributions = Vec::new();
    for (witness, gap) in witnesses.into_iter().zip(gaps) {
        if !gap.is_finite() {
            continue;
        }
        let metric = MetricTensor3::isotropic_length_m(gap / f64::from(cells_across_gap)).map_err(
            |error| {
                SharedCurveError::invalid_request("surface proximity metric", error.to_string())
            },
        )?;
        contributions.push(MetricContribution {
            source: MetricSourceKind::Proximity,
            scope: MetricContributionScope::Entity {
                entity_id: witness.face_id,
            },
            metric,
        });
    }
    Ok(contributions)
}

/// The point of the boundary piece `start`-`end` closest to `point`, with the surface normal of
/// the nearer sample.
fn closest_boundary_point(
    point: [f64; 3],
    start: &([f64; 3], [f64; 3]),
    end: &([f64; 3], [f64; 3]),
) -> ([f64; 3], [f64; 3]) {
    let along: [f64; 3] = std::array::from_fn(|axis| end.0[axis] - start.0[axis]);
    let length_squared = (0..3).map(|axis| along[axis] * along[axis]).sum::<f64>();
    let fraction = if length_squared > 0.0 {
        ((0..3)
            .map(|axis| (point[axis] - start.0[axis]) * along[axis])
            .sum::<f64>()
            / length_squared)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    (
        std::array::from_fn(|axis| start.0[axis] + fraction * along[axis]),
        if fraction <= 0.5 { start.1 } else { end.1 },
    )
}

/// Separations at or below this are one location: the edges' combined geometric tolerance,
/// and never less than the roundoff of their coordinates.
fn coincidence_tolerance_m(left: &EdgeWitness, right: &EdgeWitness) -> f64 {
    let magnitude = left
        .points_m
        .iter()
        .chain(&right.points_m)
        .flatten()
        .fold(0.0_f64, |largest, coordinate| largest.max(coordinate.abs()));
    (left.tolerance_m + right.tolerance_m).max(64.0 * f64::EPSILON * magnitude)
}

fn bounds(witness: &EdgeWitness) -> ([f64; 3], [f64; 3]) {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for point in &witness.points_m {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    (minimum, maximum)
}

fn bounds_distance(left: ([f64; 3], [f64; 3]), right: ([f64; 3], [f64; 3])) -> f64 {
    (0..3)
        .map(|axis| {
            if left.1[axis] < right.0[axis] {
                right.0[axis] - left.1[axis]
            } else if right.1[axis] < left.0[axis] {
                left.0[axis] - right.1[axis]
            } else {
                0.0
            }
        })
        .map(|distance| distance * distance)
        .sum::<f64>()
        .sqrt()
}

fn witnesses_are_contact(
    left: &EdgeWitness,
    right: &EdgeWitness,
    contact_pairs: &ContactProximityPairs,
) -> bool {
    sets_have_pair(&left.face_ids, &right.face_ids, &contact_pairs.faces)
        || sets_have_pair(
            &left.endpoint_ids,
            &right.endpoint_ids,
            &contact_pairs.vertices,
        )
}

fn sets_have_pair(
    left: &BTreeSet<PersistentEntityId>,
    right: &BTreeSet<PersistentEntityId>,
    pairs: &BTreeSet<(PersistentEntityId, PersistentEntityId)>,
) -> bool {
    left.iter().any(|left| {
        right
            .iter()
            .any(|right| pairs.contains(&ordered_pair(left.clone(), right.clone())))
    })
}

pub(super) fn ordered_pair(
    left: PersistentEntityId,
    right: PersistentEntityId,
) -> (PersistentEntityId, PersistentEntityId) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn geometry_error(
    edge_id: &PersistentEntityId,
    error: runmat_geometry_core::GeometryEvaluationError,
) -> SharedCurveError {
    SharedCurveError::new(
        SharedCurveErrorKind::GeometryEvaluation(error.kind),
        "curve proximity search",
        error.reason,
    )
    .for_edge(edge_id)
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{GeometryEvaluationError, PersistentEntityKind};

    fn face(seed: u8, vertices: &[u8], samples: Vec<([f64; 3], [f64; 3])>) -> FaceWitness {
        let mut witness = FaceWitness::new(id(PersistentEntityKind::Face, seed));
        witness.vertex_ids = vertices
            .iter()
            .map(|vertex| id(PersistentEntityKind::Vertex, *vertex))
            .collect();
        // Each sample its own boundary: the original pointwise witnesses.
        witness.boundaries = samples.into_iter().map(|sample| vec![sample]).collect();
        witness
    }

    #[test]
    fn facing_surfaces_of_a_thin_wall_get_cells_across_the_gap() {
        let grid = |z: f64, normal: [f64; 3]| {
            (0..5)
                .flat_map(|i| (0..5).map(move |j| ([i as f64 * 0.25, j as f64 * 0.25, z], normal)))
                .collect::<Vec<_>>()
        };
        let top = face(1, &[1, 2, 3, 4], grid(0.02, [0.0, 0.0, 1.0]));
        let bottom = face(2, &[5, 6, 7, 8], grid(0.0, [0.0, 0.0, -1.0]));
        // A side face meets the top through a 45 degree blend: oblique, so no gap.
        let blended = face(
            3,
            &[9, 10],
            (0..5)
                .map(|j| ([1.03, j as f64 * 0.25, -0.01], [1.0, 0.0, 0.0]))
                .collect(),
        );
        // A face sharing a vertex with the bottom is adjacent, never a gap.
        let adjacent = face(4, &[5, 11], grid(-0.01, [0.0, 0.0, 1.0]));
        let contributions = derive_face_proximity_contributions(
            vec![top, bottom, blended, adjacent],
            &ContactProximityPairs::default(),
            0.3,
            4,
            &Control,
        )
        .unwrap();
        let sized = contributions
            .iter()
            .map(|contribution| match &contribution.scope {
                MetricContributionScope::Entity { entity_id } => {
                    (entity_id.clone(), contribution.metric)
                }
                other => panic!("unexpected scope {other:?}"),
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let expected = MetricTensor3::isotropic_length_m(0.02 / 4.0).unwrap();
        assert_eq!(
            sized.get(&id(PersistentEntityKind::Face, 1)),
            Some(&expected)
        );
        assert_eq!(
            sized.get(&id(PersistentEntityKind::Face, 2)),
            Some(&expected)
        );
        // The adjacent face faces the top across 0.03 (no shared vertex), so it is a gap.
        let adjacent_size = MetricTensor3::isotropic_length_m(0.03 / 4.0).unwrap();
        assert_eq!(
            sized.get(&id(PersistentEntityKind::Face, 4)),
            Some(&adjacent_size)
        );
        assert!(!sized.contains_key(&id(PersistentEntityKind::Face, 3)));
        assert!(contributions
            .iter()
            .all(|contribution| contribution.source == MetricSourceKind::Proximity));
    }

    /// A 3 mm layer (a thermal interface pad): its top face's outline is sampled at other
    /// points than its bottom face's, and no sample pair lies within the facing cone. Each top
    /// sample finds the bottom outline straight below it, so both faces get the layer thickness
    /// as their size; a face beside the layer and inclined to it does not.
    #[test]
    fn thin_layer_faces_with_unmatched_outline_samples_face_each_other() {
        let outline = |z: f64, xs: &[f64], normal: [f64; 3]| {
            vec![xs.iter().map(|x| ([*x, 0.0, z], normal)).collect::<Vec<_>>()]
        };
        let mut top = face(1, &[1, 2], Vec::new());
        top.boundaries = outline(0.003, &[0.03, 0.04, 0.05], [0.0, 0.0, 1.0]);
        let mut bottom = face(2, &[3, 4], Vec::new());
        bottom.boundaries = outline(0.0, &[-0.1, 0.0, 0.1, 0.2], [0.0, 0.0, -1.0]);
        let mut inclined = face(3, &[5, 6], Vec::new());
        inclined.boundaries = outline(0.006, &[0.3, 0.31], [0.0, 0.6, 0.8]);
        let layer = BTreeSet::from([id(PersistentEntityKind::Solid, 30)]);
        for witness in [&mut top, &mut bottom, &mut inclined] {
            witness.solid_ids = layer.clone();
        }
        let contributions = derive_face_proximity_contributions(
            vec![top, bottom, inclined],
            &ContactProximityPairs::default(),
            0.016,
            1,
            &Control,
        )
        .unwrap();
        let expected = MetricTensor3::isotropic_length_m(0.003).unwrap();
        let sized = contributions
            .iter()
            .map(|contribution| match &contribution.scope {
                MetricContributionScope::Entity { entity_id } => {
                    (entity_id.clone(), contribution.metric)
                }
                other => panic!("unexpected scope {other:?}"),
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(sized.len(), 2);
        for seed in [1, 2] {
            let metric = sized[&id(PersistentEntityKind::Face, seed)];
            for (left, right) in [
                (metric.xx, expected.xx),
                (metric.yy, expected.yy),
                (metric.zz, expected.zz),
            ] {
                assert!((left / right - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn unrequested_cells_across_a_face_gap_are_one_layer() {
        let request = runmat_meshing_core::MetricFieldRequest {
            combination: runmat_meshing_core::MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(0.008).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
            entity_supports: Vec::new(),
            cells_across_gap: None,
        };
        assert_eq!(request.resolved_cells_across_face_gap(), 1);
        assert_eq!(request.resolved_cells_across_gap(), 2);
        let requested = runmat_meshing_core::MetricFieldRequest {
            cells_across_gap: Some(3),
            ..request
        };
        assert_eq!(requested.resolved_cells_across_face_gap(), 3);
    }

    #[test]
    fn requested_cells_across_gap_divide_the_gap_into_that_many_layers() {
        let densest = |cells| {
            let contributions = derive_proximity_contributions(
                vec![witness(1, 0.0), witness(2, 0.2)],
                &ContactProximityPairs::default(),
                MetricTensor3::isotropic_length_m(1.0).unwrap(),
                1.0,
                cells,
                &Control,
            )
            .unwrap();
            let metric = contributions[0].metric;
            metric.xx.max(metric.yy).max(metric.zz)
        };
        // The directional size across the gap is gap / cells: density scales with cells squared.
        let two = densest(2);
        let five = densest(5);
        assert!((two - 1.0 / (0.1 * 0.1)).abs() < 1.0e-9 * two);
        assert!((five - 1.0 / (0.04 * 0.04)).abs() < 1.0e-9 * five);
    }

    #[test]
    fn sweep_emits_canonical_nearby_constraints_and_skips_far_edges() {
        let mut witnesses = vec![witness(3, 5.0), witness(1, 0.0), witness(2, 0.2)];
        witnesses[0].points_m.push([6.0, 0.0, 0.0]);
        let contributions = derive_proximity_contributions(
            witnesses,
            &ContactProximityPairs::default(),
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
            1.0,
            2,
            &Control,
        )
        .unwrap();
        assert_eq!(contributions.len(), 2);
        assert!(contributions
            .iter()
            .all(|contribution| contribution.source == MetricSourceKind::Proximity));
        assert!(contributions[0].scope != contributions[1].scope);
    }

    #[test]
    fn sweep_preserves_typed_search_budget_failure() {
        let error = derive_proximity_contributions(
            vec![witness(1, 0.0), witness(2, 0.2)],
            &ContactProximityPairs::default(),
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
            1.0,
            2,
            &ExhaustedControl,
        )
        .unwrap_err();
        assert_eq!(
            error.kind,
            SharedCurveErrorKind::GeometryEvaluation(
                runmat_geometry_core::GeometryEvaluationErrorKind::SearchWorkBudgetExceeded
            )
        );
    }

    /// Densely sampled curves (rolling-ball fillets) make one edge pair's search quadratic in
    /// its samples. The sweep probes responsiveness per row, so the work between checkpoints is
    /// bounded by one row, not by a whole pair.
    #[test]
    fn dense_pair_search_checkpoints_every_row() {
        let mut left = witness(1, 0.0);
        let mut right = witness(2, 0.2);
        left.points_m = (0..400).map(|i| [0.0, i as f64 * 1.0e-3, 0.0]).collect();
        right.points_m = (0..400).map(|i| [0.2, i as f64 * 1.0e-3, 0.0]).collect();
        let control = RowControl::default();
        derive_proximity_contributions(
            vec![left, right],
            &ContactProximityPairs::default(),
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
            1.0,
            2,
            &control,
        )
        .unwrap();
        let load = |value: &AtomicU64| value.load(Ordering::Relaxed);
        assert_eq!(load(&control.total), 400 * 400);
        assert!(
            load(&control.worst) <= 400,
            "work between checkpoints {}",
            load(&control.worst)
        );
    }

    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Default)]
    struct RowControl {
        since: AtomicU64,
        worst: AtomicU64,
        total: AtomicU64,
    }

    impl GeometryEvaluationControl for RowControl {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            self.since.store(0, Ordering::Relaxed);
            Ok(())
        }
        fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError> {
            let since = self.since.fetch_add(count, Ordering::Relaxed) + count;
            self.total.fetch_add(count, Ordering::Relaxed);
            self.worst.fetch_max(since, Ordering::Relaxed);
            Ok(())
        }
        fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }

    #[test]
    fn coincident_edges_of_separate_bodies_are_contact_and_of_one_body_a_defect() {
        let mut left = witness(1, 0.0);
        let mut right = witness(2, 0.0);
        left.solid_ids.insert(id(PersistentEntityKind::Solid, 21));
        right.solid_ids.insert(id(PersistentEntityKind::Solid, 22));
        let derive = |left: EdgeWitness, right: EdgeWitness| {
            derive_proximity_contributions(
                vec![left, right],
                &ContactProximityPairs::default(),
                MetricTensor3::isotropic_length_m(1.0).unwrap(),
                1.0,
                2,
                &Control,
            )
        };
        // A part seated against another (the speaker cone in the air's recess): no gap.
        assert!(derive(left.clone(), right.clone()).unwrap().is_empty());
        // Separately evaluated coincident edges differ by coordinate roundoff, or by up to
        // their geometric tolerance; neither is a gap that sizes the mesh.
        let mut offset = right.clone();
        offset.points_m = vec![[0.08, 0.0, 9.0e-18]];
        let mut seated = left.clone();
        seated.points_m = vec![[0.08, 0.0, 0.0]];
        assert!(derive(seated.clone(), offset.clone()).unwrap().is_empty());
        offset.points_m = vec![[0.08, 0.0, 1.5e-7]];
        seated.tolerance_m = 1.0e-7;
        offset.tolerance_m = 1.0e-7;
        assert!(derive(seated.clone(), offset.clone()).unwrap().is_empty());
        // A gap beyond the tolerance between separate bodies is unmeshed void: no sizing.
        offset.points_m = vec![[0.08, 0.0, 1.0e-3]];
        assert!(derive(seated, offset).unwrap().is_empty());
        right.solid_ids = left.solid_ids.clone();
        assert!(derive(left, right).is_err());
    }

    #[test]
    fn authored_face_pair_emits_contact_constraints() {
        let mut left = witness(1, 0.0);
        let mut right = witness(2, 0.2);
        let left_face = id(PersistentEntityKind::Face, 11);
        let right_face = id(PersistentEntityKind::Face, 12);
        left.face_ids.insert(left_face.clone());
        right.face_ids.insert(right_face.clone());

        let contributions = derive_proximity_contributions(
            vec![left, right],
            &ContactProximityPairs {
                faces: BTreeSet::from([ordered_pair(left_face, right_face)]),
                vertices: BTreeSet::new(),
            },
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
            1.0,
            2,
            &Control,
        )
        .unwrap();

        assert_eq!(contributions.len(), 2);
        assert!(contributions
            .iter()
            .all(|contribution| contribution.source == MetricSourceKind::Contact));
    }

    #[test]
    fn authored_contact_boundary_vertices_allow_edges_that_only_meet_at_a_corner() {
        let mut left = witness(1, 0.0);
        let mut right = witness(2, 0.0);
        let left_vertex = left.endpoint_ids.iter().next().unwrap().clone();
        let right_vertex = right.endpoint_ids.iter().next().unwrap().clone();
        left.face_ids.insert(id(PersistentEntityKind::Face, 21));
        right.face_ids.insert(id(PersistentEntityKind::Face, 22));

        let contributions = derive_proximity_contributions(
            vec![left, right],
            &ContactProximityPairs {
                faces: BTreeSet::new(),
                vertices: BTreeSet::from([ordered_pair(left_vertex, right_vertex)]),
            },
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
            1.0,
            2,
            &Control,
        )
        .unwrap();

        assert!(contributions.is_empty());
    }

    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Three translated/rotated long-edge pairs retain half-separation resolution across a
    //   thin gap while preserving the long direction at the global baseline. Original contact and
    //   search-control cases remain separate. Sampled witnesses are not a closest-distance certificate
    //   or general local spatial-support qualification.
    #[test]
    fn thin_extrusion_gap_resolution_preserves_long_direction_under_rotation_and_translation() {
        for (long, gap) in [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.6, 0.8, 0.0], [-0.8, 0.6, 0.0]),
            ([0.0, 0.0, 1.0], [0.6, 0.8, 0.0]),
        ] {
            let separation = 0.0003535533905932738;
            let base = MetricTensor3::isotropic_length_m(0.01).unwrap();
            let mut left = witness(1, 0.0);
            let mut right = witness(2, 0.0);
            left.points_m = [0.0, 0.0175, 0.035]
                .map(|distance| {
                    std::array::from_fn(|axis| [2.0, -3.0, 4.0][axis] + distance * long[axis])
                })
                .to_vec();
            right.points_m = left
                .points_m
                .iter()
                .map(|point| std::array::from_fn(|axis| point[axis] + separation * gap[axis]))
                .collect();
            let contributions = derive_proximity_contributions(
                vec![left, right],
                &ContactProximityPairs::default(),
                base,
                0.02,
                2,
                &Control,
            )
            .unwrap();
            assert_eq!(contributions.len(), 2);
            let density = |metric: MetricTensor3, v: [f64; 3]| {
                let [x, y, z] = v;
                metric.xx * x * x
                    + metric.yy * y * y
                    + metric.zz * z * z
                    + 2.0 * (metric.xy * x * y + metric.xz * x * z + metric.yz * y * z)
            };
            for contribution in contributions {
                assert!((density(contribution.metric, long) / base.xx - 1.0).abs() < 1e-10);
                assert!(
                    (density(contribution.metric, gap) * (separation * 0.5).powi(2) - 1.0).abs()
                        < 1e-8
                );
            }
        }
    }

    fn solids(seeds: &[u8]) -> BTreeSet<PersistentEntityId> {
        seeds
            .iter()
            .map(|seed| id(PersistentEntityKind::Solid, *seed))
            .collect()
    }

    /// A box's facing edge and face witnesses across `gap` in z: an edge and a face on the
    /// bottom of the upper box (z = gap, outward normal -z) and on the top of the lower box
    /// (z = 0, outward normal +z), each bounding the given solids.
    fn facing_pair(gap: f64, lower: &[u8], upper: &[u8]) -> (Vec<EdgeWitness>, Vec<FaceWitness>) {
        let line = |z: f64| {
            (0..5)
                .map(|i| [i as f64 * 0.025, 0.0, z])
                .collect::<Vec<_>>()
        };
        let grid = |z: f64, normal: [f64; 3]| {
            (0..5)
                .flat_map(|i| {
                    (0..5).map(move |j| ([i as f64 * 0.025, j as f64 * 0.025, z], normal))
                })
                .collect::<Vec<_>>()
        };
        let mut low_edge = witness(1, 0.0);
        low_edge.points_m = line(0.0);
        low_edge.solid_ids = solids(lower);
        let mut high_edge = witness(2, 0.0);
        high_edge.points_m = line(gap);
        high_edge.solid_ids = solids(upper);
        let mut low_face = face(1, &[1, 3], grid(0.0, [0.0, 0.0, 1.0]));
        low_face.solid_ids = solids(lower);
        let mut high_face = face(2, &[2, 4], grid(gap, [0.0, 0.0, -1.0]));
        high_face.solid_ids = solids(upper);
        (vec![low_edge, high_edge], vec![low_face, high_face])
    }

    fn proximity_sizing(
        (edges, faces): (Vec<EdgeWitness>, Vec<FaceWitness>),
    ) -> Vec<MetricContribution> {
        let mut contributions = derive_proximity_contributions(
            edges,
            &ContactProximityPairs::default(),
            MetricTensor3::isotropic_length_m(0.012).unwrap(),
            0.024,
            2,
            &Control,
        )
        .unwrap();
        contributions.extend(
            derive_face_proximity_contributions(
                faces,
                &ContactProximityPairs::default(),
                0.024,
                2,
                &Control,
            )
            .unwrap(),
        );
        contributions
    }

    /// Two cells 2 mm apart (battery cells with the barrier left out): the void between
    /// them is never meshed, so neither body is refined to gap / cells_across_gap.
    #[test]
    fn disjoint_boxes_a_small_gap_apart_do_not_refine_each_other() {
        assert!(proximity_sizing(facing_pair(0.002, &[21], &[22])).is_empty());
    }

    /// A 2 mm wall of one body: both sides bound the same solid, so the material between
    /// them is resolved with cells_across_gap cells.
    #[test]
    fn one_thin_walled_body_still_refines_across_its_wall() {
        let contributions = proximity_sizing(facing_pair(0.002, &[21], &[21]));
        assert_eq!(contributions.len(), 4);
        let face_size = MetricTensor3::isotropic_length_m(0.001).unwrap();
        assert_eq!(
            contributions
                .iter()
                .filter(|contribution| contribution.metric == face_size)
                .count(),
            2
        );
        let across: [f64; 3] = [0.0, 0.0, 1.0];
        for contribution in &contributions {
            let metric = contribution.metric;
            let density = metric.xx * across[0].powi(2)
                + metric.yy * across[1].powi(2)
                + metric.zz * across[2].powi(2);
            assert!((density * 0.001_f64.powi(2) - 1.0).abs() < 1e-9);
        }
    }

    /// Two bodies bonded through a 2 mm layer of a meshed region (a shared interface on each
    /// side of the layer): the features bound the layer's region, so its thickness refines.
    /// Without the shared region the same pair is void and stays coarse.
    #[test]
    fn bodies_bonded_through_a_shared_meshed_region_still_refine() {
        assert_eq!(
            proximity_sizing(facing_pair(0.002, &[21, 23], &[22, 23])).len(),
            4
        );
        // Sheet geometry bounds no solid and keeps its gaps.
        assert_eq!(proximity_sizing(facing_pair(0.002, &[], &[22])).len(), 4);
    }

    fn witness(seed: u8, x: f64) -> EdgeWitness {
        EdgeWitness {
            edge_id: id(PersistentEntityKind::Edge, seed),
            tolerance_m: 0.0,
            endpoint_ids: BTreeSet::from([id(PersistentEntityKind::Vertex, seed)]),
            face_ids: BTreeSet::new(),
            solid_ids: BTreeSet::new(),
            points_m: vec![[x, 0.0, 0.0]],
        }
    }

    fn id(kind: PersistentEntityKind, seed: u8) -> PersistentEntityId {
        PersistentEntityId::new(kind, format!("entity-{seed}"), Vec::new())
    }

    struct Control;

    impl GeometryEvaluationControl for Control {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }

    struct ExhaustedControl;

    impl GeometryEvaluationControl for ExhaustedControl {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Err(GeometryEvaluationError::new(
                runmat_geometry_core::GeometryEvaluationErrorKind::SearchWorkBudgetExceeded,
                "test search budget exhausted",
            ))
        }
        fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }
}
