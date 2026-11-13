use super::super::PreparedEvaluation;

use runmat_geometry_core::{
    ExactCurveEvaluator, ExactPcurveEvaluator, ExactSurfaceEvaluator, ExactTrimClassifier,
    PersistentEntityId, TrimDomainLocation,
};
use runmat_meshing_core::{SolverMeshNode, SolverNodeExactParameter};

use super::SurfaceUse;
use crate::cdt::solver_topology::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyOptions, ProjectionInput,
};

pub(in crate::cdt::solver_topology::order_elevation) struct EdgeGeometryRequest<'a> {
    pub left: &'a SolverMeshNode,
    pub right: &'a SolverMeshNode,
    pub exact_edge_id: Option<&'a PersistentEntityId>,
    pub surface_uses: &'a [SurfaceUse<'a>],
    pub fraction: f64,
}

pub(in crate::cdt::solver_topology::order_elevation) struct ExactEdgePoint {
    pub coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    pub exact_parameters: Vec<SolverNodeExactParameter>,
}

pub(super) fn midpoint_geometry(
    input: &ProjectionInput<'_>,
    left: &SolverMeshNode,
    right: &SolverMeshNode,
    exact_edge_id: Option<&PersistentEntityId>,
    surface_uses: &[SurfaceUse],
    evaluation: PreparedEvaluation<'_, '_>,
    options: DelaunaySolverTopologyOptions,
) -> Result<([f64; 3], Vec<SolverNodeExactParameter>), DelaunaySolverTopologyError> {
    edge_geometry(
        input,
        evaluation,
        options,
        EdgeGeometryRequest {
            left,
            right,
            exact_edge_id,
            surface_uses,
            fraction: 0.5,
        },
    )?
    .map(|point| {
        (
            point.coordinates_m.canonical_value(),
            point.exact_parameters,
        )
    })
    .map_or_else(
        || {
            Err(invalid(format!(
                "exact midpoint leaves its source trim domain: {}",
                explain_rejected_midpoint(
                    input,
                    left,
                    right,
                    exact_edge_id,
                    surface_uses,
                    evaluation,
                    options
                )
            )))
        },
        Ok,
    )
}

/// Evidence for a rejected exact midpoint: the owning edge or faces, each candidate's trim
/// classification and its distance from the curve point.
fn explain_rejected_midpoint(
    input: &ProjectionInput<'_>,
    left: &SolverMeshNode,
    right: &SolverMeshNode,
    exact_edge_id: Option<&PersistentEntityId>,
    surface_uses: &[SurfaceUse],
    evaluation: PreparedEvaluation<'_, '_>,
    options: DelaunaySolverTopologyOptions,
) -> String {
    let mut text = format!(
        "edge {:?} -> {:?}",
        left.coordinates_m.canonical_value(),
        right.coordinates_m.canonical_value()
    );
    let classify = |surface_use: &SurfaceUse, uv: [f64; 2]| {
        let face = evaluation.sources.face(surface_use.source_face_id).ok()?;
        let location = ExactTrimClassifier::classify(
            evaluation.evaluator,
            &face.trim_classifier_id,
            uv,
            options.trim_boundary_tolerance_uv,
            evaluation.control,
        )
        .ok()?;
        let point = ExactSurfaceEvaluator::point(
            evaluation.evaluator,
            &face.surface_evaluator_id,
            uv,
            evaluation.control,
        )
        .ok()?;
        let transform = evaluation
            .sources
            .transforms
            .world_transform_for(&face.id)
            .ok()?;
        Some((location, transform.transform_point(point.canonical_value())))
    };
    let curve_point = exact_edge_id.and_then(|edge_id| {
        let (left_parameter, right_parameter) =
            compatible_curve_parameters(left, right, edge_id).ok()?;
        let parameter = 0.5 * (left_parameter + right_parameter);
        let edge = evaluation.sources.edge(edge_id).ok()?;
        let local = ExactCurveEvaluator::point(
            evaluation.evaluator,
            &edge.curve_evaluator_id,
            parameter,
            evaluation.control,
        )
        .ok()?;
        let transform = evaluation
            .sources
            .transforms
            .world_transform_for(edge_id)
            .ok()?;
        Some((
            edge_id,
            parameter,
            transform.transform_point(local.canonical_value()),
        ))
    });
    if let Some((edge_id, parameter, point)) = curve_point {
        text.push_str(&format!(
            "; curve {} at {parameter} -> {point:?}",
            edge_id.source_topology_id
        ));
    }
    let tolerance = input.request.tolerance.source_tolerance_m.canonical_value();
    for surface_use in surface_uses {
        let uvs = match curve_point {
            Some((edge_id, parameter, _)) => {
                pcurve_uvs(edge_id, parameter, surface_use, evaluation)
                    .map(Iterator::collect::<Vec<_>>)
                    .unwrap_or_default()
            }
            None => compatible_surface_parameters(left, right, surface_use, [None, None])
                .map(|(left, right)| vec![interpolate2(left, right, 0.5)])
                .unwrap_or_default(),
        };
        for uv in uvs {
            let classified = classify(surface_use, uv);
            let distance = match (classified, curve_point) {
                (Some((_, point)), Some((_, _, curve))) => (0..3)
                    .map(|axis| (point[axis] - curve[axis]).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                _ => f64::NAN,
            };
            text.push_str(&format!(
                "; face {} uv {uv:?} {:?} curve distance {distance:e} (source tolerance {tolerance:e})",
                surface_use.source_face_id.source_topology_id,
                classified.map(|(location, _)| location)
            ));
        }
    }
    text
}

pub(in crate::cdt::solver_topology::order_elevation) fn edge_geometry(
    input: &ProjectionInput<'_>,
    evaluation: PreparedEvaluation<'_, '_>,
    options: DelaunaySolverTopologyOptions,
    request: EdgeGeometryRequest<'_>,
) -> Result<Option<ExactEdgePoint>, DelaunaySolverTopologyError> {
    if !request.fraction.is_finite() || !(0.0..=1.0).contains(&request.fraction) {
        return Err(invalid("exact edge fraction is outside its source domain"));
    }
    if request.exact_edge_id.is_some() {
        return curve_midpoint(input, evaluation, request);
    }
    if !request.surface_uses.is_empty() {
        return surface_midpoint(input, evaluation, options, request);
    }
    Ok(Some(ExactEdgePoint {
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                interpolate3(
                    request.left.coordinates_m.canonical_value(),
                    request.right.coordinates_m.canonical_value(),
                    request.fraction,
                ),
            ),
        exact_parameters: Vec::new(),
    }))
}

pub(in crate::cdt::solver_topology::order_elevation) fn surface_geometry(
    evaluation: PreparedEvaluation<'_, '_>,
    options: DelaunaySolverTopologyOptions,
    surface_use: &SurfaceUse,
    evaluator_uv: [f64; 2],
) -> Result<Option<ExactEdgePoint>, DelaunaySolverTopologyError> {
    let Some(coordinates_m) =
        evaluate_surface_candidate(surface_use, evaluator_uv, evaluation, options)?
    else {
        return Ok(None);
    };
    let storage = crate::cdt::solver_topology::storage::Storage(Some(evaluation.control));
    let mut exact_parameters = storage.vector(1)?;
    exact_parameters.push(SolverNodeExactParameter::Surface {
        source_face_id: storage.entity(surface_use.source_face_id)?,
        chart_id: surface_use.chart_id,
        evaluator_uv,
    });
    Ok(Some(ExactEdgePoint {
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                coordinates_m,
            ),
        exact_parameters,
    }))
}

fn curve_midpoint(
    input: &ProjectionInput<'_>,
    evaluation: PreparedEvaluation<'_, '_>,
    request: EdgeGeometryRequest<'_>,
) -> Result<Option<ExactEdgePoint>, DelaunaySolverTopologyError> {
    let edge_id = request
        .exact_edge_id
        .ok_or_else(|| invalid("curve point has no exact edge owner"))?;
    let (left_parameter, right_parameter) =
        compatible_curve_parameters(request.left, request.right, edge_id)?;
    let parameter = interpolate(left_parameter, right_parameter, request.fraction);
    let edge = evaluation.sources.edge(edge_id)?;
    evaluation.control.checkpoint().map_err(error::geometry)?;
    let local = ExactCurveEvaluator::point(
        evaluation.evaluator,
        &edge.curve_evaluator_id,
        parameter,
        evaluation.control,
    )
    .map_err(error::geometry)?;
    let transform = evaluation
        .sources
        .transforms
        .world_transform_for(edge_id)
        .map_err(error::geometry)?;
    let coordinates_m = transform.transform_point(local.canonical_value());
    let storage = crate::cdt::solver_topology::storage::Storage(Some(evaluation.control));
    let mut parameters = storage.vector(1)?;
    parameters.push(SolverNodeExactParameter::Curve {
        source_edge_id: storage.entity(edge_id)?,
        parameter,
    });
    // A pcurve image lies on its face's trim boundary by construction, so it is not
    // trim-classified; the surface point must still agree with the curve point.
    for surface_use in request.surface_uses {
        for evaluator_uv in pcurve_uvs(edge_id, parameter, surface_use, evaluation)? {
            let surface_point = evaluate_surface_point(surface_use, evaluator_uv, evaluation)?;
            if !points_match_within(
                coordinates_m,
                surface_point,
                curve_match_tolerance_m(input, edge),
            ) {
                return Ok(None);
            }
            let parameter = SolverNodeExactParameter::Surface {
                source_face_id: storage.entity(surface_use.source_face_id)?,
                chart_id: surface_use.chart_id,
                evaluator_uv,
            };
            storage.push(&mut parameters, parameter)?;
        }
    }
    Ok(Some(ExactEdgePoint {
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                coordinates_m,
            ),
        exact_parameters: parameters,
    }))
}

fn surface_midpoint(
    input: &ProjectionInput<'_>,
    evaluation: PreparedEvaluation<'_, '_>,
    options: DelaunaySolverTopologyOptions,
    request: EdgeGeometryRequest<'_>,
) -> Result<Option<ExactEdgePoint>, DelaunaySolverTopologyError> {
    let mut coordinates = None;
    let storage = crate::cdt::solver_topology::storage::Storage(Some(evaluation.control));
    let mut parameters = storage.vector(request.surface_uses.len())?;
    for surface_use in request.surface_uses {
        let face = evaluation.sources.face(surface_use.source_face_id)?;
        let periods =
            ExactSurfaceEvaluator::periodicity(evaluation.evaluator, &face.surface_evaluator_id)
                .map_err(error::geometry)?;
        let bounds = ExactSurfaceEvaluator::parameter_bounds(
            evaluation.evaluator,
            &face.surface_evaluator_id,
        )
        .map_err(error::geometry)?;
        let (left_uv, right_uv) =
            compatible_surface_parameters(request.left, request.right, surface_use, periods)?;
        let parametric_uv = wrap_into_period(
            interpolate2(left_uv, right_uv, request.fraction),
            periods,
            bounds,
        );
        // On a concave trim the parametric interpolant of two trimmed points can leave the
        // trim; the surface point closest to the chord point is the fallback, if it is inside.
        let candidate =
            match evaluate_surface_candidate(surface_use, parametric_uv, evaluation, options)? {
                Some(point) => Some((parametric_uv, point)),
                None => closest_trimmed_chord_point(
                    input,
                    surface_use,
                    [left_uv, right_uv],
                    request.fraction,
                    evaluation,
                    options,
                )?,
            };
        // A surface edge whose ends both lie on a concave trim boundary (a hole's rim) can
        // leave the trim by up to the chordal deviation the linear surface mesh admitted.
        // The untrimmed surface point within that deviation of the chord point keeps the
        // elevated face as close to the exact face as its linear edge.
        let candidate = match candidate {
            Some(candidate) => Some(candidate),
            None => {
                let point = evaluate_surface_point(surface_use, parametric_uv, evaluation)?;
                let chord = interpolate3(
                    request.left.coordinates_m.canonical_value(),
                    request.right.coordinates_m.canonical_value(),
                    request.fraction,
                );
                let deviation = input
                    .request
                    .quality
                    .surface
                    .maximum_chordal_deviation_m
                    .canonical_value()
                    .max(match_tolerance_m(input));
                let squared = (0..3)
                    .map(|axis| (point[axis] - chord[axis]).powi(2))
                    .sum::<f64>();
                (squared <= deviation * deviation).then_some((parametric_uv, point))
            }
        };
        let Some((evaluator_uv, point)) = candidate else {
            return Ok(None);
        };
        if let Some(existing) = coordinates {
            if !points_match(input, existing, point) {
                return Ok(None);
            }
        } else {
            coordinates = Some(point);
        }
        parameters.push(SolverNodeExactParameter::Surface {
            source_face_id: storage.entity(surface_use.source_face_id)?,
            chart_id: surface_use.chart_id,
            evaluator_uv,
        });
    }
    Ok(Some(ExactEdgePoint {
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                coordinates.ok_or_else(|| invalid("surface midpoint has no exact evaluation"))?,
            ),
        exact_parameters: parameters,
    }))
}

pub(super) fn pcurve_uvs<'a>(
    edge_id: &PersistentEntityId,
    parameter: f64,
    surface_use: &SurfaceUse,
    evaluation: PreparedEvaluation<'_, 'a>,
) -> Result<PcurveImages<'a>, DelaunaySolverTopologyError> {
    let coedges = evaluation.sources.coedge_images(edge_id, surface_use)?;
    let count = coedges.clone().count();
    let requested = count
        .checked_mul(std::mem::size_of::<[f64; 2]>())
        .ok_or_else(|| invalid("pcurve image storage overflow"))?;
    let storage = runmat_geometry_core::GeometryAllocationReservation::new(
        evaluation.control,
        requested as u64,
    )
    .map_err(error::geometry)?;
    let mut images = Vec::with_capacity(count);
    for coedge_id in coedges {
        let coedge = evaluation.sources.coedge(coedge_id)?;
        images.push(
            ExactPcurveEvaluator::point(
                evaluation.evaluator,
                &coedge.pcurve_evaluator_id,
                parameter,
                evaluation.control,
            )
            .map(|uv| uv.map(|coordinate| coordinate.with_native_coordinate(|value, _axis| value)))
            .map_err(error::geometry)?,
        );
    }
    images.sort_unstable_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
    });
    images.dedup_by(|left, right| left.map(f64::to_bits) == right.map(f64::to_bits));
    Ok(PcurveImages {
        values: images.into_iter(),
        _storage: storage,
    })
}

pub(super) struct PcurveImages<'a> {
    values: std::vec::IntoIter<[f64; 2]>,
    _storage: runmat_geometry_core::GeometryAllocationReservation<'a>,
}
impl Iterator for PcurveImages<'_> {
    type Item = [f64; 2];
    fn next(&mut self) -> Option<Self::Item> {
        self.values.next()
    }
}

fn evaluate_surface_candidate(
    surface_use: &SurfaceUse,
    evaluator_uv: [f64; 2],
    evaluation: PreparedEvaluation<'_, '_>,
    options: DelaunaySolverTopologyOptions,
) -> Result<Option<[f64; 3]>, DelaunaySolverTopologyError> {
    if !inside_trim(surface_use, evaluator_uv, evaluation, options)? {
        return Ok(None);
    }
    evaluate_surface_point(surface_use, evaluator_uv, evaluation).map(Some)
}

fn inside_trim(
    surface_use: &SurfaceUse,
    evaluator_uv: [f64; 2],
    evaluation: PreparedEvaluation<'_, '_>,
    options: DelaunaySolverTopologyOptions,
) -> Result<bool, DelaunaySolverTopologyError> {
    let face = evaluation.sources.face(surface_use.source_face_id)?;
    let location = ExactTrimClassifier::classify(
        evaluation.evaluator,
        &face.trim_classifier_id,
        evaluator_uv,
        options.trim_boundary_tolerance_uv,
        evaluation.control,
    )
    .map_err(error::geometry)?;
    Ok(location != TrimDomainLocation::Outside)
}

/// A surface point as its `(u, v)` coordinates and its position in meters.
type TrimmedChordPoint = ([f64; 2], [f64; 3]);

/// The trimmed surface point closest to the point at `fraction` along the chord between two
/// surface coordinates, with its coordinate; `None` when that point lies outside the trim.
fn closest_trimmed_chord_point(
    input: &ProjectionInput<'_>,
    surface_use: &SurfaceUse,
    [left_uv, right_uv]: [[f64; 2]; 2],
    fraction: f64,
    evaluation: PreparedEvaluation<'_, '_>,
    options: DelaunaySolverTopologyOptions,
) -> Result<Option<TrimmedChordPoint>, DelaunaySolverTopologyError> {
    let face = evaluation.sources.face(surface_use.source_face_id)?;
    let local = |uv| {
        evaluation.control.checkpoint().map_err(error::geometry)?;
        ExactSurfaceEvaluator::point(
            evaluation.evaluator,
            &face.surface_evaluator_id,
            uv,
            evaluation.control,
        )
        .map(|point| point.canonical_value())
        .map_err(error::geometry)
    };
    let chord_point = interpolate3(local(left_uv)?, local(right_uv)?, fraction);
    let projection = ExactSurfaceEvaluator::closest_point(
        evaluation.evaluator,
        &face.surface_evaluator_id,
        runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
            chord_point,
        ),
        runmat_types::physical::Length::new(match_tolerance_m(input)),
        evaluation.control,
    )
    .map_err(error::geometry)?;
    let evaluator_uv = projection
        .uv
        .map(|coordinate| coordinate.with_native_coordinate(|value, _axis| value));
    if !inside_trim(surface_use, evaluator_uv, evaluation, options)? {
        return Ok(None);
    }
    Ok(Some((
        evaluator_uv,
        evaluate_surface_point(surface_use, evaluator_uv, evaluation)?,
    )))
}

pub(super) fn evaluate_surface_point(
    surface_use: &SurfaceUse,
    evaluator_uv: [f64; 2],
    evaluation: PreparedEvaluation<'_, '_>,
) -> Result<[f64; 3], DelaunaySolverTopologyError> {
    let face = evaluation.sources.face(surface_use.source_face_id)?;
    let local = ExactSurfaceEvaluator::point(
        evaluation.evaluator,
        &face.surface_evaluator_id,
        evaluator_uv,
        evaluation.control,
    )
    .map_err(error::geometry)?;
    let transform = evaluation
        .sources
        .transforms
        .world_transform_for(&face.id)
        .map_err(error::geometry)?;
    let point = transform.transform_point(local.canonical_value());
    if point.iter().any(|value| !value.is_finite()) {
        return Err(invalid("exact midpoint evaluation is not finite"));
    }
    Ok(point)
}

fn compatible_curve_parameters(
    left: &SolverMeshNode,
    right: &SolverMeshNode,
    edge_id: &PersistentEntityId,
) -> Result<(f64, f64), DelaunaySolverTopologyError> {
    curve_parameters(left, edge_id)
        .flat_map(|left| curve_parameters(right, edge_id).map(move |right| (left, right)))
        .min_by(|(left_a, right_a), (left_b, right_b)| {
            (left_a - right_a)
                .abs()
                .total_cmp(&(left_b - right_b).abs())
                .then_with(|| left_a.total_cmp(left_b))
                .then_with(|| right_a.total_cmp(right_b))
        })
        .ok_or_else(|| invalid("exact curve edge endpoint has no matching parameter"))
}

pub(super) fn curve_parameter(
    node: &SolverMeshNode,
    edge_id: &PersistentEntityId,
) -> Result<f64, DelaunaySolverTopologyError> {
    node.exact_parameters
        .iter()
        .find_map(|parameter| match parameter {
            SolverNodeExactParameter::Curve {
                source_edge_id,
                parameter,
            } if source_edge_id == edge_id => Some(*parameter),
            SolverNodeExactParameter::Curve { .. } | SolverNodeExactParameter::Surface { .. } => {
                None
            }
        })
        .ok_or_else(|| invalid("exact curve edge endpoint has no matching parameter"))
}

/// The nearest pair of the endpoints' chart images. On a periodic axis an image is the same
/// point as its translates by the period, so the right image is taken at the translate nearest
/// the left one: an edge crossing the seam interpolates across it, not around the surface.
fn compatible_surface_parameters(
    left: &SolverMeshNode,
    right: &SolverMeshNode,
    surface_use: &SurfaceUse,
    periods: [Option<f64>; 2],
) -> Result<([f64; 2], [f64; 2]), DelaunaySolverTopologyError> {
    surface_parameters(left, surface_use)
        .flat_map(|left| {
            surface_parameters(right, surface_use)
                .map(move |right| (left, nearest_translate(left, right, periods)))
        })
        .min_by(|(left_a, right_a), (left_b, right_b)| {
            squared_parameter_distance(*left_a, *right_a)
                .total_cmp(&squared_parameter_distance(*left_b, *right_b))
                .then_with(|| left_a[0].total_cmp(&left_b[0]))
                .then_with(|| left_a[1].total_cmp(&left_b[1]))
                .then_with(|| right_a[0].total_cmp(&right_b[0]))
                .then_with(|| right_a[1].total_cmp(&right_b[1]))
        })
        .ok_or_else(|| invalid("surface edge endpoints have no matching chart parameters"))
}

fn curve_parameters<'a>(
    node: &'a SolverMeshNode,
    edge: &'a PersistentEntityId,
) -> impl Iterator<Item = f64> + Clone + 'a {
    node.exact_parameters
        .iter()
        .filter_map(move |parameter| match parameter {
            SolverNodeExactParameter::Curve {
                source_edge_id,
                parameter,
            } if source_edge_id == edge => Some(*parameter),
            _ => None,
        })
}

pub(in crate::cdt::solver_topology::order_elevation) fn surface_parameters<'a>(
    node: &'a SolverMeshNode,
    usage: &'a SurfaceUse<'_>,
) -> impl Iterator<Item = [f64; 2]> + Clone + 'a {
    node.exact_parameters
        .iter()
        .filter_map(move |parameter| match parameter {
            SolverNodeExactParameter::Surface {
                source_face_id,
                chart_id,
                evaluator_uv,
            } if source_face_id == usage.source_face_id && chart_id == &usage.chart_id => {
                Some(*evaluator_uv)
            }
            _ => None,
        })
}

fn squared_parameter_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)
}

/// A curve point and its pcurve image's surface point agree within the request tolerance or
/// the source edge's own tolerance. Import sets that edge tolerance: it rebuilds the pcurves of
/// an edge whose curve/pcurve gap exceeds the written tolerance and admits the remaining gap up
/// to the request's geometric tolerance (its requested deviation). A larger gap is a geometry
/// defect, reported here with the edge, the gap and that bound.
pub(super) fn require_matching_curve_points(
    input: &ProjectionInput<'_>,
    edge: &runmat_geometry_core::ExactEdge,
    curve: [f64; 3],
    surface: [f64; 3],
) -> Result<(), DelaunaySolverTopologyError> {
    let tolerance = curve_match_tolerance_m(input, edge);
    if !points_match_within(curve, surface, tolerance) {
        return Err(invalid(format!(
            "exact curve and surface evaluations disagree beyond tolerance: edge {} curve \
             point {curve:?}, surface point {surface:?}, distance {:e} m, admitted {tolerance:e} m \
             (edge tolerance {:e} m; import admits curve/pcurve gaps up to the request's \
             geometric tolerance {:e} m)",
            edge.id.source_topology_id,
            squared_point_distance(curve, surface).sqrt(),
            edge.tolerance_m.canonical_value(),
            input
                .request
                .tolerance
                .requested_deviation_m
                .canonical_value()
        )));
    }
    Ok(())
}

/// The world point of an exact edge's curve at `parameter`.
pub(super) fn evaluate_curve_point(
    edge_id: &PersistentEntityId,
    parameter: f64,
    evaluation: PreparedEvaluation<'_, '_>,
) -> Result<[f64; 3], DelaunaySolverTopologyError> {
    let edge = evaluation.sources.edge(edge_id)?;
    let local = ExactCurveEvaluator::point(
        evaluation.evaluator,
        &edge.curve_evaluator_id,
        parameter,
        evaluation.control,
    )
    .map_err(error::geometry)?;
    let transform = evaluation
        .sources
        .transforms
        .world_transform_for(edge_id)
        .map_err(error::geometry)?;
    let point = transform.transform_point(local.canonical_value());
    if point.iter().any(|value| !value.is_finite()) {
        return Err(invalid("exact curve evaluation is not finite"));
    }
    Ok(point)
}

/// A linear boundary node lies within the request's chordal deviation of its curve point.
///
/// Surface nodes lie on the curve, but a node the volume stage inserted on a boundary segment
/// (a segment split during facet recovery) lies on the segment's chord, at most the segment's
/// sagitta off the curve. The surface stage bounds that sagitta by the chordal deviation, and the
/// published linear boundary keeps the node where it is.
pub(super) fn require_node_near_curve(
    input: &ProjectionInput<'_>,
    edge: &runmat_geometry_core::ExactEdge,
    parameter: f64,
    node: [f64; 3],
    curve: [f64; 3],
) -> Result<(), DelaunaySolverTopologyError> {
    let tolerance = input
        .request
        .quality
        .surface
        .maximum_chordal_deviation_m
        .canonical_value()
        .max(curve_match_tolerance_m(input, edge));
    if !points_match_within(node, curve, tolerance) {
        return Err(invalid(format!(
            "boundary node lies off its exact curve beyond the chordal deviation: edge {} \
             parameter {parameter}, node {node:?}, curve point {curve:?}, distance {:e} m, \
             admitted {tolerance:e} m",
            edge.id.source_topology_id,
            squared_point_distance(node, curve).sqrt()
        )));
    }
    Ok(())
}

fn curve_match_tolerance_m(
    input: &ProjectionInput<'_>,
    edge: &runmat_geometry_core::ExactEdge,
) -> f64 {
    match_tolerance_m(input).max(edge.tolerance_m.canonical_value())
}

fn match_tolerance_m(input: &ProjectionInput<'_>) -> f64 {
    input
        .request
        .tolerance
        .source_tolerance_m
        .canonical_value()
        .max(input.request.tolerance.absolute_floor_m.canonical_value())
        .max(
            input
                .request
                .tolerance
                .maximum_healing_displacement_m
                .canonical_value(),
        )
}

fn points_match(input: &ProjectionInput<'_>, left: [f64; 3], right: [f64; 3]) -> bool {
    points_match_within(left, right, match_tolerance_m(input))
}

fn points_match_within(left: [f64; 3], right: [f64; 3], tolerance: f64) -> bool {
    let squared_distance = squared_point_distance(left, right);
    squared_distance.is_finite() && squared_distance <= tolerance * tolerance
}

fn squared_point_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>()
}

/// The translate of `image` by whole periods nearest `reference` on each periodic axis: the
/// same surface point, expressed on the reference's side of the seam.
pub(super) fn nearest_translate(
    reference: [f64; 2],
    image: [f64; 2],
    periods: [Option<f64>; 2],
) -> [f64; 2] {
    std::array::from_fn(|axis| match periods[axis] {
        Some(period) if period.is_finite() && period > 0.0 => {
            image[axis] + period * ((reference[axis] - image[axis]) / period).round()
        }
        _ => image[axis],
    })
}

/// A periodic coordinate interpolated across the seam can leave the parameter domain by less
/// than a period; its translate inside the domain is the same surface point.
pub(super) fn wrap_into_period(
    uv: [f64; 2],
    periods: [Option<f64>; 2],
    bounds: [runmat_geometry_core::ParameterRange; 2],
) -> [f64; 2] {
    std::array::from_fn(|axis| match periods[axis] {
        Some(period) if period.is_finite() && period > 0.0 => {
            let start = bounds[axis].start;
            if uv[axis] < start {
                uv[axis] + period
            } else if uv[axis] > start + period {
                uv[axis] - period
            } else {
                uv[axis]
            }
        }
        _ => uv[axis],
    })
}

fn interpolate(left: f64, right: f64, fraction: f64) -> f64 {
    left * (1.0 - fraction) + right * fraction
}

fn interpolate2(left: [f64; 2], right: [f64; 2], fraction: f64) -> [f64; 2] {
    [
        interpolate(left[0], right[0], fraction),
        interpolate(left[1], right[1], fraction),
    ]
}

fn interpolate3(left: [f64; 3], right: [f64; 3], fraction: f64) -> [f64; 3] {
    [
        interpolate(left[0], right[0], fraction),
        interpolate(left[1], right[1], fraction),
        interpolate(left[2], right[2], fraction),
    ]
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidGeometry, reason)
}

#[cfg(test)]
mod tests;
