use std::collections::BTreeSet;

use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactEdge, ExactPcurveEvaluator, ExactSurfaceEvaluator,
    ExactTrimClassifier, PersistentEntityId, TrimDomainLocation,
};
use runmat_meshing_core::{SolverMeshNode, SolverNodeExactParameter};

use super::SurfaceUse;
use crate::cdt::solver_topology::{
    error, DelaunayExactEvaluation, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyInput, DelaunaySolverTopologyOptions,
};

pub(super) fn midpoint_geometry(
    input: &DelaunaySolverTopologyInput<'_>,
    left: &SolverMeshNode,
    right: &SolverMeshNode,
    exact_edge_id: Option<&PersistentEntityId>,
    surface_uses: &[SurfaceUse],
    evaluation: DelaunayExactEvaluation<'_>,
    options: DelaunaySolverTopologyOptions,
) -> Result<([f64; 3], Vec<SolverNodeExactParameter>), DelaunaySolverTopologyError> {
    if let Some(edge_id) = exact_edge_id {
        return curve_midpoint(
            input,
            left,
            right,
            edge_id,
            surface_uses,
            evaluation,
            options,
        );
    }
    if !surface_uses.is_empty() {
        return surface_midpoint(input, left, right, surface_uses, evaluation, options);
    }
    Ok((
        average3(left.coordinates_m, right.coordinates_m),
        Vec::new(),
    ))
}

fn curve_midpoint(
    input: &DelaunaySolverTopologyInput<'_>,
    left: &SolverMeshNode,
    right: &SolverMeshNode,
    edge_id: &PersistentEntityId,
    surface_uses: &[SurfaceUse],
    evaluation: DelaunayExactEvaluation<'_>,
    options: DelaunaySolverTopologyOptions,
) -> Result<([f64; 3], Vec<SolverNodeExactParameter>), DelaunaySolverTopologyError> {
    let left_parameter = curve_parameter(left, edge_id)?;
    let right_parameter = curve_parameter(right, edge_id)?;
    let parameter = left_parameter * 0.5 + right_parameter * 0.5;
    let edge = exact_edge(input.exact_topology, edge_id)?;
    evaluation.control.checkpoint().map_err(error::geometry)?;
    let local = ExactCurveEvaluator::point(
        evaluation.evaluator,
        &edge.curve_evaluator_id,
        parameter,
        evaluation.control,
    )
    .map_err(error::geometry)?;
    let transform = input
        .exact_topology
        .world_transform_for(edge_id)
        .map_err(|failure| invalid(failure.to_string()))?;
    let coordinates_m = transform.transform_point(local);
    let mut parameters = vec![SolverNodeExactParameter::Curve {
        source_edge_id: edge_id.clone(),
        parameter,
    }];
    for surface_use in surface_uses {
        let evaluator_uv = pcurve_uv(input, edge_id, parameter, surface_use, evaluation)?;
        let surface_point =
            evaluate_surface(input, surface_use, evaluator_uv, evaluation, options)?;
        require_matching_points(input, coordinates_m, surface_point)?;
        parameters.push(SolverNodeExactParameter::Surface {
            source_face_id: surface_use.source_face_id.clone(),
            chart_id: surface_use.chart_id,
            evaluator_uv,
        });
    }
    Ok((coordinates_m, parameters))
}

fn surface_midpoint(
    input: &DelaunaySolverTopologyInput<'_>,
    left: &SolverMeshNode,
    right: &SolverMeshNode,
    surface_uses: &[SurfaceUse],
    evaluation: DelaunayExactEvaluation<'_>,
    options: DelaunaySolverTopologyOptions,
) -> Result<([f64; 3], Vec<SolverNodeExactParameter>), DelaunaySolverTopologyError> {
    let mut coordinates = None;
    let mut parameters = Vec::with_capacity(surface_uses.len());
    for surface_use in surface_uses {
        let left_uv = surface_parameter(left, surface_use)?;
        let right_uv = surface_parameter(right, surface_use)?;
        let evaluator_uv = average2(left_uv, right_uv);
        let point = evaluate_surface(input, surface_use, evaluator_uv, evaluation, options)?;
        if let Some(existing) = coordinates {
            require_matching_points(input, existing, point)?;
        } else {
            coordinates = Some(point);
        }
        parameters.push(SolverNodeExactParameter::Surface {
            source_face_id: surface_use.source_face_id.clone(),
            chart_id: surface_use.chart_id,
            evaluator_uv,
        });
    }
    Ok((
        coordinates.ok_or_else(|| invalid("surface midpoint has no exact evaluation"))?,
        parameters,
    ))
}

pub(super) fn pcurve_uv(
    input: &DelaunaySolverTopologyInput<'_>,
    edge_id: &PersistentEntityId,
    parameter: f64,
    surface_use: &SurfaceUse,
    evaluation: DelaunayExactEvaluation<'_>,
) -> Result<[f64; 2], DelaunaySolverTopologyError> {
    let mut coedge_ids = input
        .exact_surface
        .nodes
        .iter()
        .flat_map(|node| &node.uses)
        .filter(|use_record| {
            use_record.source_face_id == surface_use.source_face_id
                && use_record.chart_id == surface_use.chart_id
        })
        .flat_map(|use_record| &use_record.exact_edge_parameters)
        .filter(|binding| binding.source_edge_id == *edge_id)
        .map(|binding| binding.source_coedge_id.clone())
        .collect::<BTreeSet<_>>();
    if coedge_ids.is_empty() {
        coedge_ids.extend(
            input
                .exact_topology
                .coedges
                .iter()
                .filter(|coedge| {
                    coedge.face_id == surface_use.source_face_id && coedge.edge_id == *edge_id
                })
                .map(|coedge| coedge.id.clone()),
        );
    }
    if coedge_ids.len() != 1 {
        return Err(invalid(
            "exact chart does not identify one coedge image for a curve midpoint",
        ));
    }
    let coedge_id = coedge_ids
        .into_iter()
        .next()
        .ok_or_else(|| invalid("exact chart has no coedge image"))?;
    let coedge = input
        .exact_topology
        .coedges
        .iter()
        .find(|coedge| coedge.id == coedge_id)
        .ok_or_else(|| invalid("exact chart references an absent coedge"))?;
    ExactPcurveEvaluator::point(
        evaluation.evaluator,
        &coedge.pcurve_evaluator_id,
        parameter,
        evaluation.control,
    )
    .map_err(error::geometry)
}

pub(super) fn evaluate_surface(
    input: &DelaunaySolverTopologyInput<'_>,
    surface_use: &SurfaceUse,
    evaluator_uv: [f64; 2],
    evaluation: DelaunayExactEvaluation<'_>,
    options: DelaunaySolverTopologyOptions,
) -> Result<[f64; 3], DelaunaySolverTopologyError> {
    let face = input
        .exact_topology
        .faces
        .iter()
        .find(|face| face.id == surface_use.source_face_id)
        .ok_or_else(|| invalid("boundary surface use references an absent exact face"))?;
    let location = ExactTrimClassifier::classify(
        evaluation.evaluator,
        &face.trim_classifier_id,
        evaluator_uv,
        options.trim_boundary_tolerance_uv,
        evaluation.control,
    )
    .map_err(error::geometry)?;
    if location == TrimDomainLocation::Outside {
        return Err(invalid(
            "elevated surface midpoint leaves the exact trim domain",
        ));
    }
    let local = ExactSurfaceEvaluator::point(
        evaluation.evaluator,
        &face.surface_evaluator_id,
        evaluator_uv,
        evaluation.control,
    )
    .map_err(error::geometry)?;
    let transform = input
        .exact_topology
        .world_transform_for(&face.id)
        .map_err(|failure| invalid(failure.to_string()))?;
    let point = transform.transform_point(local);
    if point.iter().any(|value| !value.is_finite()) {
        return Err(invalid("exact midpoint evaluation is not finite"));
    }
    Ok(point)
}

fn exact_edge<'a>(
    topology: &'a ExactBRepTopology,
    edge_id: &PersistentEntityId,
) -> Result<&'a ExactEdge, DelaunaySolverTopologyError> {
    topology
        .edges
        .iter()
        .find(|edge| edge.id == *edge_id)
        .ok_or_else(|| invalid("boundary edge provenance references an absent exact edge"))
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

fn surface_parameter(
    node: &SolverMeshNode,
    surface_use: &SurfaceUse,
) -> Result<[f64; 2], DelaunaySolverTopologyError> {
    node.exact_parameters
        .iter()
        .find_map(|parameter| match parameter {
            SolverNodeExactParameter::Surface {
                source_face_id,
                chart_id,
                evaluator_uv,
            } if source_face_id == &surface_use.source_face_id
                && chart_id == &surface_use.chart_id =>
            {
                Some(*evaluator_uv)
            }
            SolverNodeExactParameter::Curve { .. } | SolverNodeExactParameter::Surface { .. } => {
                None
            }
        })
        .ok_or_else(|| invalid("surface edge endpoint has no matching exact chart parameter"))
}

pub(super) fn require_matching_points(
    input: &DelaunaySolverTopologyInput<'_>,
    left: [f64; 3],
    right: [f64; 3],
) -> Result<(), DelaunaySolverTopologyError> {
    let tolerance = input
        .request
        .tolerance
        .source_tolerance_m
        .max(input.request.tolerance.absolute_floor_m)
        .max(input.request.tolerance.maximum_healing_displacement_m);
    let squared_distance = left
        .into_iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>();
    if !squared_distance.is_finite() || squared_distance > tolerance * tolerance {
        return Err(invalid(
            "exact curve and surface midpoint evaluations disagree beyond tolerance",
        ));
    }
    Ok(())
}

fn average2(left: [f64; 2], right: [f64; 2]) -> [f64; 2] {
    [
        left[0] * 0.5 + right[0] * 0.5,
        left[1] * 0.5 + right[1] * 0.5,
    ]
}

fn average3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[0] * 0.5 + right[0] * 0.5,
        left[1] * 0.5 + right[1] * 0.5,
        left[2] * 0.5 + right[2] * 0.5,
    ]
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidGeometry, reason)
}
