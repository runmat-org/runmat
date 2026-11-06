use runmat_geometry_core::{ExactEdge, ExactPcurveEvaluator, ParameterRange};

use crate::shared::{SharedCurveError, SharedCurveErrorKind, SharedCurveFaceUse};

use super::{
    error::{edge_error, geometry_error},
    types::{SharedCurveDiscretizationOptions, SharedCurveEvaluationContext},
    vertex_uv::VertexUvContext,
};

/// Face-local node coordinates of `edge` on each of its coedges. On a non-periodic face each
/// endpoint takes its vertex's canonical coordinate, admitted against the vertex tolerance.
pub(crate) fn face_uses_for_parameters(
    context: SharedCurveEvaluationContext<'_>,
    edge: &ExactEdge,
    parameter_range: ParameterRange,
    parameters: &[f64],
    options: SharedCurveDiscretizationOptions,
) -> Result<Vec<SharedCurveFaceUse>, SharedCurveError> {
    let SharedCurveEvaluationContext {
        topology,
        pcurves,
        control,
        ..
    } = context;
    let vertex_uv = VertexUvContext {
        topology,
        pcurves,
        control,
        geometry_absolute_error_m: options.geometry_absolute_error_m.canonical_value(),
    };
    let mut coedges = topology
        .coedges
        .iter()
        .filter(|coedge| coedge.edge_id == edge.id)
        .collect::<Vec<_>>();
    coedges.sort_by(|left, right| left.id.cmp(&right.id));
    let mut face_uses = Vec::with_capacity(coedges.len());
    for coedge in coedges {
        let pcurve_range = pcurves
            .parameter_range(&coedge.pcurve_evaluator_id)
            .map_err(|error| geometry_error(edge, error))?;
        if pcurve_range != parameter_range {
            return Err(edge_error(
                edge,
                SharedCurveErrorKind::GeometricMismatch,
                "pcurve parameter range",
                "edge and coedge evaluator ranges differ",
            ));
        }
        let mut node_uv = parameters
            .iter()
            .map(|parameter| {
                ExactPcurveEvaluator::point(
                    pcurves,
                    &coedge.pcurve_evaluator_id,
                    *parameter,
                    control,
                )
                .map(|uv| {
                    uv.map(|coordinate| coordinate.with_native_coordinate(|value, _axis| value))
                })
                .map_err(|error| geometry_error(edge, error))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let face = topology
            .faces
            .iter()
            .find(|face| face.id == coedge.face_id)
            .ok_or_else(|| {
                edge_error(
                    edge,
                    SharedCurveErrorKind::InvalidContract,
                    "coedge face",
                    "coedge references an absent exact face",
                )
            })?;
        if edge.start_vertex_id == edge.end_vertex_id
            && !face.periodic_u
            && !face.periodic_v
            && node_uv.len() > 1
        {
            let first = node_uv[0];
            let last = *node_uv.last().expect("node UV inventory is non-empty");
            // With a vertex, the canonical vertex coordinate below admits and joins both ends.
            if edge.start_vertex_id.is_none() {
                vertex_uv.require_within_edge_tolerance(edge, &coedge.face_id, first, last)?;
                *node_uv.last_mut().expect("node UV inventory is non-empty") = first;
            }
        }
        if !face.periodic_u && !face.periodic_v && node_uv.len() > 1 {
            for (index, vertex) in [
                (0, edge.start_vertex_id.as_ref()),
                (node_uv.len() - 1, edge.end_vertex_id.as_ref()),
            ] {
                if let Some(vertex) = vertex {
                    node_uv[index] =
                        vertex_uv.canonical_coordinate(edge, coedge, vertex, node_uv[index])?;
                }
            }
        }
        face_uses.push(SharedCurveFaceUse {
            coedge_id: coedge.id.clone(),
            face_id: coedge.face_id.clone(),
            orientation: coedge.orientation,
            seam_image: coedge.seam_image,
            node_uv,
        });
    }
    Ok(face_uses)
}
