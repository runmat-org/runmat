use runmat_geometry_core::{
    ExactBRepTopology, ExactEdge, ExactPcurveEvaluator, GeometryEvaluationControl, ParameterRange,
};

use crate::shared::{SharedCurveError, SharedCurveErrorKind, SharedCurveFaceUse};

use super::error::{edge_error, geometry_error};

pub(super) fn face_uses_for_parameters(
    topology: &ExactBRepTopology,
    edge: &ExactEdge,
    pcurves: &dyn ExactPcurveEvaluator,
    control: &dyn GeometryEvaluationControl,
    parameter_range: ParameterRange,
    parameters: &[f64],
    pcurve_absolute_error: f64,
) -> Result<Vec<SharedCurveFaceUse>, SharedCurveError> {
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
                pcurves
                    .point(&coedge.pcurve_evaluator_id, *parameter, control)
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
            if ((first[0] - last[0]).powi(2) + (first[1] - last[1]).powi(2)).sqrt()
                > pcurve_absolute_error
            {
                return Err(edge_error(
                    edge,
                    SharedCurveErrorKind::GeometricMismatch,
                    "closed pcurve endpoint",
                    "closed edge endpoints differ on a non-periodic face",
                ));
            }
            *node_uv.last_mut().expect("node UV inventory is non-empty") = first;
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
