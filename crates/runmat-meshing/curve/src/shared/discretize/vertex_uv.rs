//! One nonperiodic face-local coordinate per authoritative source vertex.
//! The smallest incident coedge supplies the representative; independent edge
//! partitions derive it identically. Distinct vertices and periodic images never merge.
//! Incident pcurve ends agree when their surface points lie within the source vertex
//! tolerance (or the geometry error, if larger) of each other: CAD vertices carry a
//! tolerance in model space, and its size in UV depends on the surface parametrization.
use super::error::{edge_error, geometry_error};
use crate::shared::{SharedCurveError, SharedCurveErrorKind};
use runmat_geometry_core::{
    ExactBRepTopology, ExactCoedge, ExactEdge, ExactPcurveEvaluator, ExactSurfaceEvaluator,
    GeometryEvaluationControl, GeometryEvaluationErrorKind, PersistentEntityId,
};

pub(super) struct VertexUvContext<'a> {
    pub topology: &'a ExactBRepTopology,
    pub pcurves: &'a dyn ExactPcurveEvaluator,
    pub surfaces: &'a dyn ExactSurfaceEvaluator,
    pub control: &'a dyn GeometryEvaluationControl,
    /// UV agreement admitted without evaluating the surface.
    pub maximum_error: f64,
    /// Model-space agreement floor; a source vertex tolerance above it governs instead.
    pub geometry_error_m: f64,
}

impl VertexUvContext<'_> {
    pub fn canonical_coordinate(
        &self,
        edge: &ExactEdge,
        coedge: &ExactCoedge,
        vertex: &PersistentEntityId,
        original: [f64; 2],
    ) -> Result<[f64; 2], SharedCurveError> {
        let Self {
            topology,
            pcurves,
            control,
            maximum_error,
            ..
        } = *self;
        let mut representative = None;
        for candidate in &topology.coedges {
            control
                .checkpoint()
                .map_err(|error| geometry_error(edge, error))?;
            control
                .consume_search_work(1)
                .map_err(|error| geometry_error(edge, error))?;
            if candidate.face_id != coedge.face_id {
                continue;
            }
            let candidate_edge = topology
                .edges
                .binary_search_by(|item| item.id.cmp(&candidate.edge_id))
                .ok()
                .map(|index| &topology.edges[index])
                .ok_or_else(|| {
                    edge_error(
                        edge,
                        SharedCurveErrorKind::InvalidContract,
                        "vertex pcurve incidence",
                        "incident coedge references an absent edge",
                    )
                })?;
            let at_start = candidate_edge.start_vertex_id.as_ref() == Some(vertex);
            if !at_start && candidate_edge.end_vertex_id.as_ref() != Some(vertex) {
                continue;
            }
            if representative
                .as_ref()
                .is_none_or(|(previous, _): &(&ExactCoedge, bool)| candidate.id < previous.id)
            {
                representative = Some((candidate, at_start));
            }
        }
        let (representative, at_start) = representative.ok_or_else(|| {
            edge_error(
                edge,
                SharedCurveErrorKind::InvalidContract,
                "vertex pcurve incidence",
                "source vertex has no coedge on its face",
            )
        })?;
        let range = pcurves
            .parameter_range(&representative.pcurve_evaluator_id)
            .map_err(|error| geometry_error(edge, error))?;
        let uv = pcurves
            .point(
                &representative.pcurve_evaluator_id,
                if at_start { range.start } else { range.end },
                control,
            )
            .map(|uv| uv.map(|coordinate| coordinate.with_native_coordinate(|value, _axis| value)))
            .map_err(|error| geometry_error(edge, error))?;
        let difference = (uv[0] - original[0]).hypot(uv[1] - original[1]);
        if difference.is_finite() && difference <= maximum_error {
            return Ok(uv);
        }
        let tolerance_m = topology
            .vertices
            .iter()
            .find(|candidate| &candidate.id == vertex)
            .map(|candidate| candidate.tolerance_m.canonical_value())
            .ok_or_else(|| {
                edge_error(
                    edge,
                    SharedCurveErrorKind::InvalidContract,
                    "vertex pcurve incidence",
                    "edge references an absent source vertex",
                )
            })?;
        let admitted_m = tolerance_m.max(self.geometry_error_m);
        let separation_m = self.model_separation_m(edge, coedge, original, uv)?;
        if !difference.is_finite() || !separation_m.is_finite() || separation_m > admitted_m {
            return Err(edge_error(
                edge,
                SharedCurveErrorKind::GeometricMismatch,
                "vertex pcurve coordinate",
                format!(
                    "incident pcurves disagree beyond the source vertex tolerance at vertex {}: \
                     {original:?} vs {uv:?} from coedge {} (UV difference {difference:e}, \
                     surface points {separation_m:e} m apart, admitted {admitted_m:e} m)",
                    vertex.source_topology_id, representative.id.source_topology_id
                ),
            ));
        }
        Ok(uv)
    }

    /// Distance in model space between the surface points of two coordinates on a coedge's face.
    pub fn model_separation_m(
        &self,
        edge: &ExactEdge,
        coedge: &ExactCoedge,
        first: [f64; 2],
        second: [f64; 2],
    ) -> Result<f64, SharedCurveError> {
        let face = self
            .topology
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
        let point = |uv| {
            self.surfaces
                .point(&face.surface_evaluator_id, uv, self.control)
                .map(|point| Some(point.canonical_value()))
                .or_else(|error| match error.kind {
                    // A pcurve end outside the surface domain has no surface point to agree.
                    GeometryEvaluationErrorKind::ParameterOutsideDomain => Ok(None),
                    _ => Err(geometry_error(edge, error)),
                })
        };
        let (Some(first), Some(second)) = (point(first)?, point(second)?) else {
            return Ok(f64::INFINITY);
        };
        Ok(((first[0] - second[0]).powi(2)
            + (first[1] - second[1]).powi(2)
            + (first[2] - second[2]).powi(2))
        .sqrt())
    }
}

#[cfg(test)]
mod tests;
