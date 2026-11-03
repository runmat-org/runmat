//! One nonperiodic face-local coordinate per authoritative source vertex.
//! The smallest incident coedge supplies the representative; independent edge
//! partitions derive it identically. Distinct vertices and periodic images never merge.
use super::error::{edge_error, geometry_error};
use crate::shared::{SharedCurveError, SharedCurveErrorKind};
use runmat_geometry_core::{
    ExactBRepTopology, ExactCoedge, ExactEdge, ExactPcurveEvaluator, GeometryEvaluationControl,
    PersistentEntityId,
};

pub(super) struct VertexUvContext<'a> {
    pub topology: &'a ExactBRepTopology,
    pub pcurves: &'a dyn ExactPcurveEvaluator,
    pub control: &'a dyn GeometryEvaluationControl,
    pub maximum_error: f64,
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
        if !difference.is_finite() || difference > maximum_error {
            return Err(edge_error(
                edge,
                SharedCurveErrorKind::GeometricMismatch,
                "vertex pcurve coordinate",
                format!(
                    "incident pcurves disagree beyond the admitted UV error at vertex {}: \
                     {original:?} vs {uv:?} from coedge {} (difference {difference:e}, \
                     admitted {maximum_error:e})",
                    vertex.source_topology_id, representative.id.source_topology_id
                ),
            ));
        }
        Ok(uv)
    }
}

#[cfg(test)]
mod tests;
