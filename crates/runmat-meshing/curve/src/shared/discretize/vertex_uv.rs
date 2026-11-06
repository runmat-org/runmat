//! One nonperiodic face-local coordinate per authoritative source vertex.
//! The smallest incident coedge supplies the representative; independent edge
//! partitions derive it identically. Distinct vertices and periodic images never merge.
//!
//! Incident pcurves meet a vertex within that vertex's 3D tolerance, which a CAD kernel
//! records with the vertex (OCCT widens it after booleans). Their endpoint UVs may therefore
//! differ by that tolerance mapped through the face's parametric scale at the vertex. Each
//! endpoint is admitted when its image on the face lies within the vertex tolerance of the
//! vertex point, plus the numerical geometry error the request admits.
use super::error::{edge_error, geometry_error};
use super::types::ExactFacePcurveEvaluator;
use crate::shared::{SharedCurveError, SharedCurveErrorKind};
use runmat_geometry_core::{
    ExactBRepTopology, ExactCoedge, ExactEdge, ExactPcurveEvaluator, ExactSurfaceEvaluator,
    GeometryEvaluationControl, PersistentEntityId,
};

pub(super) struct VertexUvContext<'a> {
    pub topology: &'a ExactBRepTopology,
    pub pcurves: &'a dyn ExactFacePcurveEvaluator,
    pub control: &'a dyn GeometryEvaluationControl,
    /// Numerical geometry error the request admits, in meters, added to the vertex tolerance.
    pub geometry_absolute_error_m: f64,
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
        let uv = ExactPcurveEvaluator::point(
            pcurves,
            &representative.pcurve_evaluator_id,
            if at_start { range.start } else { range.end },
            control,
        )
        .map(|uv| uv.map(|coordinate| coordinate.with_native_coordinate(|value, _axis| value)))
        .map_err(|error| geometry_error(edge, error))?;
        for endpoint in [original, uv] {
            self.require_within_vertex_tolerance(edge, &coedge.face_id, vertex, endpoint)?;
        }
        Ok(uv)
    }

    /// Admits a face-local coordinate at `vertex` when its image on the face lies within the
    /// vertex's 3D tolerance of the vertex point.
    pub fn require_within_vertex_tolerance(
        &self,
        edge: &ExactEdge,
        face_id: &PersistentEntityId,
        vertex_id: &PersistentEntityId,
        uv: [f64; 2],
    ) -> Result<(), SharedCurveError> {
        let vertex = self
            .topology
            .vertices
            .binary_search_by(|item| item.id.cmp(vertex_id))
            .ok()
            .map(|index| &self.topology.vertices[index])
            .ok_or_else(|| {
                edge_error(
                    edge,
                    SharedCurveErrorKind::InvalidContract,
                    "vertex pcurve incidence",
                    "coedge endpoint vertex is absent from admitted topology",
                )
            })?;
        let image = self.face_image(edge, face_id, uv)?;
        let point = vertex.point_m.canonical_value();
        let distance_m = distance(image, point);
        let admitted_m = vertex.tolerance_m.canonical_value() + self.geometry_absolute_error_m;
        if !distance_m.is_finite() || distance_m > admitted_m {
            return Err(edge_error(
                edge,
                SharedCurveErrorKind::GeometricMismatch,
                "vertex pcurve coordinate",
                format!(
                    "incident pcurve endpoint lies {distance_m:.3e} m from its vertex on the face, \
                     beyond the vertex tolerance {admitted_m:.3e} m"
                ),
            ));
        }
        Ok(())
    }

    /// Admits two face-local coordinates of one closed vertex-free edge when their images on
    /// the face lie within the edge's 3D tolerance of each other.
    pub fn require_within_edge_tolerance(
        &self,
        edge: &ExactEdge,
        face_id: &PersistentEntityId,
        first: [f64; 2],
        last: [f64; 2],
    ) -> Result<(), SharedCurveError> {
        let distance_m = distance(
            self.face_image(edge, face_id, first)?,
            self.face_image(edge, face_id, last)?,
        );
        let admitted_m = edge.tolerance_m.canonical_value() + self.geometry_absolute_error_m;
        if !distance_m.is_finite() || distance_m > admitted_m {
            return Err(edge_error(
                edge,
                SharedCurveErrorKind::GeometricMismatch,
                "closed pcurve endpoint",
                format!(
                    "closed edge endpoints lie {distance_m:.3e} m apart on a non-periodic face, \
                     beyond the edge tolerance {admitted_m:.3e} m"
                ),
            ));
        }
        Ok(())
    }

    fn face_image(
        &self,
        edge: &ExactEdge,
        face_id: &PersistentEntityId,
        uv: [f64; 2],
    ) -> Result<[f64; 3], SharedCurveError> {
        let face = self
            .topology
            .faces
            .iter()
            .find(|face| &face.id == face_id)
            .ok_or_else(|| {
                edge_error(
                    edge,
                    SharedCurveErrorKind::InvalidContract,
                    "coedge face",
                    "coedge references an absent exact face",
                )
            })?;
        ExactSurfaceEvaluator::point(self.pcurves, &face.surface_evaluator_id, uv, self.control)
            .map(|point| point.canonical_value())
            .map_err(|error| geometry_error(edge, error))
    }
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2))
        .sqrt()
}

#[cfg(test)]
mod tests;
