//! One canonical boundary-edge inventory per selection pass, independent of triangle count.
use crate::{ExactFaceGeometry, ExactFacePslg};
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};

use super::{checkpoint, ExactFaceRefinementError, ExactFaceRefinementErrorKind};

pub(super) struct ProtectedEdges<'a> {
    edges: Vec<[u32; 2]>,
    _storage: Option<GeometryAllocationReservation<'a>>,
}

impl<'a> ProtectedEdges<'a> {
    pub(super) fn prepare(
        pslg: &ExactFacePslg,
        geometry: &ExactFaceGeometry,
        control: Option<&'a dyn GeometryEvaluationControl>,
    ) -> Result<Self, ExactFaceRefinementError> {
        let storage = if let Some(control) = control {
            Some(
                GeometryAllocationReservation::new(
                    control,
                    (pslg.segments.len() as u64)
                        .saturating_mul(std::mem::size_of::<[u32; 2]>() as u64),
                )
                .map_err(|failure| {
                    ExactFaceRefinementError::new(
                        ExactFaceRefinementErrorKind::Metric(
                            crate::ExactFaceMetricErrorKind::GeometryEvaluation(failure.kind),
                        ),
                        &geometry.source_face_id,
                        failure.reason,
                    )
                })?,
            )
        } else {
            None
        };
        let mut edges = Vec::with_capacity(pslg.segments.len());
        for segment in &pslg.segments {
            checkpoint(control, &geometry.source_face_id)?;
            let mut edge = segment.vertex_indices;
            edge.sort_unstable();
            edges.push(edge);
        }
        edges.sort_unstable();
        edges.dedup();
        checkpoint(control, &geometry.source_face_id)?;
        Ok(Self {
            edges,
            _storage: storage,
        })
    }

    pub(super) fn contains(&self, mut edge: [u32; 2]) -> bool {
        edge.sort_unstable();
        self.edges.binary_search(&edge).is_ok()
    }
}
