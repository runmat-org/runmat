//! One admitted node index per selection pass, reused for every witness and relocation.
use super::{checkpoint, metric_distance};
use crate::refinement::{spatial, ExactFaceRefinementError};
use crate::{ExactFacePslg, ParametricMetricTensor};
use runmat_geometry_core::GeometryEvaluationControl;
use runmat_meshing_core::quality::spatial_index::{PackedBoundsSpatialIndex, SpatialEntry};

pub(super) struct VertexQueries<'a> {
    pslg: &'a ExactFacePslg,
    index: PackedBoundsSpatialIndex<usize>,
    extent: f64,
}
impl<'a> VertexQueries<'a> {
    pub(super) fn prepare(
        pslg: &'a ExactFacePslg,
        control: Option<&dyn GeometryEvaluationControl>,
    ) -> Result<Self, ExactFaceRefinementError> {
        let index = spatial::build(
            pslg.vertices.len(),
            pslg.vertices
                .iter()
                .enumerate()
                .map(|(payload, vertex)| SpatialEntry {
                    bounds: spatial::point(vertex.uv),
                    payload,
                }),
            control,
            &pslg.source_face_id,
        )?;
        let extent = pslg
            .vertices
            .iter()
            .flat_map(|vertex| vertex.uv)
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        Ok(Self {
            pslg,
            index,
            extent,
        })
    }
    pub(super) fn contains(
        &self,
        uv: [f64; 2],
        control: Option<&dyn GeometryEvaluationControl>,
    ) -> Result<bool, ExactFaceRefinementError> {
        let mut found = false;
        self.index.visit_bounds_with_checkpoint(
            spatial::point(uv),
            &mut |entry| {
                found |= self.pslg.vertices[entry.payload].uv == uv;
            },
            &mut || checkpoint(control, &self.pslg.source_face_id),
        )?;
        Ok(found)
    }
    pub(super) fn nearest_within(
        &self,
        uv: [f64; 2],
        metric: ParametricMetricTensor,
        distance: f64,
        control: Option<&dyn GeometryEvaluationControl>,
    ) -> Result<f64, ExactFaceRefinementError> {
        let extent = self.extent.max(uv[0].abs()).max(uv[1].abs());
        let mut nearest = f64::INFINITY;
        self.index.visit_bounds_with_checkpoint(
            spatial::metric_bounds(uv, metric, distance, extent),
            &mut |entry| {
                nearest = nearest.min(metric_distance(
                    uv,
                    self.pslg.vertices[entry.payload].uv,
                    metric,
                ));
            },
            &mut || checkpoint(control, &self.pslg.source_face_id),
        )?;
        Ok(nearest)
    }
}
