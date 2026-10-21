//! Bounded, cooperative construction of a separated point in one triangle.
use super::{checkpoint, metric_distance};
use crate::refinement::ExactFaceRefinementError;
use crate::{ExactFaceGeometry, ExactFacePslg, ParametricMetricTensor};
use runmat_geometry_core::GeometryEvaluationControl;

/// Preserve the original witness when feasible, then move toward its centroid.
/// A fixed barycentric inventory provides a bounded fallback. Boundary classification
/// and independent geometric/quality acceptance remain mandatory after relocation.
pub(crate) fn separate_refinement_candidate(
    witness: [f64; 2],
    triangle_index: usize,
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    minimum_separation: f64,
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<Option<[f64; 2]>, ExactFaceRefinementError> {
    let storage = control.map(runmat_geometry_core::GeometryAllocationScope::new);
    let admitted = storage
        .as_ref()
        .map(|scope| scope as &dyn GeometryEvaluationControl);
    let vertices = super::vertices::VertexQueries::prepare(pslg, admitted)?;
    separate_with_queries(
        witness,
        triangle_index,
        geometry,
        &vertices,
        minimum_separation,
        control,
    )
}

pub(super) fn separate_with_queries(
    witness: [f64; 2],
    triangle_index: usize,
    geometry: &ExactFaceGeometry,
    vertices: &super::vertices::VertexQueries<'_>,
    minimum_separation: f64,
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<Option<[f64; 2]>, ExactFaceRefinementError> {
    let Some(triangle) = geometry.triangles.get(triangle_index) else {
        return Ok(None);
    };
    let metric = triangle.deviation_samples[3].sizing_metric;
    separate_in_metric(
        witness,
        triangle_index,
        geometry,
        vertices,
        minimum_separation,
        control,
        &|_| Ok(metric),
    )
}

/// As `separate_with_queries`, measuring each trial point in the sizing metric that
/// `metric_at` reports at that point.
pub(super) fn separate_in_metric(
    witness: [f64; 2],
    triangle_index: usize,
    geometry: &ExactFaceGeometry,
    vertices: &super::vertices::VertexQueries<'_>,
    minimum_separation: f64,
    control: Option<&dyn GeometryEvaluationControl>,
    metric_at: &dyn Fn([f64; 2]) -> Result<ParametricMetricTensor, ExactFaceRefinementError>,
) -> Result<Option<[f64; 2]>, ExactFaceRefinementError> {
    checkpoint(control, &geometry.source_face_id)?;
    let Some(triangle) = geometry.triangles.get(triangle_index) else {
        return Ok(None);
    };
    let centroid = triangle.deviation_samples[3].uv;
    let metric = metric_at(witness)?;
    for toward_centroid in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let candidate = std::array::from_fn(|axis| {
            witness[axis] * (1.0 - toward_centroid) + centroid[axis] * toward_centroid
        });
        if separated(
            candidate,
            vertices,
            geometry,
            metric_at(candidate)?,
            minimum_separation,
            control,
        )? {
            return Ok(Some(candidate));
        }
    }
    let corners = triangle
        .triangle
        .vertex_indices
        .map(|index| geometry.vertices[index as usize].evaluation.uv);
    let mut best = None::<[f64; 2]>;
    const DENOMINATOR: u32 = 8;
    for first in 0..=DENOMINATOR {
        for second in 0..=(DENOMINATOR - first) {
            let third = DENOMINATOR - first - second;
            if [first, second, third].contains(&DENOMINATOR) {
                continue;
            }
            let weights = [first, second, third].map(|value| value as f64 / DENOMINATOR as f64);
            let candidate = std::array::from_fn(|axis| {
                corners[0][axis] * weights[0]
                    + corners[1][axis] * weights[1]
                    + corners[2][axis] * weights[2]
            });
            if separated(
                candidate,
                vertices,
                geometry,
                metric_at(candidate)?,
                minimum_separation,
                control,
            )? && best.is_none_or(|current| {
                metric_distance(candidate, witness, metric)
                    .total_cmp(&metric_distance(current, witness, metric))
                    .then_with(|| candidate[0].total_cmp(&current[0]))
                    .then_with(|| candidate[1].total_cmp(&current[1]))
                    .is_lt()
            }) {
                best = Some(candidate);
            }
        }
    }
    Ok(best)
}

fn separated(
    candidate: [f64; 2],
    vertices: &super::vertices::VertexQueries<'_>,
    geometry: &ExactFaceGeometry,
    metric: ParametricMetricTensor,
    minimum_separation: f64,
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<bool, ExactFaceRefinementError> {
    checkpoint(control, &geometry.source_face_id)?;
    if vertices.contains(candidate, control)? {
        return Ok(false);
    }
    Ok(
        vertices.nearest_within(candidate, metric, minimum_separation, control)? + f64::EPSILON
            >= minimum_separation,
    )
}
