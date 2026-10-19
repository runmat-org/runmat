//! Deterministic spacing and cavity independence with cooperative inventory walks.
use super::{checkpoint, metric_distance};
use crate::refinement::spatial;
use crate::refinement::{
    ExactFaceRefinementCandidate, ExactFaceRefinementError, ExactFaceRefinementErrorKind,
    ExactFaceRefinementReason,
};
use crate::{ExactFaceGeometry, ExactFacePslg};
use runmat_geometry_core::{GeometryAllocationScope, GeometryEvaluationControl};
use runmat_meshing_core::quality::spatial_index::SpatialEntry;

/// Selects a deterministic maximal set of refinement cavities that cannot touch through a
/// triangle vertex. Inserting circumcenters from adjacent triangles against one stale topology
/// can manufacture arbitrarily short edges even when every individual candidate is valid.
pub(crate) fn independent_candidate_batch(
    candidates: Vec<ExactFaceRefinementCandidate>,
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    minimum_candidate_separation: f64,
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<Vec<ExactFaceRefinementCandidate>, ExactFaceRefinementError> {
    checkpoint(control, &geometry.source_face_id)?;
    let storage = control.map(GeometryAllocationScope::new);
    let admitted = storage
        .as_ref()
        .map(|scope| scope as &dyn GeometryEvaluationControl);
    let nodes = spatial::build(
        pslg.vertices.len(),
        pslg.vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| SpatialEntry {
                bounds: spatial::point(vertex.uv),
                payload: index,
            }),
        admitted,
        &geometry.source_face_id,
    )?;
    let points = spatial::build(
        candidates.len(),
        candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| SpatialEntry {
                bounds: spatial::point(candidate.uv),
                payload: index,
            }),
        admitted,
        &geometry.source_face_id,
    )?;
    if let Some(control) = admitted {
        control
            .consume_allocation_bytes((candidates.len() as u64).saturating_mul(16))
            .map_err(|error| {
                ExactFaceRefinementError::new(
                    ExactFaceRefinementErrorKind::Metric(
                        crate::ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
                    ),
                    &geometry.source_face_id,
                    error.reason,
                )
            })?;
    }
    let mut selected_flags = vec![false; candidates.len()];
    let mut identities = Vec::with_capacity(candidates.len().saturating_mul(3));
    for candidate in &candidates {
        checkpoint(control, &geometry.source_face_id)?;
        identities.extend(candidate.triangle.vertex_indices);
    }
    runmat_cpu_execution::sort_in_place_with_checkpoint(&mut identities, u32::cmp, || {
        checkpoint(control, &geometry.source_face_id)
    })?;
    identities.dedup();
    let mut occupied = vec![false; identities.len()];
    let coordinate_extent = pslg
        .vertices
        .iter()
        .map(|vertex| vertex.uv)
        .chain(candidates.iter().map(|candidate| candidate.uv))
        .flatten()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let mut selected = Vec::<ExactFaceRefinementCandidate>::new();
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        checkpoint(control, &geometry.source_face_id)?;
        let metric = geometry.triangles[candidate.triangle_index as usize].deviation_samples[3]
            .sizing_metric;
        let query = spatial::metric_bounds(
            candidate.uv,
            metric,
            minimum_candidate_separation,
            coordinate_extent,
        );
        let mut nearest_existing_distance = f64::INFINITY;
        nodes.visit_bounds_with_checkpoint(
            query,
            &mut |entry| {
                nearest_existing_distance = nearest_existing_distance.min(metric_distance(
                    candidate.uv,
                    pslg.vertices[entry.payload].uv,
                    metric,
                ));
            },
            &mut || checkpoint(control, &geometry.source_face_id),
        )?;
        if nearest_existing_distance < minimum_candidate_separation {
            if matches!(
                candidate.reason,
                ExactFaceRefinementReason::ChordalDeviation
                    | ExactFaceRefinementReason::NormalDeviation
                    | ExactFaceRefinementReason::MetricEdgeLength
                    | ExactFaceRefinementReason::BoundaryMetricSupport
            ) {
                return Err(ExactFaceRefinementError::new(
                    ExactFaceRefinementErrorKind::UnsatisfiedQuality,
                    &candidate.source_face_id,
                    format!(
                        "surface {:?} target requires candidate {:?} at metric distance {nearest_existing_distance} from the nearest node, below the {minimum_candidate_separation} insertion separation",
                        candidate.reason, candidate.uv
                    ),
                ));
            }
            continue;
        }
        if candidate.triangle.vertex_indices.iter().any(|vertex| {
            occupied[identities
                .binary_search(vertex)
                .expect("candidate vertex inventory")]
        }) {
            continue;
        }
        let mut independent = true;
        points.visit_bounds_with_checkpoint(
            query,
            &mut |entry| {
                if selected_flags[entry.payload]
                    && metric_distance(candidate.uv, candidates[entry.payload].uv, metric)
                        < minimum_candidate_separation
                {
                    independent = false;
                }
            },
            &mut || checkpoint(control, &geometry.source_face_id),
        )?;
        if independent {
            selected_flags[candidate_index] = true;
            for vertex in candidate.triangle.vertex_indices {
                occupied[identities
                    .binary_search(&vertex)
                    .expect("candidate vertex inventory")] = true;
            }
            selected.push(candidate.clone());
        }
    }
    Ok(selected)
}
