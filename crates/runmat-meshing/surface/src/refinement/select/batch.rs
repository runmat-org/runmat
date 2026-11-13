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

type CandidateMetric<'a> = dyn Fn(
        &ExactFaceRefinementCandidate,
        [f64; 2],
    ) -> Result<crate::ParametricMetricTensor, ExactFaceRefinementError>
    + 'a;

/// Selects a deterministic maximal set of refinement cavities that cannot touch through a
/// triangle vertex. Inserting circumcenters from adjacent triangles against one stale topology
/// can manufacture arbitrarily short edges even when every individual candidate is valid.
#[cfg(test)]
pub(crate) fn independent_candidate_batch(
    candidates: Vec<ExactFaceRefinementCandidate>,
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    minimum_candidate_separation: f64,
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<Vec<ExactFaceRefinementCandidate>, ExactFaceRefinementError> {
    independent_candidate_batch_in_metric(
        candidates,
        geometry,
        pslg,
        minimum_candidate_separation,
        control,
        &|candidate, _| {
            Ok(
                geometry.triangles[candidate.triangle_index as usize].deviation_samples[3]
                    .sizing_metric,
            )
        },
    )
}

/// As `independent_candidate_batch`, measuring separation in the sizing metric that
/// `metric_at` reports for each candidate. Refinement evaluates the face field at the
/// candidate itself: sizing graded from edges can vary within one triangle.
pub(crate) fn independent_candidate_batch_in_metric(
    candidates: Vec<ExactFaceRefinementCandidate>,
    geometry: &ExactFaceGeometry,
    pslg: &ExactFacePslg,
    minimum_candidate_separation: f64,
    control: Option<&dyn GeometryEvaluationControl>,
    metric_at: &CandidateMetric<'_>,
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
    let mut vertex_queries = None;
    let mut moved_selected = Vec::<[f64; 2]>::new();
    let mut relocated = 0u32;
    let mut dropped = 0u32;
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        checkpoint(control, &geometry.source_face_id)?;
        let mut candidate = candidate.clone();
        let mut metric = metric_at(&candidate, candidate.uv)?;
        let nearest = |uv: [f64; 2], metric| -> Result<f64, ExactFaceRefinementError> {
            let mut nearest_existing_distance = f64::INFINITY;
            nodes.visit_bounds_with_checkpoint(
                spatial::metric_bounds(uv, metric, minimum_candidate_separation, coordinate_extent),
                &mut |entry| {
                    nearest_existing_distance = nearest_existing_distance.min(metric_distance(
                        uv,
                        pslg.vertices[entry.payload].uv,
                        metric,
                    ));
                },
                &mut || checkpoint(control, &geometry.source_face_id),
            )?;
            Ok(nearest_existing_distance)
        };
        let nearest_existing_distance = nearest(candidate.uv, metric)?;
        if nearest_existing_distance < minimum_candidate_separation {
            match candidate.reason {
                // Geometric fidelity targets have no admissible substitute point.
                ExactFaceRefinementReason::ChordalDeviation
                | ExactFaceRefinementReason::NormalDeviation => {
                    return Err(ExactFaceRefinementError::new(
                        ExactFaceRefinementErrorKind::UnsatisfiedQuality,
                        &candidate.source_face_id,
                        format!(
                            "surface {:?} target requires candidate {:?} at metric distance {nearest_existing_distance} from the nearest node, below the {minimum_candidate_separation} insertion separation",
                            candidate.reason, candidate.uv
                        ),
                    ));
                }
                // Sizing and support targets accept any separated point in their triangle,
                // measured in the sizing at that point. When none exists, the existing node
                // already resolves the target at this spacing: the candidate is dropped for
                // this iteration, as spacing-suppressed shape candidates are.
                ExactFaceRefinementReason::MetricEdgeLength
                | ExactFaceRefinementReason::BoundaryMetricSupport => {
                    if vertex_queries.is_none() {
                        vertex_queries =
                            Some(super::vertices::VertexQueries::prepare(pslg, admitted)?);
                    }
                    let queries = vertex_queries.as_ref().expect("prepared above");
                    let source = candidate.clone();
                    let Some(uv) = super::separation::separate_in_metric(
                        candidate.uv,
                        candidate.triangle_index as usize,
                        geometry,
                        queries,
                        minimum_candidate_separation,
                        control,
                        &|uv| metric_at(&source, uv),
                    )?
                    else {
                        dropped += 1;
                        continue;
                    };
                    candidate.uv = uv;
                    metric = metric_at(&candidate, uv)?;
                    if nearest(uv, metric)? < minimum_candidate_separation {
                        dropped += 1;
                        continue;
                    }
                    relocated += 1;
                }
                _ => continue,
            }
        }
        let query = spatial::metric_bounds(
            candidate.uv,
            metric,
            minimum_candidate_separation,
            coordinate_extent,
        );
        if candidate.triangle.vertex_indices.iter().any(|vertex| {
            occupied[identities
                .binary_search(vertex)
                .expect("candidate vertex inventory")]
        }) {
            continue;
        }
        let moved = candidate.uv != candidates[candidate_index].uv;
        // Relocated selections are not at their indexed position; compare them directly.
        let mut independent = moved_selected
            .iter()
            .all(|uv| metric_distance(candidate.uv, *uv, metric) >= minimum_candidate_separation);
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
            if moved {
                moved_selected.push(candidate.uv);
            } else {
                selected_flags[candidate_index] = true;
            }
            for vertex in candidate.triangle.vertex_indices {
                occupied[identities
                    .binary_search(&vertex)
                    .expect("candidate vertex inventory")] = true;
            }
            selected.push(candidate);
        }
    }
    if relocated > 0 || dropped > 0 {
        tracing::debug!(
            event_code = "meshing.surface_refinement.separation",
            face = ?geometry.source_face_id,
            relocated,
            dropped,
            "near-node refinement candidates resolved"
        );
    }
    Ok(selected)
}
