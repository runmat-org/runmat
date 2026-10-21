//! Validate candidate membership cooperatively before rebuilding the chart.
use super::{delaunay_error, invalid};
use crate::refinement::{ExactFaceRefinementCandidate, ExactFaceRefinementError};
use crate::{ExactFaceDelaunayErrorKind, ExactFacePslg, ExactFaceTrimmedDelaunay};
use runmat_meshing_core::{predicate::orient2d, MeshingCancellationSignal, PredicateSign};

pub(super) fn validate_candidate(
    pslg: &ExactFacePslg,
    trimmed: &ExactFaceTrimmedDelaunay,
    candidate: &ExactFaceRefinementCandidate,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), ExactFaceRefinementError> {
    if candidate.source_face_id != pslg.source_face_id
        || candidate.triangle_index as usize >= trimmed.triangles.len()
        || trimmed.triangles[candidate.triangle_index as usize] != candidate.triangle
        || candidate
            .uv
            .iter()
            .any(|coordinate| !coordinate.is_finite())
    {
        return Err(invalid(
            pslg,
            "refinement candidate does not identify current face topology",
        ));
    }
    for vertex in &pslg.vertices {
        checkpoint(pslg, cancellation)?;
        if vertex.uv == candidate.uv {
            return Err(invalid(
                pslg,
                "refinement candidate duplicates an existing vertex",
            ));
        }
    }
    for segment in &pslg.segments {
        checkpoint(pslg, cancellation)?;
        let endpoints = segment
            .vertex_indices
            .map(|index| pslg.vertices[index as usize].uv);
        let sign = orient2d([endpoints[0], endpoints[1], candidate.uv])
            .map_err(|_| invalid(pslg, "refinement candidate has invalid predicate input"))?;
        if sign == PredicateSign::Zero && within_segment_bounds(candidate.uv, endpoints) {
            return Err(invalid(
                pslg,
                &format!(
                    "{:?} refinement candidate {:?} lies on protected segment {:?} ({:?}) and requires a curve split; source triangle {:?}",
                    candidate.reason, candidate.uv, endpoints, segment.source, candidate.triangle.vertex_indices.map(|index| (index, pslg.vertices[index as usize].uv, pslg.vertices[index as usize].node_id))
                ),
            ));
        }
    }
    let mut inside = false;
    for triangle in &trimmed.triangles {
        checkpoint(pslg, cancellation)?;
        let vertices = triangle
            .vertex_indices
            .map(|index| pslg.vertices[index as usize].uv);
        let mut within_triangle = true;
        for [start, finish] in [[0, 1], [1, 2], [2, 0]] {
            let sign = orient2d([vertices[start], vertices[finish], candidate.uv])
                .map_err(|_| invalid(pslg, "refinement candidate has invalid predicate input"))?;
            within_triangle &= sign != PredicateSign::Negative;
        }
        if within_triangle {
            inside = true;
            break;
        }
    }
    if !inside {
        return Err(invalid(
            pslg,
            "refinement candidate is outside the current trimmed face domain",
        ));
    }
    Ok(())
}

fn within_segment_bounds(point: [f64; 2], endpoints: [[f64; 2]; 2]) -> bool {
    (0..2).all(|axis| {
        point[axis] >= endpoints[0][axis].min(endpoints[1][axis])
            && point[axis] <= endpoints[0][axis].max(endpoints[1][axis])
    })
}

#[track_caller]
fn checkpoint(
    pslg: &ExactFacePslg,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), ExactFaceRefinementError> {
    if cancellation.is_cancelled() {
        Err(delaunay_error(
            pslg,
            ExactFaceDelaunayErrorKind::Cancelled,
            "refinement candidate validation cancelled",
        ))
    } else {
        Ok(())
    }
}

#[path = "candidate/indexed.rs"]
mod indexed;
pub(super) use indexed::validate_candidates;

#[cfg(test)]
#[path = "candidate/tests.rs"]
mod tests;
