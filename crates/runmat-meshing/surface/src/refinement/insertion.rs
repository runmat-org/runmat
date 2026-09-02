use runmat_meshing_core::{predicate::orient2d, MeshingCancellationSignal, PredicateSign};
use std::collections::BTreeSet;

use crate::{
    exact_face_interior_node_id, validate_exact_face_trimmed_delaunay, ExactFaceBoundary,
    ExactFaceDelaunayError, ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions, ExactFacePslg,
    ExactFacePslgVertex, ExactFaceTrimmedDelaunay,
};

use super::{
    ExactFaceRefinedTopology, ExactFaceRefinementCandidate, ExactFaceRefinementError,
    ExactFaceRefinementErrorKind,
};

pub fn insert_exact_face_refinement_candidate(
    boundary: &ExactFaceBoundary,
    topology: &ExactFaceRefinedTopology,
    candidate: &ExactFaceRefinementCandidate,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceRefinedTopology, ExactFaceRefinementError> {
    insert_exact_face_refinement_candidates(
        boundary,
        topology,
        std::slice::from_ref(candidate),
        cancellation,
        options,
    )
    .map(|(topology, _)| topology)
}

pub(crate) fn insert_exact_face_refinement_candidates(
    boundary: &ExactFaceBoundary,
    topology: &ExactFaceRefinedTopology,
    candidates: &[ExactFaceRefinementCandidate],
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<(ExactFaceRefinedTopology, u32), ExactFaceRefinementError> {
    let pslg = &topology.pslg;
    validate_exact_face_trimmed_delaunay(
        &topology.trimmed,
        &topology.constrained,
        pslg,
        boundary,
        cancellation,
        options,
    )
    .map_err(map_delaunay)?;
    insert_validated_face_refinement_candidates(topology, candidates, cancellation, options)
}

pub(crate) fn insert_validated_face_refinement_candidates(
    topology: &ExactFaceRefinedTopology,
    candidates: &[ExactFaceRefinementCandidate],
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<(ExactFaceRefinedTopology, u32), ExactFaceRefinementError> {
    let pslg = &topology.pslg;
    if candidates.is_empty() {
        return Err(invalid(pslg, "refinement candidate batch is empty"));
    }
    crate::exact_cdt::validate_face_trimmed_topology(
        &topology.trimmed,
        &topology.constrained,
        pslg,
        cancellation,
        options,
    )
    .map_err(map_delaunay)?;
    for candidate in candidates {
        validate_candidate(pslg, &topology.trimmed, candidate)?;
    }

    let mut seen = BTreeSet::new();
    let additions = candidates
        .iter()
        .filter(|candidate| seen.insert(candidate.uv.map(f64::to_bits)))
        .map(|candidate| ExactFacePslgVertex {
            node_id: exact_face_interior_node_id(&pslg.source_face_id, candidate.uv),
            seam_image: None,
            uv: candidate.uv,
        })
        .collect::<Vec<_>>();
    let inserted_count = u32::try_from(additions.len()).map_err(|_| {
        delaunay_error(
            pslg,
            ExactFaceDelaunayErrorKind::ElementLimit,
            "face refinement candidate batch exceeds the vertex identity space",
        )
    })?;
    let (refined_pslg, _) = crate::exact_cdt::insert_pslg_vertices(pslg, &additions)
        .map_err(|reason| delaunay_error(pslg, ExactFaceDelaunayErrorKind::ElementLimit, reason))?;
    rebuild_refined_topology(refined_pslg, cancellation, options)
        .map(|topology| (topology, inserted_count))
}

fn rebuild_refined_topology(
    pslg: ExactFacePslg,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceRefinedTopology, ExactFaceRefinementError> {
    let delaunay = crate::exact_cdt::triangulate_validated_face_pslg(&pslg, cancellation, options)
        .map_err(map_delaunay)?;
    let refined_constrained =
        crate::exact_cdt::recover_validated_face_segments(&delaunay, &pslg, cancellation, options)
            .map_err(map_delaunay)?;
    let refined_trimmed = crate::exact_cdt::carve_validated_face_domain(
        &refined_constrained,
        &pslg,
        cancellation,
        options,
    )
    .map_err(map_delaunay)?;
    Ok(ExactFaceRefinedTopology {
        pslg,
        constrained: refined_constrained,
        trimmed: refined_trimmed,
    })
}

fn validate_candidate(
    pslg: &ExactFacePslg,
    trimmed: &ExactFaceTrimmedDelaunay,
    candidate: &ExactFaceRefinementCandidate,
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
    if pslg.vertices.iter().any(|vertex| vertex.uv == candidate.uv) {
        return Err(invalid(
            pslg,
            "refinement candidate duplicates an existing vertex",
        ));
    }
    for segment in &pslg.segments {
        let endpoints = segment
            .vertex_indices
            .map(|index| pslg.vertices[index as usize].uv);
        let sign = orient2d([endpoints[0], endpoints[1], candidate.uv])
            .map_err(|_| invalid(pslg, "refinement candidate has invalid predicate input"))?;
        if sign == PredicateSign::Zero && within_segment_bounds(candidate.uv, endpoints) {
            return Err(invalid(
                pslg,
                "refinement candidate lies on a protected segment and requires a curve split",
            ));
        }
    }
    let mut inside = false;
    for triangle in &trimmed.triangles {
        let vertices = triangle
            .vertex_indices
            .map(|index| pslg.vertices[index as usize].uv);
        let signs = [
            orient2d([vertices[0], vertices[1], candidate.uv]),
            orient2d([vertices[1], vertices[2], candidate.uv]),
            orient2d([vertices[2], vertices[0], candidate.uv]),
        ];
        let signs = signs
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid(pslg, "refinement candidate has invalid predicate input"))?;
        if signs.iter().all(|sign| *sign != PredicateSign::Negative) {
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

fn map_delaunay(error: ExactFaceDelaunayError) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Delaunay(error.kind),
        &error.source_face_id,
        error.reason,
    )
}

fn delaunay_error(
    pslg: &ExactFacePslg,
    kind: ExactFaceDelaunayErrorKind,
    reason: &str,
) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Delaunay(kind),
        &pslg.source_face_id,
        reason,
    )
}

fn invalid(pslg: &ExactFacePslg, reason: &str) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::InvalidGeometry,
        &pslg.source_face_id,
        reason,
    )
}
