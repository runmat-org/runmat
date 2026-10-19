mod candidate;
use candidate::validate_candidates;
use runmat_meshing_core::MeshingCancellationSignal;
use std::collections::BTreeSet;

use crate::{
    exact_face_interior_node_id, ExactFaceBoundary, ExactFaceDelaunayError,
    ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions, ExactFacePslg, ExactFacePslgVertex,
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
    insert_exact_face_refinement_candidates_with_control(
        boundary,
        topology,
        candidates,
        cancellation,
        options,
        None,
    )
}

pub(super) fn insert_exact_face_refinement_candidates_with_control(
    boundary: &ExactFaceBoundary,
    topology: &ExactFaceRefinedTopology,
    candidates: &[ExactFaceRefinementCandidate],
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
    geometry_control: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<(ExactFaceRefinedTopology, u32), ExactFaceRefinementError> {
    // Validate boundary/PSLG provenance once. The common insertion entry validates
    // the complete constrained and trimmed topology before any mutation.
    crate::validate_exact_face_pslg(&topology.pslg, boundary).map_err(|error| {
        delaunay_error(
            &topology.pslg,
            ExactFaceDelaunayErrorKind::InvalidPslg,
            &error.to_string(),
        )
    })?;
    insert_validated_face_refinement_candidates(
        topology,
        candidates,
        cancellation,
        options,
        geometry_control,
    )
}

pub(crate) fn insert_validated_face_refinement_candidates(
    topology: &ExactFaceRefinedTopology,
    candidates: &[ExactFaceRefinementCandidate],
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
    geometry_control: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
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
        geometry_control,
    )
    .map_err(map_delaunay)?;
    validate_candidates(
        pslg,
        &topology.trimmed,
        candidates,
        cancellation,
        geometry_control,
    )?;

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
    rebuild_refined_topology(refined_pslg, cancellation, options, geometry_control)
        .map(|topology| (topology, inserted_count))
}

fn rebuild_refined_topology(
    pslg: ExactFacePslg,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
    geometry_control: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<ExactFaceRefinedTopology, ExactFaceRefinementError> {
    let delaunay = crate::exact_cdt::triangulate_validated_face_pslg(&pslg, cancellation, options)
        .map_err(map_delaunay)?;
    let refined_constrained = crate::exact_cdt::recover_validated_face_segments(
        &delaunay,
        &pslg,
        cancellation,
        options,
        geometry_control,
    )
    .map_err(map_delaunay)?;
    let refined_trimmed = crate::exact_cdt::carve_validated_face_domain(
        &refined_constrained,
        &pslg,
        cancellation,
        options,
        geometry_control,
    )
    .map_err(map_delaunay)?;
    Ok(ExactFaceRefinedTopology {
        pslg,
        constrained: refined_constrained,
        trimmed: refined_trimmed,
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
