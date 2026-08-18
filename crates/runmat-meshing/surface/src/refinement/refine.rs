use crate::{
    evaluate_exact_face_geometry, validate_exact_face_geometry, ExactFaceBoundary,
    ExactFaceGeometryError,
};

use super::{
    classify_exact_face_refinement_candidate, derive_exact_face_feature_collars,
    insert_exact_face_refinement_candidate, select_exact_face_refinement_candidate,
    ExactFaceCandidateDisposition, ExactFaceRefinedMesh, ExactFaceRefinedTopology,
    ExactFaceRefinementContext, ExactFaceRefinementError, ExactFaceRefinementErrorKind,
    ExactFaceRefinementOutcome, ExactFaceRefinementPolicy,
};

pub fn refine_exact_face_until_blocked(
    boundary: &ExactFaceBoundary,
    initial: &ExactFaceRefinedTopology,
    context: ExactFaceRefinementContext<'_>,
    policy: ExactFaceRefinementPolicy,
) -> Result<ExactFaceRefinementOutcome, ExactFaceRefinementError> {
    policy.quality.validate().map_err(|error| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidQuality,
            &initial.pslg.source_face_id,
            error.to_string(),
        )
    })?;
    if policy.refinement.maximum_interior_insertions == 0 || policy.delaunay.validate().is_err() {
        return Err(ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidOptions,
            &initial.pslg.source_face_id,
            "refinement and Delaunay operational limits must be nonzero",
        ));
    }
    let mut current = initial.clone();
    let mut insertion_count = 0u32;
    loop {
        let geometry = evaluate_exact_face_geometry(
            &current.trimmed,
            &current.pslg,
            context.topology,
            context.metric_request,
            context.evaluator,
            context.geometry_control,
        )
        .map_err(map_geometry)?;
        validate_exact_face_geometry(
            &geometry,
            &current.trimmed,
            &current.pslg,
            context.topology,
            context.metric_request,
            context.evaluator,
            context.geometry_control,
        )
        .map_err(map_geometry)?;
        let collars = derive_exact_face_feature_collars(&current.pslg, &geometry, policy.quality)?;
        let Some(candidate) = select_exact_face_refinement_candidate(
            &geometry,
            &current.pslg,
            &collars,
            policy.quality,
        )?
        else {
            return Ok(ExactFaceRefinementOutcome::Converged(Box::new(
                ExactFaceRefinedMesh {
                    topology: current,
                    geometry,
                    feature_collars: collars,
                    interior_insertion_count: insertion_count,
                },
            )));
        };
        match classify_exact_face_refinement_candidate(
            &candidate,
            &current.pslg,
            context.topology,
            context.metric_request,
            context.evaluator,
            context.geometry_control,
        )? {
            ExactFaceCandidateDisposition::SplitProtectedSegment(split) => {
                return Ok(ExactFaceRefinementOutcome::RequiresCurveSplit {
                    split,
                    completed_interior_insertions: insertion_count,
                });
            }
            ExactFaceCandidateDisposition::Insert => {
                if insertion_count >= policy.refinement.maximum_interior_insertions {
                    return Err(ExactFaceRefinementError::new(
                        ExactFaceRefinementErrorKind::ResourceLimit,
                        &current.pslg.source_face_id,
                        "exact face refinement exceeded its interior insertion hard limit",
                    ));
                }
                current = insert_exact_face_refinement_candidate(
                    boundary,
                    &current,
                    &candidate,
                    context.cancellation,
                    policy.delaunay,
                )?;
                insertion_count += 1;
            }
        }
    }
}

fn map_geometry(error: ExactFaceGeometryError) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Geometry(error.kind),
        &error.source_face_id,
        error.reason,
    )
}
