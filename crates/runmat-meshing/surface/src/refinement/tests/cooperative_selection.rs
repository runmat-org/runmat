//! Cancellation must reach the potentially long exact trim-membership walk.
use super::{empty_collars, geometry, quality, triangle_pslg, Control};
use crate::refinement::{
    select::select_exact_face_refinement_candidates, ExactFaceRefinementErrorKind,
};
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
};
use std::sync::atomic::{AtomicBool, Ordering};

struct CancelDuringDomainWalk(AtomicBool, &'static std::path::Path);
impl GeometryEvaluationControl for CancelDuringDomainWalk {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if std::path::Path::new(std::panic::Location::caller().file()).ends_with(self.1) {
            self.0.store(true, Ordering::Relaxed);
            Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::Cancelled,
                "cancel exact domain walk",
            ))
        } else {
            Ok(())
        }
    }
    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

#[test]
fn controlled_selection_preserves_candidates_and_cancels_inside_domain_walk() {
    let geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    let pslg = triangle_pslg(&geometry);
    let collars = empty_collars(&geometry);
    let original =
        select_exact_face_refinement_candidates(&geometry, &pslg, &collars, quality(), None)
            .unwrap();
    assert!(!original.is_empty());
    assert_eq!(
        original,
        select_exact_face_refinement_candidates(
            &geometry,
            &pslg,
            &collars,
            quality(),
            Some(&Control)
        )
        .unwrap()
    );
    let control = CancelDuringDomainWalk(
        AtomicBool::new(false),
        std::path::Path::new("select/domain.rs"),
    );
    let failure = select_exact_face_refinement_candidates(
        &geometry,
        &pslg,
        &collars,
        quality(),
        Some(&control),
    )
    .unwrap_err();
    assert!(control.0.load(Ordering::Relaxed));
    assert_eq!(
        failure.kind,
        ExactFaceRefinementErrorKind::Metric(crate::ExactFaceMetricErrorKind::GeometryEvaluation(
            GeometryEvaluationErrorKind::Cancelled
        ))
    );
}

#[test]
fn independent_batch_preserves_order_and_observes_cancellation() {
    use crate::refinement::select::independent_candidate_batch;
    let geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    let pslg = triangle_pslg(&geometry);
    let candidates = select_exact_face_refinement_candidates(
        &geometry,
        &pslg,
        &empty_collars(&geometry),
        quality(),
        None,
    )
    .unwrap();
    assert!(!candidates.is_empty());
    let original =
        independent_candidate_batch(candidates.clone(), &geometry, &pslg, 0.0, None).unwrap();
    assert_eq!(
        original,
        independent_candidate_batch(candidates.clone(), &geometry, &pslg, 0.0, Some(&Control),)
            .unwrap()
    );
    let control = CancelDuringDomainWalk(
        AtomicBool::new(false),
        std::path::Path::new("select/batch.rs"),
    );
    let failure =
        independent_candidate_batch(candidates, &geometry, &pslg, 0.0, Some(&control)).unwrap_err();
    assert!(control.0.load(Ordering::Relaxed));
    assert_eq!(
        failure.kind,
        ExactFaceRefinementErrorKind::Metric(crate::ExactFaceMetricErrorKind::GeometryEvaluation(
            GeometryEvaluationErrorKind::Cancelled
        ))
    );
}
