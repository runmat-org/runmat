//! Pairwise classification must remain interruptible after metric preparation.
use super::{candidate, metric_request, one_segment_pslg, Control};
use crate::refinement::{
    classify_exact_face_refinement_candidates_in_parameterization, ExactFaceRefinementErrorKind,
};
use crate::ExactFaceChartParameterization;
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind, GeometryModel,
    PortableExactEvaluator,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CancelDuringPairWalk {
    probes: AtomicUsize,
    cancel_at: usize,
}
impl GeometryEvaluationControl for CancelDuringPairWalk {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        if std::path::Path::new(std::panic::Location::caller().file())
            .ends_with(std::path::Path::new("encroachment/classification.rs"))
            && self.probes.fetch_add(1, Ordering::Relaxed) + 1 == self.cancel_at
        {
            return Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::Cancelled,
                "cancel protected-segment pair walk",
            ));
        }
        Ok(())
    }
    fn consume_iterations(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, _: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

#[test]
fn protected_segment_classification_cancels_after_metric_preparation() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("exact fixture")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let face = topology.faces[0].id.clone();
    let pslg = one_segment_pslg(face.clone());
    let candidates = [candidate(face, [0.0, 0.0])];
    let request = metric_request();
    let parameterization = ExactFaceChartParameterization::EvaluatorParameters;
    let original = classify_exact_face_refinement_candidates_in_parameterization(
        &candidates,
        &pslg,
        &topology,
        &request,
        &parameterization,
        &evaluator,
        &Control,
    )
    .unwrap();
    assert_eq!(original.protected_splits.len(), 1);
    let control = CancelDuringPairWalk {
        probes: AtomicUsize::new(0),
        // Entry, candidate admission, segment metrics, candidate entry, then its first pair.
        cancel_at: 1 + candidates.len() + pslg.segments.len() + 2,
    };
    let failure = classify_exact_face_refinement_candidates_in_parameterization(
        &candidates,
        &pslg,
        &topology,
        &request,
        &parameterization,
        &evaluator,
        &control,
    )
    .err()
    .expect("pair walk must observe cancellation");
    assert_eq!(control.probes.load(Ordering::Relaxed), control.cancel_at);
    assert_eq!(
        failure.kind,
        ExactFaceRefinementErrorKind::Metric(crate::ExactFaceMetricErrorKind::GeometryEvaluation(
            GeometryEvaluationErrorKind::Cancelled
        ))
    );
}
