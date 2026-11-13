//! Exercise real preparation failure before the shared study commit boundary.
use super::*;
use crate::analysis::storage::AnalysisPublicationError;

fn manifest(
    publication: &mut PreparedStudyPublication,
    compiled: &runmat_analysis_fea::study::CompiledAnalysisStudy,
) -> (
    StudyResultManifest<(), ()>,
    crate::analysis::storage::publication::OwnedCopyReservation,
) {
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(
        runmat_meshing_core::ElementOrder::Tet4,
    );
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
    StudyResultManifest::new(
        compiled.definition().unwrap(),
        Digest::sha256(b"original-workload"),
        &(),
        &mesh,
        &evidence,
        (),
        &mut publication.completion,
    )
    .unwrap()
}

#[test]
fn authored_assessment_limit_is_capacity_and_preserves_hidden_original_children() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let (mut publication, store) = fixture();
    let (mut compiled, _, definition) = crate::analysis::assessment::tests::fixture();
    let limit =
        crate::analysis::assessment::AssessmentEvaluationLimits::default().maximum_assessments;
    compiled.assessments = (0..=limit)
        .map(|ordinal| {
            let mut assessment = definition.clone();
            assessment.assessment_id = format!("criterion-{ordinal}");
            assessment
        })
        .collect();
    publication.completion.published.data.study_fingerprint =
        compiled.definition_digest().unwrap().to_string();
    let manifest = manifest(&mut publication, &compiled);
    let error = PreparedStudyPublication::new(publication.completion, manifest)
        .err()
        .unwrap();
    assert!(matches!(
        error,
        AnalysisPublicationError::ResourceLimit { .. }
    ));
    let envelope = super::super::failure::publication_failure(
        error,
        &crate::operations::OperationContext::new(None, None),
        |_| panic!("assessment capacity treated as invalid numerical result"),
    );
    assert_eq!(
        envelope.error_type,
        crate::operations::OperationErrorType::Capacity
    );
    assert!(!envelope.retryable);
    assert!(store.list_runs().unwrap().is_empty());
}

#[test]
fn invalid_identity_and_cancelled_preparation_never_publish_staged_children() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    for cancelled in [false, true] {
        let (mut publication, store) = fixture();
        let (compiled, _, _) = crate::analysis::assessment::tests::fixture();
        let manifest = manifest(&mut publication, &compiled);
        if !cancelled {
            publication.completion.published.data.study_fingerprint =
                Digest::sha256(b"foreign-study").to_string();
        }
        let _interrupt = crate::interrupt::replace_interrupt(Some(
            runmat_execution::CancellationToken::from_shared(Arc::new(
                std::sync::atomic::AtomicBool::new(cancelled),
            )),
        ));
        let error = PreparedStudyPublication::new(publication.completion, manifest)
            .err()
            .unwrap();
        let AnalysisPublicationError::Rejected { message } = error else {
            panic!("non-capacity assessment rejection was reclassified")
        };
        if cancelled {
            assert!(message.contains("cancelled"));
        } else {
            assert!(message.contains("another checked study revision"));
        }
        assert!(store.list_runs().unwrap().is_empty());
    }
}

#[test]
fn admitted_assessment_phase_keeps_missing_scalar_evidence_indeterminate() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let (mut publication, store) = fixture();
    let (compiled, _, _) = crate::analysis::assessment::tests::fixture();
    let original = publication
        .completion
        .staging
        .load_assessment_metadata("immutable-run", 1 << 20)
        .unwrap()
        .unwrap();
    assert!(original.observation_scalars.is_empty());
    let manifest = manifest(&mut publication, &compiled);
    let prepared = PreparedStudyPublication::new(publication.completion, manifest).unwrap();
    let completed = prepared.publish_without_function_execution().unwrap();
    assert_eq!(completed.data.assessments.len(), 1);
    let report = &completed.data.assessments[0];
    assert_eq!(
        report.study_definition,
        compiled.definition_digest().unwrap()
    );
    assert_eq!(report.result.status, AssessmentStatus::Indeterminate);
    assert!(report.result.measured.is_none());
    assert!(report.evidence.is_none());
    let retained = store.load_run("immutable-run").unwrap().unwrap();
    assert_eq!(retained.run.identity, original.identity);
    assert!(retained.observation_scalars.is_empty());
    assert_eq!(store.list_runs().unwrap().len(), 1);
}
