use super::*;
use crate::analysis::storage::AnalysisPublicationReceipt;
use runmat_execution::Digest;
use runmat_filesystem::workspace::{
    WorkspacePublicationId, WorkspacePublicationReceipt, WorkspaceRevision,
};

#[test]
fn uncertain_publication_is_retryable_backend_evidence_not_invalid_physics() {
    let workspace = WorkspacePublicationReceipt {
        id: WorkspacePublicationId([29; 16]),
        revision: WorkspaceRevision([67; 32]),
    };
    let publication = AnalysisPublicationReceipt {
        identity: Digest::sha256(b"original-root"),
        definition: Digest::sha256(b"original-study"),
        manifest_digest: Digest::sha256(b"original-manifest"),
        run_ids: vec!["original-child".into()],
        manifest_artifact_path: Some("original/manifest.json".into()),
    };
    for publication in [None, Some(Box::new(publication))] {
        let original = AnalysisPublicationError::DurabilityIndeterminate {
            workspace: workspace.clone(),
            publication,
        };
        let envelope =
            publication_failure(original.clone(), &OperationContext::new(None, None), |_| {
                panic!("uncertainty treated as invalid physics")
            });
        assert_eq!(
            envelope.error_code,
            "RM.FEA.RUN_STUDY.PUBLICATION_DURABILITY_INDETERMINATE"
        );
        assert_eq!(envelope.error_type, OperationErrorType::Backend);
        assert!(envelope.retryable);
        assert_eq!(
            serde_json::from_str::<AnalysisPublicationError>(
                &envelope.context["publication_outcome"]
            )
            .unwrap(),
            original
        );
    }
}

#[test]
fn resource_exhaustion_is_capacity_and_semantic_rejection_keeps_family_handler() {
    let original = AnalysisPublicationError::ResourceLimit {
        message: "original admitted inventory exceeds allowance".into(),
    };
    let context = OperationContext::new(None, None);
    let envelope = publication_failure(original.clone(), &context, |_| {
        panic!("resource limit routed to invalid physics")
    });
    assert_eq!(
        envelope.error_code,
        "RM.FEA.RUN_STUDY.PUBLICATION_RESOURCE_LIMIT"
    );
    assert_eq!(envelope.error_type, OperationErrorType::Capacity);
    assert!(!envelope.retryable);
    assert_eq!(
        serde_json::from_str::<AnalysisPublicationError>(&envelope.message).unwrap(),
        original
    );
    let envelope = publication_failure(
        AnalysisPublicationError::Rejected {
            message: "malformed original result".into(),
        },
        &context,
        |message| {
            assert_eq!(message, "malformed original result");
            crate::operations::operation_error(
                crate::analysis::ANALYSIS_RUN_STUDY_OPERATION,
                crate::analysis::ANALYSIS_RUN_STUDY_OP_VERSION,
                &context,
                OperationErrorSpec {
                    error_code: "TEST.INVALID_RESULT",
                    error_type: OperationErrorType::Contract,
                    retryable: false,
                    severity: OperationErrorSeverity::Error,
                },
                message,
                Default::default(),
            )
        },
    );
    assert_eq!(envelope.error_code, "TEST.INVALID_RESULT");
}

#[test]
fn textual_budget_wording_does_not_authorize_resource_classification() {
    let error = AnalysisPublicationError::from(std::io::Error::other(
        "invalid content says byte budget exceeded",
    ));
    assert!(matches!(error, AnalysisPublicationError::Rejected { .. }));
}

#[test]
fn evidence_sync_uncertainty_keeps_original_path_without_inventing_study_receipt() {
    let original =
        crate::analysis::storage::AnalysisPublicationError::EvidenceDurabilityIndeterminate {
            path: std::path::PathBuf::from("studies/original.json"),
            cause: "directory synchronization denied".into(),
        };
    let context = crate::operations::OperationContext::new(None, None);
    let envelope = super::publication_failure(original.clone(), &context, |_| {
        panic!("uncertainty reached invalid physics handler")
    });
    assert_eq!(
        envelope.error_type,
        crate::operations::OperationErrorType::Backend
    );
    assert!(envelope.retryable);
    let wire = serde_json::to_value(original).unwrap();
    assert_eq!(wire["path"], "studies/original.json");
    assert!(wire.get("publication").is_none());
    assert!(wire.get("workspace").is_none());
}
