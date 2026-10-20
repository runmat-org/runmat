use super::*;

#[tokio::test]
async fn original_program_borrows_validation_and_worker_wire_without_payload_copies() {
    let temp = tempfile::tempdir().unwrap();
    let supervisor = LocalSupervisor::open(test_config(&temp)).unwrap();
    let context = supervisor
        .execution_context(Some("borrowed-program"))
        .await
        .unwrap();
    let submission = original_program_submission(context);
    submission.validate().unwrap();
    let mut stable = submission.clone();
    stable.idempotency_key = None;
    let mut original_identity = b"runmat-local-program-batch-request-v1\0".to_vec();
    original_identity.extend(serde_json::to_vec(&stable).unwrap());
    assert_eq!(
        submission.request_digest().unwrap(),
        runmat_execution::Digest::sha256(original_identity)
    );
    let request = submission.program_request();
    assert!(std::ptr::eq(request.recipe, &submission.recipe));
    assert!(std::ptr::eq(request.artifact, &submission.artifact));
    assert_eq!(request.arguments.as_ptr(), submission.arguments.as_ptr());
    #[derive(serde::Serialize)]
    struct BorrowedStoredDirectProgram<'a> {
        recipe: &'a runmat_execution_artifact::ProgramBuildRecipe,
        artifact: &'a runmat_execution_artifact::ProgramArtifact,
    }
    let borrowed = BorrowedStoredDirectProgram {
        recipe: request.recipe,
        artifact: request.artifact,
    };
    let owned =
        crate::protocol::StoredProgram::Direct(Box::new(crate::protocol::StoredDirectProgram {
            recipe: submission.recipe.clone(),
            artifact: submission.artifact.clone(),
        }));
    assert_eq!(request.callable, submission.callable);
    let bytes = runmat_canonical_codec::encode_json_bounded(
        &borrowed,
        u64::from(crate::supervisor::protocol::SUPERVISOR_MAX_MESSAGE_BYTES),
    )
    .unwrap();
    assert_eq!(bytes, serde_json::to_vec(&owned).unwrap());
    let crate::protocol::StoredProgram::Direct(cold) = serde_json::from_slice(&bytes).unwrap()
    else {
        panic!("original direct stored program became graph work");
    };
    assert_eq!(cold.recipe, submission.recipe);
    assert_eq!(cold.artifact, submission.artifact);
    cold.artifact.validate_against(&cold.recipe).unwrap();
    assert!(supervisor.list().await.unwrap().is_empty());
}
