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
    let request = submission.program_request();
    assert!(std::ptr::eq(request.recipe, &submission.recipe));
    assert!(std::ptr::eq(request.artifact, &submission.artifact));
    assert_eq!(request.arguments.as_ptr(), submission.arguments.as_ptr());
    let borrowed = crate::protocol::StoredProgram {
        recipe: request.recipe,
        artifact: request.artifact,
        function: request.function,
    };
    let owned = crate::protocol::StoredProgram {
        recipe: submission.recipe.clone(),
        artifact: submission.artifact.clone(),
        function: submission.function,
    };
    let bytes = runmat_canonical_codec::encode_json_bounded(
        &borrowed,
        u64::from(crate::supervisor::protocol::SUPERVISOR_MAX_MESSAGE_BYTES),
    )
    .unwrap();
    assert_eq!(bytes, serde_json::to_vec(&owned).unwrap());
    let cold: crate::protocol::StoredProgram = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cold.recipe, submission.recipe);
    assert_eq!(cold.artifact, submission.artifact);
    assert_eq!(cold.function, submission.function);
    cold.artifact.validate_against(&cold.recipe).unwrap();
    assert!(supervisor.list().await.unwrap().is_empty());
}
