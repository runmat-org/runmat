//! A failed launch cannot certify that no child or artifact producer ran.
use super::*;

pub(super) async fn run() {
    let directory = tempfile::tempdir().unwrap();
    let (host, request) = fixture();
    let mut configuration = config(directory.path(), "--slow-child");
    configuration.executable = directory.path().join("absent-executable");
    let session = NativeProgramSession::new(configuration).unwrap();
    seed_host_invocation(&mut session.object_store(), &host, &request);
    let original = submission(&session, &host, &request);
    let task = session.submit(request, original.clone()).unwrap();
    let result = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(result) = task.try_result() {
                break result;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(result.is_err());
    let evidence = task.try_execution_evidence().unwrap();
    evidence.validate().unwrap();
    assert_eq!(evidence.task.submission, original);
    assert_eq!(evidence.attempts.len(), 1);
    let terminal = evidence.attempts[0].terminal.as_ref().unwrap();
    assert!(matches!(
        terminal.disposition,
        AttemptDisposition::Failed { .. }
    ));
    assert_eq!(
        terminal.execution.as_ref().unwrap().termination,
        AttemptExecutionTermination::Unconfirmed
    );
    assert!(terminal.resources.is_none());
    assert!(evidence.resource_aggregate(10).is_err());
    let cold: runmat_execution_runner::TaskExecutionEvidence =
        serde_json::from_slice(&serde_json::to_vec(&evidence).unwrap()).unwrap();
    assert_eq!(cold, evidence);
    cold.validate().unwrap();
}
