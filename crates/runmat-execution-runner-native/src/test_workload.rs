use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use runmat_execution_artifact::{
    ExecutableForm, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_test_runner::worker::WorkerExecution;
use runmat_test_runner_execution::TestAttemptWorkload;

pub async fn execute_host_program_request(
    request: ProgramExecutionRequest,
) -> ProgramExecutionResponse {
    execute_host_program_request_with_project(request, None).await
}

pub async fn execute_host_program_request_with_project(
    request: ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
) -> ProgramExecutionResponse {
    if request.artifact.form != ExecutableForm::TestAttemptV1 {
        return runmat_vm::execute_program_request(request).await;
    }
    match execute_test_attempt(&request, project).await {
        Ok(execution) => match runmat_test_runner_execution::encode_execution(&execution) {
            Ok(value) => ProgramExecutionResponse::Success { value },
            Err(message) => ProgramExecutionResponse::Failure { message },
        },
        Err(message) => ProgramExecutionResponse::Failure { message },
    }
}

async fn execute_test_attempt(
    request: &ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
) -> Result<WorkerExecution, String> {
    let workload = TestAttemptWorkload::from_program_request(request)?;
    if let Some(project) = project {
        let revision = project.revision();
        if request.recipe.program_revision.graph_digest().bytes() != revision.graph_digest.bytes()
            || workload.submission.snapshot.base_source_digest
                != revision.source_revision.to_string()
        {
            return Err(
                "test workload base project revision differs from the installed bundle".into(),
            );
        }
    }
    let mut session = runmat_core::RunMatSession::with_options(true, false)
        .map_err(|error| format!("failed to initialize test execution session: {error}"))?;
    if let Some(project) = project {
        session
            .install_project_handoff(project.clone())
            .map_err(|error| format!("failed to install exact test project: {error}"))?;
    }
    let execution = session
        .execute_planned_test(
            &workload.submission.snapshot,
            &workload.submission.plan,
            &workload.test_id,
            workload.attempt,
            Arc::new(AtomicBool::new(false)),
        )
        .await
        .map_err(|error| error.to_string())?;
    Ok(WorkerExecution {
        result: execution.result,
        events: execution.events,
        coverage: execution.coverage,
    })
}

#[cfg(test)]
mod tests {
    use runmat_execution::Digest;
    use runmat_execution_artifact::ProgramExecutionResponse;
    use runmat_test::descriptor::TestSelector;
    use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
    use runmat_test::result::TerminalDisposition;
    use runmat_test_runner::worker::RunSubmission;
    use runmat_test_runner_execution::{decode_execution, TestAttemptWorkload};

    use super::execute_host_program_request;

    #[tokio::test]
    async fn host_executes_an_exact_test_workload_and_returns_canonical_result() {
        let snapshot = FrozenTestRunSnapshot::freeze(
            Digest::sha256(b"graph").to_string(),
            "sha256:base-sources",
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
            Digest::sha256(b"test-config").to_string(),
            vec![SavedRunSource {
                owner_identity: "path:workspace".into(),
                relative_path: "tests/arithmeticTest.m".into(),
                content: "function tests = arithmeticTest()\n tests = functiontests(localfunctions);\nend\nfunction testAddition(testCase)\n testCase.verifyEqual(1 + 1, 2);\nend\n".into(),
            }],
            Vec::new(),
        )
        .unwrap();
        let session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        let prepared = session
            .prepare_tests(&snapshot, &TestSelector::default())
            .unwrap();
        let test_id = prepared.plan.tests().next().unwrap().id.clone();
        let workload = TestAttemptWorkload::new(
            RunSubmission::new(prepared.plan, snapshot).unwrap(),
            test_id,
            1,
        )
        .unwrap();
        let response = execute_host_program_request(workload.program_request().unwrap()).await;
        let ProgramExecutionResponse::Success { value } = response else {
            panic!("test-capable host rejected a valid workload: {response:?}");
        };
        let execution = decode_execution(&value).unwrap();
        assert_eq!(
            execution.result.state.disposition,
            TerminalDisposition::Passed,
            "{execution:#?}"
        );
        assert!(!execution.events.is_empty());
        assert!(!execution.coverage.is_empty());
    }
}
