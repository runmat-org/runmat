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
    execute_host_program_request_in_context(request, project, None, None, None).await
}

pub(crate) async fn execute_host_program_request_in_context(
    request: ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    environment: Option<&[(String, String)]>,
    cancellation: Option<Arc<AtomicBool>>,
    resources: Option<&runmat_execution::resource::ResourceRequest>,
) -> ProgramExecutionResponse {
    if let Err(error) = request.validate_for_portable_host() {
        return ProgramExecutionResponse::Failure {
            message: error.to_string(),
        };
    }
    if environment.is_some_and(|entries| entries != request.environment.entries()) {
        return ProgramExecutionResponse::Failure {
            message: "portable invocation environment differs from admitted bundle context".into(),
        };
    }
    if request.artifact.form == ExecutableForm::AnalysisWorkload {
        // This branch is synchronous: the thread-affine context remains installed
        // throughout the physical solve. Local process cancellation remains owned
        // by the parent guard; callers that supply a token also reach the solver.
        let _progress = crate::analysis_progress::install(
            cancellation
                .as_ref()
                .map(|flag| {
                    runmat_analysis_fea::FeaCancellationToken::from_shared(Arc::clone(flag))
                })
                .unwrap_or_default(),
            None,
        );
        let Some(root) = std::env::var_os(crate::NATIVE_OBJECT_STORE_ROOT_ENV) else {
            return ProgramExecutionResponse::Failure {
                message: "analysis worker object store is not configured".into(),
            };
        };
        let mut store = match crate::FilesystemObjectStore::open(
            root,
            crate::NativeAnalysisHostLimits::default()
                .inventory
                .max_object_bytes,
        ) {
            Ok(store) => store,
            Err(error) => {
                return ProgramExecutionResponse::Failure {
                    message: error.to_string(),
                }
            }
        };
        return match resources {
            Some(resources) => crate::execute_reserved_analysis_program_request(
                &request,
                &mut store,
                &crate::NativeAnalysisKernel,
                crate::NativeAnalysisHostLimits::default(),
                resources,
            ),
            None => crate::execute_analysis_program_request(
                &request,
                &mut store,
                &crate::NativeAnalysisKernel,
                crate::NativeAnalysisHostLimits::default(),
            ),
        };
    }
    if request.artifact.form == ExecutableForm::GeometryWorkload {
        let Some(root) = std::env::var_os(crate::NATIVE_OBJECT_STORE_ROOT_ENV) else {
            return ProgramExecutionResponse::Failure {
                message: "geometry worker object store is not configured".into(),
            };
        };
        let limits = crate::NativeGeometryMaterializationLimits::default();
        let mut store =
            match crate::FilesystemObjectStore::open(root, limits.inventory.max_object_bytes) {
                Ok(store) => store,
                Err(error) => {
                    return ProgramExecutionResponse::Failure {
                        message: error.to_string(),
                    }
                }
            };
        return crate::execute_geometry_materialization_program_request(
            &request, &mut store, limits,
        );
    }
    if request.artifact.form == ExecutableForm::ExecutableUnitV4 {
        return crate::program_workload::execute_unit(&request, project, environment, cancellation)
            .await;
    }
    if request.artifact.form != ExecutableForm::TestAttemptV1 {
        return runmat_vm::execute_program_request(request).await;
    }
    match execute_test_attempt(&request, project, cancellation).await {
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
    cancellation: Option<Arc<AtomicBool>>,
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
    let mut session = runmat_core::RunMatSession::with_frozen_environment(
        true,
        false,
        request.environment.entries().to_vec(),
    )
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
            cancellation.unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
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

    fn original_request(source: &str) -> runmat_execution_artifact::ProgramExecutionRequest {
        let snapshot = FrozenTestRunSnapshot::freeze(
            Digest::sha256(b"graph").to_string(),
            "sha256:base-sources",
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
            Digest::sha256(b"test-config").to_string(),
            vec![SavedRunSource {
                owner_identity: "path:workspace".into(),
                relative_path: "tests/arithmeticTest.m".into(),
                content: source.into(),
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
        workload.program_request().unwrap()
    }

    #[tokio::test]
    async fn host_executes_an_exact_test_workload_and_returns_canonical_result() {
        let request = original_request("function tests = arithmeticTest()\n tests = functiontests(localfunctions);\nend\nfunction testAddition(testCase)\n testCase.verifyEqual(1 + 1, 2);\nend\n");
        let response = execute_host_program_request(request).await;
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

    #[tokio::test]
    async fn original_test_attempt_uses_request_context_and_rejects_bundle_substitution() {
        let source = "function tests = arithmeticTest()\n tests = functiontests(localfunctions);\nend\nfunction testEnvironment(testCase)\n testCase.verifyEqual(getenv('RUNMAT_TEST_ORIGINAL'), 'captured');\n testCase.verifyEmpty(getenv('HOME'));\n setenv('RUNMAT_TEST_ORIGINAL','child');\nend\n";
        let host = std::env::var_os("RUNMAT_TEST_ORIGINAL");
        let mut request = original_request(source);
        request.environment = runmat_execution::EnvironmentSnapshot::new(vec![(
            "RUNMAT_TEST_ORIGINAL".into(),
            "captured".into(),
        )])
        .unwrap();
        let conflicting = [("RUNMAT_TEST_ORIGINAL".into(), "substituted".into())];
        assert!(matches!(
            super::execute_host_program_request_in_context(
                request.clone(),
                None,
                Some(&conflicting),
                None,
                None
            )
            .await,
            ProgramExecutionResponse::Failure { .. }
        ));
        let ProgramExecutionResponse::Success { value } =
            execute_host_program_request(request).await
        else {
            panic!("original test invocation failed");
        };
        assert_eq!(
            decode_execution(&value).unwrap().result.state.disposition,
            TerminalDisposition::Passed
        );
        assert_eq!(std::env::var_os("RUNMAT_TEST_ORIGINAL"), host);
    }
}
