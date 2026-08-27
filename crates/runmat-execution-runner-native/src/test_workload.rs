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

pub(crate) async fn execute_host_program_request_with_project(
    request: ProgramExecutionRequest,
    materialized: Option<&crate::materialized_project::MaterializedProject>,
) -> ProgramExecutionResponse {
    if request.artifact.form != ExecutableForm::TestAttemptV1 {
        return execute_portable_request(request, materialized).await;
    }
    match execute_test_attempt(&request, materialized.and_then(|value| value.handoff())).await {
        Ok(execution) => match runmat_test_runner_execution::encode_execution(&execution) {
            Ok(value) => ProgramExecutionResponse::Success { value },
            Err(message) => ProgramExecutionResponse::Failure { message },
        },
        Err(message) => ProgramExecutionResponse::Failure { message },
    }
}

async fn execute_portable_request(
    request: ProgramExecutionRequest,
    materialized: Option<&crate::materialized_project::MaterializedProject>,
) -> ProgramExecutionResponse {
    let interop = match request.artifact.executable_unit() {
        Ok(Some(envelope)) => envelope.manifest.interop.clone(),
        Ok(None) => runmat_types::InteropManifest::empty(),
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("worker rejected an invalid executable unit: {error}"),
            }
        }
    };
    if interop.adapters.is_empty() && interop.foreign_types.is_empty() {
        return runmat_vm::execute_program_request(request).await;
    }

    let mut session = match runmat_core::RunMatSession::with_options(true, false) {
        Ok(session) => session,
        Err(error) => {
            return ProgramExecutionResponse::Failure {
                message: format!("failed to initialize foreign execution session: {error}"),
            }
        }
    };
    if let Err(error) = session.set_native_ffi_isolation_policy(
        runmat_runtime::foreign::NativeFfiIsolationPolicy::isolated(None),
    ) {
        return ProgramExecutionResponse::Failure {
            message: format!("failed to configure native-library isolation: {error}"),
        };
    }
    for requirement in &interop.adapters {
        if requirement.adapter != runmat_native_ffi::NATIVE_FFI_ADAPTER_ID {
            continue;
        }
        for identity in &requirement.artifact_identities {
            let Some(interface) =
                materialized.and_then(|materialized| materialized.native_interface(identity))
            else {
                return ProgramExecutionResponse::Failure {
                    message: format!(
                        "worker has no materialized native interface for required artifact {identity}"
                    ),
                };
            };
            if let Err(error) = session.install_native_interface_artifact(
                &interface.library_path,
                &interface.manifest_path,
            ) {
                return ProgramExecutionResponse::Failure {
                    message: format!("worker could not install native interface: {error}"),
                };
            }
        }
    }
    if let Some(requirement) = interop
        .adapters
        .iter()
        .find(|requirement| requirement.adapter == runmat_java::JAVA_ADAPTER_ID)
    {
        let Some(materialized) = materialized else {
            return ProgramExecutionResponse::Failure {
                message: "worker has no materialized Java artifacts".into(),
            };
        };
        for identity in &requirement.artifact_identities {
            if materialized.java_artifact(identity).is_none() {
                return ProgramExecutionResponse::Failure {
                    message: format!(
                        "worker has no materialized Java artifact for required identity {identity}"
                    ),
                };
            }
        }
        let required = requirement
            .artifact_identities
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let artifacts = materialized
            .java_artifacts()
            .iter()
            .filter(|artifact| required.contains(&artifact.identity.to_string()))
            .map(|artifact| (artifact.identity.clone(), artifact.path.clone()))
            .collect::<Vec<_>>();
        if let Err(error) = session.install_java_project_artifacts(&artifacts) {
            return ProgramExecutionResponse::Failure {
                message: format!("worker could not install Java artifacts: {error}"),
            };
        }
    }
    if let Err(error) = session.admit_interop_manifest(&interop) {
        return ProgramExecutionResponse::Failure {
            message: format!("worker rejected foreign interoperability requirements: {error}"),
        };
    }
    let runtime = session.execution_runtime_context(request.recipe.program_revision.clone());
    let response = runmat_vm::execute_program_request_with_context(request, runtime).await;
    if let Err(error) = session.shutdown_foreign_runtime().await {
        return ProgramExecutionResponse::Failure {
            message: format!("worker could not shut down foreign execution session: {error}"),
        };
    }
    response
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
        .map_err(|error| error.to_string());
    let shutdown = session
        .shutdown_foreign_runtime()
        .await
        .map_err(|error| format!("failed to shut down test execution session: {error}"));
    match (execution, shutdown) {
        (Ok(execution), Ok(())) => Ok(WorkerExecution {
            result: execution.result,
            events: execution.events,
            coverage: execution.coverage,
        }),
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::process::Command;

    use runmat_execution::value::{InlineValue, ValuePayload};
    use runmat_execution::Digest;
    use runmat_execution_artifact::{
        ExecutableForm, ExecutionBundleBuilder, LogicalObject, ObjectNamespace, ProgramArtifact,
        ProgramBuildRecipe, ProgramExecutionRequest, ProgramExecutionResponse, ProgramTarget,
        PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
    };
    use runmat_test::descriptor::TestSelector;
    use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
    use runmat_test::result::TerminalDisposition;
    use runmat_test_runner::worker::RunSubmission;
    use runmat_test_runner_execution::{decode_execution, TestAttemptWorkload};

    use super::{execute_host_program_request, execute_host_program_request_with_project};

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

    #[tokio::test]
    async fn host_rejects_a_required_native_interface_before_program_execution() {
        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        let unit = session
            .compile_executable_unit(
                runmat_core::ExecutableSource::new(
                    "runner-native-interface-test@1",
                    "native_interface.m",
                    "answer = 42;\n",
                ),
                None,
            )
            .await
            .unwrap();
        let interop = runmat_types::InteropManifest {
            schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![runmat_types::ForeignAdapterRequirement {
                adapter: runmat_native_ffi::NATIVE_FFI_ADAPTER_ID.into(),
                minimum_version: runmat_native_ffi::NATIVE_FFI_ADAPTER_VERSION,
                capabilities: runmat_types::CapabilitySet(BTreeSet::from([
                    runmat_types::CapabilityRequirement::NativeCode,
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                ])),
                artifact_identities: vec!["native-ffi:v1:missing-fixture".into()],
            }],
        };
        let envelope = unit
            .portable_envelope_for_with_interop(None, interop)
            .unwrap();
        let function = usize::try_from(envelope.manifest.identity.entrypoint_function.0).unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: unit.revision().program_revision.clone(),
            entrypoint: envelope.manifest.identity.entrypoint.clone(),
            outputs: runmat_execution::OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target: ProgramTarget::portable("runner-native-interface-test"),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::ExecutableUnitV3,
            envelope.canonical_bytes().unwrap(),
        )
        .unwrap();
        let response = execute_host_program_request(ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe,
            artifact,
            function,
            arguments: Vec::new(),
            requested_outputs: 1,
        })
        .await;
        let ProgramExecutionResponse::Failure { message } = response else {
            panic!("host executed a program before satisfying its native interface");
        };
        assert!(message.contains("missing-fixture"), "{message}");
    }

    #[tokio::test]
    async fn host_executes_a_packaged_java_artifact_from_the_exact_bundle() {
        let Some(java_home) =
            std::env::var_os("RUNMAT_TEST_JAVA_HOME").map(std::path::PathBuf::from)
        else {
            eprintln!("RUNMAT_TEST_JAVA_HOME is unset; remote Java execution was not requested");
            return;
        };
        let javac = java_home
            .join("bin")
            .join(if cfg!(windows) { "javac.exe" } else { "javac" });
        let jar = java_home
            .join("bin")
            .join(if cfg!(windows) { "jar.exe" } else { "jar" });
        if !javac.is_file() || !jar.is_file() {
            eprintln!("remote Java execution requires javac and jar");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let project_root = temp.path().join("project");
        let java_source = temp.path().join("java/fixture/packaged/Value.java");
        let java_classes = temp.path().join("classes");
        std::fs::create_dir_all(java_source.parent().unwrap()).unwrap();
        std::fs::create_dir_all(&java_classes).unwrap();
        std::fs::create_dir_all(project_root.join("src")).unwrap();
        std::fs::create_dir_all(project_root.join("lib")).unwrap();
        std::fs::write(
            &java_source,
            "package fixture.packaged; public final class Value { private Value() {} public static int read() { return 73; } }",
        )
        .unwrap();
        assert!(Command::new(&javac)
            .args(["-d", java_classes.to_str().unwrap()])
            .arg(&java_source)
            .status()
            .unwrap()
            .success());
        let jar_path = project_root.join("lib/fixture.jar");
        assert!(Command::new(&jar)
            .args(["--create", "--file"])
            .arg(&jar_path)
            .args(["-C", java_classes.to_str().unwrap(), "."])
            .status()
            .unwrap()
            .success());
        let source_text = "function value = main(); value = readPackaged() + readPackaged(); end\nfunction value = readPackaged(); value = fixture.packaged.Value.read(); end\n";
        std::fs::write(project_root.join("src/main.m"), source_text).unwrap();
        std::fs::write(
            project_root.join("runmat.toml"),
            "[package]\nname = \"remote-java-application\"\nversion = \"1.0.0\"\n[sources]\nroots = [\"src\"]\n[java-artifacts.fixture]\npath = \"lib/fixture.jar\"\n",
        )
        .unwrap();

        let project = runmat_package::build_frozen_project(
            &project_root.join("runmat.toml"),
            BTreeSet::from([runmat_package::HostCapability::Jvm]),
        )
        .unwrap();
        let project_revision = project.revision();
        let revision = runmat_execution::ProgramRevision::new(
            Digest::from_bytes(*project_revision.graph_digest.bytes()),
            Digest::from_bytes(*project_revision.source_revision.bytes()),
            runmat_core::program_environment(runmat_core::CompatMode::RunMat),
        )
        .unwrap();
        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        session
            .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
            .unwrap();
        let unit = session
            .compile_executable_unit(
                runmat_core::ExecutableSource::new("root", "src/main.m", source_text),
                Some(revision.clone()),
            )
            .await
            .unwrap();
        let jar_bytes = std::fs::read(&jar_path).unwrap();
        let java_identity = runmat_java::JavaArtifactIdentity::for_bytes(&jar_bytes);
        let interop = runmat_types::InteropManifest {
            schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
            foreign_types: Vec::new(),
            adapters: vec![runmat_types::ForeignAdapterRequirement {
                adapter: runmat_java::JAVA_ADAPTER_ID.into(),
                minimum_version: runmat_java::JAVA_ADAPTER_VERSION,
                capabilities: runmat_types::CapabilitySet(BTreeSet::from([
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                ])),
                artifact_identities: vec![java_identity.to_string()],
            }],
        };
        let envelope = unit
            .portable_envelope_for_with_interop(Some("main"), interop)
            .unwrap();
        let function = usize::try_from(envelope.manifest.identity.entrypoint_function.0).unwrap();
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision: revision.clone(),
            entrypoint: function.to_string(),
            outputs: runmat_execution::OutputContract {
                requested_outputs: 1,
            },
            execution_mode: "interpreter".into(),
            target: ProgramTarget::portable("remote-java-execution"),
            features: BTreeSet::new(),
            compile_options: BTreeSet::new(),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let java_object = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            "java/fixture.jar",
            runmat_java::JAVA_ARCHIVE_MEDIA_TYPE,
            jar_bytes,
        )
        .unwrap();
        let bundle = ExecutionBundleBuilder::native(&project, revision)
            .unwrap()
            .with_compiled_package_closure()
            .with_foreign_artifact(java_object)
            .unwrap()
            .with_materialized_program(
                recipe,
                ExecutableForm::ExecutableUnitV3,
                envelope.canonical_bytes().unwrap(),
            )
            .build()
            .unwrap();
        let materialized =
            crate::materialized_project::MaterializedProject::from_bundle(&bundle).unwrap();
        let recipe = bundle.manifest.recipes.first().cloned().unwrap();
        let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
        let response = execute_host_program_request_with_project(
            ProgramExecutionRequest {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
                recipe,
                artifact,
                function,
                arguments: Vec::new(),
                requested_outputs: 1,
            },
            Some(&materialized),
        )
        .await;
        assert_eq!(
            response,
            ProgramExecutionResponse::Success {
                value: ValuePayload::Inline(Box::new(InlineValue::I32(146))),
            }
        );
    }
}
