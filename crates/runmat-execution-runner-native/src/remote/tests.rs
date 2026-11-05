use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use runmat_execution::security::{
    EndpointIdentityEvidence, EndpointRecipientKey, ExecutionTrustTier,
};
use runmat_execution::{
    Digest, OutputContract, ProgramCallable, ProgramFunctionId, ProgramRevision,
};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::encryption::{
    encode_run_key_envelope, EncryptionPurpose, PortableExecutionEncryption, RunKeyMaterial,
};
use runmat_execution_artifact::{
    ExecutableForm, ExecutionBundle, ExecutionBundleBuilder, ProgramArtifact, ProgramBuildRecipe,
    ProgramExecutionDescriptor, ProgramExecutionInputs, ProgramExecutionObject,
    ProgramExecutionObjectClosure, ProgramExecutionObjectLocation, ProgramExecutionOutputs,
    ProgramExecutionRequest, ProgramExecutionResponse, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_execution_transport_native::control::{
    CheckpointDisposition, DriverArtifactDownload, DriverArtifactKind, DriverAuthority,
    DriverBootstrap, DriverControlPlane, DriverHeartbeat, DriverRunTarget, DriverRunTransition,
    StoreDriverArtifact, StoredDriverArtifact,
};
use runmat_execution_transport_native::identity::EndpointIdentityMaterial;
use runmat_execution_transport_native::{TransportError, TransportResult};
use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
use runmat_test::result::TerminalDisposition;
use runmat_test_runner::worker::RunSubmission;
use runmat_test_runner_execution::{decode_execution, TestAttemptWorkload};

use super::config::RemoteDriverConfig;
use super::crypto::{ciphertext_digest, open_object, project_revision_identity, seal_object};
use super::driver::run_remote_driver;

mod mixed_recovery;

fn fixture_org_id() -> String {
    runmat_server_client::ids::OrgId::from_uuid(uuid::Uuid::from_u128(1)).to_string()
}

fn fixture_driver_lease_id(fence: u64) -> String {
    runmat_server_client::ids::ExecutionDriverLeaseId::from_uuid(uuid::Uuid::from_u128(u128::from(
        fence,
    )))
    .to_string()
}

#[derive(Default)]
struct FakeState {
    downloads: BTreeMap<String, Vec<u8>>,
    artifact_kinds: BTreeMap<String, String>,
    stored: Vec<FakeStoredArtifact>,
    transitions: Vec<(DriverRunTarget, Option<String>)>,
    diagnostics: Vec<(Option<String>, Option<String>)>,
    usage: Vec<u64>,
    fail_store_at: Option<usize>,
}

struct FakeStoredArtifact {
    kind: DriverArtifactKind,
    ciphertext: Vec<u8>,
    checkpoint: Option<(u64, CheckpointDisposition)>,
}

struct FakeControl {
    bootstrap: DriverBootstrap,
    state: Mutex<FakeState>,
}

#[async_trait]
impl DriverControlPlane for FakeControl {
    async fn bootstrap(&self, _authority: &DriverAuthority) -> TransportResult<DriverBootstrap> {
        Ok(self.bootstrap.clone())
    }

    async fn heartbeat(
        &self,
        _authority: &DriverAuthority,
        _ttl_seconds: u64,
    ) -> TransportResult<DriverHeartbeat> {
        Ok(DriverHeartbeat {
            run_state: "running".into(),
            cancellation_requested: false,
            expires_at_millis: i64::MAX,
        })
    }

    async fn record_usage(
        &self,
        _authority: &DriverAuthority,
        source_sequence: u64,
    ) -> TransportResult<bool> {
        self.state.lock().unwrap().usage.push(source_sequence);
        Ok(true)
    }

    async fn download(&self, artifact: &DriverArtifactDownload) -> TransportResult<Vec<u8>> {
        let bytes = self
            .state
            .lock()
            .unwrap()
            .downloads
            .get(&artifact.artifact_id)
            .cloned()
            .ok_or_else(|| TransportError::Unavailable("missing fixture artifact".into()))?;
        if bytes.len() as u64 != artifact.ciphertext_size_bytes
            || ciphertext_digest(&bytes) != artifact.ciphertext_digest
        {
            return Err(TransportError::Integrity);
        }
        Ok(bytes)
    }

    async fn execution_artifact(
        &self,
        _authority: &DriverAuthority,
        artifact_id: &str,
    ) -> TransportResult<DriverArtifactDownload> {
        let state = self.state.lock().unwrap();
        let bytes = state
            .downloads
            .get(artifact_id)
            .ok_or_else(|| TransportError::Unavailable("missing fixture artifact".into()))?;
        let kind = state
            .artifact_kinds
            .get(artifact_id)
            .ok_or(TransportError::Integrity)?;
        Ok(DriverArtifactDownload {
            artifact_id: artifact_id.into(),
            kind: kind.clone(),
            ciphertext_digest: ciphertext_digest(bytes),
            ciphertext_size_bytes: bytes.len() as u64,
            media_type: "application/vnd.runmat.execution+ciphertext".into(),
            method: "GET".into(),
            url: format!("memory://{artifact_id}"),
            headers: BTreeMap::new(),
        })
    }

    async fn store_artifact(
        &self,
        _authority: &DriverAuthority,
        artifact: StoreDriverArtifact<'_>,
    ) -> TransportResult<StoredDriverArtifact> {
        if ciphertext_digest(artifact.ciphertext) != artifact.ciphertext_digest {
            return Err(TransportError::Integrity);
        }
        let mut state = self.state.lock().unwrap();
        let store_index = state.stored.len() + 1;
        if state.fail_store_at == Some(store_index) {
            return Err(TransportError::Unavailable(
                "injected artifact store failure".into(),
            ));
        }
        let artifact_id = format!("stored-{store_index}");
        let kind = format!("{:?}", artifact.kind).to_ascii_lowercase();
        state.stored.push(FakeStoredArtifact {
            kind: artifact.kind,
            ciphertext: artifact.ciphertext.to_vec(),
            checkpoint: artifact.checkpoint,
        });
        state
            .downloads
            .insert(artifact_id.clone(), artifact.ciphertext.to_vec());
        state
            .artifact_kinds
            .insert(artifact_id.clone(), kind.clone());
        Ok(StoredDriverArtifact {
            artifact_id,
            kind,
            ciphertext_size_bytes: artifact.ciphertext.len() as u64,
        })
    }

    async fn transition(
        &self,
        _authority: &DriverAuthority,
        transition: DriverRunTransition,
    ) -> TransportResult<()> {
        let mut state = self.state.lock().unwrap();
        state
            .diagnostics
            .push((transition.reason_code, transition.diagnostic_artifact_id));
        state
            .transitions
            .push((transition.target, transition.result_artifact_id));
        Ok(())
    }
}

#[tokio::test]
async fn remote_driver_executes_exact_encrypted_program_and_commits_once() {
    let run_id = "run-remote-conformance";
    let (program, _, bytes) = super::pool_tests::executable_bundle().await;
    let bundle =
        runmat_execution_artifact::archive::read_bundle(bytes.as_slice(), ArchiveLimits::default())
            .unwrap();
    let response = run_encrypted_remote_request_with_prior_sequence(
        run_id,
        bundle,
        None,
        Some(program.callable),
    )
    .await;
    assert!(
        matches!(response, ProgramExecutionResponse::Success { .. }),
        "{response:?}"
    );
}

#[tokio::test]
async fn replacement_driver_consumes_an_exact_resume_safe_checkpoint() {
    let run_id = "run-remote-replacement";
    let (program, _, bytes) = super::pool_tests::executable_bundle().await;
    let bundle =
        runmat_execution_artifact::archive::read_bundle(bytes.as_slice(), ArchiveLimits::default())
            .unwrap();
    let response = run_encrypted_remote_request_with_prior_sequence(
        run_id,
        bundle,
        Some(7),
        Some(program.callable),
    )
    .await;
    assert!(
        matches!(response, ProgramExecutionResponse::Success { .. }),
        "{response:?}"
    );
}

#[tokio::test]
async fn remote_driver_commits_verified_externalized_results() {
    let run_id = "run-externalized-result";
    let bytes = b"canonical analysis result object".to_vec();
    let reference = runmat_execution::value::ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: runmat_execution::identity::ValueId::derive(&[b"remote-result", &bytes]),
        logical_digest: Digest::sha256(&bytes),
        encoded_length: bytes.len() as u64,
        media_type: "application/vnd.runmat.analysis-object".into(),
        value_schema: "runmat.analysis-object.v1".into(),
        encryption_context: Digest::sha256(b"remote-result-encryption"),
        kind: runmat_execution::value::ValueRefKind::ResultObject,
        authorization_scope: run_id.into(),
        resident_fence: None,
    };
    let objects = super::object_transfer::RemoteObjectStore::new().unwrap();
    objects
        .register(ProgramExecutionObject {
            reference: reference.clone(),
            bytes,
        })
        .unwrap();
    let result = super::pool_execution::RemotePoolExecutionResult {
        console: None,
        cancelled: false,
        response: ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![runmat_execution::value::ValuePayload::Object(Box::new(
                reference.clone(),
            ))],
            result_objects: vec![reference.clone()],
        },
        objects: super::result_objects::RemoteResultObjects::Local(objects),
    };
    let control = FakeControl {
        bootstrap: DriverBootstrap {
            driver_accelerator_devices: Vec::new(),
            project_revision: "unused".into(),
            endpoint_fingerprint: "unused".into(),
            run_key_envelope: Vec::new(),
            artifacts: Vec::new(),
            checkpoint: None,
            cancellation_requested: false,
            driver_resources: driver_resource_request(),
            desired_worker_count: 0,
            worker_resources: resource_request(),
        },
        state: Mutex::new(FakeState::default()),
    };
    let run_key = RunKeyMaterial::from_entropy([41; 32]).unwrap();
    let config = RemoteDriverConfig {
        ready_v1: false,
        clock_parent: None,
        authority: DriverAuthority {
            server_url: "https://control.invalid".into(),
            run_id: run_id.into(),
            org_id: fixture_org_id(),
            project_id: "project-1".into(),
            allocation_lease_id: "allocation-1".into(),
            driver_lease_id: fixture_driver_lease_id(1),
            fencing_token: 1,
            credential: "driver-secret".into(),
        },
        endpoint_identity_file: PathBuf::from("unused"),
    };
    let (grant, _) = super::completion::RemoteCompletionGrant::admit(4096).unwrap();
    super::driver::commit_response(&control, &config, &run_key, grant.retain(result).unwrap())
        .await
        .unwrap();
    let state = control.state.lock().unwrap();
    assert_eq!(
        state.transitions,
        vec![(DriverRunTarget::Succeeded, Some("stored-2".into()))]
    );
    let object_plaintext = open_object(
        &run_key,
        &state.stored[0].ciphertext,
        run_id,
        EncryptionPurpose::Result,
    )
    .unwrap();
    assert_eq!(object_plaintext, b"canonical analysis result object");
    let plaintext = open_object(
        &run_key,
        &state.stored[1].ciphertext,
        run_id,
        EncryptionPurpose::Result,
    )
    .unwrap();
    let committed: ProgramExecutionOutputs = serde_json::from_slice(&plaintext).unwrap();
    committed.validate().unwrap();
    assert_eq!(committed.objects.len(), 1);
    assert_eq!(committed.objects[0].artifact_id, "stored-1");
}

#[tokio::test]
async fn cancelled_console_free_completion_does_not_publish_a_result() {
    let run_id = "run-cancelled-completion";
    let config = RemoteDriverConfig {
        ready_v1: false,
        clock_parent: None,
        authority: DriverAuthority {
            server_url: "https://control.invalid".into(),
            run_id: run_id.into(),
            org_id: fixture_org_id(),
            project_id: "project-1".into(),
            allocation_lease_id: "allocation-1".into(),
            driver_lease_id: fixture_driver_lease_id(1),
            fencing_token: 1,
            credential: "driver-secret".into(),
        },
        endpoint_identity_file: PathBuf::from("unused"),
    };
    let run_key = RunKeyMaterial::from_entropy([42; 32]).unwrap();
    let responses = [
        ProgramExecutionResponse::Failure {
            message: "remote attempt was cancelled".into(),
        },
        ProgramExecutionResponse::GraphSuccess {
            graph: runmat_execution::value::PayloadGraph {
                roots: Vec::new(),
                nodes: Vec::new(),
            },
        },
    ];
    for response in responses {
        let control = FakeControl {
            bootstrap: DriverBootstrap {
                driver_accelerator_devices: Vec::new(),
                project_revision: "unused".into(),
                endpoint_fingerprint: "unused".into(),
                run_key_envelope: Vec::new(),
                artifacts: Vec::new(),
                checkpoint: None,
                cancellation_requested: false,
                driver_resources: driver_resource_request(),
                desired_worker_count: 0,
                worker_resources: resource_request(),
            },
            state: Mutex::new(FakeState::default()),
        };
        let mut result =
            super::pool_execution::RemotePoolExecutionResult::without_objects(response);
        result.cancelled = true;
        let (grant, _) = super::completion::RemoteCompletionGrant::admit(4096).unwrap();
        super::driver::commit_response(&control, &config, &run_key, grant.retain(result).unwrap())
            .await
            .unwrap();
        let state = control.state.lock().unwrap();
        assert_eq!(state.transitions, vec![(DriverRunTarget::Cancelled, None)]);
        assert_eq!(
            state.diagnostics,
            vec![(Some("cancelled-during-execution".into()), None)]
        );
        assert!(state.stored.is_empty());
    }
}

#[tokio::test]
async fn replacement_restores_a_verified_run_scoped_object_closure() {
    let run_id = "run-recovery-closure";
    let run_key = RunKeyMaterial::from_entropy([57; 32]).unwrap();
    let first = result_object(run_id, b"first recovered object");
    let second = result_object(run_id, b"second recovered object");
    let first_location = ProgramExecutionObjectLocation {
        reference: first.reference.clone(),
        artifact_id: "recovered-object-1".into(),
    };
    let second_location = ProgramExecutionObjectLocation {
        reference: second.reference.clone(),
        artifact_id: "recovered-object-2".into(),
    };
    let closure = ProgramExecutionObjectClosure::new(
        second.reference.clone(),
        vec![second_location.clone(), first_location.clone()],
    )
    .unwrap();
    let mut catalog = result_object(run_id, &closure.canonical_bytes().unwrap());
    catalog.reference.media_type =
        runmat_execution_artifact::PROGRAM_EXECUTION_OBJECT_CLOSURE_MEDIA_TYPE.into();
    catalog.reference.value_schema =
        runmat_execution_artifact::PROGRAM_EXECUTION_OBJECT_CLOSURE_VALUE_SCHEMA.into();
    let catalog_location = ProgramExecutionObjectLocation {
        reference: catalog.reference.clone(),
        artifact_id: "recovery-catalog".into(),
    };
    let encrypted = [
        (&catalog_location, catalog.bytes.as_slice()),
        (&first_location, first.bytes.as_slice()),
        (&second_location, second.bytes.as_slice()),
    ]
    .into_iter()
    .map(|(location, bytes)| {
        (
            location.artifact_id.clone(),
            seal_object(&run_key, run_id, EncryptionPurpose::Result, bytes).unwrap(),
        )
    })
    .collect::<BTreeMap<_, _>>();
    let control = FakeControl {
        bootstrap: unused_bootstrap(),
        state: Mutex::new(FakeState {
            artifact_kinds: encrypted
                .keys()
                .map(|artifact_id| (artifact_id.clone(), "result".into()))
                .collect(),
            downloads: encrypted,
            ..Default::default()
        }),
    };
    let authority = DriverAuthority {
        server_url: "https://control.invalid".into(),
        run_id: run_id.into(),
        org_id: fixture_org_id(),
        project_id: "project-1".into(),
        allocation_lease_id: "allocation-1".into(),
        driver_lease_id: fixture_driver_lease_id(2),
        fencing_token: 2,
        credential: "credential".into(),
    };

    let restored = super::recovery::restore(&control, &run_key, &authority, &catalog_location)
        .await
        .unwrap();
    let mut expected = vec![first, second];
    expected.sort_by_key(|object| object.reference.id);
    assert_eq!(restored.root, second_location.reference);
    assert_eq!(restored.objects, expected);

    control
        .state
        .lock()
        .unwrap()
        .artifact_kinds
        .insert(second_location.artifact_id.clone(), "input".into());
    assert!(
        super::recovery::restore(&control, &run_key, &authority, &catalog_location)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn recovery_publication_commits_objects_then_catalog_then_checkpoint() {
    let run_id = "run-recovery-publication";
    let run_key = RunKeyMaterial::from_entropy([67; 32]).unwrap();
    let authority = DriverAuthority {
        server_url: "https://control.invalid".into(),
        run_id: run_id.into(),
        org_id: fixture_org_id(),
        project_id: "project-1".into(),
        allocation_lease_id: "allocation-1".into(),
        driver_lease_id: fixture_driver_lease_id(4),
        fencing_token: 4,
        credential: "credential".into(),
    };
    let revision = ProgramRevision::new(
        Digest::sha256("recovery-graph"),
        Digest::sha256("recovery-source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let mut recipe = recipe(revision);
    recipe.entrypoint = "analysis_workload".into();
    recipe.execution_mode = "recovery-publication-test".into();
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::AnalysisWorkload, vec![1]).unwrap();
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    let control = FakeControl {
        bootstrap: unused_bootstrap(),
        state: Mutex::new(FakeState::default()),
    };
    let first = result_object(run_id, b"first transaction object");
    let second = result_object(run_id, b"second transaction object");
    let third = result_object(run_id, b"third transaction object");
    let fourth = result_object(run_id, b"fourth transaction object");
    let mut publisher = super::recovery::RemoteRecoveryPublisher::new(
        &control,
        &run_key,
        &authority,
        "project-revision",
        &request,
        4,
        None,
    )
    .unwrap();

    publisher
        .publish_objects(
            second.reference.clone(),
            vec![second.clone(), first.clone()],
        )
        .await
        .unwrap();
    assert_eq!(publisher.checkpoint_sequence(), 5);
    let first_catalog = {
        let state = control.state.lock().unwrap();
        assert_eq!(state.stored.len(), 4);
        assert_eq!(
            state
                .stored
                .iter()
                .map(|stored| stored.kind)
                .collect::<Vec<_>>(),
            vec![
                DriverArtifactKind::Result,
                DriverArtifactKind::Result,
                DriverArtifactKind::Result,
                DriverArtifactKind::Checkpoint,
            ]
        );
        assert_eq!(
            state.stored[3].checkpoint,
            Some((5, CheckpointDisposition::ResumeSafe))
        );
        super::checkpoint::RemoteDriverCheckpoint::from_canonical_bytes(
            &open_object(
                &run_key,
                &state.stored[3].ciphertext,
                run_id,
                EncryptionPurpose::Checkpoint,
            )
            .unwrap(),
        )
        .unwrap()
        .closure_catalog
        .unwrap()
    };
    let restored = super::recovery::restore(&control, &run_key, &authority, &first_catalog)
        .await
        .unwrap();
    assert_eq!(restored.root, second.reference);
    assert_eq!(restored.objects.len(), 2);

    publisher
        .publish_objects(
            third.reference.clone(),
            vec![third.clone(), first.clone(), second.clone()],
        )
        .await
        .unwrap();
    assert_eq!(publisher.checkpoint_sequence(), 6);
    {
        let state = control.state.lock().unwrap();
        assert_eq!(state.stored.len(), 7);
        assert_eq!(state.stored[4].kind, DriverArtifactKind::Result);
        assert_eq!(state.stored[5].kind, DriverArtifactKind::Result);
        assert_eq!(state.stored[6].kind, DriverArtifactKind::Checkpoint);
        assert_eq!(
            state.stored[6].checkpoint,
            Some((6, CheckpointDisposition::ResumeSafe))
        );
    }

    control.state.lock().unwrap().fail_store_at = Some(10);
    assert!(publisher
        .publish_objects(
            fourth.reference.clone(),
            vec![first.clone(), second.clone(), third.clone(), fourth.clone()],
        )
        .await
        .is_err());
    assert_eq!(publisher.checkpoint_sequence(), 6);
    {
        let state = control.state.lock().unwrap();
        assert_eq!(state.stored.len(), 9);
        assert_eq!(state.stored[8].kind, DriverArtifactKind::Result);
        assert_eq!(state.stored[6].kind, DriverArtifactKind::Checkpoint);
    }

    control.state.lock().unwrap().fail_store_at = None;
    let mut clone = publisher.clone();
    let clone_retry = clone
        .publish_objects(fourth.reference.clone(), vec![fourth.clone()])
        .await
        .expect_err("every publisher clone must be fenced after an uncertain write");
    assert!(clone_retry.to_string().contains("replace the driver"));
    let retry = publisher
        .publish_objects(fourth.reference.clone(), vec![first, second, third, fourth])
        .await
        .expect_err("uncertain checkpoint write must fence this publisher generation");
    assert!(retry.to_string().contains("replace the driver"));
    assert_eq!(publisher.checkpoint_sequence(), 6);
    let state = control.state.lock().unwrap();
    assert_eq!(state.stored.len(), 9);
    assert_eq!(
        state.stored[6].checkpoint,
        Some((6, CheckpointDisposition::ResumeSafe))
    );
}

#[tokio::test]
async fn remote_driver_preserves_exact_test_result_events_and_coverage() {
    let test_source = "function tests = remoteTest()\n tests = functiontests(localfunctions);\nend\nfunction testRemote(testCase)\n testCase.verifyEqual(6 * 7, 42);\nend\n";
    let (_project_root, project, revision) = project_fixture(test_source);
    let snapshot = FrozenTestRunSnapshot::freeze(
        revision.graph_digest().to_string(),
        revision.source_digest().to_string(),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        Digest::sha256(b"test-config").to_string(),
        vec![SavedRunSource {
            owner_identity: "path:workspace".into(),
            relative_path: "src/remoteTest.m".into(),
            content: test_source.into(),
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
    let request = workload
        .program_request(runmat_types::InteropManifest::empty())
        .unwrap();
    let bundle = ExecutionBundleBuilder::native(&project, request.recipe.program_revision.clone())
        .unwrap()
        .with_materialized_program(
            request.recipe.clone(),
            ExecutableForm::TestAttemptV1,
            request.artifact.executable_bytes.clone(),
        )
        .build()
        .unwrap();
    let response = run_encrypted_remote_request("run-remote-test-conformance", bundle).await;
    let ProgramExecutionResponse::Success { outputs } = response else {
        panic!("remote driver rejected an exact test workload: {response:?}");
    };
    let [value] = outputs.as_slice() else {
        panic!("expected one test output")
    };
    let execution = decode_execution(value).unwrap();
    assert_eq!(
        execution.result.state.disposition,
        TerminalDisposition::Passed,
        "{execution:#?}"
    );
    assert!(!execution.events.is_empty());
    assert!(!execution.coverage.is_empty());
}

#[tokio::test]
async fn encrypted_registry_dependency_executes_without_worker_registry_credentials() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("root");
    let registry_root = temp.path().join("decrypted-registry-release");
    let test_source = "function tests = privatePackageTest()\n tests = functiontests(localfunctions);\nend\nfunction testPrivatePackage(testCase)\n testCase.verifyEqual(registry_answer(), 42);\nend\n";
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(registry_root.join("src")).unwrap();
    std::fs::write(root.join("src/privatePackageTest.m"), test_source).unwrap();
    std::fs::write(
        root.join("runmat.toml"),
        "[package]\nname = \"private-package-app\"\nversion = \"1.0.0\"\n[sources]\nroots = [\"src\"]\n[dependencies]\ntools = { package = \"acme/private-tools\", version = \"=1.0.0\" }\n",
    )
    .unwrap();
    std::fs::write(
        registry_root.join("src/registry_answer.m"),
        "function y = registry_answer(); y = 42; end\n",
    )
    .unwrap();
    std::fs::write(
        registry_root.join("runmat.toml"),
        "[package]\nname = \"private-tools\"\norganization = \"acme\"\nversion = \"1.0.0\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let provider = EncryptedRegistryFixture::new(registry_root);
    let resolved = runmat_package::resolve_project_async(
        &root.join("runmat.toml"),
        None,
        package_options(),
        &provider,
    )
    .await
    .unwrap();
    assert_eq!(
        provider.acquisitions.lock().unwrap().as_slice(),
        ["authorized"]
    );
    assert!(resolved
        .frozen
        .graph
        .packages
        .values()
        .any(|package| matches!(
            package.instance.source,
            runmat_package::SourceId::Registry(_)
        )));

    let project_revision = resolved.frozen.revision();
    let snapshot = FrozenTestRunSnapshot::freeze(
        project_revision.graph_digest.to_string(),
        project_revision.source_revision.to_string(),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        Digest::sha256(b"private-package-test-config").to_string(),
        vec![SavedRunSource {
            owner_identity: resolved.frozen.graph.root.to_string(),
            relative_path: "src/privatePackageTest.m".into(),
            content: test_source.into(),
        }],
        Vec::new(),
    )
    .unwrap();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(
            resolved.frozen.clone(),
        ))
        .await
        .unwrap();
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
    let request = workload
        .program_request(runmat_types::InteropManifest::empty())
        .unwrap();
    let bundle =
        ExecutionBundleBuilder::native(&resolved.frozen, request.recipe.program_revision.clone())
            .unwrap()
            .with_materialized_program(
                request.recipe.clone(),
                ExecutableForm::TestAttemptV1,
                request.artifact.executable_bytes.clone(),
            )
            .build()
            .unwrap();
    drop(provider);
    drop(resolved);

    let response =
        run_encrypted_remote_request("run-private-registry-package-conformance", bundle).await;
    let ProgramExecutionResponse::Success { outputs } = response else {
        panic!("credential-free remote private package execution failed: {response:?}");
    };
    let [value] = outputs.as_slice() else {
        panic!("expected one test output")
    };
    let execution = decode_execution(value).unwrap();
    assert_eq!(
        execution.result.state.disposition,
        TerminalDisposition::Passed,
        "{execution:#?}"
    );
}

struct EncryptedRegistryFixture {
    root: PathBuf,
    candidate: runmat_package::RegistryCandidateRecord,
    acquisitions: Mutex<Vec<&'static str>>,
}

impl EncryptedRegistryFixture {
    fn new(root: PathBuf) -> Self {
        let package =
            runmat_package::CanonicalPackageId::from_str("default:acme/private-tools").unwrap();
        let mut source = runmat_package::RegistrySourceId {
            registry_origin: runmat_package::RegistryOrigin::new("https://packages.runmat.test")
                .unwrap(),
            package,
            release: runmat_package::RegistryReleaseId::new("rel_0123456789abcdef0123456789abcdef")
                .unwrap(),
            version: runmat_package::PackageVersion::from_str("1.0.0").unwrap(),
            release_digest: runmat_package::ContentDigest::sha256("placeholder"),
            artifact_digest: runmat_package::ContentDigest::sha256("encrypted-artifact"),
            tree_digest: runmat_package::ContentDigest::sha256("private-package-tree"),
        };
        let encryption = runmat_package::EncryptedArtifactMetadata::new(
            1,
            b"deterministic-private-package-archive",
            [9; runmat_package::AES_256_GCM_NONCE_BYTE_LEN],
            &source,
        )
        .unwrap();
        let metadata = runmat_package::RegistryReleaseMetadata {
            singleton: false,
            runmat_requirement: None,
            dependencies: Vec::new(),
            features: Default::default(),
            required_capabilities: Vec::new(),
            optional_capabilities: Vec::new(),
            readme_digest: None,
            license: Some("Proprietary".into()),
            encryption: Some(encryption),
            supply_chain: None,
        };
        source.release_digest = metadata.compute_digest(&source).unwrap();
        metadata
            .encryption
            .as_ref()
            .unwrap()
            .validate(&source)
            .unwrap();
        Self {
            root,
            candidate: runmat_package::RegistryCandidateRecord {
                source,
                metadata,
                yanked: false,
            },
            acquisitions: Mutex::new(Vec::new()),
        }
    }
}

impl runmat_package::PackageSourceProvider for EncryptedRegistryFixture {
    fn acquire_git<'a>(
        &'a self,
        _plan: &'a runmat_package::GitAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<runmat_package::GitPackageMount, String>> + 'a>,
    > {
        Box::pin(async { Err("worker must not acquire Git sources".into()) })
    }

    fn registry_candidates<'a>(
        &'a self,
        _plan: &'a runmat_package::RegistryCandidatePlan,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<Vec<runmat_package::RegistryCandidateRecord>, String>,
                > + 'a,
        >,
    > {
        Box::pin(async { Ok(vec![self.candidate.clone()]) })
    }

    fn acquire_registry<'a>(
        &'a self,
        plan: &'a runmat_package::RegistryAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<runmat_package::RegistryPackageMount, String>>
                + 'a,
        >,
    > {
        Box::pin(async move {
            if plan.expected.as_ref() != Some(&self.candidate.source) {
                return Err("resolver requested a different private release".into());
            }
            self.acquisitions.lock().unwrap().push("authorized");
            Ok(runmat_package::RegistryPackageMount {
                source: self.candidate.source.clone(),
                root: self.root.clone(),
                metadata: Some(self.candidate.metadata.clone()),
            })
        })
    }
}

fn package_options() -> runmat_package::ProjectResolveOptions {
    runmat_package::ProjectResolveOptions {
        target: "x86_64-unknown-linux-gnu".into(),
        default_server_origin: "https://api.runmat.test".into(),
        default_registry_index: "https://packages.runmat.test".into(),
        groups: [runmat_package::DependencyGroup::Runtime]
            .into_iter()
            .collect(),
        root_features: Default::default(),
        host_capabilities: Default::default(),
        source_intent: runmat_package::SourceAcquisitionIntent::Execute,
        source_policy: Default::default(),
    }
}

async fn run_encrypted_remote_request(
    run_id: &str,
    bundle: ExecutionBundle,
) -> ProgramExecutionResponse {
    run_encrypted_remote_request_with_prior_sequence(run_id, bundle, None, None).await
}

async fn run_encrypted_remote_request_with_prior_sequence(
    run_id: &str,
    bundle: ExecutionBundle,
    prior_checkpoint_sequence: Option<u64>,
    callable: Option<ProgramCallable>,
) -> ProgramExecutionResponse {
    let bundle = if bundle.manifest.project_runtime_policy.is_none() {
        let policy =
            crate::remote::runtime_settings::fixture_policy(&bundle.manifest.program_revision);
        bundle.with_project_runtime_policy_object(policy).unwrap()
    } else {
        bundle
    };
    let revision = bundle.manifest.program_revision.clone();
    let recipe = bundle.manifest.recipes.first().cloned().unwrap();
    let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
    let mut bundle_bytes = Vec::new();
    write_bundle(&bundle, &mut bundle_bytes, ArchiveLimits::default()).unwrap();
    let descriptor_contract = ProgramExecutionDescriptor {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::new(
            bundle.manifest.portable_environment.clone(),
        )
        .unwrap(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: callable.unwrap_or_else(|| ProgramCallable::semantic(ProgramFunctionId(0), None)),
        requested_outputs: 1,
    };
    let inputs_contract = ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        context: runmat_execution::ProgramInvocationContext::Direct,
        arguments: Vec::new(),
        objects: Vec::new(),
    };
    let request =
        ProgramExecutionRequest::from_parts(descriptor_contract.clone(), inputs_contract.clone())
            .unwrap();
    let descriptor = serde_json::to_vec(&descriptor_contract).unwrap();
    let inputs = serde_json::to_vec(&inputs_contract).unwrap();

    let run_key = RunKeyMaterial::from_entropy([9; 32]).unwrap();
    let recipient_entropy = [7; 32];
    let (recipient, _) = PortableExecutionEncryption
        .recipient_from_entropy_with_derived_fingerprint(recipient_entropy, 1, u64::MAX - 1)
        .unwrap();
    let material = endpoint_material(recipient_entropy, &recipient, run_id);
    let temp = tempfile::tempdir().unwrap();
    let identity_path = temp.path().join("endpoint.json");
    std::fs::write(&identity_path, serde_json::to_vec(&material).unwrap()).unwrap();
    let envelope = PortableExecutionEncryption
        .seal_run_key_with_entropy([8; 32], &recipient, &run_key, run_id, 1)
        .unwrap();

    let encrypted = [
        ("bundle", EncryptionPurpose::Bundle, bundle_bytes),
        ("program", EncryptionPurpose::Program, descriptor),
        ("input", EncryptionPurpose::Input, inputs),
    ]
    .into_iter()
    .map(|(kind, purpose, plaintext)| {
        let bytes = seal_object(&run_key, run_id, purpose, &plaintext).unwrap();
        (kind, bytes)
    })
    .collect::<Vec<_>>();
    let downloads = encrypted
        .iter()
        .enumerate()
        .map(|(index, (kind, bytes))| download(index, kind, bytes))
        .collect::<Vec<_>>();
    let project_revision = project_revision_identity(&revision);
    let driver_fence = if prior_checkpoint_sequence.is_some() {
        2
    } else {
        1
    };
    let authority = DriverAuthority {
        server_url: "https://control.invalid".into(),
        run_id: run_id.into(),
        org_id: fixture_org_id(),
        project_id: "project-1".into(),
        allocation_lease_id: "allocation-1".into(),
        driver_lease_id: fixture_driver_lease_id(driver_fence),
        fencing_token: driver_fence,
        credential: "driver-secret".into(),
    };
    let prior_checkpoint = prior_checkpoint_sequence.map(|sequence| {
        let prior_authority = DriverAuthority {
            driver_lease_id: fixture_driver_lease_id(1),
            fencing_token: 1,
            ..authority.clone()
        };
        let checkpoint = super::checkpoint::RemoteDriverCheckpoint::new(
            sequence,
            &prior_authority,
            &project_revision,
            &request,
            super::checkpoint::RemoteDriverCheckpointPhase::ReadyToExecute,
            None,
        )
        .unwrap();
        let ciphertext = seal_object(
            &run_key,
            run_id,
            EncryptionPurpose::Checkpoint,
            &checkpoint.canonical_bytes().unwrap(),
        )
        .unwrap();
        (download(99, "checkpoint", &ciphertext), ciphertext)
    });
    let control = Arc::new(FakeControl {
        bootstrap: DriverBootstrap {
            project_revision,
            endpoint_fingerprint: recipient.fingerprint.clone(),
            run_key_envelope: encode_run_key_envelope(&envelope).unwrap(),
            artifacts: downloads.clone(),
            checkpoint: prior_checkpoint
                .as_ref()
                .map(|(download, _)| download.clone()),
            cancellation_requested: false,
            driver_resources: driver_resource_request(),
            driver_accelerator_devices: Vec::new(),
            desired_worker_count: 0,
            worker_resources: resource_request(),
        },
        state: Mutex::new(FakeState {
            downloads: downloads
                .iter()
                .zip(encrypted)
                .map(|(download, (_, bytes))| (download.artifact_id.clone(), bytes))
                .chain(
                    prior_checkpoint
                        .iter()
                        .map(|(download, bytes)| (download.artifact_id.clone(), bytes.clone())),
                )
                .collect(),
            ..Default::default()
        }),
    });
    let config = RemoteDriverConfig {
        authority,
        endpoint_identity_file: identity_path,
        ready_v1: false,
        clock_parent: None,
    };

    run_remote_driver(config, control.clone()).await.unwrap();

    let state = control.state.lock().unwrap();
    let diagnostics = state
        .diagnostics
        .iter()
        .map(|(reason, id)| {
            (
                reason,
                id.as_ref().map(|id| {
                    let encrypted = state
                        .downloads
                        .get(id)
                        .expect("original diagnostic artifact retained");
                    open_object(
                        &run_key,
                        encrypted,
                        run_id,
                        EncryptionPurpose::DetailedEvent,
                    )
                    .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                }),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        state.transitions,
        vec![
            (DriverRunTarget::Running, None),
            (DriverRunTarget::Succeeded, Some("stored-3".into()))
        ],
        "original remote driver diagnostics: {diagnostics:?}"
    );
    assert_eq!(state.usage.len(), 1);
    assert_eq!(state.usage[0], 1);
    assert_eq!(state.stored.len(), 3);
    assert_eq!(state.stored[0].kind, DriverArtifactKind::Checkpoint);
    assert_eq!(state.stored[1].kind, DriverArtifactKind::Checkpoint);
    assert_eq!(state.stored[2].kind, DriverArtifactKind::Result);
    let first_sequence = prior_checkpoint_sequence.map_or(1, |sequence| sequence + 1);
    assert_eq!(
        state.stored[0].checkpoint,
        Some((first_sequence, CheckpointDisposition::ResumeSafe))
    );
    assert_eq!(
        state.stored[1].checkpoint,
        Some((first_sequence + 1, CheckpointDisposition::Indeterminate))
    );
    let ready_checkpoint = super::checkpoint::RemoteDriverCheckpoint::from_canonical_bytes(
        &open_object(
            &run_key,
            &state.stored[0].ciphertext,
            run_id,
            EncryptionPurpose::Checkpoint,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        ready_checkpoint.phase,
        super::checkpoint::RemoteDriverCheckpointPhase::ReadyToExecute
    );
    let started_checkpoint = super::checkpoint::RemoteDriverCheckpoint::from_canonical_bytes(
        &open_object(
            &run_key,
            &state.stored[1].ciphertext,
            run_id,
            EncryptionPurpose::Checkpoint,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        started_checkpoint.phase,
        super::checkpoint::RemoteDriverCheckpointPhase::EffectsMayHaveStarted
    );
    let result = open_object(
        &run_key,
        &state.stored[2].ciphertext,
        run_id,
        EncryptionPurpose::Result,
    )
    .unwrap();
    let outputs = serde_json::from_slice::<ProgramExecutionOutputs>(&result).unwrap();
    outputs.validate().unwrap();
    outputs.response
}

fn resource_request() -> runmat_execution_transport_native::control::ResourceRequest {
    runmat_execution_transport_native::control::ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1 << 30,
        scratch_bytes: 1 << 30,
        accelerators: Vec::new(),
        maximum_wall_millis: 60_000,
    }
}

fn driver_resource_request() -> runmat_execution_transport_native::control::ResourceRequest {
    let mut resources = resource_request();
    resources.memory_bytes =
        runmat_execution::resource::remote_driver_memory_bytes(resources.memory_bytes).unwrap();
    resources
}

fn recipe(revision: ProgramRevision) -> ProgramBuildRecipe {
    ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: "0".into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("remote-test"),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    }
}

fn project_fixture(
    source: &str,
) -> (
    tempfile::TempDir,
    runmat_package::FrozenProject,
    ProgramRevision,
) {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join("src")).unwrap();
    std::fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"remote-fixture\"\nversion = \"1.0.0\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    std::fs::write(temp.path().join("src/remoteTest.m"), source).unwrap();
    let project =
        runmat_package::build_frozen_project(&temp.path().join("runmat.toml"), Default::default())
            .unwrap();
    let project_revision = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_revision.graph_digest.bytes()),
        Digest::from_bytes(*project_revision.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    (temp, project, revision)
}

fn endpoint_material(
    entropy: [u8; 32],
    recipient: &runmat_execution_artifact::encryption::ExecutionRecipientKey,
    run_id: &str,
) -> EndpointIdentityMaterial {
    EndpointIdentityMaterial::new(
        entropy,
        EndpointIdentityEvidence {
            schema_version: 1,
            org_id: fixture_org_id(),
            cluster_id: "cluster-1".into(),
            node_id: "node-1".into(),
            allocation_lease_id: "allocation-1".into(),
            fencing_token: 1,
            run_identity: run_id.into(),
            identity_public_key: vec![1; 32],
            identity_fingerprint: "identity".into(),
            recipient: EndpointRecipientKey {
                suite: "x25519-hkdf-sha256-aes128gcm-v1".into(),
                public_key: recipient.public_key.clone(),
                fingerprint: recipient.fingerprint.clone(),
                valid_after_unix_millis: recipient.valid_after_unix_millis,
                valid_before_unix_millis: recipient.valid_before_unix_millis,
            },
            direct_quic_endpoints: Vec::new(),
            trust_tier: ExecutionTrustTier::CustomerTrusted,
            attestation_class: None,
            attestation_evidence: None,
            issued_at_unix_millis: 1,
            expires_at_unix_millis: u64::MAX - 1,
            signature: vec![0; 64],
        },
    )
}

fn download(index: usize, kind: &str, bytes: &[u8]) -> DriverArtifactDownload {
    DriverArtifactDownload {
        artifact_id: format!("download-{index}"),
        kind: kind.into(),
        ciphertext_digest: ciphertext_digest(bytes),
        ciphertext_size_bytes: bytes.len() as u64,
        media_type: "application/vnd.runmat.execution+ciphertext".into(),
        method: "GET".into(),
        url: format!("memory://download-{index}"),
        headers: BTreeMap::new(),
    }
}

fn result_object(run_id: &str, bytes: &[u8]) -> ProgramExecutionObject {
    ProgramExecutionObject {
        reference: runmat_execution::value::ValueRef {
            schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
            id: runmat_execution::identity::ValueId::derive(&[b"recovery-object", bytes]),
            logical_digest: Digest::sha256(bytes),
            encoded_length: bytes.len() as u64,
            media_type: "application/vnd.runmat.analysis-object".into(),
            value_schema: "runmat.analysis-object.v1".into(),
            encryption_context: Digest::sha256(b"recovery-object-encryption"),
            kind: runmat_execution::value::ValueRefKind::ResultObject,
            authorization_scope: run_id.into(),
            resident_fence: None,
        },
        bytes: bytes.to_vec(),
    }
}

fn unused_bootstrap() -> DriverBootstrap {
    DriverBootstrap {
        driver_accelerator_devices: Vec::new(),
        project_revision: "unused".into(),
        endpoint_fingerprint: "unused".into(),
        run_key_envelope: Vec::new(),
        artifacts: Vec::new(),
        checkpoint: None,
        cancellation_requested: false,
        driver_resources: driver_resource_request(),
        desired_worker_count: 0,
        worker_resources: resource_request(),
    }
}

mod environment;
