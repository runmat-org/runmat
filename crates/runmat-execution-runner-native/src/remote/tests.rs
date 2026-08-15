use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use runmat_execution::security::{
    EndpointIdentityEvidence, EndpointRecipientKey, ExecutionTrustTier,
};
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::encryption::{
    encode_run_key_envelope, EncryptionPurpose, PortableExecutionEncryption, RunKeyMaterial,
};
use runmat_execution_artifact::{
    ExecutableForm, ExecutionBundle, ExecutionBundleBuilder, ProgramArtifact, ProgramBuildRecipe,
    ProgramExecutionDescriptor, ProgramExecutionInputs, ProgramExecutionResponse,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_transport_native::control::{
    DriverArtifactDownload, DriverArtifactKind, DriverAuthority, DriverBootstrap,
    DriverControlPlane, DriverHeartbeat, DriverRunTarget, DriverRunTransition, StoreDriverArtifact,
    StoredDriverArtifact,
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

#[derive(Default)]
struct FakeState {
    downloads: BTreeMap<String, Vec<u8>>,
    stored: Vec<(String, DriverArtifactKind, Vec<u8>)>,
    transitions: Vec<(DriverRunTarget, Option<String>)>,
    usage: Vec<u64>,
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

    async fn store_artifact(
        &self,
        _authority: &DriverAuthority,
        artifact: StoreDriverArtifact<'_>,
    ) -> TransportResult<StoredDriverArtifact> {
        if ciphertext_digest(artifact.ciphertext) != artifact.ciphertext_digest {
            return Err(TransportError::Integrity);
        }
        let mut state = self.state.lock().unwrap();
        let artifact_id = format!("stored-{}", state.stored.len() + 1);
        state.stored.push((
            artifact_id.clone(),
            artifact.kind,
            artifact.ciphertext.to_vec(),
        ));
        Ok(StoredDriverArtifact {
            artifact_id,
            kind: format!("{:?}", artifact.kind).to_ascii_lowercase(),
            ciphertext_size_bytes: artifact.ciphertext.len() as u64,
        })
    }

    async fn transition(
        &self,
        _authority: &DriverAuthority,
        transition: DriverRunTransition,
    ) -> TransportResult<()> {
        self.state
            .lock()
            .unwrap()
            .transitions
            .push((transition.target, transition.result_artifact_id));
        Ok(())
    }
}

#[tokio::test]
async fn remote_driver_executes_exact_encrypted_program_and_commits_once() {
    let run_id = "run-remote-conformance";
    let (_project_root, project, revision) =
        project_fixture("function y = remote_answer(); y = 42; end\n");
    let mut registry = runmat_vm::FunctionRegistry::default();
    registry.insert_replacing_name(runmat_vm::FunctionBytecode {
        display_name: "remote_answer".into(),
        instructions: vec![
            runmat_vm::Instr::LoadConst(42.0),
            runmat_vm::Instr::StoreVar(0),
            runmat_vm::Instr::Return,
        ],
        var_count: 1,
        output_slots: vec![0],
        ..Default::default()
    });
    let recipe = recipe(revision.clone());
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV1,
        serde_json::to_vec(&registry).unwrap(),
    )
    .unwrap();
    let bundle = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_materialized_program(
            recipe.clone(),
            ExecutableForm::InterpreterBytecodeV1,
            artifact.executable_bytes.clone(),
        )
        .build()
        .unwrap();
    let response = run_encrypted_remote_request(run_id, bundle).await;
    assert!(matches!(response, ProgramExecutionResponse::Success { .. }));
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
    let request = workload.program_request().unwrap();
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
    let ProgramExecutionResponse::Success { value } = response else {
        panic!("remote driver rejected an exact test workload: {response:?}");
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
    let request = workload.program_request().unwrap();
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
    let ProgramExecutionResponse::Success { value } = response else {
        panic!("credential-free remote private package execution failed: {response:?}");
    };
    let execution = decode_execution(&value).unwrap();
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
    let revision = bundle.manifest.program_revision.clone();
    let recipe = bundle.manifest.recipes.first().cloned().unwrap();
    let artifact = bundle.manifest.artifacts.first().cloned().unwrap();
    let mut bundle_bytes = Vec::new();
    write_bundle(&bundle, &mut bundle_bytes, ArchiveLimits::default()).unwrap();
    let descriptor = serde_json::to_vec(&ProgramExecutionDescriptor {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe,
        artifact,
        function: 0,
        requested_outputs: 1,
    })
    .unwrap();
    let inputs = serde_json::to_vec(&ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        arguments: Vec::new(),
    })
    .unwrap();

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
    let control = Arc::new(FakeControl {
        bootstrap: DriverBootstrap {
            project_revision: project_revision_identity(&revision),
            endpoint_fingerprint: recipient.fingerprint.clone(),
            run_key_envelope: encode_run_key_envelope(&envelope).unwrap(),
            artifacts: downloads.clone(),
            checkpoint: None,
            cancellation_requested: false,
            driver_resources: resource_request(),
            desired_worker_count: 0,
            worker_resources: resource_request(),
        },
        state: Mutex::new(FakeState {
            downloads: downloads
                .iter()
                .zip(encrypted)
                .map(|(download, (_, bytes))| (download.artifact_id.clone(), bytes))
                .collect(),
            ..Default::default()
        }),
    });
    let config = RemoteDriverConfig {
        authority: DriverAuthority {
            server_url: "https://control.invalid".into(),
            run_id: run_id.into(),
            org_id: "org-1".into(),
            project_id: "project-1".into(),
            allocation_lease_id: "allocation-1".into(),
            driver_lease_id: "driver-1".into(),
            fencing_token: 1,
            credential: "driver-secret".into(),
        },
        endpoint_identity_file: identity_path,
    };

    run_remote_driver(config, control.clone()).await.unwrap();

    let state = control.state.lock().unwrap();
    assert_eq!(
        state.transitions,
        vec![
            (DriverRunTarget::Running, None),
            (DriverRunTarget::Succeeded, Some("stored-3".into()))
        ]
    );
    assert_eq!(state.usage.len(), 1);
    assert_eq!(state.usage[0], 1);
    assert_eq!(state.stored.len(), 3);
    assert_eq!(state.stored[0].1, DriverArtifactKind::Checkpoint);
    assert_eq!(state.stored[1].1, DriverArtifactKind::Checkpoint);
    assert_eq!(state.stored[2].1, DriverArtifactKind::Result);
    let result = open_object(
        &run_key,
        &state.stored[2].2,
        run_id,
        EncryptionPurpose::Result,
    )
    .unwrap();
    serde_json::from_slice::<ProgramExecutionResponse>(&result).unwrap()
}

fn resource_request() -> runmat_execution_transport_native::control::ResourceRequest {
    runmat_execution_transport_native::control::ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1 << 30,
        scratch_bytes: 1 << 30,
        accelerator_count: 0,
        accelerator_class: None,
        accelerator_memory_bytes: 0,
        maximum_wall_millis: 60_000,
    }
}

fn recipe(revision: ProgramRevision) -> ProgramBuildRecipe {
    ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: "0".into(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("remote-test"),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
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
            org_id: "org-1".into(),
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
