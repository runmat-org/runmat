use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use runmat_execution::security::{
    EndpointIdentityEvidence, EndpointRecipientKey, ExecutionTrustTier,
};
use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::encryption::{
    encode_run_key_envelope, EncryptionPurpose, PortableExecutionEncryption, RunKeyMaterial,
};
use runmat_execution_artifact::{
    BuildResourceDeclaration, BundleManifest, ExecutableForm, ExecutionBundle, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionDescriptor, ProgramExecutionInputs,
    ProgramExecutionResponse, ProjectRevisionRecord, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_transport_native::control::{
    DriverArtifactDownload, DriverArtifactKind, DriverAuthority, DriverBootstrap,
    DriverControlPlane, DriverHeartbeat, DriverRunTarget, DriverRunTransition, StoreDriverArtifact,
    StoredDriverArtifact,
};
use runmat_execution_transport_native::identity::EndpointIdentityMaterial;
use runmat_execution_transport_native::{TransportError, TransportResult};

use super::config::RemoteDriverConfig;
use super::crypto::{ciphertext_digest, open_object, project_revision_identity, seal_object};
use super::driver::run_remote_driver;

#[derive(Default)]
struct FakeState {
    downloads: BTreeMap<String, Vec<u8>>,
    stored: Vec<(String, DriverArtifactKind, Vec<u8>)>,
    transitions: Vec<(DriverRunTarget, Option<String>)>,
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
    let revision = revision();
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
    let bundle = ExecutionBundle {
        manifest: BundleManifest {
            schema_version: 1,
            program_revision: revision.clone(),
            project_revision: ProjectRevisionRecord {
                graph_digest: *revision.graph_digest(),
                source_digest: *revision.source_digest(),
            },
            sources: Vec::new(),
            callables: Vec::new(),
            recipes: vec![recipe.clone()],
            artifacts: vec![artifact.clone()],
            requested_capabilities: Default::default(),
            resources: BuildResourceDeclaration {
                cpu_millicores: 1000,
                memory_bytes: 1024 * 1024,
                scratch_bytes: 1024 * 1024,
            },
            portable_environment: Vec::new(),
        },
        objects: Vec::new(),
    };
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
            desired_worker_count: 0,
            worker_resources: runmat_execution_transport_native::control::ResourceRequest {
                cpu_millicores: 1_000,
                memory_bytes: 1 << 30,
                scratch_bytes: 1 << 30,
                accelerator_count: 0,
                accelerator_class: None,
                accelerator_memory_bytes: 0,
                maximum_wall_millis: 60_000,
            },
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
    assert!(matches!(
        serde_json::from_slice::<ProgramExecutionResponse>(&result).unwrap(),
        ProgramExecutionResponse::Success { .. }
    ));
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

fn recipe(revision: ProgramRevision) -> ProgramBuildRecipe {
    ProgramBuildRecipe {
        schema_version: 1,
        program_revision: revision,
        entrypoint: "0".into(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target_profile: "remote-test".into(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    }
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
