use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use rand::RngCore as _;
use runmat_execution::security::{EndpointTrustPolicy, ExecutionTrustTier};
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::encryption::{
    encode_encrypted_run_object, encode_run_key_envelope, EncryptionContext, EncryptionPurpose,
    ExecutionRecipientKey, PortableExecutionEncryption, RunKeyMaterial, RunObjectEncryption,
};
use runmat_execution_artifact::{
    ExecutableForm, ExecutionBundleBuilder, ProgramBuildRecipe, ProgramExecutionDescriptor,
    ProgramExecutionInputs, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_server_client::execution::{endpoint_evidence, public_error, ExecutionArtifactUpload};
use runmat_server_client::public_api::types;
use sha2::{Digest as _, Sha256};
use uuid::Uuid;

use super::secret::{self, SavedRemoteRun};
use crate::cli::Cli;
use crate::commands::session::create_session;

const RETENTION_SECONDS: u64 = 7 * 24 * 60 * 60;
const MEDIA_TYPE: &str = "application/vnd.runmat.execution+ciphertext";
const ENCRYPTION_SUITE: &str = "hkdf-sha256-aes256gcm-v1";

#[allow(clippy::too_many_arguments)]
pub async fn submit(
    file: PathBuf,
    project: Option<Uuid>,
    cluster: String,
    queue: String,
    trust_identity: String,
    function: Option<String>,
    idempotency_key: Option<String>,
    workers: u32,
    detach: bool,
    json: bool,
    args: Vec<String>,
    cli: &Cli,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
) -> Result<()> {
    let file = std::fs::canonicalize(&file)
        .with_context(|| format!("resolve remote job source {}", file.display()))?;
    let resolved = crate::commands::package::resolve_for_source(&file, cli)
        .await?
        .context("remote execution requires a project runmat.toml with an exact frozen graph")?;
    let frozen = &resolved.resolved.frozen;
    let project_revision = frozen.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_revision.graph_digest.bytes()),
        Digest::from_bytes(*project_revision.source_revision.bytes()),
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let mut session = create_session(false, false, config, "create remote compilation session")?;
    session.install_project_handoff(runmat_package::FrozenProjectHandoff::new(frozen.clone()))?;
    let source_text = std::fs::read_to_string(&file)
        .with_context(|| format!("read remote job source {}", file.display()))?;
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new("root", file.to_string_lossy(), source_text),
            Some(revision.clone()),
        )
        .await?;
    let executable = unit
        .portable_executable(function.as_deref())
        .map_err(anyhow::Error::msg)?;
    if executable.kind == runmat_core::PortableExecutableKind::Script && !args.is_empty() {
        bail!("remote script jobs do not accept positional function arguments");
    }
    let form = match executable.kind {
        runmat_core::PortableExecutableKind::Function => ExecutableForm::InterpreterBytecodeV1,
        runmat_core::PortableExecutableKind::Script => ExecutableForm::InterpreterScriptV1,
    };
    let recipe = ProgramBuildRecipe {
        schema_version: 1,
        program_revision: revision.clone(),
        entrypoint: executable.entrypoint,
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target_profile: "portable-interpreter-v1".into(),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let bundle = ExecutionBundleBuilder::native(frozen, revision.clone())?
        .with_materialized_program(recipe, form, executable.bytes)
        .build()?;
    let recipe = bundle
        .manifest
        .recipes
        .first()
        .cloned()
        .context("compiled bundle has no recipe")?;
    let artifact = bundle
        .manifest
        .artifacts
        .first()
        .cloned()
        .context("compiled bundle has no program artifact")?;
    let descriptor = serde_json::to_vec(&ProgramExecutionDescriptor {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe,
        artifact,
        function: executable.function,
        requested_outputs: 1,
    })?;
    let inputs = serde_json::to_vec(&ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        arguments: args
            .into_iter()
            .map(|value| ValuePayload::Inline(Box::new(InlineValue::String(value))))
            .collect(),
    })?;
    let mut bundle_archive = Vec::new();
    write_bundle(&bundle, &mut bundle_archive, ArchiveLimits::default())?;

    let (client, server_url, project_id) = super::client(project).await?;
    let idempotency_key = idempotency_key.unwrap_or_else(|| Uuid::new_v4().to_string());
    let request_digest = request_digest(&idempotency_key, &revision, &cluster, &queue, workers);
    let run = client
        .api()
        .submit_run(
            &project_id,
            &idempotency_key,
            &types::SubmitRunRequest {
                cluster_id: cluster.clone(),
                queue,
                request_digest,
                project_revision: project_revision_identity(&revision),
                bundle_ciphertext_size_class: size_class(bundle_archive.len()),
                compatibility_fingerprints: HashMap::from([
                    (
                        "runtime".into(),
                        revision.environment().runtime_fingerprint.to_string(),
                    ),
                    (
                        "catalog".into(),
                        revision.environment().catalog_fingerprint.to_string(),
                    ),
                ]),
                resources: types::ResourceRequestBody {
                    cpu_millicores: 1_000,
                    memory_bytes: 1024 * 1024 * 1024,
                    scratch_bytes: 1024 * 1024 * 1024,
                    accelerator_count: 0,
                    accelerator_memory_bytes: 0,
                    accelerator_class: None,
                    maximum_wall_millis: 60 * 60 * 1_000,
                },
                worker_count: Some(
                    i32::try_from(workers).context("worker count exceeds API range")?,
                ),
                worker_resources: (workers > 0).then_some(types::ResourceRequestBody {
                    cpu_millicores: 1_000,
                    memory_bytes: 1024 * 1024 * 1024,
                    scratch_bytes: 1024 * 1024 * 1024,
                    accelerator_count: 0,
                    accelerator_memory_bytes: 0,
                    accelerator_class: None,
                    maximum_wall_millis: 60 * 60 * 1_000,
                }),
            },
        )
        .await
        .map_err(public_error)?
        .into_inner();
    let admission = wait_for_admission(&client, &project_id, &run.id).await?;
    let identity = admission
        .endpoint_identity
        .context("allocation did not publish endpoint identity")?;
    let evidence = endpoint_evidence(&identity.evidence)?;
    verify_evidence(
        &evidence,
        &run,
        &admission.allocation_lease_id,
        &trust_identity,
    )?;
    let evidence_digest = admission
        .evidence_digest
        .context("admission omitted its signed evidence digest")?;
    let recipient =
        ExecutionRecipientKey::from_verified_endpoint(&evidence, &trust_policy(&trust_identity)?)?;
    client
        .api()
        .confirm_run_admission(
            &project_id,
            &run.id,
            &types::ConfirmEndpointIdentityRequest {
                endpoint_fingerprint: recipient.fingerprint.clone(),
                evidence_digest,
            },
        )
        .await
        .map_err(public_error)?;

    let run_key = random_run_key()?;
    let endpoint_fingerprint = format!("submitter-{}", Uuid::new_v4());
    let bundle = seal(
        &run_key,
        &run.id,
        EncryptionPurpose::Bundle,
        &bundle_archive,
    )?;
    let program = seal(&run_key, &run.id, EncryptionPurpose::Program, &descriptor)?;
    let input = seal(&run_key, &run.id, EncryptionPurpose::Input, &inputs)?;
    let bundle_id = upload(
        &client,
        &project_id,
        &endpoint_fingerprint,
        types::ArtifactKindRequest::Bundle,
        bundle,
    )
    .await?;
    let program_id = upload(
        &client,
        &project_id,
        &endpoint_fingerprint,
        types::ArtifactKindRequest::Program,
        program,
    )
    .await?;
    let input_id = upload(
        &client,
        &project_id,
        &endpoint_fingerprint,
        types::ArtifactKindRequest::Input,
        input,
    )
    .await?;
    let envelope = PortableExecutionEncryption.seal_run_key_with_entropy(
        random_entropy()?,
        &recipient,
        &run_key,
        &run.id,
        1,
    )?;
    let committed = client
        .api()
        .commit_run_content(
            &project_id,
            &run.id,
            &types::CommitRunContentRequest {
                bundle_artifact_id: bundle_id,
                program_artifact_id: program_id,
                input_artifact_id: Some(input_id),
                endpoint_fingerprint: recipient.fingerprint.clone(),
                run_key_envelopes: vec![types::RunKeyEnvelopeBody {
                    recipient_role: types::RunKeyRecipientRoleBody::Driver,
                    envelope: base64::engine::general_purpose::URL_SAFE_NO_PAD
                        .encode(encode_run_key_envelope(&envelope)?),
                }],
            },
        )
        .await
        .map_err(public_error)?
        .into_inner();
    secret::save(&SavedRemoteRun::new(
        committed.id.clone(),
        server_url,
        project_id,
        endpoint_fingerprint,
        &run_key,
    ))?;
    if json {
        println!("{}", serde_json::to_string(&committed)?);
    } else {
        println!(
            "{} {} ({})",
            crate::presentation::stdout().success("Job"),
            committed.id,
            committed.state
        );
    }
    if !detach {
        super::attach::attach(&committed.id, false, json).await?;
    }
    Ok(())
}

async fn wait_for_admission(
    client: &runmat_server_client::execution::ExecutionClient,
    project_id: &str,
    run_id: &str,
) -> Result<types::RunAdmissionResponse> {
    let mut last_error = None;
    for _ in 0..120 {
        match client.api().get_run_admission(project_id, run_id).await {
            Ok(response) => {
                let admission = response.into_inner();
                if admission.endpoint_identity.is_some() {
                    return Ok(admission);
                }
            }
            Err(error) => last_error = Some(error.to_string()),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    bail!(
        "timed out waiting for an execution endpoint{}",
        last_error
            .map(|error| format!(": {error}"))
            .unwrap_or_default()
    )
}

fn verify_evidence(
    evidence: &runmat_execution::security::EndpointIdentityEvidence,
    run: &types::RunResponse,
    allocation_id: &str,
    trusted_identity: &str,
) -> Result<()> {
    trust_policy(trusted_identity)?.verify(evidence)?;
    if evidence.org_id != run.org_id
        || evidence.cluster_id != run.cluster_id
        || evidence.allocation_lease_id != allocation_id
        || evidence.run_identity != run.id
    {
        bail!("signed endpoint evidence does not match the admitted run");
    }
    Ok(())
}

fn trust_policy(identity: &str) -> Result<EndpointTrustPolicy> {
    if identity.is_empty() {
        bail!("a pinned endpoint identity fingerprint is required");
    }
    Ok(EndpointTrustPolicy {
        permitted_tiers: [
            ExecutionTrustTier::CustomerTrusted,
            ExecutionTrustTier::HostedOrdinary,
            ExecutionTrustTier::AttestedConfidential,
        ]
        .into_iter()
        .collect(),
        trusted_identity_fingerprints: [identity.to_string()].into_iter().collect(),
        allowed_attestation_classes: BTreeSet::new(),
        require_pinned_identity: true,
        now_unix_millis: SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis()
            .try_into()
            .context("system clock exceeds execution timestamp range")?,
        maximum_clock_skew_millis: 30_000,
    })
}

fn random_run_key() -> Result<RunKeyMaterial> {
    RunKeyMaterial::from_entropy(random_entropy()?).map_err(anyhow::Error::from)
}

fn random_entropy() -> Result<[u8; 32]> {
    let mut bytes = [0_u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    if bytes.iter().all(|byte| *byte == 0) {
        bail!("operating system random source returned invalid entropy");
    }
    Ok(bytes)
}

fn seal(
    key: &RunKeyMaterial,
    run_id: &str,
    purpose: EncryptionPurpose,
    plaintext: &[u8],
) -> Result<Vec<u8>> {
    let object = RunObjectEncryption.seal_with_entropy(
        key,
        random_entropy()?,
        EncryptionContext {
            schema_version: 1,
            run_identity: run_id.into(),
            purpose,
            object_digest: Digest::sha256(plaintext),
            task_identity: None,
            attempt_identity: None,
            chunk_index: 0,
            total_length: plaintext.len() as u64,
            key_epoch: 1,
        },
        plaintext,
    )?;
    encode_encrypted_run_object(&object).map_err(anyhow::Error::from)
}

async fn upload(
    client: &runmat_server_client::execution::ExecutionClient,
    project_id: &str,
    endpoint_fingerprint: &str,
    kind: types::ArtifactKindRequest,
    ciphertext: Vec<u8>,
) -> Result<String> {
    Ok(client
        .upload_artifact(
            project_id,
            endpoint_fingerprint,
            ExecutionArtifactUpload {
                kind,
                ciphertext,
                media_type: MEDIA_TYPE.into(),
                encryption_suite: ENCRYPTION_SUITE.into(),
                retain_for_seconds: RETENTION_SECONDS,
            },
        )
        .await?
        .id)
}

fn request_digest(
    idempotency_key: &str,
    revision: &ProgramRevision,
    cluster: &str,
    queue: &str,
    workers: u32,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"runmat-remote-run-request-v1\0");
    digest.update(idempotency_key.as_bytes());
    digest.update(revision.canonical_identity().as_bytes());
    digest.update(cluster.as_bytes());
    digest.update(queue.as_bytes());
    digest.update(workers.to_be_bytes());
    format!("{:x}", digest.finalize())
}

fn project_revision_identity(revision: &ProgramRevision) -> String {
    format!("sha256:{}", hex(revision.source_digest().bytes()))
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn size_class(bytes: usize) -> types::ArtifactSizeClassRequest {
    match bytes {
        0..=1_048_576 => types::ArtifactSizeClassRequest::Small,
        1_048_577..=67_108_864 => types::ArtifactSizeClassRequest::Medium,
        67_108_865..=1_073_741_824 => types::ArtifactSizeClassRequest::Large,
        _ => types::ArtifactSizeClassRequest::ExtraLarge,
    }
}
