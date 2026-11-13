use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use rand::RngCore as _;
use runmat_execution::security::EndpointTrustPolicy;
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::encryption::{
    encode_encrypted_run_object, encode_run_key_envelope, EncryptionContext, EncryptionPurpose,
    ExecutionRecipientKey, PortableExecutionEncryption, RunKeyMaterial, RunObjectEncryption,
    TRANSFER_FRAME_ENCRYPTION_SUITE,
};
use runmat_execution_artifact::{
    ExecutableForm, ExecutionBundleBuilder, ProgramBuildRecipe, ProgramExecutionDescriptor,
    ProgramExecutionInputs, ProgramExecutionObject, ProgramExecutionObjectLocation,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_server_client::execution::RemoteEndpointTrust;
use runmat_server_client::execution::{endpoint_evidence, public_error, ExecutionArtifactUpload};
use runmat_server_client::public_api::types;
use sha2::{Digest as _, Sha256};
use uuid::Uuid;

use super::secret::{self, SavedRemoteRun};
use crate::cli::Cli;
use crate::commands::session::create_session;

const RETENTION_SECONDS: u64 = 7 * 24 * 60 * 60;
const MEDIA_TYPE: &str = "application/vnd.runmat.execution+ciphertext";
const DRIVER_MEMORY_BYTES: i64 = 1024 * 1024 * 1024;

pub(crate) struct PreparedRemoteExecution {
    pub revision: ProgramRevision,
    pub bundle_archive: Vec<u8>,
    pub descriptor: Vec<u8>,
    pub inputs: ProgramExecutionInputs,
    pub objects: Vec<ProgramExecutionObject>,
}

pub(crate) struct RemoteSubmissionOptions {
    pub console: Option<runmat_execution::console::ConsoleLimits>,
    pub project: Option<Uuid>,
    pub cluster: String,
    pub queue: String,
    pub trust_identity: RemoteEndpointTrust,
    pub idempotency_key: Option<String>,
    pub workers: u32,
    pub accelerator: Option<runmat_execution::resource::AcceleratorRequest>,
    pub on_run_created: Option<Arc<dyn Fn(String) + Send + Sync>>,
    pub output_intent: super::secret::SavedOutputIntent,
}

pub(crate) struct JobSubmission {
    pub file: PathBuf,
    pub project: Option<Uuid>,
    pub cluster: String,
    pub queue: String,
    pub trust_identity: RemoteEndpointTrust,
    pub function: Option<String>,
    pub idempotency_key: Option<String>,
    pub workers: u32,
    pub detach: bool,
    pub json: bool,
    pub args: Vec<String>,
}

pub async fn submit(
    submission: JobSubmission,
    cli: &Cli,
    _config: &runmat_config::runtime::RunMatRuntimeConfig,
) -> Result<()> {
    let JobSubmission {
        file,
        project,
        cluster,
        queue,
        trust_identity,
        function,
        idempotency_key,
        workers,
        detach,
        json,
        args,
    } = submission;
    let file = std::fs::canonicalize(&file)
        .with_context(|| format!("resolve remote job source {}", file.display()))?;
    let resolved = crate::commands::package::resolve_for_source(&file, cli)
        .await?
        .context("remote execution requires a project runmat.toml with an exact frozen graph")?;
    let frozen = &resolved.resolved.frozen;
    let project_policy =
        runmat_config::runtime::ClusterRuntimePolicy::from_project_file(&frozen.manifest_path)?;
    let project_runtime = project_policy.runtime_config()?;
    let config = &project_runtime;
    let runtime_policy = runmat_core::cluster_settings::capture_project_policy(&project_policy)
        .map_err(anyhow::Error::msg)?;
    let project_revision = frozen.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_revision.graph_digest.bytes()),
        Digest::from_bytes(*project_revision.source_revision.bytes()),
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let mut session = create_session(false, false, config, "create remote compilation session")?;
    session.apply_cluster_runtime_config(config)?;
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(frozen.clone()))
        .await?;
    let unit = session
        .compile_executable_unit(
            crate::commands::package::executable_source(Some(frozen), &file).await?,
            Some(revision.clone()),
        )
        .await?;
    let python_required = unit.requires_python_runtime();
    let foreign_artifacts =
        crate::commands::package::prepare_foreign_artifacts(frozen, config, python_required)?;
    let foreign_policy =
        runmat_core::foreign_artifacts::capture_foreign_policy_object(&config.foreign)
            .map_err(anyhow::Error::msg)?;
    let interop = foreign_artifacts.interop.clone();
    let foreign_accelerators = foreign_artifacts.accelerators.clone();
    let executable = unit
        .portable_envelope_for_with_interop(function.as_deref(), interop.clone())
        .map_err(anyhow::Error::msg)?;
    let accelerators = runmat_execution::resource::merge_accelerator_requirements(
        runmat_execution::resource::accelerator_requirements_for_capabilities(
            &executable.manifest.capabilities,
        )?
        .into_iter()
        .chain(foreign_accelerators),
    )?;
    let identity = &executable.manifest.identity;
    if identity.entrypoint_kind == runmat_execution::ExecutableEntrypointKind::Script
        && !args.is_empty()
    {
        bail!("remote script jobs do not accept positional function arguments");
    }
    let requested_outputs = match identity.entrypoint_kind {
        runmat_execution::ExecutableEntrypointKind::Script => 0,
        runmat_execution::ExecutableEntrypointKind::Function => 1,
        runmat_execution::ExecutableEntrypointKind::DeclarationOnly => {
            bail!("declaration-only executables cannot be submitted as jobs")
        }
    };
    let callable_function = identity.entrypoint_function.ok_or_else(|| {
        anyhow::anyhow!("declaration-only executables cannot be submitted as callable jobs")
    })?;
    let executable_bytes = executable.canonical_bytes()?;
    let form = ExecutableForm::ExecutableUnitV4;
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision.clone(),
        entrypoint: identity.entrypoint.clone(),
        outputs: OutputContract {
            requested_outputs,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("portable-executable-unit-v3"),
        interop,
        foreign_policy: Some(foreign_policy.descriptor.clone()),
        accelerators,
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let mut builder = ExecutionBundleBuilder::native(frozen, revision.clone())?
        .with_compiled_package_closure()
        .with_materialized_program(recipe, form, executable_bytes)
        .with_project_runtime_policy_object(runtime_policy)?
        .with_foreign_policy_object(foreign_policy)?;
    for object in foreign_artifacts.objects {
        builder = builder.with_foreign_object(object)?;
    }
    for closure in foreign_artifacts.closures {
        builder = builder.with_foreign_artifact_closure(closure)?;
    }
    let bundle = builder.build()?;
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
        console: Some(super::console::TERMINAL_CONSOLE_LIMITS),
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: runmat_execution::ProgramCallable::semantic(callable_function, None),
        requested_outputs,
    })?;
    let inputs = ProgramExecutionInputs {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        context: runmat_execution::ProgramInvocationContext::Direct,
        arguments: args
            .into_iter()
            .map(|value| ValuePayload::Inline(Box::new(InlineValue::String(value.into()))))
            .collect(),
        objects: Vec::new(),
    };
    inputs.validate()?;
    let mut bundle_archive = Vec::new();
    write_bundle(&bundle, &mut bundle_archive, ArchiveLimits::default())?;

    let committed = submit_prepared(
        PreparedRemoteExecution {
            revision,
            bundle_archive,
            descriptor,
            inputs,
            objects: Vec::new(),
        },
        RemoteSubmissionOptions {
            console: Some(super::console::TERMINAL_CONSOLE_LIMITS),
            project,
            cluster,
            queue,
            trust_identity,
            idempotency_key,
            workers,
            accelerator: None,
            on_run_created: None,
            output_intent: secret::SavedOutputIntent::Generic,
        },
    )
    .await?;
    if detach {
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
        return Ok(());
    }
    if !json {
        eprintln!(
            "{} {}",
            crate::presentation::stderr().muted("Attached to remote job"),
            committed.id
        );
    }
    super::attach::attach(&committed.id, false, json).await
}

pub(crate) async fn submit_prepared(
    prepared: PreparedRemoteExecution,
    options: RemoteSubmissionOptions,
) -> Result<types::RunResponse> {
    let program_descriptor: ProgramExecutionDescriptor =
        serde_json::from_slice(&prepared.descriptor)
            .context("decode prepared program descriptor")?;
    program_descriptor
        .validate()
        .context("validate prepared program descriptor")?;
    if options.workers == 0 && !program_descriptor.recipe.accelerators.is_empty() {
        bail!("remote accelerator workloads require at least one worker");
    }
    let accelerators = runmat_execution::resource::merge_accelerator_requirements(
        program_descriptor
            .recipe
            .accelerators
            .clone()
            .into_iter()
            .chain(options.accelerator.clone()),
    )?;
    let revision = prepared.revision.clone();
    let bundle_archive_size = prepared.bundle_archive.len();
    let mut prepared = Some(prepared);
    submit_scoped_with_accelerators(
        revision,
        bundle_archive_size,
        move |_| {
            prepared
                .take()
                .context("remote execution content was prepared more than once")
        },
        options,
        accelerators,
    )
    .await
}

/// Submit content whose access-bound references can only be finalized after the Server assigns
/// the run identity. The declared revision and bundle size class are admission inputs; the factory
/// must preserve both when it binds the encrypted content to `run_id`.
pub(crate) async fn submit_scoped<F>(
    revision: ProgramRevision,
    bundle_archive_size: usize,
    prepare: F,
    options: RemoteSubmissionOptions,
) -> Result<types::RunResponse>
where
    F: FnOnce(&str) -> Result<PreparedRemoteExecution>,
{
    let accelerators = options.accelerator.clone().into_iter().collect();
    submit_scoped_with_accelerators(
        revision,
        bundle_archive_size,
        prepare,
        options,
        accelerators,
    )
    .await
}

async fn submit_scoped_with_accelerators<F>(
    revision: ProgramRevision,
    bundle_archive_size: usize,
    prepare: F,
    mut options: RemoteSubmissionOptions,
    accelerators: Vec<runmat_execution::resource::AcceleratorRequest>,
) -> Result<types::RunResponse>
where
    F: FnOnce(&str) -> Result<PreparedRemoteExecution>,
{
    for accelerator in &accelerators {
        accelerator.validate().map_err(anyhow::Error::msg)?;
    }
    let digest_accelerators = accelerators.clone();
    let accelerator_bodies = accelerators
        .into_iter()
        .map(runmat_server_client::execution::accelerator_request_to_api)
        .collect::<Result<Vec<_>>>()?;
    // Explicit accelerator placement runs on the driver when the job has no workers;
    // otherwise accelerator work is placed on the workers.
    let (driver_accelerators, worker_accelerators) = if options.workers == 0 {
        (accelerator_bodies, Vec::new())
    } else {
        (Vec::new(), accelerator_bodies)
    };
    let (client, server_url, project_id) = super::client(options.project).await?;
    let cluster_id = runmat_server_client::execution::project_cluster_by_name(
        client.api(),
        &project_id,
        &options.cluster,
    )
    .await?;
    let recovery_recipient = client
        .api()
        .get_project_execution_policy(project_id.as_str())
        .await
        .map_err(public_error)?
        .into_inner()
        .recovery_recipient
        .map(recovery_recipient)
        .transpose()?;
    let recovery_fingerprint = recovery_recipient
        .as_ref()
        .map(|recipient| recipient.fingerprint.clone());
    let idempotency_key = options
        .idempotency_key
        .take()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    // Ask the Server to authorize the child and its terminal control envelope
    // together. The captured numerical policy and console budget keep the
    // original child limit after the driver partitions the parent grant.
    let driver_memory_bytes = i64::try_from(
        runmat_execution::resource::remote_driver_memory_bytes(DRIVER_MEMORY_BYTES as u64)?,
    )
    .context("driver parent memory exceeds API range")?;
    let request_digest = request_digest(
        &idempotency_key,
        &revision,
        cluster_id.as_str(),
        &options,
        &digest_accelerators,
        recovery_fingerprint.as_deref(),
        driver_memory_bytes,
    );
    let run = client
        .api()
        .submit_run(
            project_id.as_str(),
            &idempotency_key,
            &types::SubmitRunRequest {
                cluster_id: cluster_id.to_string(),
                queue: options.queue,
                request_digest,
                recovery_recipient_fingerprint: recovery_fingerprint,
                project_revision: project_revision_identity(&revision),
                bundle_ciphertext_size_class: size_class(bundle_archive_size),
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
                    memory_bytes: driver_memory_bytes,
                    scratch_bytes: 1024 * 1024 * 1024,
                    accelerators: driver_accelerators,
                    maximum_wall_millis: 60 * 60 * 1_000,
                },
                worker_count: Some(
                    i32::try_from(options.workers).context("worker count exceeds API range")?,
                ),
                worker_resources: (options.workers > 0).then_some(types::ResourceRequestBody {
                    cpu_millicores: 1_000,
                    memory_bytes: 1024 * 1024 * 1024,
                    scratch_bytes: 1024 * 1024 * 1024,
                    accelerators: worker_accelerators,
                    maximum_wall_millis: 60 * 60 * 1_000,
                }),
            },
        )
        .await
        .map_err(public_error)?
        .into_inner();
    if let Some(observer) = &options.on_run_created {
        observer(run.id.clone());
    }
    let prepared = prepare(&run.id)?;
    let console = {
        let program: runmat_execution_artifact::RemoteProgramObject =
            serde_json::from_slice(&prepared.descriptor)
                .context("decode run-scoped program object")?;
        program.validate()?;
        match &program {
            runmat_execution_artifact::RemoteProgramObject::Direct(descriptor) => {
                descriptor.console
            }
            runmat_execution_artifact::RemoteProgramObject::Graph(request) => request
                .invocation_controls
                .as_ref()
                .and_then(|controls| controls.console),
        }
    };
    anyhow::ensure!(
        console == options.console,
        "run-scoped console controls changed after submission admission"
    );
    if prepared.revision != revision
        || size_class(prepared.bundle_archive.len()) != size_class(bundle_archive_size)
    {
        bail!("run-scoped execution content changed its admitted revision or bundle size class");
    }
    let admission = wait_for_admission(&client, project_id.as_str(), &run.id).await?;
    let identity = admission
        .endpoint_identity
        .context("allocation did not publish endpoint identity")?;
    let evidence = endpoint_evidence(&identity.evidence)?;
    if admission.run_id != run.id
        || identity.node_id != evidence.node_id
        || identity.lease_id != admission.allocation_lease_id
        || u64::try_from(admission.fencing_token)? != evidence.fencing_token
    {
        bail!("endpoint evidence does not match the authenticated allocation");
    }
    verify_evidence(
        &evidence,
        &run,
        &admission.allocation_lease_id,
        &options.trust_identity,
    )?;
    let evidence_digest = admission
        .evidence_digest
        .context("admission omitted its signed evidence digest")?;
    let recipient = ExecutionRecipientKey::from_verified_endpoint(
        &evidence,
        &trust_policy(&options.trust_identity)?,
    )?;
    client
        .api()
        .confirm_run_admission(
            project_id.as_str(),
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
        &prepared.bundle_archive,
    )?;
    let program = seal(
        &run_key,
        &run.id,
        EncryptionPurpose::Program,
        &prepared.descriptor,
    )?;
    let bundle_id = upload(
        &client,
        project_id.as_str(),
        &endpoint_fingerprint,
        types::ArtifactKindRequest::Bundle,
        bundle,
    )
    .await?;
    let program_id = upload(
        &client,
        project_id.as_str(),
        &endpoint_fingerprint,
        types::ArtifactKindRequest::Program,
        program,
    )
    .await?;
    let mut object_locations = Vec::with_capacity(prepared.objects.len());
    for object in prepared.objects {
        object.validate()?;
        if object.reference.authorization_scope != run.id
            || object.reference.kind != runmat_execution::value::ValueRefKind::DriverObject
        {
            bail!("prepared input object is outside the admitted run authority");
        }
        let ciphertext = seal(&run_key, &run.id, EncryptionPurpose::Input, &object.bytes)?;
        let artifact_id = upload(
            &client,
            project_id.as_str(),
            &endpoint_fingerprint,
            types::ArtifactKindRequest::Input,
            ciphertext,
        )
        .await?;
        object_locations.push(ProgramExecutionObjectLocation {
            reference: object.reference,
            artifact_id,
        });
    }
    let inputs = ProgramExecutionInputs {
        schema_version: prepared.inputs.schema_version,
        context: prepared.inputs.context,
        arguments: prepared.inputs.arguments,
        objects: object_locations,
    };
    inputs.validate()?;
    let encoded_inputs = serde_json::to_vec(&inputs)?;
    let input = seal(&run_key, &run.id, EncryptionPurpose::Input, &encoded_inputs)?;
    let input_id = upload(
        &client,
        project_id.as_str(),
        &endpoint_fingerprint,
        types::ArtifactKindRequest::Input,
        input,
    )
    .await?;
    let driver_envelope = PortableExecutionEncryption.seal_run_key_with_entropy(
        random_entropy()?,
        &recipient,
        &run_key,
        &run.id,
        1,
    )?;
    let mut envelopes = vec![types::RunKeyEnvelopeBody {
        recipient_role: types::RunKeyRecipientRoleBody::Driver,
        envelope: base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(encode_run_key_envelope(&driver_envelope)?),
    }];
    if let Some(recovery_recipient) = recovery_recipient {
        let recovery_envelope = PortableExecutionEncryption.seal_run_key_with_entropy(
            random_entropy()?,
            &recovery_recipient,
            &run_key,
            &run.id,
            1,
        )?;
        envelopes.push(types::RunKeyEnvelopeBody {
            recipient_role: types::RunKeyRecipientRoleBody::Recovery,
            envelope: base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(encode_run_key_envelope(&recovery_envelope)?),
        });
    }
    let committed = client
        .api()
        .commit_run_content(
            project_id.as_str(),
            &run.id,
            &types::CommitRunContentRequest {
                bundle_artifact_id: bundle_id,
                program_artifact_id: program_id,
                input_artifact_id: Some(input_id),
                endpoint_fingerprint: recipient.fingerprint.clone(),
                run_key_envelopes: envelopes,
            },
        )
        .await
        .map_err(public_error)?
        .into_inner();
    let mut saved = SavedRemoteRun::new(
        committed.id.clone(),
        server_url,
        project_id.to_string(),
        endpoint_fingerprint,
        &run_key,
        options.output_intent,
    );
    saved.console = console;
    saved.console_budget = saved.console.map(|_| {
        runmat_execution_artifact::ProgramExecutionConsoleBudget::for_driver_allocation(
            DRIVER_MEMORY_BYTES as u64,
        )
    });
    secret::save(&saved)?;
    Ok(committed)
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
    trusted_identity: &RemoteEndpointTrust,
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

fn trust_policy(identity: &RemoteEndpointTrust) -> Result<EndpointTrustPolicy> {
    identity.policy_at(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis()
            .try_into()
            .context("system clock exceeds execution timestamp range")?,
    )
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
                encryption_suite: TRANSFER_FRAME_ENCRYPTION_SUITE.into(),
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
    options: &RemoteSubmissionOptions,
    accelerators: &[runmat_execution::resource::AcceleratorRequest],
    recovery_recipient_fingerprint: Option<&str>,
    driver_memory_bytes: i64,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"runmat-remote-run-request-v2\0");
    digest.update(idempotency_key.as_bytes());
    digest.update(revision.canonical_identity().as_bytes());
    digest.update(cluster.as_bytes());
    digest.update(options.queue.as_bytes());
    digest.update(options.workers.to_be_bytes());
    digest.update(b"\0console\0");
    match options.console {
        None => digest.update([0]),
        Some(limits) => {
            digest.update([1]);
            digest.update(limits.max_records.to_be_bytes());
            digest.update(limits.max_text_bytes.to_be_bytes());
        }
    }
    digest.update(driver_memory_bytes.to_be_bytes());
    for accelerator in accelerators {
        digest.update(b"\0accelerator\0");
        digest.update(
            serde_json::to_vec(accelerator).expect("accelerator requests serialize canonically"),
        );
    }
    if let Some(fingerprint) = recovery_recipient_fingerprint {
        digest.update(b"\0recovery-recipient\0");
        digest.update(fingerprint.as_bytes());
    }
    hex::encode(digest.finalize())
}

fn recovery_recipient(
    value: types::ExecutionRecoveryRecipientBody,
) -> Result<ExecutionRecipientKey> {
    const RECOVERY_SUITE: &str = "x25519-hkdf-sha256-aes128gcm-v1";
    if value.suite != RECOVERY_SUITE {
        bail!("Server returned an unsupported recovery recipient suite");
    }
    let public_key = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(value.public_key)
        .context("Server returned malformed recovery recipient key material")?;
    if runmat_execution::security::recipient_fingerprint(&public_key) != value.fingerprint {
        bail!("Server returned a recovery recipient with a mismatched fingerprint");
    }
    let recipient = ExecutionRecipientKey {
        suite:
            runmat_execution_artifact::encryption::ExecutionHpkeSuite::X25519HkdfSha256Aes128GcmV1,
        public_key,
        fingerprint: value.fingerprint,
        valid_after_unix_millis: u64::try_from(value.valid_after_unix_millis)
            .context("recovery recipient validity starts before the Unix epoch")?,
        valid_before_unix_millis: u64::try_from(value.valid_before_unix_millis)
            .context("recovery recipient validity ends before the Unix epoch")?,
        custodian_uri: value.custodian_uri,
    };
    recipient.validate()?;
    let now: u64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()
        .context("system clock exceeds execution timestamp range")?;
    if now < recipient.valid_after_unix_millis || now >= recipient.valid_before_unix_millis {
        bail!("organization recovery recipient is outside its configured validity window");
    }
    Ok(recipient)
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
