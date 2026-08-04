use std::sync::Arc;
use std::time::Duration;

use runmat_execution_artifact::archive::{read_bundle, ArchiveLimits};
use runmat_execution_artifact::encryption::{
    decode_run_key_envelope, EncryptionPurpose, PortableExecutionEncryption,
};
use runmat_execution_artifact::{
    ProgramExecutionDescriptor, ProgramExecutionInputs, ProgramExecutionRequest,
    ProgramExecutionResponse,
};
use runmat_execution_transport_native::control::{
    CheckpointDisposition, DriverArtifactDownload, DriverArtifactKind, DriverControlPlane,
    DriverRunTarget, DriverRunTransition, HttpDriverControlPlane, StoreDriverArtifact,
};
use runmat_execution_transport_native::identity::EndpointIdentityMaterial;

use super::config::RemoteDriverConfig;
use super::crypto::{ciphertext_digest, open_object, project_revision_identity, seal_object};
use crate::{NativeExecutionError, NativeExecutionResult};

const MEDIA_TYPE: &str = "application/vnd.runmat.execution+ciphertext";
const RETAIN_FOR_SECONDS: u64 = 7 * 24 * 60 * 60;

pub async fn run_remote_driver_from_env() -> NativeExecutionResult<()> {
    let config = RemoteDriverConfig::from_env()?;
    let control = Arc::new(
        HttpDriverControlPlane::new(config.authority.server_url.clone()).map_err(protocol)?,
    );
    run_remote_driver(config, control).await
}

pub(super) async fn run_remote_driver(
    config: RemoteDriverConfig,
    control: Arc<dyn DriverControlPlane>,
) -> NativeExecutionResult<()> {
    let material_bytes = std::fs::read(&config.endpoint_identity_file).map_err(protocol)?;
    let material: EndpointIdentityMaterial =
        serde_json::from_slice(&material_bytes).map_err(protocol)?;
    if material.evidence.allocation_lease_id != config.authority.allocation_lease_id
        || material.evidence.run_identity != config.authority.run_id
        || material.evidence.recipient.fingerprint.is_empty()
    {
        return Err(protocol(
            "endpoint identity does not match driver authority",
        ));
    }
    let (_, private_key) = material.recipient_private_key().map_err(protocol)?;
    let bootstrap = control
        .bootstrap(&config.authority)
        .await
        .map_err(protocol)?;
    if bootstrap.endpoint_fingerprint != material.evidence.recipient.fingerprint {
        return Err(protocol("driver endpoint fingerprint was substituted"));
    }
    let envelope =
        decode_run_key_envelope(&bootstrap.run_key_envelope, 64 * 1024).map_err(protocol)?;
    let run_key = PortableExecutionEncryption
        .open_run_key(
            &private_key,
            &envelope,
            &bootstrap.endpoint_fingerprint,
            &config.authority.run_id,
            1,
        )
        .map_err(protocol)?;
    if bootstrap.cancellation_requested {
        transition(
            control.as_ref(),
            &config,
            DriverRunTarget::Cancelled,
            Some("cancelled-before-start"),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    let bundle = required_artifact(&bootstrap.artifacts, "bundle")?;
    let program = required_artifact(&bootstrap.artifacts, "program")?;
    let input = required_artifact(&bootstrap.artifacts, "input")?;
    let bundle_archive = open_download(
        control.as_ref(),
        &run_key,
        bundle,
        &config.authority.run_id,
        EncryptionPurpose::Bundle,
    )
    .await?;
    let bundle =
        read_bundle(bundle_archive.as_slice(), ArchiveLimits::default()).map_err(protocol)?;
    let descriptor: ProgramExecutionDescriptor = serde_json::from_slice(
        &open_download(
            control.as_ref(),
            &run_key,
            program,
            &config.authority.run_id,
            EncryptionPurpose::Program,
        )
        .await?,
    )
    .map_err(protocol)?;
    let inputs: ProgramExecutionInputs = serde_json::from_slice(
        &open_download(
            control.as_ref(),
            &run_key,
            input,
            &config.authority.run_id,
            EncryptionPurpose::Input,
        )
        .await?,
    )
    .map_err(protocol)?;
    descriptor.validate().map_err(protocol)?;
    inputs.validate().map_err(protocol)?;
    bundle.validate().map_err(protocol)?;
    if bundle.manifest.program_revision != descriptor.recipe.program_revision
        || !bundle
            .manifest
            .recipes
            .iter()
            .any(|recipe| recipe == &descriptor.recipe)
        || !bundle
            .manifest
            .artifacts
            .iter()
            .any(|artifact| artifact == &descriptor.artifact)
        || project_revision_identity(&bundle.manifest.program_revision)
            != bootstrap.project_revision
    {
        return Err(protocol(
            "bundle, program, and admitted project revision do not converge",
        ));
    }
    let request = ProgramExecutionRequest::from_parts(descriptor, inputs).map_err(protocol)?;
    let materialized_project = if bootstrap.desired_worker_count == 0 {
        Some(crate::materialized_project::MaterializedProject::from_bundle(&bundle)?)
    } else {
        None
    };

    store_checkpoint(
        control.as_ref(),
        &config,
        &run_key,
        1,
        CheckpointDisposition::ResumeSafe,
        b"ready",
    )
    .await?;
    transition(
        control.as_ref(),
        &config,
        DriverRunTarget::Running,
        None,
        None,
        None,
    )
    .await?;
    store_checkpoint(
        control.as_ref(),
        &config,
        &run_key,
        2,
        CheckpointDisposition::Indeterminate,
        b"execution-started",
    )
    .await?;

    let mut usage_sequence = 0_u64;
    let (cancellation_sender, cancellation_receiver) = tokio::sync::watch::channel(false);
    let uses_remote_pool = bootstrap.desired_worker_count > 0;
    let execution = async {
        if bootstrap.desired_worker_count == 0 {
            Ok(
                super::pool_execution::RemotePoolExecutionOutcome::Completed(
                    crate::execute_host_program_request_with_project(
                        request,
                        materialized_project
                            .as_ref()
                            .map(|project| project.handoff()),
                    )
                    .await,
                ),
            )
        } else {
            super::pool_execution::execute(
                control.as_ref(),
                &config.authority,
                &run_key,
                bundle_archive,
                request,
                bootstrap.desired_worker_count,
                bootstrap.worker_resources,
                cancellation_receiver,
            )
            .await
        }
    };
    tokio::pin!(execution);
    let response = loop {
        tokio::select! {
            response = &mut execution => {
                match response? {
                    super::pool_execution::RemotePoolExecutionOutcome::Completed(response) => {
                        break response;
                    }
                    super::pool_execution::RemotePoolExecutionOutcome::Cancelled => {
                        report_usage(
                            control.as_ref(),
                            &config,
                            &mut usage_sequence,
                        ).await?;
                        transition(
                            control.as_ref(),
                            &config,
                            DriverRunTarget::Indeterminate,
                            Some("cancelled-during-execution"),
                            None,
                            None,
                        ).await?;
                        return Ok(());
                    }
                    super::pool_execution::RemotePoolExecutionOutcome::Indeterminate(message) => {
                        let diagnostic = store_encrypted(
                            control.as_ref(),
                            &config,
                            &run_key,
                            DriverArtifactKind::Diagnostic,
                            EncryptionPurpose::DetailedEvent,
                            message.as_bytes(),
                            None,
                        )
                        .await?;
                        report_usage(
                            control.as_ref(),
                            &config,
                            &mut usage_sequence,
                        ).await?;
                        transition(
                            control.as_ref(),
                            &config,
                            DriverRunTarget::Indeterminate,
                            Some("worker-lost"),
                            None,
                            Some(diagnostic),
                        ).await?;
                        return Ok(());
                    }
                }
            },
            _ = tokio::time::sleep(Duration::from_secs(5)) => {
                report_usage(
                    control.as_ref(),
                    &config,
                    &mut usage_sequence,
                ).await?;
                let heartbeat = control
                    .heartbeat(&config.authority, 60)
                    .await
                    .map_err(protocol)?;
                if heartbeat.cancellation_requested {
                    if uses_remote_pool {
                        cancellation_sender.send_replace(true);
                        continue;
                    }
                    report_usage(
                        control.as_ref(),
                        &config,
                        &mut usage_sequence,
                    ).await?;
                    transition(
                        control.as_ref(),
                        &config,
                        DriverRunTarget::Indeterminate,
                        Some("cancelled-during-execution"),
                        None,
                        None,
                    ).await?;
                    return Ok(());
                }
            }
        }
    };
    report_usage(control.as_ref(), &config, &mut usage_sequence).await?;
    commit_response(control.as_ref(), &config, &run_key, response).await
}

async fn report_usage(
    control: &dyn DriverControlPlane,
    config: &RemoteDriverConfig,
    sequence: &mut u64,
) -> NativeExecutionResult<()> {
    *sequence = sequence
        .checked_add(1)
        .ok_or_else(|| protocol("remote usage sequence overflow"))?;
    control
        .record_usage(&config.authority, *sequence)
        .await
        .map_err(protocol)?;
    Ok(())
}

async fn commit_response(
    control: &dyn DriverControlPlane,
    config: &RemoteDriverConfig,
    run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    response: ProgramExecutionResponse,
) -> NativeExecutionResult<()> {
    match response {
        ProgramExecutionResponse::Success { value } => {
            let plaintext = serde_json::to_vec(&ProgramExecutionResponse::Success { value })
                .map_err(protocol)?;
            let result = store_encrypted(
                control,
                config,
                run_key,
                DriverArtifactKind::Result,
                EncryptionPurpose::Result,
                &plaintext,
                None,
            )
            .await?;
            transition(
                control,
                config,
                DriverRunTarget::Succeeded,
                None,
                Some(result),
                None,
            )
            .await
        }
        ProgramExecutionResponse::Failure { message } => {
            let diagnostic = store_encrypted(
                control,
                config,
                run_key,
                DriverArtifactKind::Diagnostic,
                EncryptionPurpose::DetailedEvent,
                message.as_bytes(),
                None,
            )
            .await?;
            transition(
                control,
                config,
                DriverRunTarget::Failed,
                Some("execution-failed"),
                None,
                Some(diagnostic),
            )
            .await
        }
    }
}

async fn open_download(
    control: &dyn DriverControlPlane,
    run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    artifact: &DriverArtifactDownload,
    run_id: &str,
    purpose: EncryptionPurpose,
) -> NativeExecutionResult<Vec<u8>> {
    let ciphertext = control.download(artifact).await.map_err(protocol)?;
    open_object(run_key, &ciphertext, run_id, purpose)
}

async fn store_checkpoint(
    control: &dyn DriverControlPlane,
    config: &RemoteDriverConfig,
    run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    sequence: u64,
    disposition: CheckpointDisposition,
    plaintext: &[u8],
) -> NativeExecutionResult<String> {
    store_encrypted(
        control,
        config,
        run_key,
        DriverArtifactKind::Checkpoint,
        EncryptionPurpose::Checkpoint,
        plaintext,
        Some((sequence, disposition)),
    )
    .await
}

async fn store_encrypted(
    control: &dyn DriverControlPlane,
    config: &RemoteDriverConfig,
    run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    kind: DriverArtifactKind,
    purpose: EncryptionPurpose,
    plaintext: &[u8],
    checkpoint: Option<(u64, CheckpointDisposition)>,
) -> NativeExecutionResult<String> {
    let ciphertext = seal_object(run_key, &config.authority.run_id, purpose, plaintext)?;
    let artifact = control
        .store_artifact(
            &config.authority,
            StoreDriverArtifact {
                kind,
                ciphertext_digest: ciphertext_digest(&ciphertext),
                ciphertext: &ciphertext,
                media_type: MEDIA_TYPE.into(),
                retain_for_seconds: RETAIN_FOR_SECONDS,
                checkpoint,
            },
        )
        .await
        .map_err(protocol)?;
    Ok(artifact.artifact_id)
}

async fn transition(
    control: &dyn DriverControlPlane,
    config: &RemoteDriverConfig,
    target: DriverRunTarget,
    reason_code: Option<&str>,
    result_artifact_id: Option<String>,
    diagnostic_artifact_id: Option<String>,
) -> NativeExecutionResult<()> {
    control
        .transition(
            &config.authority,
            DriverRunTransition {
                target,
                reason_code: reason_code.map(str::to_string),
                result_artifact_id,
                diagnostic_artifact_id,
            },
        )
        .await
        .map_err(protocol)
}

fn required_artifact<'a>(
    artifacts: &'a [DriverArtifactDownload],
    kind: &str,
) -> NativeExecutionResult<&'a DriverArtifactDownload> {
    let matches = artifacts
        .iter()
        .filter(|artifact| artifact.kind == kind)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(protocol(format!(
            "driver bootstrap requires exactly one {kind} artifact"
        )));
    }
    Ok(matches[0])
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
