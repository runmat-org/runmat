use std::sync::Arc;
use std::time::Duration;

use runmat_execution_artifact::encryption::{
    decode_run_key_envelope, EncryptionPurpose, PortableExecutionEncryption,
};
use runmat_execution_artifact::{
    ProgramExecutionObject, ProgramExecutionObjectLocation, ProgramExecutionOutputs,
    ProgramExecutionResponse,
};
use runmat_execution_transport_native::control::{
    CheckpointDisposition, DriverArtifactDownload, DriverArtifactKind, DriverAuthority,
    DriverControlPlane, DriverRunTarget, DriverRunTransition, HttpDriverControlPlane,
    StoreDriverArtifact,
};
use runmat_execution_transport_native::identity::EndpointIdentityMaterial;

use super::checkpoint::{RemoteDriverCheckpoint, RemoteDriverCheckpointPhase};
use super::config::RemoteDriverConfig;
use super::crypto::{ciphertext_digest, open_object, seal_object};
use crate::{NativeExecutionError, NativeExecutionResult};

pub(super) mod program;
pub(super) use program::AdmittedRemoteProgram;
#[path = "local_graph.rs"]
pub(super) mod local_graph;

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
    let original_clock = super::fea_admission::AllocationClock::begin();
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

    // This is a parent control component, separate from the child's signed
    // numerical limits. Admission below must refuse a child that cannot fit
    // the remaining capacity; it must not resize the program's request.
    let (completion_grant, child_memory_bytes) =
        super::completion::RemoteCompletionGrant::admit(bootstrap.driver_resources.memory_bytes)?;
    let mut execution_resources = bootstrap.driver_resources.clone();
    execution_resources.memory_bytes = child_memory_bytes;

    let admitted = program::admit(
        control.as_ref(),
        &run_key,
        &bootstrap.artifacts,
        &config.authority.run_id,
        &bootstrap.project_revision,
        bootstrap.desired_worker_count,
        bootstrap.driver_resources.scratch_bytes,
    )
    .await?;
    let program_identity = admitted.checkpoint_identity()?;
    let input_locations = admitted.input_locations().to_vec();
    let (mut checkpoint_sequence, closure_catalog, pool_catalog, original_checkpoint_identity) =
        match bootstrap.checkpoint.as_ref() {
            Some(download) => {
                if download.kind != "checkpoint" {
                    return Err(protocol(
                        "driver bootstrap checkpoint has the wrong artifact kind",
                    ));
                }
                let plaintext = open_download(
                    control.as_ref(),
                    &run_key,
                    download,
                    &config.authority.run_id,
                    EncryptionPurpose::Checkpoint,
                )
                .await?;
                let checkpoint = RemoteDriverCheckpoint::from_canonical_bytes(&plaintext)?;
                checkpoint.validate_for_program(
                    &config.authority,
                    &bootstrap.project_revision,
                    program_identity,
                )?;
                let source_driver_fence = checkpoint.source_driver_fence();
                (
                    checkpoint.sequence,
                    checkpoint.closure_catalog,
                    checkpoint.pool_catalog,
                    Some((
                        download.artifact_id.clone(),
                        download.ciphertext_digest.clone(),
                        source_driver_fence,
                    )),
                )
            }
            None => (0, None, None, None),
        };
    let mut input_objects = Vec::with_capacity(input_locations.len());
    for location in input_locations {
        let artifact = control
            .execution_artifact(&config.authority, &location.artifact_id)
            .await
            .map_err(protocol)?;
        if artifact.artifact_id != location.artifact_id || artifact.kind != "input" {
            return Err(protocol(
                "Server resolved the wrong independent input artifact",
            ));
        }
        let object = ProgramExecutionObject {
            reference: location.reference,
            bytes: open_download(
                control.as_ref(),
                &run_key,
                &artifact,
                &config.authority.run_id,
                EncryptionPurpose::Input,
            )
            .await?,
        };
        object.validate().map_err(protocol)?;
        input_objects.push(object);
    }
    let recovered = match closure_catalog.as_ref() {
        Some(location) => Some(
            super::recovery::restore(control.as_ref(), &run_key, &config.authority, location)
                .await?,
        ),
        None => None,
    };
    let mut recovered_pool = match pool_catalog.as_ref() {
        Some(location) => {
            if bootstrap.desired_worker_count == 0 {
                return Err(protocol(
                    "a pool checkpoint requires a remote worker allocation",
                ));
            }
            Some(
                super::pool_checkpoint::restore(
                    control.as_ref(),
                    &run_key,
                    &config.authority,
                    location,
                )
                .await?,
            )
        }
        None => None,
    };
    if let Some(pool) = recovered_pool.as_mut() {
        let (artifact_id, ciphertext_digest, source_fence) = original_checkpoint_identity
            .as_ref()
            .ok_or_else(|| protocol("restored pool lacks original checkpoint identity"))?;
        pool.reconcile_closed_clocks(
            control.as_ref(),
            &config.authority,
            checkpoint_sequence,
            artifact_id,
            ciphertext_digest,
            *source_fence,
        )
        .await?;
    }
    if let Some(recovered) = &recovered {
        input_objects.extend(recovered.objects.iter().cloned());
        if !input_objects
            .iter()
            .any(|object| object.reference == recovered.root)
        {
            return Err(protocol("restored recovery closure omits its root object"));
        }
    }

    checkpoint_sequence = checkpoint_sequence
        .checked_add(1)
        .ok_or_else(|| protocol("remote driver checkpoint sequence overflow"))?;
    store_driver_checkpoint(
        control.as_ref(),
        &config.authority,
        &run_key,
        &RemoteDriverCheckpoint::for_program(
            checkpoint_sequence,
            &config.authority,
            &bootstrap.project_revision,
            program_identity,
            RemoteDriverCheckpointPhase::ReadyToExecute,
            closure_catalog.clone(),
        )?
        .with_pool_catalog(pool_catalog.clone())?,
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
    checkpoint_sequence = checkpoint_sequence
        .checked_add(1)
        .ok_or_else(|| protocol("remote driver checkpoint sequence overflow"))?;
    store_driver_checkpoint(
        control.as_ref(),
        &config.authority,
        &run_key,
        &RemoteDriverCheckpoint::for_program(
            checkpoint_sequence,
            &config.authority,
            &bootstrap.project_revision,
            program_identity,
            admitted.execution_start_phase(),
            closure_catalog,
        )?
        .with_pool_catalog(pool_catalog)?,
    )
    .await?;

    let mut usage_sequence = 0_u64;
    let (cancellation_sender, cancellation_receiver) = tokio::sync::watch::channel(false);
    let direct_cancellation = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let supports_owned_cancellation = bootstrap.desired_worker_count > 0
        || matches!(&admitted, AdmittedRemoteProgram::Graph(_))
        || matches!(&admitted, AdmittedRemoteProgram::Direct(program) if matches!(program.request.artifact.form,
            runmat_execution_artifact::ExecutableForm::ExecutableUnitV4 | runmat_execution_artifact::ExecutableForm::InterpreterRegionSourcesV1));
    let execution = async {
        if bootstrap.desired_worker_count == 0 {
            match admitted {
                AdmittedRemoteProgram::Direct(admitted) => {
                    let response = execute_local_direct(
                        *admitted,
                        input_objects,
                        &execution_resources,
                        Some(Arc::clone(&direct_cancellation)),
                        &config.authority,
                        original_clock,
                    )
                    .await?;
                    Ok(Ok(response))
                }
                AdmittedRemoteProgram::Graph(admitted) => {
                    let program::AdmittedGraphProgram {
                        request, _staged, ..
                    } = *admitted;
                    local_graph::execute(
                        &config.authority.run_id,
                        request,
                        _staged,
                        &execution_resources,
                        &bootstrap.driver_accelerator_devices,
                        cancellation_receiver,
                    )
                    .await
                }
            }
        } else {
            super::pool_execution::execute(super::pool_execution::RemotePoolExecution {
                control: control.as_ref(),
                authority: &config.authority,
                run_key: &run_key,
                program: admitted,
                input_objects,
                recovered,
                recovered_pool,
                project_revision: bootstrap.project_revision.clone(),
                checkpoint_sequence,
                desired_workers: bootstrap.desired_worker_count,
                resources: bootstrap.worker_resources,
                driver_resources: execution_resources,
                cancellation: cancellation_receiver,
                ready_v1: config.ready_v1,
                clock_parent: config.clock_parent.clone(),
            })
            .await
        }
    };
    tokio::pin!(execution);
    let response = loop {
        tokio::select! {
            response = &mut execution => {
                match response? {
                    Ok(response) => {
                        break completion_grant.retain(response)?;
                    }
                    Err(crate::remote::pool_execution::RemoteExecutionStop::Cancelled) => {
                        report_usage(
                            control.as_ref(),
                            &config,
                            &mut usage_sequence,
                        ).await?;
                        transition(
                            control.as_ref(),
                            &config,
                            DriverRunTarget::Cancelled,
                            Some("cancelled-during-execution"),
                            None,
                            None,
                        ).await?;
                        return Ok(());
                    }
                    Err(crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(message)) => {
                        let diagnostic = store_encrypted(
                            control.as_ref(),
                            &config.authority,
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
                    if supports_owned_cancellation {
                        direct_cancellation.store(true, std::sync::atomic::Ordering::Release);
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

/// Execute an admitted Direct program on the driver host with its independent
/// input objects; values are materialized only for language programs.
async fn execute_local_direct(
    admitted: program::AdmittedDirectProgram,
    input_objects: Vec<ProgramExecutionObject>,
    allocation: &runmat_execution_transport_native::control::ResourceRequest,
    cancellation: Option<Arc<std::sync::atomic::AtomicBool>>,
    authority: &DriverAuthority,
    original_clock: super::fea_admission::AllocationClock,
) -> NativeExecutionResult<super::pool_execution::RemotePoolExecutionResult> {
    let program::AdmittedDirectProgram {
        mut request,
        materialized,
        bundle,
        ..
    } = admitted;
    let config = runmat_core::cluster_settings::runtime_config(&bundle)
        .map_err(protocol)?
        .ok_or_else(|| protocol("cluster Direct bundle lacks project-only runtime settings"))?;
    super::runtime_settings::admit_limits(
        &config,
        allocation.cpu_millicores,
        allocation.memory_bytes,
        !allocation.accelerators.is_empty(),
    )
    .map_err(protocol)?;
    if !allocation.accelerators.is_empty() && request.assignment.is_none() {
        return Err(protocol("zero-worker GPU execution requires an authenticated exact driver device assignment; requested accelerator resources are not device authority"));
    }
    let objects = super::object_transfer::RemoteObjectStore::new()?;
    let mut values = std::collections::HashMap::new();
    let mut console = None;
    let resources = super::pool_resources::task_resources(allocation)?;
    let fea_admission = super::fea_admission::for_verified_driver(
        authority,
        allocation,
        original_clock,
        runmat_execution::cancellation::CancellationToken::from_shared(
            cancellation.clone().ok_or_else(|| {
                protocol("verified driver FEA authority lacks physical cancellation")
            })?,
        ),
    )
    .map_err(protocol)?;
    let console_owner = request.console.map(|_| {
        let owner = crate::console_retention::ConsoleRetentionOwner::new(allocation.memory_bytes);
        crate::test_workload::HostConsoleOwner {
            budget: owner.limits(),
            owner,
            identity: runmat_runtime::context::RuntimeConsoleScopeIdentity {
                scope_id: runmat_execution::ExecutionScopeId::derive(&[
                    authority.run_id.as_bytes(),
                    b"remote-root",
                ]),
                pool_id: runmat_execution::PoolId::derive(&[
                    authority.run_id.as_bytes(),
                    b"driver-console",
                ]),
                task_id: runmat_execution::TaskId::derive(&[
                    authority.run_id.as_bytes(),
                    b"remote-root",
                ]),
                attempt_id: runmat_execution::identity::AttemptId::derive(&[
                    authority.run_id.as_bytes(),
                    &authority.fencing_token.to_be_bytes(),
                    b"driver-console",
                ]),
            },
        }
    });
    for object in input_objects {
        if let Some(value) = object.decode_externalized_value().map_err(protocol)? {
            values.insert(object.reference.id, value);
        } else {
            objects.register(object)?;
        }
    }
    let response = match request.artifact.form {
        runmat_execution_artifact::ExecutableForm::AnalysisWorkload => {
            let mut objects = objects.clone();
            super::runtime_settings::with_domain_policy(
                &config,
                request.assignment.clone(),
                || {
                    fea_admission.run_sync(|| {
                        execute_driver_analysis(
                            &request,
                            &mut objects,
                            &crate::NativeAnalysisKernel,
                            allocation,
                        )
                    })
                },
            )?
        }
        runmat_execution_artifact::ExecutableForm::GeometryWorkload => {
            let mut objects = objects.clone();
            fea_admission.run_sync(|| {
                crate::execute_geometry_materialization_program_request(
                    &request,
                    &mut objects,
                    crate::NativeGeometryMaterializationLimits::default(),
                )
            })
        }
        _ => {
            request.arguments = request
                .arguments
                .iter()
                .map(|argument| super::value_transfer::materialize(argument, &values))
                .collect::<NativeExecutionResult<Vec<_>>>()?;
            let environment = bundle.manifest.portable_environment.clone();
            let (response, captured) = fea_admission
                .run_future(
                    crate::test_workload::execute_host_program_request_with_console(
                        request,
                        crate::test_workload::HostProgramContext {
                            materialized: materialized.as_ref(),
                            collective: None,
                            bundle: Some(bundle),
                            environment: Some(&environment),
                            cancellation: cancellation.clone(),
                            resources: Some(&resources),
                            console_owner,
                            console_sink: None,
                        },
                    ),
                )
                .await;
            console = captured.map(super::pool_execution::RemoteTerminalConsole::Owned);
            response
        }
    };
    let mut result = collect_result_objects(response, objects)?;
    result.console = console;
    result.cancelled = cancellation
        .as_ref()
        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire));
    Ok(result)
}

/// The driver is the execution owner when a Direct analysis has no workers.
/// Its authenticated allocation must cover the governed plan before kernel effects.
pub(crate) fn execute_driver_analysis<
    S: runmat_execution_artifact::cache::CacheImport + runmat_execution_artifact::cache::CacheExport,
>(
    request: &runmat_execution_artifact::ProgramExecutionRequest,
    store: &mut S,
    kernel: &dyn runmat_analysis_execution::AnalysisKernel,
    allocation: &runmat_execution_transport_native::control::ResourceRequest,
) -> NativeExecutionResult<ProgramExecutionResponse> {
    let resources = super::pool_resources::task_resources(allocation)?;
    Ok(crate::execute_reserved_analysis_program_request(
        request,
        store,
        kernel,
        crate::NativeAnalysisHostLimits::default(),
        &resources,
    ))
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

pub(super) async fn commit_response(
    control: &dyn DriverControlPlane,
    config: &RemoteDriverConfig,
    run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    mut completion: super::completion::OwnedRemoteCompletion,
) -> NativeExecutionResult<()> {
    let super::pool_execution::RemotePoolExecutionResult {
        response,
        objects,
        console,
        cancelled,
    } = completion.take_result();
    // A proven cancellation without a requested console has no terminal
    // result to publish. Resolve it before the response-specific fast paths;
    // a worker may have produced either a failure or a Graph result while
    // cancellation and physical retirement were being observed.
    if cancelled && console.is_none() {
        return transition(
            control,
            config,
            DriverRunTarget::Cancelled,
            Some("cancelled-during-execution"),
            None,
            None,
        )
        .await;
    }
    let failed = matches!(
        &response,
        ProgramExecutionResponse::Failure { .. } | ProgramExecutionResponse::RuntimeFailure { .. }
    );
    let response = match response {
        response @ ProgramExecutionResponse::GraphSuccess { .. } if console.is_none() => {
            let plaintext = serde_json::to_vec(&response).map_err(protocol)?;
            let result = store_encrypted(
                control,
                &config.authority,
                run_key,
                DriverArtifactKind::Result,
                EncryptionPurpose::Result,
                &plaintext,
                None,
            )
            .await?;
            return transition(
                control,
                config,
                DriverRunTarget::Succeeded,
                None,
                Some(result),
                None,
            )
            .await;
        }
        ProgramExecutionResponse::SpmdSuccess { .. } => {
            return fail_with_diagnostic(
                control,
                config,
                run_key,
                b"SPMD worker response requires the typed remote gang driver",
                "spmd-gang-driver-required",
            )
            .await;
        }
        response @ ProgramExecutionResponse::RuntimeFailure { .. } if console.is_none() => {
            let plaintext = serde_json::to_vec(&response).map_err(protocol)?;
            return fail_with_diagnostic(
                control,
                config,
                run_key,
                &plaintext,
                "runtime-execution-failed",
            )
            .await;
        }
        response => response,
    };
    if console.is_none() {
        if let ProgramExecutionResponse::Failure { message } = response {
            return fail_with_diagnostic(
                control,
                config,
                run_key,
                message.as_bytes(),
                "execution-failed",
            )
            .await;
        }
    }

    let references: &[runmat_execution::value::ValueRef] = match &response {
        ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } => result_objects,
        _ => &[],
    };
    let mut locations = Vec::with_capacity(references.len());
    for reference in references {
        let object = ProgramExecutionObject {
            reference: reference.clone(),
            bytes: objects.read(reference, &config.authority.run_id)?,
        };
        object.validate().map_err(protocol)?;
        if object.reference.authorization_scope != config.authority.run_id {
            return Err(protocol("result object is outside the run authority"));
        }
        let ciphertext = seal_object(
            run_key,
            &config.authority.run_id,
            EncryptionPurpose::Result,
            &object.bytes,
        )?;
        let artifact = control
            .store_artifact(
                &config.authority,
                StoreDriverArtifact {
                    kind: DriverArtifactKind::Result,
                    ciphertext_digest: ciphertext_digest(&ciphertext),
                    ciphertext: &ciphertext,
                    media_type: MEDIA_TYPE.into(),
                    retain_for_seconds: RETAIN_FOR_SECONDS,
                    checkpoint: None,
                },
            )
            .await
            .map_err(protocol)?;
        locations.push(ProgramExecutionObjectLocation {
            reference: object.reference,
            artifact_id: artifact.artifact_id,
        });
    }
    let outputs = ProgramExecutionOutputs::new(response, locations).map_err(protocol)?;
    let plaintext = match console {
        Some(console) => console.encode(&outputs)?,
        None => serde_json::to_vec(&outputs).map_err(protocol)?,
    };
    let result = store_encrypted(
        control,
        &config.authority,
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
        if cancelled {
            DriverRunTarget::Cancelled
        } else if failed {
            DriverRunTarget::Failed
        } else {
            DriverRunTarget::Succeeded
        },
        if cancelled {
            Some("cancelled-during-execution")
        } else if failed {
            Some("execution-failed")
        } else {
            None
        },
        Some(result),
        None,
    )
    .await
}

async fn fail_with_diagnostic(
    control: &dyn DriverControlPlane,
    config: &RemoteDriverConfig,
    run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    diagnostic: &[u8],
    reason: &str,
) -> NativeExecutionResult<()> {
    let diagnostic = store_encrypted(
        control,
        &config.authority,
        run_key,
        DriverArtifactKind::Diagnostic,
        EncryptionPurpose::DetailedEvent,
        diagnostic,
        None,
    )
    .await?;
    transition(
        control,
        config,
        DriverRunTarget::Failed,
        Some(reason),
        None,
        Some(diagnostic),
    )
    .await
}

fn collect_result_objects(
    response: ProgramExecutionResponse,
    store: super::object_transfer::RemoteObjectStore,
) -> NativeExecutionResult<super::pool_execution::RemotePoolExecutionResult> {
    let references: &[runmat_execution::value::ValueRef] = match &response {
        ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } => result_objects,
        _ => &[],
    };
    for reference in references {
        if store
            .object(reference, &reference.authorization_scope)
            .map_err(protocol)?
            .is_none()
        {
            return Err(protocol("result object is absent from the execution store"));
        }
    }
    Ok(super::pool_execution::RemotePoolExecutionResult {
        response,
        console: None,
        cancelled: false,
        objects: super::result_objects::RemoteResultObjects::Local(store),
    })
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

pub(super) async fn store_driver_checkpoint(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    checkpoint: &RemoteDriverCheckpoint,
) -> NativeExecutionResult<String> {
    Ok(
        store_driver_checkpoint_with_identity(control, authority, run_key, checkpoint)
            .await?
            .artifact_id,
    )
}

pub(super) struct StoredDriverCheckpoint {
    pub(super) artifact_id: String,
    pub(super) ciphertext_digest: String,
}

pub(super) async fn store_driver_checkpoint_with_identity(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    checkpoint: &RemoteDriverCheckpoint,
) -> NativeExecutionResult<StoredDriverCheckpoint> {
    let plaintext = checkpoint.canonical_bytes()?;
    let ciphertext = seal_object(
        run_key,
        &authority.run_id,
        EncryptionPurpose::Checkpoint,
        &plaintext,
    )?;
    let digest = ciphertext_digest(&ciphertext);
    let artifact = control
        .store_artifact(
            authority,
            StoreDriverArtifact {
                kind: DriverArtifactKind::Checkpoint,
                ciphertext_digest: digest.clone(),
                ciphertext: &ciphertext,
                media_type: MEDIA_TYPE.into(),
                retain_for_seconds: RETAIN_FOR_SECONDS,
                checkpoint: Some((checkpoint.sequence, checkpoint.phase.disposition())),
            },
        )
        .await
        .map_err(protocol)?;
    Ok(StoredDriverCheckpoint {
        artifact_id: artifact.artifact_id,
        ciphertext_digest: digest,
    })
}

pub(super) async fn store_encrypted(
    control: &dyn DriverControlPlane,
    authority: &DriverAuthority,
    run_key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    kind: DriverArtifactKind,
    purpose: EncryptionPurpose,
    plaintext: &[u8],
    checkpoint: Option<(u64, CheckpointDisposition)>,
) -> NativeExecutionResult<String> {
    let ciphertext = seal_object(run_key, &authority.run_id, purpose, plaintext)?;
    let artifact = control
        .store_artifact(
            authority,
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
