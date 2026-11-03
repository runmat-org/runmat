use std::num::NonZeroU64;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use runmat_execution_artifact::encryption::EncryptionPurpose;
use runmat_execution_artifact::{
    ProgramExecutionObject, ProgramExecutionOutputs, ProgramExecutionResponse,
};
use runmat_server_client::auth::{
    resolve_auth_token, resolve_public_project_id, resolve_server_url, RemoteConfig,
};
use runmat_server_client::execution::{public_error, ExecutionClient};

use super::secret;

pub async fn list(
    project: Option<uuid::Uuid>,
    limit: Option<u32>,
    cursor: Option<String>,
    json: bool,
) -> Result<()> {
    let (client, _, project_id) = super::client(project).await?;
    let limit = limit
        .map(|value| NonZeroU64::new(u64::from(value)).context("limit must be greater than zero"))
        .transpose()?;
    let response = client
        .api()
        .list_runs(project_id.as_str(), cursor.as_deref(), limit)
        .await
        .map_err(public_error)?
        .into_inner();
    if json {
        println!("{}", serde_json::to_string(&response)?);
    } else {
        for run in response.runs {
            print_run(&run);
        }
        if let Some(cursor) = response.next_cursor {
            println!("next_cursor\t{cursor}");
        }
    }
    Ok(())
}

pub async fn show(run_id: &str, json: bool) -> Result<()> {
    let (client, saved) = saved_client(run_id).await?;
    let run = client
        .api()
        .get_run(&saved.project_id, run_id)
        .await
        .map_err(public_error)?
        .into_inner();
    if json {
        println!("{}", serde_json::to_string(&run)?);
    } else {
        print_run(&run);
        if let Some(reason) = run.reason_code {
            println!("  {}", crate::presentation::stdout().muted(&reason));
        }
    }
    Ok(())
}

pub async fn cancel(run_id: &str, json: bool) -> Result<()> {
    let (client, saved) = saved_client(run_id).await?;
    let run = client
        .api()
        .cancel_run(&saved.project_id, run_id)
        .await
        .map_err(public_error)?
        .into_inner();
    if json {
        println!("{}", serde_json::to_string(&run)?);
    } else {
        print_run(&run);
    }
    Ok(())
}

pub(crate) async fn await_result(run_id: &str) -> Result<ProgramExecutionResponse> {
    Ok(await_outputs(run_id).await?.response)
}

pub(crate) async fn await_outputs(run_id: &str) -> Result<ProgramExecutionOutputs> {
    let (client, saved) = saved_client(run_id).await?;
    loop {
        let run = client
            .api()
            .get_run(&saved.project_id, run_id)
            .await
            .map_err(public_error)?
            .into_inner();
        if !terminal(&run.state) {
            tokio::time::sleep(Duration::from_secs(1)).await;
            continue;
        }
        if let Some(artifact_id) = &run.result_artifact_id {
            let ciphertext = client
                .download_artifact(&saved.project_id, artifact_id, &saved.endpoint_fingerprint)
                .await?;
            let plaintext = super::crypto::open_run_object(
                &saved.key()?,
                &run.id,
                EncryptionPurpose::Result,
                &ciphertext,
            )?;
            let outputs: ProgramExecutionOutputs =
                serde_json::from_slice(&plaintext).context("remote result payload is malformed")?;
            outputs
                .validate()
                .context("remote result payload failed integrity validation")?;
            super::console::validate_requested_capture(
                &outputs,
                saved.console,
                saved.console_budget,
            )?;
            return Ok(outputs);
        }
        if let Some(artifact_id) = &run.diagnostic_artifact_id {
            let ciphertext = client
                .download_artifact(&saved.project_id, artifact_id, &saved.endpoint_fingerprint)
                .await?;
            let diagnostic = super::crypto::open_run_object(
                &saved.key()?,
                &run.id,
                EncryptionPurpose::DetailedEvent,
                &ciphertext,
            )?;
            bail!(
                "{}",
                String::from_utf8(diagnostic).context("remote diagnostic is not valid UTF-8")?
            );
        }
        bail!(
            "remote test attempt ended in state '{}' without a result",
            run.state
        );
    }
}

pub(crate) async fn await_downloaded_outputs(
    run_id: &str,
) -> Result<(ProgramExecutionOutputs, Vec<ProgramExecutionObject>)> {
    let outputs = await_outputs(run_id).await?;
    let (client, saved) = saved_client(run_id).await?;
    let objects = download_output_objects(&client, &saved, &outputs).await?;
    Ok((outputs, objects))
}

async fn download_output_objects(
    client: &ExecutionClient,
    saved: &secret::SavedRemoteRun,
    outputs: &ProgramExecutionOutputs,
) -> Result<Vec<ProgramExecutionObject>> {
    let key = saved.key()?;
    let mut objects = Vec::with_capacity(outputs.objects.len());
    for location in &outputs.objects {
        let ciphertext = client
            .download_artifact(
                &saved.project_id,
                &location.artifact_id,
                &saved.endpoint_fingerprint,
            )
            .await?;
        let bytes = super::crypto::open_run_object(
            &key,
            &saved.run_id,
            EncryptionPurpose::Result,
            &ciphertext,
        )?;
        let object = ProgramExecutionObject {
            reference: location.reference.clone(),
            bytes,
        };
        object
            .validate()
            .context("remote result object failed integrity validation")?;
        objects.push(object);
    }
    Ok(objects)
}

pub async fn attach(run_id: &str, no_follow: bool, json: bool) -> Result<()> {
    let (client, saved) = saved_client(run_id).await?;
    loop {
        let run = client
            .api()
            .get_run(&saved.project_id, run_id)
            .await
            .map_err(public_error)?
            .into_inner();
        if terminal(&run.state) {
            return print_terminal(&client, &saved, run, json).await;
        }
        if no_follow {
            if json {
                println!("{}", serde_json::to_string(&run)?);
            } else {
                print_run(&run);
            }
            return Ok(());
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(1)) => {}
            _ = tokio::signal::ctrl_c() => {
                eprintln!("{}", crate::presentation::stderr().muted(
                    "Detached; the remote job is still running."
                ));
                return Ok(());
            }
        }
    }
}

async fn print_terminal(
    client: &ExecutionClient,
    saved: &secret::SavedRemoteRun,
    run: runmat_server_client::execution::RunResponse,
    json: bool,
) -> Result<()> {
    let key = saved.key()?;
    if let Some(artifact_id) = &run.result_artifact_id {
        let ciphertext = client
            .download_artifact(&saved.project_id, artifact_id, &saved.endpoint_fingerprint)
            .await?;
        let plaintext =
            super::crypto::open_run_object(&key, &run.id, EncryptionPurpose::Result, &ciphertext)?;
        let outputs: ProgramExecutionOutputs =
            serde_json::from_slice(&plaintext).context("remote result payload is malformed")?;
        outputs
            .validate()
            .context("remote result payload failed integrity validation")?;
        super::console::validate_requested_capture(&outputs, saved.console, saved.console_budget)?;
        if !json {
            super::console::print_outputs(&outputs)?;
        }
        if let secret::SavedOutputIntent::FeaStudy {
            study_id,
            expectation,
            assessment_functions,
        } = &saved.output_intent
        {
            let objects = download_output_objects(client, saved, &outputs).await?;
            let result = crate::commands::fea::remote::materialize_fea_study(
                &run.id,
                outputs.response,
                objects,
                (**expectation).clone(),
                assessment_functions,
            )
            .await?;
            return crate::commands::fea::result_presentation::print_fea_result(
                study_id,
                &result.result,
                &result.publication,
                json,
            );
        }
        if let secret::SavedOutputIntent::FeaSweep {
            sweep_id,
            case_ids,
            expectation,
            assessment_functions,
        } = &saved.output_intent
        {
            let objects = download_output_objects(client, saved, &outputs).await?;
            let result = crate::commands::fea::remote::materialize_fea_sweep(
                &run.id,
                outputs.response,
                objects,
                (**expectation).clone(),
                assessment_functions,
            )
            .await?;
            return crate::commands::fea::result_presentation::print_fea_sweep_result(
                sweep_id, case_ids, &result, json,
            );
        }
        if let secret::SavedOutputIntent::Mesh {
            solver_mesh_path,
            evidence_path,
            force,
            materialization,
        } = &saved.output_intent
        {
            let objects = download_output_objects(client, saved, &outputs).await?;
            let result = crate::commands::mesh::materialize_remote_outputs(
                outputs.response,
                objects,
                solver_mesh_path,
                evidence_path,
                *force,
            )?;
            if *materialization == secret::OutputMaterialization::Pending {
                secret::mark_output_materialized(&run.id)?;
            }
            return print_materialized_mesh(&run, solver_mesh_path, evidence_path, &result, json);
        }
        if let secret::SavedOutputIntent::StructuralStudy {
            continuation,
            input_materializations,
            revision,
            materialization,
            result,
        } = &saved.output_intent
        {
            if *materialization == secret::OutputMaterialization::Materialized {
                let result = result.as_ref().context(
                    "saved structural study materialization omitted its completed result",
                )?;
                if json {
                    crate::commands::fea::print_envelope(result, true)?;
                } else {
                    crate::commands::fea::print_study_run_summary(&result.data, false);
                }
                return Ok(());
            }
            let objects = download_output_objects(client, saved, &outputs).await?;
            let envelope = crate::commands::fea::remote::materialize_remote_outputs(
                &run.id,
                outputs.response,
                objects,
                continuation,
                input_materializations,
                revision.clone(),
            )?;
            secret::mark_structural_study_materialized(&run.id, envelope.clone())?;
            if json {
                crate::commands::fea::print_envelope(&envelope, true)?;
            } else {
                crate::commands::fea::print_study_run_summary(&envelope.data, false);
            }
            return Ok(());
        }
        if let secret::SavedOutputIntent::StructuralSweep {
            spec,
            continuations,
            input_materializations,
            revision,
            materialization,
            result,
        } = &saved.output_intent
        {
            if *materialization == secret::OutputMaterialization::Materialized {
                let result = result.as_ref().context(
                    "saved structural sweep materialization omitted its completed result",
                )?;
                if json {
                    crate::commands::fea::print_envelope(result, true)?;
                } else {
                    crate::commands::fea::print_sweep_run_summary(&result.data);
                }
                return Ok(());
            }
            let objects = download_output_objects(client, saved, &outputs).await?;
            let envelope = crate::commands::fea::remote::materialize_remote_sweep_outputs(
                &run.id,
                outputs.response,
                objects,
                spec,
                continuations,
                input_materializations,
                revision.clone(),
            )?;
            secret::mark_structural_sweep_materialized(&run.id, envelope.clone())?;
            if json {
                crate::commands::fea::print_envelope(&envelope, true)?;
            } else {
                crate::commands::fea::print_sweep_run_summary(&envelope.data);
            }
            return Ok(());
        }
        let result_object_count = outputs.objects.len();
        let response = outputs.response;
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "run": run,
                    "result": response,
                    "resultObjectCount": result_object_count,
                    "console": outputs.console,
                    "consoleScopes": outputs.console_scopes,
                    "consoleBudget": outputs.console_budget,
                    "delegatedConsoleRoot": outputs.delegated_console_root,
                }))?
            );
            super::result::ensure_terminal_success(
                response,
                super::result::ResultOrigin::Attached,
            )?;
        } else {
            print_run(&run);
            println!(
                "{}",
                super::result::format_terminal_result(
                    response,
                    super::result::ResultOrigin::Attached,
                )?
            );
        }
        return Ok(());
    }
    if let Some(artifact_id) = &run.diagnostic_artifact_id {
        let ciphertext = client
            .download_artifact(&saved.project_id, artifact_id, &saved.endpoint_fingerprint)
            .await?;
        let diagnostic = super::crypto::open_run_object(
            &key,
            &run.id,
            EncryptionPurpose::DetailedEvent,
            &ciphertext,
        )?;
        let message =
            String::from_utf8(diagnostic).context("remote diagnostic is not valid UTF-8")?;
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "run": run,
                    "diagnostic": message,
                }))?
            );
        } else {
            print_run(&run);
        }
        bail!("{message}");
    }
    if json {
        println!("{}", serde_json::to_string(&run)?);
    } else {
        print_run(&run);
    }
    bail!(
        "remote run ended in state '{}' without a terminal result",
        run.state
    )
}

fn print_materialized_mesh(
    run: &runmat_server_client::execution::RunResponse,
    solver_mesh_path: &std::path::Path,
    evidence_path: &std::path::Path,
    result: &crate::commands::mesh::MeshOutputSummary,
    json: bool,
) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "run": run,
                "solverMeshArtifactPath": solver_mesh_path,
                "meshingEvidenceArtifactPath": evidence_path,
                "canonicalDigest": crate::commands::mesh::hex_digest(
                    result.canonical_digest.bytes()
                ),
                "nodeCount": result.node_count,
                "elementCount": result.element_count,
                "boundaryFaceCount": result.boundary_face_count,
                "materialized": true,
            }))?
        );
    } else {
        print_run(run);
        println!("Verified solver mesh available");
        println!("  mesh: {}", solver_mesh_path.display());
        println!("  evidence: {}", evidence_path.display());
    }
    Ok(())
}

async fn saved_client(run_id: &str) -> Result<(ExecutionClient, secret::SavedRemoteRun)> {
    let mut saved = secret::load(run_id)?;
    let mut config = RemoteConfig::load()?;
    let current_server = resolve_server_url(&config, None)?;
    if current_server.trim_end_matches('/') != saved.server_url.trim_end_matches('/') {
        bail!(
            "saved run belongs to {}, but the active Server is {}",
            saved.server_url,
            current_server
        );
    }
    let selected = resolve_public_project_id(&config, None)?;
    let saved_project = runmat_server_client::ids::ProjectId::from_uuid(
        runmat_server_client::auth::parse_project_id_input(&saved.project_id)
            .context("saved run contains an invalid project identity")?,
    );
    if selected != saved_project {
        bail!(
            "saved run belongs to project {}, but the active project is {}",
            saved.project_id,
            selected
        );
    }
    let token = resolve_auth_token(&mut config, &current_server).await?;
    // Legacy persisted UUID shorthand and current public IDs denote the same project.
    // Normalize only the transport representation after checking server/project ownership.
    saved.project_id = saved_project.to_string();
    Ok((ExecutionClient::new(&current_server, &token)?, saved))
}

fn terminal(state: &str) -> bool {
    matches!(
        state,
        "succeeded" | "failed" | "cancelled" | "expired" | "indeterminate"
    )
}

fn print_run(run: &runmat_server_client::execution::RunResponse) {
    println!(
        "{}\t{}\t{}\t{}",
        run.id, run.state, run.cluster_id, run.updated_at
    );
}
