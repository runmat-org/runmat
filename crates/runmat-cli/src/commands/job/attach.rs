use std::num::NonZeroU64;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use runmat_execution_artifact::encryption::{
    decode_encrypted_run_object, EncryptionPurpose, RunObjectEncryption,
};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_server_client::auth::{
    resolve_auth_token, resolve_project_id, resolve_server_url, RemoteConfig,
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
        .list_runs(&project_id, cursor.as_deref(), limit)
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
            let plaintext = open(
                &saved.key()?,
                &run.id,
                EncryptionPurpose::Result,
                &ciphertext,
            )?;
            return serde_json::from_slice(&plaintext)
                .context("remote result payload is malformed");
        }
        if let Some(artifact_id) = &run.diagnostic_artifact_id {
            let ciphertext = client
                .download_artifact(&saved.project_id, artifact_id, &saved.endpoint_fingerprint)
                .await?;
            let diagnostic = open(
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
        let plaintext = open(&key, &run.id, EncryptionPurpose::Result, &ciphertext)?;
        let response: ProgramExecutionResponse =
            serde_json::from_slice(&plaintext).context("remote result payload is malformed")?;
        if json {
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "run": run,
                    "result": response,
                }))?
            );
        } else {
            print_run(&run);
            match response {
                ProgramExecutionResponse::Success { value } => {
                    println!("{}", serde_json::to_string_pretty(&value)?);
                }
                ProgramExecutionResponse::Failure { message } => bail!("{message}"),
            }
        }
        return Ok(());
    }
    if let Some(artifact_id) = &run.diagnostic_artifact_id {
        let ciphertext = client
            .download_artifact(&saved.project_id, artifact_id, &saved.endpoint_fingerprint)
            .await?;
        let diagnostic = open(&key, &run.id, EncryptionPurpose::DetailedEvent, &ciphertext)?;
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
            bail!("{message}");
        }
        return Ok(());
    }
    if json {
        println!("{}", serde_json::to_string(&run)?);
    } else {
        print_run(&run);
    }
    Ok(())
}

fn open(
    key: &runmat_execution_artifact::encryption::RunKeyMaterial,
    run_id: &str,
    purpose: EncryptionPurpose,
    ciphertext: &[u8],
) -> Result<Vec<u8>> {
    let object = decode_encrypted_run_object(ciphertext, 64 * 1024 * 1024)?;
    if object.context.run_identity != run_id || object.context.purpose != purpose {
        bail!("encrypted remote artifact has the wrong run scope or purpose");
    }
    RunObjectEncryption
        .open(key, &object)
        .map_err(anyhow::Error::from)
}

async fn saved_client(run_id: &str) -> Result<(ExecutionClient, secret::SavedRemoteRun)> {
    let saved = secret::load(run_id)?;
    let mut config = RemoteConfig::load()?;
    let current_server = resolve_server_url(&config, None)?;
    if current_server.trim_end_matches('/') != saved.server_url.trim_end_matches('/') {
        bail!(
            "saved run belongs to {}, but the active Server is {}",
            saved.server_url,
            current_server
        );
    }
    let selected = resolve_project_id(&config, None)?;
    if selected.to_string() != saved.project_id {
        bail!(
            "saved run belongs to project {}, but the active project is {}",
            saved.project_id,
            selected
        );
    }
    let token = resolve_auth_token(&mut config, &current_server).await?;
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
