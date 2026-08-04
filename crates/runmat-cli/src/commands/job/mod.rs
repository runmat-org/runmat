pub(crate) mod attach;
mod crypto;
mod recovery;
mod secret;
pub(crate) mod submit;

use anyhow::{Context, Result};
use runmat_server_client::auth::{
    resolve_auth_token, resolve_project_id, resolve_server_url, RemoteConfig,
};
use runmat_server_client::execution::ExecutionClient;

use crate::cli::{Cli, JobCommand};

pub async fn execute(
    command: JobCommand,
    cli: &Cli,
    config: &runmat_config::runtime::RunMatRuntimeConfig,
) -> Result<()> {
    match command {
        JobCommand::Submit {
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
        } => {
            submit::submit(
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
                cli,
                config,
            )
            .await
        }
        JobCommand::List {
            project,
            limit,
            cursor,
            json,
        } => attach::list(project, limit, cursor, json).await,
        JobCommand::Show { run_id, json } => attach::show(&run_id, json).await,
        JobCommand::Attach {
            run_id,
            no_follow,
            json,
        } => attach::attach(&run_id, no_follow, json).await,
        JobCommand::Cancel { run_id, json } => attach::cancel(&run_id, json).await,
        JobCommand::Recovery { command } => recovery::execute(command).await,
    }
}

async fn client(project: Option<uuid::Uuid>) -> Result<(ExecutionClient, String, String)> {
    let (client, server_url, config) = authenticated_context().await?;
    let project_id = resolve_project_id(&config, project)?.to_string();
    Ok((client, server_url, project_id))
}

async fn authenticated_context() -> Result<(ExecutionClient, String, RemoteConfig)> {
    let mut config = RemoteConfig::load()?;
    let server_url = resolve_server_url(&config, None)?;
    let token = resolve_auth_token(&mut config, &server_url).await?;
    let client =
        ExecutionClient::new(&server_url, &token).context("initialize execution client")?;
    Ok((client, server_url, config))
}

pub(crate) async fn cancel_remote_run(project: Option<uuid::Uuid>, run_id: &str) -> Result<()> {
    let (client, _, project_id) = client(project).await?;
    client
        .api()
        .cancel_run(&project_id, run_id)
        .await
        .map_err(runmat_server_client::execution::public_error)?;
    Ok(())
}
