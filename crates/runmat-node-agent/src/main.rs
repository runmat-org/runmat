use std::sync::Arc;

use clap::Parser as _;
use runmat_node_agent::cli::{Cli, Command, TrustTier};
use runmat_node_agent::enrollment::CredentialStore;
use runmat_node_agent::service::{HttpNodeControlPlane, NodeAgentService, Shutdown};
use runmat_node_agent::AgentConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Inventory => {
            let inventory = runmat_node_agent::inventory::collect()?;
            println!("{}", serde_json::to_string_pretty(&inventory)?);
            Ok(())
        }
        Command::Enroll { token } => {
            let config = config(&cli)?;
            config.validate()?;
            let control = Arc::new(HttpNodeControlPlane::new(config.server_url.clone())?);
            let credential = runmat_node_agent::enrollment::enroll(
                control,
                &CredentialStore::new(&config.state_directory),
                token.clone(),
                runmat_node_agent::inventory::collect()?,
                config.heartbeat_ttl.as_secs(),
            )
            .await?;
            println!("enrolled node {}", credential.node_id);
            Ok(())
        }
        Command::Run => {
            let config = config(&cli)?;
            let control = Arc::new(HttpNodeControlPlane::new(config.server_url.clone())?);
            let service = NodeAgentService::load(config, control)?;
            let shutdown = Shutdown::default();
            let receiver = shutdown.subscribe();
            let signal = tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                shutdown.trigger();
            });
            let result = service.run(receiver).await;
            signal.abort();
            result.map_err(Into::into)
        }
        Command::RotateCredential => {
            let config = config(&cli)?;
            let control = Arc::new(HttpNodeControlPlane::new(config.server_url.clone())?);
            let mut service = NodeAgentService::load(config, control)?;
            service.rotate_credential().await?;
            println!("node credential rotated");
            Ok(())
        }
    }
}

fn config(cli: &Cli) -> anyhow::Result<AgentConfig> {
    let server_url = cli
        .server
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--server or RUNMAT_SERVER_URL is required"))?;
    let runmat_executable = cli
        .runmat
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--runmat or RUNMAT_EXECUTABLE is required"))?;
    Ok(AgentConfig {
        state_directory: cli
            .state_directory
            .clone()
            .map(Ok)
            .unwrap_or_else(AgentConfig::default_state_directory)?,
        server_url,
        runmat_executable,
        heartbeat_interval: AgentConfig::DEFAULT_HEARTBEAT_INTERVAL,
        heartbeat_ttl: AgentConfig::DEFAULT_HEARTBEAT_TTL,
        drain_timeout: AgentConfig::DEFAULT_DRAIN_TIMEOUT,
        maximum_allocations: AgentConfig::DEFAULT_MAXIMUM_ALLOCATIONS,
        trust_tier: match cli.trust_tier {
            TrustTier::CustomerTrusted => {
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted
            }
            TrustTier::HostedOrdinary => {
                runmat_execution::security::ExecutionTrustTier::HostedOrdinary
            }
        },
    })
}
