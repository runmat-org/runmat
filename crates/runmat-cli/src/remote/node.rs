use std::sync::Arc;

use anyhow::{anyhow, Result};
use runmat_node_agent::enrollment::CredentialStore;
use runmat_node_agent::service::{HttpNodeControlPlane, NodeAgentService, Shutdown};
use runmat_node_agent::{AgentConfig, AgentFileConfig};

use crate::cli::{NodeJoinArgs, NodeJoinCommand, NodeServiceCommand, NodeTrustTier};

pub(super) async fn execute(args: NodeJoinArgs) -> Result<()> {
    match &args.command {
        NodeJoinCommand::Inventory => {
            let inventory = runmat_node_agent::inventory::collect()?;
            println!("{}", serde_json::to_string_pretty(&inventory)?);
            Ok(())
        }
        NodeJoinCommand::Enroll { token } => {
            let config = config(&args, false)?;
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
        NodeJoinCommand::Run => run(config(&args, false)?).await,
        NodeJoinCommand::RotateCredential => {
            let config = config(&args, false)?;
            let control = Arc::new(HttpNodeControlPlane::new(config.server_url.clone())?);
            let mut service = NodeAgentService::load(config, control)?;
            service.rotate_credential().await?;
            println!("node credential rotated");
            Ok(())
        }
        NodeJoinCommand::Service { command } => execute_service(&args, command),
        NodeJoinCommand::WindowsServiceRun => {
            #[cfg(windows)]
            {
                runmat_node_agent::dispatch_windows_service(config(&args, false)?)
            }
            #[cfg(not(windows))]
            {
                anyhow::bail!("the Windows service entry point is unavailable on this platform")
            }
        }
    }
}

async fn run(config: AgentConfig) -> Result<()> {
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

fn config(args: &NodeJoinArgs, service_defaults: bool) -> Result<AgentConfig> {
    let mut config = if let Some(path) = &args.node_config {
        AgentFileConfig::load(path)?.into_runtime()?
    } else {
        AgentConfig {
            state_directory: if service_defaults {
                runmat_node_agent::service_install::service_state_directory()?
            } else {
                AgentConfig::default_state_directory()?
            },
            server_url: args
                .server
                .clone()
                .ok_or_else(|| anyhow!("--server or RUNMAT_SERVER_URL is required"))?,
            runmat_executable: args.runmat.clone().unwrap_or(std::env::current_exe()?),
            heartbeat_interval: AgentConfig::DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_ttl: AgentConfig::DEFAULT_HEARTBEAT_TTL,
            drain_timeout: AgentConfig::DEFAULT_DRAIN_TIMEOUT,
            maximum_allocations: AgentConfig::DEFAULT_MAXIMUM_ALLOCATIONS,
            trust_tier: runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        }
    };
    if let Some(value) = &args.server {
        config.server_url.clone_from(value);
    }
    if let Some(value) = &args.runmat {
        config.runmat_executable.clone_from(value);
    }
    if let Some(value) = &args.state_directory {
        config.state_directory.clone_from(value);
    }
    if let Some(value) = args.trust_tier {
        config.trust_tier = match value {
            NodeTrustTier::CustomerTrusted => {
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted
            }
            NodeTrustTier::HostedOrdinary => {
                runmat_execution::security::ExecutionTrustTier::HostedOrdinary
            }
        };
    }
    config.validate()?;
    Ok(config)
}

fn execute_service(args: &NodeJoinArgs, command: &NodeServiceCommand) -> Result<()> {
    match command {
        NodeServiceCommand::Install { dry_run } => install_service(args, *dry_run),
        NodeServiceCommand::Print => install_service(args, true),
        NodeServiceCommand::Uninstall { dry_run } => {
            let plan = runmat_node_agent::service_install::uninstall_plan()?;
            if *dry_run {
                println!("{}", serde_json::to_string_pretty(&plan)?);
                return Ok(());
            }
            runmat_node_agent::service_install::apply_uninstall(&plan)?;
            println!("stopped and removed RunMat execution-node service");
            Ok(())
        }
    }
}

fn install_service(args: &NodeJoinArgs, dry_run: bool) -> Result<()> {
    let config = config(args, true)?;
    let file_config = AgentFileConfig::from_runtime(&config);
    let plan =
        runmat_node_agent::service_install::install_plan(&file_config, &std::env::current_exe()?)?;
    if dry_run {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else {
        runmat_node_agent::service_install::apply_install(&plan, &config.state_directory)?;
        println!("installed and started RunMat execution-node service");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser as _;

    use crate::cli::{Cli, ClusterCommand, Commands, NodeJoinCommand};

    #[test]
    fn cluster_join_parses_node_lifecycle_commands() {
        let cli = Cli::try_parse_from([
            "runmat",
            "cluster",
            "join",
            "--server",
            "https://api.runmat.com",
            "enroll",
            "--token",
            "secret",
        ])
        .expect("cluster join command");
        let Some(Commands::Cluster {
            cluster_command: ClusterCommand::Join(args),
        }) = cli.command
        else {
            panic!("expected cluster join command");
        };
        assert!(matches!(
            args.command,
            NodeJoinCommand::Enroll { token } if token == "secret"
        ));
    }
}
