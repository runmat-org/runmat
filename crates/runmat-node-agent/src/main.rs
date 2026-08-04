use std::sync::Arc;

use clap::Parser as _;
use runmat_node_agent::cli::{Cli, Command, ServiceCommand, TrustTier};
use runmat_node_agent::enrollment::CredentialStore;
use runmat_node_agent::service::{HttpNodeControlPlane, NodeAgentService, Shutdown};
use runmat_node_agent::AgentConfig;

#[cfg(windows)]
mod windows_service;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if matches!(cli.command, Command::WindowsServiceRun) {
        #[cfg(windows)]
        {
            return windows_service::dispatch(config(&cli, false)?);
        }
        #[cfg(not(windows))]
        {
            anyhow::bail!("the Windows service entry point is unavailable on this platform");
        }
    }
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(execute(cli))
}

async fn execute(cli: Cli) -> anyhow::Result<()> {
    match &cli.command {
        Command::Inventory => {
            let inventory = runmat_node_agent::inventory::collect()?;
            println!("{}", serde_json::to_string_pretty(&inventory)?);
            Ok(())
        }
        Command::Enroll { token } => {
            let config = config(&cli, false)?;
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
            let config = config(&cli, false)?;
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
            let config = config(&cli, false)?;
            let control = Arc::new(HttpNodeControlPlane::new(config.server_url.clone())?);
            let mut service = NodeAgentService::load(config, control)?;
            service.rotate_credential().await?;
            println!("node credential rotated");
            Ok(())
        }
        Command::Service { command } => execute_service(&cli, command),
        Command::WindowsServiceRun => unreachable!("handled before creating the Tokio runtime"),
    }
}

fn config(cli: &Cli, service_defaults: bool) -> anyhow::Result<AgentConfig> {
    let mut config = if let Some(path) = &cli.config {
        runmat_node_agent::AgentFileConfig::load(path)?.into_runtime()?
    } else {
        AgentConfig {
            state_directory: if service_defaults {
                runmat_node_agent::service_install::service_state_directory()?
            } else {
                AgentConfig::default_state_directory()?
            },
            server_url: cli
                .server
                .clone()
                .ok_or_else(|| anyhow::anyhow!("--server or RUNMAT_SERVER_URL is required"))?,
            runmat_executable: cli
                .runmat
                .clone()
                .ok_or_else(|| anyhow::anyhow!("--runmat or RUNMAT_EXECUTABLE is required"))?,
            heartbeat_interval: AgentConfig::DEFAULT_HEARTBEAT_INTERVAL,
            heartbeat_ttl: AgentConfig::DEFAULT_HEARTBEAT_TTL,
            drain_timeout: AgentConfig::DEFAULT_DRAIN_TIMEOUT,
            maximum_allocations: AgentConfig::DEFAULT_MAXIMUM_ALLOCATIONS,
            trust_tier: runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        }
    };
    if let Some(value) = &cli.server {
        config.server_url.clone_from(value);
    }
    if let Some(value) = &cli.runmat {
        config.runmat_executable.clone_from(value);
    }
    if let Some(value) = &cli.state_directory {
        config.state_directory.clone_from(value);
    }
    if let Some(value) = cli.trust_tier {
        config.trust_tier = match value {
            TrustTier::CustomerTrusted => {
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted
            }
            TrustTier::HostedOrdinary => {
                runmat_execution::security::ExecutionTrustTier::HostedOrdinary
            }
        };
    }
    config.validate()?;
    Ok(config)
}

fn execute_service(cli: &Cli, command: &ServiceCommand) -> anyhow::Result<()> {
    match command {
        ServiceCommand::Install { dry_run } => install_service(cli, *dry_run),
        ServiceCommand::Print => install_service(cli, true),
        ServiceCommand::Uninstall { dry_run } => {
            let plan = runmat_node_agent::service_install::uninstall_plan()?;
            if *dry_run {
                println!("{}", serde_json::to_string_pretty(&plan)?);
                return Ok(());
            }
            runmat_node_agent::service_install::apply_uninstall(&plan)?;
            println!("stopped and removed RunMat node agent service");
            Ok(())
        }
    }
}

fn install_service(cli: &Cli, dry_run: bool) -> anyhow::Result<()> {
    let config = config(cli, true)?;
    let file_config = runmat_node_agent::AgentFileConfig::from_runtime(&config);
    let plan =
        runmat_node_agent::service_install::install_plan(&file_config, &std::env::current_exe()?)?;
    if dry_run {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else {
        runmat_node_agent::service_install::apply_install(&plan, &config.state_directory)?;
        println!("installed and started RunMat node agent service");
    }
    Ok(())
}
