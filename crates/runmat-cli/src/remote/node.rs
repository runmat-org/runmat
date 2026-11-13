use std::sync::Arc;

use anyhow::{Context as _, Result};
use runmat_node_agent::enrollment::CredentialStore;
use runmat_node_agent::service::{HttpNodeControlPlane, NodeAgentService, Shutdown};
use runmat_node_agent::{AgentConfig, AgentFileConfig};

use crate::cli::{
    ClusterJoinArgs, NodeArgs, NodeCommand, NodeOptions, NodeServiceCommand, NodeTrustTier,
};

pub(super) async fn join(args: ClusterJoinArgs) -> Result<()> {
    use runmat_node_agent::config::ServerEndpoint;
    use runmat_node_agent::enrollment::{register_authenticated, RegistrationTarget};
    use runmat_server_client::auth::{build_public_client, resolve_auth_token, RemoteConfig};

    let config = config(&args.options, args.service)?;
    let endpoint = ServerEndpoint::parse(&config.server_url)?;
    let mut account = RemoteConfig::load()?;
    let token = resolve_auth_token(&mut account, endpoint.as_str()).await?;
    let client = build_public_client(endpoint.as_str(), &token)?;
    let org = super::organization::resolve(&client, &account, args.org, false).await?;
    let cluster =
        runmat_server_client::execution::org_cluster_by_name(&client, &org, &args.cluster).await?;
    register_authenticated(
        &CredentialStore::new(&config.state_directory),
        RegistrationTarget::new(endpoint, org, cluster),
        runmat_node_agent::inventory::collect(config.trust_tier)?,
        config.heartbeat_ttl.as_secs(),
        &token,
    )
    .await?;
    // Account authority is bootstrap-only. The foreground/service agent loads
    // its own endpoint-bound machine credential from the registered directory.
    drop(client);
    drop(token);
    if args.service {
        install_service(&args.options, false)
    } else {
        run(config).await
    }
}

pub(super) async fn execute(args: NodeArgs) -> Result<()> {
    let NodeArgs {
        options: args,
        command,
    } = args;
    match &command {
        NodeCommand::Status => {
            #[derive(serde::Serialize)]
            #[serde(rename_all = "camelCase")]
            struct RegistrationStatus<'a> {
                server: &'a runmat_node_agent::config::ServerEndpoint,
                node_id: &'a str,
                cluster_id: &'a str,
                org_id: &'a str,
            }
            let config = config(&args, false)?;
            let credential = CredentialStore::new(&config.state_directory).load()?;
            println!(
                "{}",
                serde_json::to_string(&RegistrationStatus {
                    server: &credential.server_endpoint,
                    node_id: &credential.node_id,
                    cluster_id: &credential.cluster_id,
                    org_id: &credential.org_id,
                })?
            );
            Ok(())
        }
        NodeCommand::Inventory => {
            let inventory = runmat_node_agent::inventory::collect(trust_tier(&args))?;
            println!("{}", serde_json::to_string_pretty(&inventory)?);
            Ok(())
        }
        NodeCommand::Enroll { token, token_stdin } => {
            let token = if *token_stdin {
                use std::io::Read as _;
                let mut value = String::new();
                std::io::stdin()
                    .take(16 * 1024 + 1)
                    .read_to_string(&mut value)?;
                if value.len() > 16 * 1024 || value.trim().is_empty() {
                    anyhow::bail!("stdin enrollment token is empty or exceeds 16 KiB");
                }
                value.trim().to_owned()
            } else {
                token.clone().context("an enrollment token is required")?
            };
            let config = config(&args, false)?;
            let inventory = runmat_node_agent::inventory::collect(config.trust_tier)?;
            let control = Arc::new(HttpNodeControlPlane::new(config.server_url.clone())?);
            runmat_node_agent::enrollment::enroll(
                control,
                &CredentialStore::new(&config.state_directory),
                runmat_node_agent::config::ServerEndpoint::parse(&config.server_url)?,
                token,
                inventory,
                config.heartbeat_ttl.as_secs(),
            )
            .await?;
            println!("node enrolled");
            Ok(())
        }
        NodeCommand::Run => run(config(&args, false)?).await,
        NodeCommand::RotateCredential => {
            let config = config(&args, false)?;
            let control = Arc::new(HttpNodeControlPlane::new(config.server_url.clone())?);
            let mut service = NodeAgentService::load(config, control)?;
            service.rotate_credential().await?;
            println!("node credential rotated");
            Ok(())
        }
        NodeCommand::Service { command } => execute_service(&args, command),
        NodeCommand::WindowsServiceRun => {
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

fn config(args: &NodeOptions, service_defaults: bool) -> Result<AgentConfig> {
    let mut config = if let Some(path) = &args.node_config {
        AgentFileConfig::load(path)?.into_runtime()?
    } else {
        AgentConfig {
            state_directory: if service_defaults {
                runmat_node_agent::service_install::service_state_directory()?
            } else {
                AgentConfig::default_state_directory()?
            },
            server_url: runmat_server_client::auth::resolve_server_url(
                &runmat_server_client::auth::RemoteConfig::load()?,
                args.server.clone(),
            )?,
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

fn trust_tier(args: &NodeOptions) -> runmat_execution::security::ExecutionTrustTier {
    match args.trust_tier {
        Some(NodeTrustTier::HostedOrdinary) => {
            runmat_execution::security::ExecutionTrustTier::HostedOrdinary
        }
        Some(NodeTrustTier::CustomerTrusted) | None => {
            runmat_execution::security::ExecutionTrustTier::CustomerTrusted
        }
    }
}

fn execute_service(args: &NodeOptions, command: &NodeServiceCommand) -> Result<()> {
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

fn install_service(args: &NodeOptions, dry_run: bool) -> Result<()> {
    let config = config(args, true)?;
    let file_config = AgentFileConfig::from_runtime(&config);
    let plan =
        runmat_node_agent::service_install::install_plan(&file_config, &std::env::current_exe()?)?;
    if dry_run {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else {
        let credential = CredentialStore::new(&config.state_directory)
            .load()
            .context(
            "enroll this machine using the same --state-directory before installing its service",
        )?;
        if credential.server_endpoint
            != runmat_node_agent::config::ServerEndpoint::parse(&config.server_url)?
        {
            anyhow::bail!("node credentials belong to a different Server; service configuration was not installed");
        }
        runmat_node_agent::service_install::apply_install(&plan, &config.state_directory)?;
        println!("installed and started RunMat execution-node service");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser as _;

    use crate::cli::{Cli, ClusterCommand, Commands, NodeCommand};

    #[test]
    fn named_join_and_node_lifecycle_have_distinct_entrypoints() {
        let cli = Cli::try_parse_from([
            "runmat",
            "node",
            "--server",
            "https://api.runmat.com",
            "enroll",
            "--token",
            "secret",
        ])
        .expect("node enrollment command");
        let Some(Commands::Node(args)) = cli.command else {
            panic!("expected node command");
        };
        assert!(matches!(
            args.command,
            NodeCommand::Enroll { token: Some(token), token_stdin: false } if token == "secret"
        ));
        for service in [false, true] {
            let mut argv = vec!["runmat", "cluster", "join", "engineering"];
            if service {
                argv.push("--service");
            }
            let cli = Cli::try_parse_from(argv).expect("named join command");
            let Some(Commands::Cluster {
                cluster_command: ClusterCommand::Join(args),
            }) = cli.command
            else {
                panic!("expected named join");
            };
            assert_eq!(args.cluster, "engineering");
            assert_eq!(args.service, service);
            assert!(args.org.is_none());
        }
        assert!(Cli::try_parse_from([
            "runmat",
            "cluster",
            "join",
            "engineering",
            "--service",
            "--foreground",
        ])
        .is_err());
    }
}
