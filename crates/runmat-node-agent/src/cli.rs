use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum TrustTier {
    CustomerTrusted,
    HostedOrdinary,
}

#[derive(Debug, Parser)]
#[command(name = "runmat-node-agent", version)]
pub struct Cli {
    /// Load the canonical node-agent JSON configuration
    #[arg(long, env = "RUNMAT_NODE_AGENT_CONFIG")]
    pub config: Option<PathBuf>,
    #[arg(long, env = "RUNMAT_SERVER_URL")]
    pub server: Option<String>,
    #[arg(long)]
    pub state_directory: Option<PathBuf>,
    #[arg(long, env = "RUNMAT_EXECUTABLE")]
    pub runmat: Option<PathBuf>,
    #[arg(long, value_enum)]
    pub trust_tier: Option<TrustTier>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Enroll {
        #[arg(long, env = "RUNMAT_NODE_ENROLLMENT_TOKEN")]
        token: String,
    },
    Run,
    RotateCredential,
    Inventory,
    /// Install, inspect, or remove the operating-system service
    Service {
        #[command(subcommand)]
        command: ServiceCommand,
    },
    /// Internal Windows Service Control Manager entry point
    #[command(hide = true)]
    WindowsServiceRun,
}

#[derive(Debug, Subcommand)]
pub enum ServiceCommand {
    /// Persist the validated configuration and start the service at boot
    Install {
        /// Print the exact installation plan without changing the host
        #[arg(long)]
        dry_run: bool,
    },
    /// Stop and remove the operating-system service
    Uninstall {
        /// Print the exact removal plan without changing the host
        #[arg(long)]
        dry_run: bool,
    },
    /// Print the exact service files and commands for this host
    Print,
}
