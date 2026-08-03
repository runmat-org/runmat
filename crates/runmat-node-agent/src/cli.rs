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
    #[arg(long, env = "RUNMAT_SERVER_URL")]
    pub server: Option<String>,
    #[arg(long)]
    pub state_directory: Option<PathBuf>,
    #[arg(long, env = "RUNMAT_EXECUTABLE")]
    pub runmat: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "customer-trusted")]
    pub trust_tier: TrustTier,
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
}
