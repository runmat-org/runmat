use clap::{Args, Subcommand};
use std::path::PathBuf;

use super::ExecutionTargetArgs;

#[derive(Args, Clone, Debug, Default)]
pub struct FeaRootArgs {
    /// Select a study declared by the root module
    #[arg(long, conflicts_with = "sweep")]
    pub study: Option<String>,
    /// Select a sweep declared by the root module
    #[arg(long, conflicts_with = "study")]
    pub sweep: Option<String>,
}

#[derive(Subcommand, Clone, Debug)]
pub enum FeaCommand {
    /// Resolve and type-check a complete FEA source graph without executing it
    Check {
        /// Root .fea source file
        file: PathBuf,
        #[command(flatten)]
        root: FeaRootArgs,
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
    },
    /// Print the canonical logical phases and analysis definition for one study or sweep
    Plan {
        /// Root .fea source file
        file: PathBuf,
        #[command(flatten)]
        root: FeaRootArgs,
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
    },
    /// Materialize geometry, mesh, compile, solve, and publish one study or sweep
    Run {
        /// Root .fea source file
        file: PathBuf,
        #[command(flatten)]
        root: FeaRootArgs,
        #[command(flatten)]
        execution: ExecutionTargetArgs,
        /// Emit structured JSON
        #[arg(long)]
        json: bool,
    },
}

impl FeaCommand {
    pub fn machine_output(&self) -> bool {
        match self {
            Self::Check { json, .. } | Self::Plan { json, .. } | Self::Run { json, .. } => *json,
        }
    }
}
