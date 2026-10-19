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
    /// Run one study and capture portable scenes and previews for an example package
    Snapshot {
        /// Root .fea source in the example project
        file: PathBuf,
        /// Select one study declared by the root module
        #[arg(long)]
        study: Option<String>,
        /// Snapshot metadata output; scenes and previews are written beside it
        #[arg(long)]
        out: PathBuf,
    },
    /// Render an original saved numerical result without meshing or solving
    #[command(hide = true)]
    RenderSnapshot {
        snapshot: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Verify a documentation project's original saved field pages without execution
    #[command(hide = true)]
    VerifySnapshot {
        /// Original snapshot metadata, relative to the project working directory
        snapshot: PathBuf,
        /// Cold field replay receipt
        #[arg(long)]
        out: PathBuf,
    },
    /// Evaluate checked criteria against an immutable saved result without solving again
    Reassess {
        /// Checked .fea source containing the revised assessment declarations
        file: PathBuf,
        /// Select one study declared by the root module
        #[arg(long)]
        study: Option<String>,
        /// Original saved study definition (sha256: followed by 64 lowercase hex digits)
        #[arg(long)]
        original_definition: runmat_execution::Digest,
        /// Content-addressed whole-study result manifest
        #[arg(long)]
        result_manifest: PathBuf,
        /// Emit the typed reassessment publication as JSON
        #[arg(long)]
        json: bool,
    },
}

impl FeaCommand {
    pub fn machine_output(&self) -> bool {
        match self {
            Self::Snapshot { .. } | Self::RenderSnapshot { .. } | Self::VerifySnapshot { .. } => {
                true
            }
            Self::Check { json, .. }
            | Self::Plan { json, .. }
            | Self::Run { json, .. }
            | Self::Reassess { json, .. } => *json,
        }
    }
}
