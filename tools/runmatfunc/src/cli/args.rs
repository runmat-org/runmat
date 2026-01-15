use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Root CLI arguments for runmatfunc.
#[derive(Debug, Parser)]
#[command(author, version, about = "RunMat Function Manager", long_about = None)]
pub struct CliArgs {
    /// Optional subcommand
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Enable verbose logging
    #[arg(long)]
    pub verbose: bool,

    /// Override configuration file path
    #[arg(long = "config", value_name = "PATH")]
    pub config_path: Option<PathBuf>,
}

/// High-level command enum. Additional subcommands will be fleshed out during implementation.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Launch interactive TUI
    Browse,
    /// Run metadata discovery and print builtin manifest
    Manifest,
    /// Emit documentation bundle (JSON + d.ts)
    Docs {
        #[arg(long = "out-dir", value_name = "PATH")]
        out_dir: Option<String>,
    },
    /// Builtin authoring helpers
    Builtin {
        #[command(subcommand)]
        action: BuiltinAction,
    },
    /// Manage the headless job queue
    Queue {
        #[command(subcommand)]
        action: QueueAction,
    },
}

pub fn parse() -> CliArgs {
    CliArgs::parse()
}

#[derive(Debug, Subcommand)]
pub enum BuiltinAction {
    /// Generate or migrate a builtin implementation
    Generate {
        #[arg(long = "name", value_name = "NAME")]
        name: String,
        #[arg(long)]
        category: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        codex: bool,
        #[arg(long)]
        diff: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum QueueAction {
    /// Add a builtin to the queue
    Add {
        builtin: String,
        /// Optional category path like "array" or "math/reduction"
        #[arg(long)]
        category: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        codex: bool,
    },
    /// List pending jobs
    List,
    /// Run queued jobs headlessly
    Run {
        #[arg(long = "limit")]
        max: Option<usize>,
    },
    /// Clear all queued jobs
    Clear,
}
