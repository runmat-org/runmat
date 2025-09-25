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
        #[arg(long, default_value = "docs/generated")]
        out_dir: String,
    },
    /// Execute a builtin authoring job headlessly (placeholder)
    Builtin {
        name: String,
        #[arg(long)]
        model: Option<String>,
    },
}

pub fn parse() -> CliArgs {
    CliArgs::parse()
}
