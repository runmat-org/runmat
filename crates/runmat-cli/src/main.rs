use anyhow::Result;
use clap::{CommandFactory, FromArgMatches};

#[tokio::main]
async fn main() -> Result<()> {
    let matches = runmat::Cli::command().get_matches();
    let cli = runmat::Cli::from_arg_matches(&matches)?;
    let sources = runmat::CliOverrideSources::from_matches(&matches);
    runmat::run_cli(cli, sources).await
}
