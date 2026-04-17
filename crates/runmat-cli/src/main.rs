use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = runmat::Cli::parse();
    runmat::run_cli(cli).await
}
