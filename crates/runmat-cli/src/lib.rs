pub mod app;
pub mod cli;
pub mod commands;
pub mod diagnostics;
pub mod logging;
pub mod remote;
pub mod telemetry;

pub use cli::Cli;

pub async fn run_cli(cli: Cli) -> anyhow::Result<()> {
    app::bootstrap::run_cli(cli).await
}
