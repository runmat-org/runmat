pub mod app;
pub mod cli;
pub mod commands;
pub mod diagnostics;
pub mod logging;
pub mod remote;
pub mod telemetry;

pub use cli::{Cli, CliOverrideSources};

pub async fn run_cli(cli: Cli, sources: CliOverrideSources) -> anyhow::Result<()> {
    let result = app::bootstrap::run_cli(cli, sources).await;
    telemetry::shutdown();
    result
}
