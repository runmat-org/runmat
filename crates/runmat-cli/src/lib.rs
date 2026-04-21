pub mod app;
pub mod cli;
pub mod commands;
pub mod diagnostics;
pub mod logging;
pub mod remote;
pub mod telemetry;

use std::fmt;

pub use cli::{Cli, CliOverrideSources};

#[derive(Debug)]
pub struct AlreadyReportedCliError;

impl fmt::Display for AlreadyReportedCliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("command failed")
    }
}

impl std::error::Error for AlreadyReportedCliError {}

pub async fn run_cli(cli: Cli, sources: CliOverrideSources) -> anyhow::Result<()> {
    let result = app::bootstrap::run_cli(cli, sources).await;
    telemetry::shutdown();
    result
}
