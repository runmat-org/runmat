use anyhow::Result;
use tracing_subscriber::{fmt, EnvFilter};

use crate::cli::CliArgs;

pub fn init(args: &CliArgs) -> Result<()> {
    let level = if args.verbose { "debug" } else { "info" };
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| format!("runmatfunc={level}"));
    fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_writer(std::io::stderr)
        .init();
    Ok(())
}
