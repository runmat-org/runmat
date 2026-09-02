//! Governed end-to-end exact-CAD meshing corpus runner.

#[path = "support/meshing_corpus/mod.rs"]
mod meshing_corpus;
#[path = "support/native_meshing_host.rs"]
mod native_meshing_host;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use meshing_corpus::{CorpusReport, SelectedTier};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TierArgument {
    Small,
    Complete,
}

#[derive(Debug, Parser)]
struct Arguments {
    #[arg(long, default_value = "verification/meshing/integration.json")]
    manifest: PathBuf,
    #[arg(long, default_value = "target/meshing-corpus-report.json")]
    report: PathBuf,
    #[arg(long, value_enum, default_value_t = TierArgument::Small)]
    tier: TierArgument,
    #[arg(long)]
    fixture: Option<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let _otel = runmat::observability::init_from_env();
    let args = std::env::args_os().collect::<Vec<_>>();
    match native_meshing_host::dispatch_hidden_mode(&args).await {
        Ok(Some(exit_code)) => return exit_code,
        Ok(None) => {}
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    }

    let arguments = Arguments::parse_from(args);
    let tier = match arguments.tier {
        TierArgument::Small => SelectedTier::Small,
        TierArgument::Complete => SelectedTier::Complete,
    };
    let (exit_code, report) =
        match meshing_corpus::run(&arguments.manifest, tier, arguments.fixture.as_deref()) {
            Ok(report) => (u8::from(!report.passed), Some(report)),
            Err(error) => {
                eprintln!("meshing corpus failed before completion: {error:#}");
                (2, None)
            }
        };
    if let Some(report) = report {
        if let Err(error) = write_report(&arguments.report, &report) {
            eprintln!("failed to write meshing corpus report: {error:#}");
            return ExitCode::from(2);
        }
        eprintln!(
            "meshing corpus {}: {}",
            if report.passed { "passed" } else { "failed" },
            arguments.report.display()
        );
    }
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    runmat_logging::shutdown_otel();
    #[cfg(unix)]
    // SAFETY: all output is flushed above; bypassing process-wide native CAD teardown matches the
    // production mesh command's native worker boundary.
    unsafe {
        libc::_exit(exit_code.into());
    }
    #[cfg(not(unix))]
    std::process::exit(exit_code.into());
}

fn write_report(path: &std::path::Path, report: &CorpusReport) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(report)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
