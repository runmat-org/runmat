//! Verification-only native meshing entrypoint with process allocation accounting.

#[path = "support/allocation_observer.rs"]
mod allocation_observer;
#[path = "support/native_meshing_host.rs"]
mod native_meshing_host;

use clap::{Parser, ValueEnum};
use runmat::cli::{MeshElementOrderArg, MeshSourceUnitsArg};
use runmat::commands::mesh::MeshCommand;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ElementOrderArgument {
    Tet4,
    Tet10,
}

#[derive(Debug, Parser)]
struct Arguments {
    source: PathBuf,
    #[arg(long, value_enum, default_value_t = MeshSourceUnitsArg::Meter)]
    source_units: MeshSourceUnitsArg,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    evidence: PathBuf,
    #[arg(long, default_value_t = 0.01)]
    target_size: f64,
    #[arg(long, default_value_t = 0.0001)]
    deviation: f64,
    #[arg(long, default_value_t = 10_000_000)]
    max_elements: u64,
    #[arg(long, value_enum, default_value_t = ElementOrderArgument::Tet4)]
    element_order: ElementOrderArgument,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[tokio::main]
async fn main() -> ExitCode {
    let _otel = runmat::observability::init_from_env();
    let _report = allocation_observer::ReportGuard::from_environment();
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
    let command = MeshCommand {
        source: arguments.source,
        source_units: arguments.source_units,
        output: Some(arguments.output),
        evidence: Some(arguments.evidence),
        target_size_m: arguments.target_size,
        maximum_deviation_m: arguments.deviation,
        element_order: match arguments.element_order {
            ElementOrderArgument::Tet4 => MeshElementOrderArg::Tet4,
            ElementOrderArgument::Tet10 => MeshElementOrderArg::Tet10,
        },
        maximum_elements: arguments.max_elements,
        deterministic_seed: arguments.seed,
        shared_interfaces: false,
        force: false,
        json: true,
    };
    let exit_code = match runmat::commands::mesh::execute(command) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("meshing verification failed: {error:#}");
            1
        }
    };
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    allocation_observer::write_environment_report();
    runmat_logging::shutdown_otel();
    #[cfg(unix)]
    // SAFETY: all benchmark outputs and allocation evidence are flushed above; bypassing native
    // CAD teardown mirrors the production CLI's process boundary after native CAD use.
    unsafe {
        libc::_exit(exit_code);
    }
    #[cfg(not(unix))]
    std::process::exit(exit_code);
}
