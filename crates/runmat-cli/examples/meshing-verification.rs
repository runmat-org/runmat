//! Verification-only native meshing entrypoint with process allocation accounting.

#[path = "support/allocation_observer.rs"]
mod allocation_observer;

use clap::{Parser, ValueEnum};
use runmat::cli::MeshElementOrderArg;
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
    #[arg(long, default_value = "material")]
    material: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[tokio::main]
async fn main() -> ExitCode {
    let _report = allocation_observer::ReportGuard::from_environment();
    let args = std::env::args_os().collect::<Vec<_>>();
    match runmat_process_host::HiddenModeRegistry::standard().detect(args.clone()) {
        Ok(Some(runmat_process_host::HiddenMode::MeshingWorker)) => {
            return run_meshing_worker().await;
        }
        Ok(Some(_)) => {
            eprintln!("meshing verification does not support this hidden host mode");
            return ExitCode::from(2);
        }
        Ok(None) => {}
        Err(error) => {
            eprintln!("invalid RunMat host mode: {error}");
            return ExitCode::from(2);
        }
    }

    let arguments = Arguments::parse_from(args);
    let command = MeshCommand {
        source: arguments.source,
        output: Some(arguments.output),
        evidence: Some(arguments.evidence),
        target_size_m: arguments.target_size,
        maximum_deviation_m: arguments.deviation,
        element_order: match arguments.element_order {
            ElementOrderArgument::Tet4 => MeshElementOrderArg::Tet4,
            ElementOrderArgument::Tet10 => MeshElementOrderArg::Tet10,
        },
        material: arguments.material,
        maximum_elements: arguments.max_elements,
        deterministic_seed: arguments.seed,
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
    #[cfg(unix)]
    // SAFETY: all benchmark outputs and allocation evidence are flushed above; bypassing native
    // CAD teardown mirrors the production CLI's process boundary after native CAD use.
    unsafe {
        libc::_exit(exit_code);
    }
    #[cfg(not(unix))]
    std::process::exit(exit_code);
}

async fn run_meshing_worker() -> ExitCode {
    let result =
        match std::env::var_os(runmat_execution_runner_native::NATIVE_OBJECT_STORE_ROOT_ENV) {
            Some(root) => {
                runmat_execution_runner_native::run_meshing_worker_stdio(
                    std::sync::Arc::new(
                        runmat_execution_runner_native::native_meshing_kernel_dispatcher(),
                    ),
                    std::path::Path::new(&root),
                    runmat_execution_runner_native::NativeMeshingHostLimits::default(),
                )
                .await
            }
            None => Err(
                runmat_execution_runner_native::NativeExecutionError::Configuration(
                    "meshing worker object-store root is missing".into(),
                ),
            ),
        };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("runmat meshing verification worker failed: {error}");
            ExitCode::from(2)
        }
    }
}
