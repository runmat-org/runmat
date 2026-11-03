use clap::{CommandFactory, FromArgMatches};
use std::io::Write;
use std::process::ExitCode;

/// Deep async study pipelines (geometry, meshing, analysis publication) nest large futures. The
/// main thread's default stack is too small for them in unoptimized builds, so the command runs
/// on a thread with an explicit stack, and runtime workers get the same budget.
const COMMAND_STACK_BYTES: usize = 64 * 1024 * 1024;

fn main() -> ExitCode {
    let command = std::thread::Builder::new()
        .name("runmat-main".into())
        .stack_size(COMMAND_STACK_BYTES)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(COMMAND_STACK_BYTES)
                .build()
                .expect("start the tokio runtime")
                .block_on(run())
        })
        .expect("start the command thread");
    command
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

async fn run() -> ExitCode {
    let args = std::env::args_os().collect::<Vec<_>>();
    match runmat_process_host::HiddenModeRegistry::standard().detect(args.clone()) {
        Ok(Some(runmat_process_host::HiddenMode::TestWorker)) => {
            let _otel = runmat::observability::init_from_env();
            let _fea_admission = match runmat::app::bootstrap::install_local_fea_admission(None) {
                Ok(guard) => guard,
                Err(error) => {
                    eprintln!("runmat test worker admission failed: {error}");
                    return ExitCode::from(2);
                }
            };
            return match runmat::commands::test::worker::run_stdio().await {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("runmat test worker failed: {error:#}");
                    ExitCode::from(2)
                }
            };
        }
        Ok(Some(runmat_process_host::HiddenMode::ExecutionWorker)) => {
            let _otel = runmat::observability::init_from_env();
            let remote = std::env::var_os("RUNMAT_EXECUTION_WORKER_REMOTE").is_some();
            let _fea_admission = if remote {
                None
            } else {
                match runmat::app::bootstrap::install_local_fea_admission(None) {
                    Ok(guard) => Some(guard),
                    Err(error) => {
                        eprintln!("runmat local execution worker admission failed: {error}");
                        return ExitCode::from(2);
                    }
                }
            };
            if runmat_execution_runner_native::cooperative::is_cooperative_rank_process() {
                let result = tokio::task::spawn_blocking(
                    runmat_execution_runner_native::cooperative::run_cooperative_rank_from_env,
                )
                .await;
                return match result {
                    Ok(Ok(())) => ExitCode::SUCCESS,
                    Ok(Err(error)) => {
                        eprintln!("runmat cooperative rank failed: {error}");
                        ExitCode::from(2)
                    }
                    Err(error) => {
                        eprintln!("runmat cooperative rank panicked: {error}");
                        ExitCode::from(2)
                    }
                };
            }
            let result = if remote {
                runmat_execution_runner_native::run_remote_worker_from_env().await
            } else {
                runmat_execution_runner_native::run_worker_stdio().await
            };
            return match result {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("runmat execution worker failed: {error}");
                    ExitCode::from(2)
                }
            };
        }
        Ok(Some(runmat_process_host::HiddenMode::MeshingWorker)) => {
            let _otel = runmat::observability::init_from_env();
            let result = match std::env::var_os(
                runmat_execution_runner_native::NATIVE_OBJECT_STORE_ROOT_ENV,
            ) {
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
            return match result {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("runmat meshing worker failed: {error}");
                    ExitCode::from(2)
                }
            };
        }
        Ok(Some(runmat_process_host::HiddenMode::ExecutionDriver)) => {
            let _otel = runmat::observability::init_from_env();
            if std::env::var_os("RUNMAT_EXECUTION_DRIVER_LEASE_ID").is_some() {
                return match runmat_execution_runner_native::run_remote_driver_from_env().await {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(error) => {
                        eprintln!("runmat remote execution driver failed: {error}");
                        ExitCode::from(2)
                    }
                };
            }
            let _fea_admission = match runmat::app::bootstrap::install_local_fea_admission(None) {
                Ok(guard) => guard,
                Err(error) => {
                    eprintln!("runmat local batch admission failed: {error}");
                    return ExitCode::from(2);
                }
            };
            return runmat::commands::batch::run_driver().await;
        }
        Ok(Some(runmat_process_host::HiddenMode::LocalSupervisor)) => {
            let _otel = runmat::observability::init_from_env();
            return match runmat_execution_runner_native::supervisor::LocalSupervisorConfig::for_current_executable(
            ) {
                Ok(config) => {
                    match runmat_execution_runner_native::supervisor::run_local_supervisor(config)
                        .await
                    {
                        Ok(()) => ExitCode::SUCCESS,
                        Err(error) => {
                            eprintln!("runmat local execution supervisor failed: {error}");
                            ExitCode::from(2)
                        }
                    }
                }
                Err(error) => {
                    eprintln!("runmat local execution supervisor failed: {error}");
                    ExitCode::from(2)
                }
            };
        }
        Ok(Some(runmat_process_host::HiddenMode::ExtensionHost)) => {
            return match runmat_runtime::foreign::run_extension_host().await {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("runmat extension host failed: {error}");
                    ExitCode::from(2)
                }
            };
        }
        Ok(None) => {}
        Err(error) => {
            eprintln!("invalid RunMat host mode: {error}");
            return ExitCode::from(2);
        }
    }
    let requested_color = runmat::presentation::requested_color_mode(&args).unwrap_or_default();
    let matches = runmat::Cli::command()
        .color(runmat::presentation::clap_color_choice(requested_color))
        .styles(runmat::presentation::clap_styles())
        .get_matches_from(args);
    let cli = match runmat::Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(err) => {
            let styles = runmat::presentation::Presentation::detect(requested_color).stderr();
            eprintln!("{}: {err}", styles.error("Error"));
            return ExitCode::from(1);
        }
    };
    runmat::presentation::initialize(cli.color, runmat::presentation::cli_output_mode(&cli));
    let sources = runmat::CliOverrideSources::from_matches(&matches);
    let exit_code = match runmat::run_cli(cli, sources).await {
        Ok(()) => 0,
        Err(err) => {
            if let Some(status) = err.downcast_ref::<runmat::commands::test::TestCommandError>() {
                status.code()
            } else {
                if err
                    .downcast_ref::<runmat::AlreadyReportedCliError>()
                    .is_none()
                {
                    eprintln!("{}: {err:#}", runmat::presentation::stderr().error("Error"));
                }
                1
            }
        }
    };
    exit_after_native_cad_if_needed(exit_code);
    ExitCode::from(exit_code)
}

fn exit_after_native_cad_if_needed(exit_code: u8) {
    if !runmat_runtime::geometry::native_cad_backend_was_used() {
        return;
    }

    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    #[cfg(unix)]
    unsafe {
        libc::_exit(i32::from(exit_code));
    }

    #[cfg(not(unix))]
    std::process::exit(i32::from(exit_code));
}
