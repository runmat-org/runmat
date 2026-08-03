use clap::{CommandFactory, FromArgMatches};
use std::io::Write;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    let args = std::env::args_os().collect::<Vec<_>>();
    match runmat_process_host::HiddenModeRegistry::standard().detect(args.clone()) {
        Ok(Some(runmat_process_host::HiddenMode::TestWorker)) => {
            return match runmat::commands::test::worker::run_stdio().await {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("runmat test worker failed: {error:#}");
                    ExitCode::from(2)
                }
            };
        }
        Ok(Some(runmat_process_host::HiddenMode::ExecutionWorker)) => {
            return match runmat_execution_runner_native::run_worker_stdio().await {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("runmat execution worker failed: {error}");
                    ExitCode::from(2)
                }
            };
        }
        Ok(Some(mode)) => {
            eprintln!(
                "RunMat host mode '{}' is not available in this build",
                mode.marker()
            );
            return ExitCode::from(2);
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
                    eprintln!("{}: {err}", runmat::presentation::stderr().error("Error"));
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
