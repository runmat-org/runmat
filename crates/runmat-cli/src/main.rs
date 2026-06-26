use clap::{CommandFactory, FromArgMatches};
use std::io::Write;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    let matches = runmat::Cli::command().get_matches();
    let cli = match runmat::Cli::from_arg_matches(&matches) {
        Ok(cli) => cli,
        Err(err) => {
            eprintln!("Error: {err}");
            return ExitCode::from(1);
        }
    };
    let sources = runmat::CliOverrideSources::from_matches(&matches);
    let exit_code = match runmat::run_cli(cli, sources).await {
        Ok(()) => 0,
        Err(err) => {
            if err
                .downcast_ref::<runmat::AlreadyReportedCliError>()
                .is_none()
            {
                eprintln!("Error: {err}");
            }
            1
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
