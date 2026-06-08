use clap::{CommandFactory, FromArgMatches};
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
    match runmat::run_cli(cli, sources).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            if err
                .downcast_ref::<runmat::AlreadyReportedCliError>()
                .is_none()
            {
                eprintln!("Error: {err}");
            }
            ExitCode::from(1)
        }
    }
}
