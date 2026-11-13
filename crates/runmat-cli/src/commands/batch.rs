use std::path::Path;

use anyhow::{Context, Result};
use runmat_execution_runner_native::supervisor::{
    complete_batch_driver, complete_batch_driver_with_response, execute_and_complete_graph_batch,
    execute_and_complete_program_batch, execute_graph_batch, prepare_batch_driver,
    BatchDriverInvocation, BatchSubmission, JobSubmissionDisposition, LocalJobRecord,
    LocalJobState, LocalSupervisorClient,
};
use runmat_execution_runner_native::WorkerResponse;
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::{HostCommand, StdioPolicy};

use crate::cli::BatchCommand;
use crate::presentation;

mod attach;

pub async fn execute(command: BatchCommand) -> Result<()> {
    let client = LocalSupervisorClient::for_current_executable()?;
    match command {
        BatchCommand::Submit {
            file,
            idempotency_key,
            retention_hours,
            json,
            args,
        } => {
            crate::commands::script::validate_local_script_arguments(&args)?;
            let file = std::fs::canonicalize(&file)
                .with_context(|| format!("failed to resolve batch source {}", file.display()))?;
            let source_name = source_name(&file)?;
            let source = std::fs::read(&file)
                .with_context(|| format!("failed to read batch source {}", file.display()))?;
            let working_directory = std::fs::canonicalize(std::env::current_dir()?)
                .context("resolve working directory")?;
            let retention_millis = retention_hours
                .checked_mul(60 * 60 * 1000)
                .context("batch retention exceeds supported duration")?;
            let (record, submission_disposition) = client
                .submit(BatchSubmission {
                    source_name,
                    source,
                    arguments: args,
                    working_directory: working_directory.to_string_lossy().into_owned(),
                    idempotency_key,
                    retention_millis,
                })
                .await?;
            let duplicate = submission_disposition == JobSubmissionDisposition::Duplicate;
            if json {
                println!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "job": record,
                        "duplicate": duplicate,
                    }))?
                );
            } else {
                let styles = presentation::stdout();
                let disposition = if duplicate {
                    "existing idempotent job"
                } else {
                    "submitted"
                };
                println!(
                    "{} {} ({disposition})",
                    styles.success("Job"),
                    record.handle.id
                );
            }
        }
        BatchCommand::List { json } => {
            let records = client.list().await?;
            if json {
                println!("{}", serde_json::to_string(&records)?);
            } else if records.is_empty() {
                println!("{}", presentation::stdout().muted("No local batch jobs."));
            } else {
                for record in records {
                    print_record(&record);
                }
            }
        }
        BatchCommand::Show { job_id, json } => {
            let record = client.show(job_id).await?;
            if json {
                println!("{}", serde_json::to_string(&record)?);
            } else {
                print_record(&record);
                if let Some(message) = &record.message {
                    println!("  {}", presentation::stdout().muted(message));
                }
            }
        }
        BatchCommand::Attach { job_id, no_follow } => {
            attach::run(&client, job_id, no_follow).await?;
        }
        BatchCommand::Cancel { job_id, json } => {
            let record = client.cancel(job_id).await?;
            if json {
                println!("{}", serde_json::to_string(&record)?);
            } else {
                print_record(&record);
            }
        }
    }
    Ok(())
}

pub async fn run_driver() -> std::process::ExitCode {
    let invocation = match prepare_batch_driver() {
        Ok(invocation) => invocation,
        Err(error) => {
            eprintln!("runmat batch driver failed to prepare: {error}");
            return std::process::ExitCode::from(2);
        }
    };
    let job_directory = invocation.job_directory().to_path_buf();
    let outcome = match invocation {
        BatchDriverInvocation::Script {
            source_path,
            arguments,
            working_directory,
            ..
        } => run_script_driver(source_path, arguments, working_directory).await,
        BatchDriverInvocation::Program {
            attempt,
            submission,
            lease,
            ..
        } => {
            let execution =
                match execute_and_complete_program_batch(*attempt, *submission, *lease).await {
                    Ok(execution) => execution,
                    Err(error) => {
                        let message = error.to_string();
                        // Original provider ownership/report recovery records this loss.
                        // A script completion cannot stand in for a portable attempt.
                        eprintln!("runmat batch driver failed: {message}");
                        return std::process::ExitCode::from(1);
                    }
                };
            let success = matches!(
                &execution.response,
                WorkerResponse::Success { .. } | WorkerResponse::ExternalizedSuccess { .. }
            );
            return if success {
                std::process::ExitCode::SUCCESS
            } else {
                std::process::ExitCode::from(1)
            };
        }
        BatchDriverInvocation::GraphTask {
            attempt,
            submission,
            lease,
            ..
        } => {
            let execution =
                match execute_and_complete_graph_batch(*attempt, *submission, *lease).await {
                    Ok(execution) => execution,
                    Err(error) => {
                        eprintln!("runmat Graph batch driver failed: {error}");
                        return std::process::ExitCode::from(1);
                    }
                };
            return if matches!(execution.response, WorkerResponse::GraphSuccess { .. }) {
                std::process::ExitCode::SUCCESS
            } else {
                std::process::ExitCode::from(1)
            };
        }
        BatchDriverInvocation::Graph {
            job_id, submission, ..
        } => {
            let response = execute_graph_batch(*submission, job_id).await;
            let success = matches!(response, WorkerResponse::GraphSuccess { .. });
            if let Err(error) = complete_batch_driver_with_response(&job_directory, response) {
                Err(error.to_string())
            } else if success {
                return std::process::ExitCode::SUCCESS;
            } else {
                return std::process::ExitCode::from(1);
            }
        }
    };
    let (success, exit_code, message) = match outcome {
        Ok(exit) => (
            exit.success,
            exit.code,
            (!exit.success).then(|| "batch script process failed".into()),
        ),
        Err(error) => (false, None, Some(error)),
    };
    if let Err(error) = complete_batch_driver(&job_directory, success, exit_code, message) {
        eprintln!("runmat batch driver failed to commit completion: {error}");
        return std::process::ExitCode::from(2);
    }
    match exit_code {
        Some(code) if (0..=255).contains(&code) => std::process::ExitCode::from(code as u8),
        _ if success => std::process::ExitCode::SUCCESS,
        _ => std::process::ExitCode::from(1),
    }
}

async fn run_script_driver(
    source_path: std::path::PathBuf,
    arguments: Vec<String>,
    working_directory: std::path::PathBuf,
) -> Result<runmat_process_host::ProcessExit, String> {
    let mut command = match std::env::current_exe() {
        Ok(executable) => HostCommand::new(executable),
        Err(error) => return Err(error.to_string()),
    };
    command.arguments = vec!["run".into(), source_path.to_string_lossy().into_owned()];
    if !arguments.is_empty() {
        command.arguments.push("--".into());
        command.arguments.extend(arguments);
    }
    command.working_directory = Some(working_directory);
    command.environment_policy = EnvironmentPolicy::Inherit;
    command.stdio = StdioPolicy::Inherit;
    match command.spawn().await {
        Ok(mut child) => child.wait().await.map_err(|error| error.to_string()),
        Err(error) => Err(error.to_string()),
    }
}

fn source_name(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .context("batch source name must be valid UTF-8")
}

fn print_record(record: &LocalJobRecord) {
    println!(
        "{}  {:<13}  submitted={}  updated={}",
        record.handle.id,
        state_name(record.state),
        record.submitted_unix_millis,
        record.updated_unix_millis
    );
}

const fn state_name(state: LocalJobState) -> &'static str {
    match state {
        LocalJobState::Queued => "queued",
        LocalJobState::Starting => "starting",
        LocalJobState::Running => "running",
        LocalJobState::Cancelling => "cancelling",
        LocalJobState::Succeeded => "succeeded",
        LocalJobState::Failed => "failed",
        LocalJobState::Cancelled => "cancelled",
        LocalJobState::Indeterminate => "indeterminate",
    }
}
