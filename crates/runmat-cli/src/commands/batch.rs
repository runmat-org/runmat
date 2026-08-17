use std::io::Write;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use runmat_execution_runner_native::supervisor::{
    complete_batch_driver, complete_batch_driver_with_value, execute_program_batch,
    prepare_batch_driver, BatchDriverInvocation, BatchSubmission, LocalJobRecord, LocalJobState,
    LocalSupervisorClient,
};
use runmat_execution_runner_native::WorkerResponse;
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::{HostCommand, StdioPolicy};

use crate::cli::BatchCommand;
use crate::presentation;

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
            let (record, duplicate) = client
                .submit(BatchSubmission {
                    source_name,
                    source,
                    arguments: args,
                    working_directory: working_directory.to_string_lossy().into_owned(),
                    idempotency_key,
                    retention_millis,
                })
                .await?;
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
            attach(&client, job_id, no_follow).await?;
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
        BatchDriverInvocation::Program { submission, .. } => {
            match execute_program_batch(*submission).await {
                WorkerResponse::Success { value } => {
                    if let Err(error) = complete_batch_driver_with_value(
                        &job_directory,
                        true,
                        Some(0),
                        None,
                        Some(value),
                    ) {
                        Err(error.to_string())
                    } else {
                        return std::process::ExitCode::SUCCESS;
                    }
                }
                WorkerResponse::ExternalizedSuccess { .. } => Err(
                    "batch result uses externalized objects unsupported by the legacy batch sink"
                        .into(),
                ),
                WorkerResponse::Failure { message } => Err(message),
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

async fn attach(
    client: &LocalSupervisorClient,
    job_id: runmat_execution::JobId,
    no_follow: bool,
) -> Result<()> {
    let mut stdout_offset = 0;
    let mut stderr_offset = 0;
    loop {
        let attachment = client.attach(job_id, stdout_offset, stderr_offset).await?;
        std::io::stdout().write_all(&attachment.stdout)?;
        std::io::stdout().flush()?;
        std::io::stderr().write_all(&attachment.stderr)?;
        std::io::stderr().flush()?;
        stdout_offset = attachment.next_stdout_offset;
        stderr_offset = attachment.next_stderr_offset;
        if no_follow || attachment.record.state.is_terminal() {
            if attachment.record.state.is_terminal() {
                eprintln!(
                    "{} {}",
                    presentation::stderr().heading("Job"),
                    state_name(attachment.record.state)
                );
            }
            return Ok(());
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            _ = tokio::signal::ctrl_c() => {
                eprintln!("{}", presentation::stderr().muted("Detached; the batch job is still running."));
                return Ok(());
            }
        }
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
