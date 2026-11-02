//! Bounded original-attempt output pages, independent of numerical payloads.
use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;

use anyhow::{bail, Result};
use runmat_execution::identity::AttemptId;
use runmat_execution_runner_native::supervisor::{
    JobAttachment, LocalSupervisorClient, RawProcessOutputCursor,
};
use runmat_execution_runner_native::RawOutputState;

use crate::presentation;

pub(super) async fn run(
    client: &LocalSupervisorClient,
    job_id: runmat_execution::JobId,
    no_follow: bool,
) -> Result<()> {
    let mut stdout_offset = 0;
    let mut stderr_offset = 0;
    let mut cursors = HashMap::new();
    loop {
        let attachment = client.attach(job_id, stdout_offset, stderr_offset).await?;
        publish(&attachment.stdout, &attachment.stderr)?;
        let driver_progress = !attachment.stdout.is_empty() || !attachment.stderr.is_empty();
        stdout_offset = attachment.next_stdout_offset;
        stderr_offset = attachment.next_stderr_offset;
        let raw_progress = publish_attempt_pages(client, &attachment, &mut cursors).await?;
        // A terminal record is not an empty output page. Keep reading bounded
        // pages until both driver logs and original attempt journals are drained.
        if no_follow || (attachment.record.state.is_terminal() && !driver_progress && !raw_progress)
        {
            if attachment.record.state.is_terminal() {
                eprintln!(
                    "{} {}",
                    presentation::stderr().heading("Job"),
                    super::state_name(attachment.record.state)
                );
            }
            return Ok(());
        }
        let delay = if driver_progress || raw_progress {
            Duration::ZERO
        } else {
            Duration::from_millis(100)
        };
        tokio::select! {
            _ = tokio::time::sleep(delay) => {}
            _ = tokio::signal::ctrl_c() => {
                eprintln!("{}", presentation::stderr().muted("Detached; the batch job is still running."));
                return Ok(());
            }
        }
    }
}

async fn publish_attempt_pages(
    client: &LocalSupervisorClient,
    attachment: &JobAttachment,
    cursors: &mut HashMap<AttemptId, RawProcessOutputCursor>,
) -> Result<bool> {
    let Some(execution) = &attachment.execution else {
        return Ok(false);
    };
    let mut progress = false;
    for attempt in &execution.attempts {
        let cursor = cursors.entry(attempt.request.id).or_default();
        let Some(page) = client
            .raw_output_page(
                attachment.record.handle.clone(),
                attempt.request.id,
                *cursor,
            )
            .await?
        else {
            if attachment.record.state
                == runmat_execution_runner_native::supervisor::LocalJobState::Succeeded
                && !attempt.certifies_not_started()
            {
                bail!(
                    "successful job has no original application output journal for attempt {}",
                    attempt.request.id
                );
            }
            continue;
        };
        publish(page.stdout.as_slice(), page.stderr.as_slice())?;
        let advanced = page.next != *cursor;
        progress |= advanced;
        *cursor = page.next;
        if [page.stdout_state, page.stderr_state].iter().any(|state| {
            matches!(
                state,
                RawOutputState::ReadFailed | RawOutputState::CaptureFailed
            )
        }) {
            bail!(
                "application output for attempt {} was not completely retained",
                attempt.request.id
            );
        }
        if attachment.record.state.is_terminal()
            && !advanced
            && (!page.retired
                || [page.stdout_state, page.stderr_state].contains(&RawOutputState::Reading))
        {
            bail!(
                "terminal job lacks completed output and retirement for attempt {}",
                attempt.request.id
            );
        }
    }
    Ok(progress)
}

fn publish(stdout: &[u8], stderr: &[u8]) -> Result<()> {
    let mut out = std::io::stdout().lock();
    out.write_all(stdout)?;
    out.flush()?;
    drop(out);
    let mut err = std::io::stderr().lock();
    err.write_all(stderr)?;
    err.flush()?;
    Ok(())
}
