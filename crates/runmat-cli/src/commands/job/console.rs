use std::io::Write;

use anyhow::Result;
use runmat_execution::console::{
    ConsoleCompletion, ConsoleLimits, ConsoleStream, ConsoleTranscript,
};
use runmat_execution_artifact::ProgramExecutionOutputs;

// Per-source capture bounds. The terminal transport also bounds the aggregate
// inventory across worker attempts before retaining their transcripts.
pub(super) const TERMINAL_CONSOLE_LIMITS: ConsoleLimits = ConsoleLimits {
    max_records: 4096,
    max_text_bytes: 1024 * 1024,
};

pub(super) fn validate_requested_capture(
    outputs: &ProgramExecutionOutputs,
    requested: Option<ConsoleLimits>,
    requested_budget: Option<runmat_execution_artifact::ProgramExecutionConsoleBudget>,
) -> Result<()> {
    outputs.validate()?;
    anyhow::ensure!(
        outputs.console_budget == requested_budget,
        "remote console aggregate budget differs from the submitted allocation"
    );
    match (requested, outputs.root_console()) {
        (Some(limits), Some(root)) if root.limits() == limits => {}
        (None, None) if outputs.console_scopes.is_empty() => {}
        _ => anyhow::bail!("remote console capture differs from the submitted request"),
    }
    Ok(())
}

pub(super) fn print_outputs(outputs: &ProgramExecutionOutputs) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    let mut stderr = std::io::stderr().lock();
    if let Some(root) = outputs.console.as_deref() {
        write_transcript(root, &mut stdout, &mut stderr)?;
    }
    for scope in &outputs.console_scopes {
        writeln!(
            stderr,
            "Remote task {} (attempt {}):",
            scope.task_id, scope.attempt_id
        )?;
        stderr.flush()?;
        write_transcript(&scope.transcript, &mut stdout, &mut stderr)?;
    }
    Ok(())
}

pub(super) fn write_transcript(
    transcript: &ConsoleTranscript,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
) -> Result<()> {
    transcript.validate()?;
    for record in transcript.records() {
        match record.stream {
            ConsoleStream::Stdout => {
                stdout.write_all(record.text.as_bytes())?;
                stdout.flush()?;
            }
            ConsoleStream::Stderr => {
                stderr.write_all(record.text.as_bytes())?;
                stderr.flush()?;
            }
            // Preserve the caller's terminal and redirected output when replaying
            // a clear-screen control record.
            ConsoleStream::ClearScreen => {}
        }
    }
    match transcript.completion() {
        ConsoleCompletion::Complete => {}
        ConsoleCompletion::Interrupted => {
            writeln!(
                stderr,
                "Remote console capture was interrupted; output is a retained prefix."
            )?;
        }
        ConsoleCompletion::Rejected { reason } => {
            writeln!(stderr, "Remote console capture stopped at its bound ({reason:?}); output is a retained prefix.")?;
        }
    }
    stdout.flush()?;
    stderr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::console::ConsoleCapture;

    #[test]
    fn attached_capture_matches_submitted_source_and_aggregate_controls() {
        let budget =
            runmat_execution_artifact::ProgramExecutionConsoleBudget::for_driver_allocation(
                1024 * 1024 * 1024,
            );
        let empty = ProgramExecutionOutputs::new(
            runmat_execution_artifact::ProgramExecutionResponse::Success { outputs: vec![] },
            vec![],
        )
        .unwrap();
        assert!(validate_requested_capture(&empty, None, None).is_ok());
        assert!(
            validate_requested_capture(&empty, Some(TERMINAL_CONSOLE_LIMITS), Some(budget))
                .is_err()
        );
        let transcript = ConsoleCapture::new(TERMINAL_CONSOLE_LIMITS)
            .unwrap()
            .finish(true);
        let mut outputs = ProgramExecutionOutputs::with_console_scopes(
            runmat_execution_artifact::ProgramExecutionResponse::Success { outputs: vec![] },
            vec![],
            Some(Box::new(transcript)),
            vec![],
            Some(budget),
        )
        .unwrap();
        assert!(
            validate_requested_capture(&outputs, Some(TERMINAL_CONSOLE_LIMITS), Some(budget))
                .is_ok()
        );
        assert!(validate_requested_capture(&outputs, None, None).is_err());
        let other_budget =
            runmat_execution_artifact::ProgramExecutionConsoleBudget::for_driver_allocation(
                64 * 1024 * 1024,
            );
        assert!(validate_requested_capture(
            &outputs,
            Some(TERMINAL_CONSOLE_LIMITS),
            Some(other_budget)
        )
        .is_err());
        let scope = runmat_execution_artifact::ProgramExecutionConsoleScope {
            scope_id: runmat_execution::ExecutionScopeId::derive(&[b"delegated"]),
            pool_id: runmat_execution::PoolId::derive(&[b"delegated"]),
            task_id: runmat_execution::TaskId::derive(&[b"delegated"]),
            attempt_id: runmat_execution::identity::AttemptId::derive(&[b"delegated"]),
            transcript: outputs.console.take().unwrap(),
        };
        outputs.delegated_console_root = Some(scope.identity());
        outputs.console_scopes.push(scope);
        assert!(
            validate_requested_capture(&outputs, Some(TERMINAL_CONSOLE_LIMITS), Some(budget))
                .is_ok()
        );
    }

    #[test]
    fn terminal_console_preserves_streams_unicode_and_incomplete_prefixes() {
        let mut capture = ConsoleCapture::new(TERMINAL_CONSOLE_LIMITS).unwrap();
        capture
            .push(ConsoleStream::Stdout, "result: λ\n".into())
            .unwrap();
        capture
            .push(ConsoleStream::ClearScreen, String::new())
            .unwrap();
        capture
            .push(ConsoleStream::Stderr, "warning\n".into())
            .unwrap();
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        write_transcript(&capture.finish(false), &mut stdout, &mut stderr).unwrap();
        assert_eq!(stdout, "result: λ\n".as_bytes());
        let stderr = String::from_utf8(stderr).unwrap();
        assert!(stderr.starts_with("warning\n"));
        assert!(stderr.contains("retained prefix"));
        assert!(!stderr.contains('\u{1b}'));
    }
}
