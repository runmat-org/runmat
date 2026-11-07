//! Borrow original runtime child sources only while terminal encoding uses them.
use runmat_execution::console::{ConsoleBudget, ConsoleTranscript};
use runmat_execution_artifact::{
    ArtifactError, ArtifactResult, ProgramExecutionConsoleScopeRef,
    ProgramExecutionConsoleScopeSequence, ProgramExecutionConsoleSource,
};
use runmat_runtime::context::RuntimeRetainedConsoleSource;
use std::sync::Arc;

/// Build the production console admission port from a trusted host's original
/// admitted or subdelegated quota. Validation checks finite bounds; this
/// constructor does not authenticate the caller or mint resource authority.
pub fn console_admission_from_subgrant(
    budget: ConsoleBudget,
) -> crate::NativeExecutionResult<
    std::sync::Arc<dyn runmat_runtime::context::RuntimeConsoleAdmission>,
> {
    Ok(crate::console_retention::ConsoleRetentionOwner::from_subgrant(budget)?)
}

/// The console subtree grant a placed worker receives with its console controls:
/// the quote the native provider sends in its launch envelope. It derives from the
/// task's reserved memory and fits within the task's reserved artifact bytes.
pub fn console_source_subgrant(
    resources: &runmat_execution::resource::ResourceRequest,
    controls: runmat_execution::console::ConsoleLimits,
) -> crate::NativeExecutionResult<ConsoleBudget> {
    crate::console_retention::ConsoleRetentionOwner::reserved_source_subgrant(resources, controls)
}

/// Root capture plus original child-driver owners. Moving this value transfers
/// handles and the original root prefix, never a transcript-sized copy.
pub(crate) struct NativeCapturedConsole {
    pub transcript: Box<ConsoleTranscript>,
    pub scopes: Arc<dyn RuntimeRetainedConsoleSource>,
    pub budget: ConsoleBudget,
}

pub(crate) struct RuntimeConsoleSourceAdapter<'a>(pub &'a dyn RuntimeRetainedConsoleSource);

impl ProgramExecutionConsoleSource for RuntimeConsoleSourceAdapter<'_> {
    fn visit(
        &self,
        visitor: &mut dyn FnMut(ProgramExecutionConsoleScopeRef<'_>) -> ArtifactResult<()>,
    ) -> ArtifactResult<()> {
        self.0
            .visit(&mut |scope| {
                visitor(ProgramExecutionConsoleScopeRef {
                    scope_id: scope.identity.scope_id,
                    pool_id: scope.identity.pool_id,
                    task_id: scope.identity.task_id,
                    attempt_id: scope.identity.attempt_id,
                    transcript: scope.transcript,
                })
                .map_err(|error| error.to_string())
            })
            .map_err(ArtifactError::Invalid)
    }
}

impl NativeCapturedConsole {
    pub fn source(&self) -> RuntimeConsoleSourceAdapter<'_> {
        RuntimeConsoleSourceAdapter(self.scopes.as_ref())
    }
}

pub(crate) fn validate_imported_subtree(
    root: Option<&ConsoleTranscript>,
    scopes: &[runmat_execution::console::ScopedConsoleTranscript],
    budget: Option<ConsoleBudget>,
) -> Result<(), String> {
    struct Imported<'a>(&'a [runmat_execution::console::ScopedConsoleTranscript]);
    impl ProgramExecutionConsoleSource for Imported<'_> {
        fn visit(
            &self,
            visitor: &mut dyn FnMut(ProgramExecutionConsoleScopeRef<'_>) -> ArtifactResult<()>,
        ) -> ArtifactResult<()> {
            for scope in self.0 {
                visitor(ProgramExecutionConsoleScopeRef {
                    scope_id: scope.scope_id,
                    pool_id: scope.pool_id,
                    task_id: scope.task_id,
                    attempt_id: scope.attempt_id,
                    transcript: &scope.transcript,
                })?;
            }
            Ok(())
        }
    }
    // Reuse canonical aggregate/duplicate validation without building another
    // transcript inventory or encoded transcript buffer.
    serde_json::to_writer(
        std::io::sink(),
        &ProgramExecutionConsoleScopeSequence::new(root, &Imported(scopes), budget),
    )
    .map_err(|error| error.to_string())
}

pub(crate) fn encode_worker_terminal(
    response: &crate::protocol::WorkerResponse,
    console: Option<&runmat_execution::console::ConsoleReceipt>,
    captured: Option<&NativeCapturedConsole>,
) -> Result<Vec<u8>, String> {
    #[derive(serde::Serialize)]
    struct Terminal<'a> {
        message: &'static str,
        response: &'a crate::protocol::WorkerResponse,
        console: Option<&'a runmat_execution::console::ConsoleReceipt>,
        console_scopes: ProgramExecutionConsoleScopeSequence<'a>,
    }
    struct Empty;
    impl ProgramExecutionConsoleSource for Empty {
        fn visit(
            &self,
            _: &mut dyn FnMut(ProgramExecutionConsoleScopeRef<'_>) -> ArtifactResult<()>,
        ) -> ArtifactResult<()> {
            Ok(())
        }
    }
    let adapter = captured.map(NativeCapturedConsole::source);
    let source: &dyn ProgramExecutionConsoleSource = adapter
        .as_ref()
        .map(|source| source as &dyn ProgramExecutionConsoleSource)
        .unwrap_or(&Empty);
    let scopes = ProgramExecutionConsoleScopeSequence::new(
        captured.map(|capture| capture.transcript.as_ref()),
        source,
        captured.map(|capture| capture.budget),
    );
    runmat_canonical_codec::encode_json_bounded(
        &Terminal {
            message: "completed",
            response,
            console,
            console_scopes: scopes,
        },
        u64::from(super::worker::WORKER_MAX_MESSAGE_BYTES),
    )
    .map_err(|error| error.to_string())
}
