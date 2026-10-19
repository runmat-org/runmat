//! Restore request-local session policy and source authority on completion or
//! cancellation. The guard borrows the session; it never snapshots workspace
//! values or retains an execution future.

use crate::session::source_scope::SourceCompositionScope;
use crate::session::RunMatSession;

pub(super) struct RequestExecutionScope<'a> {
    source: SourceCompositionScope<'a>,
    previous: RequestState,
}

struct RequestState {
    compatibility: runmat_parser::CompatMode,
    top_level_await: bool,
    dynamic_eval: bool,
    source_identity: Option<crate::abi::SourceIdentity>,
    primary: Option<crate::executable::CapturedPrimarySource>,
    workspace: crate::abi::WorkspaceHandle,
}

impl<'a> RequestExecutionScope<'a> {
    pub(super) fn enter(session: &'a mut RunMatSession) -> Self {
        let previous = RequestState {
            compatibility: session.compat_mode,
            top_level_await: session.top_level_await_enabled,
            dynamic_eval: session.dynamic_eval_enabled,
            source_identity: session.active_source_identity.take(),
            primary: session.active_primary_source.take(),
            workspace: session.abi_workspace_handle,
        };
        let source = SourceCompositionScope::enter(session);
        Self { source, previous }
    }

    pub(super) fn session(&mut self) -> &mut RunMatSession {
        self.source.session()
    }
}

impl Drop for RequestExecutionScope<'_> {
    fn drop(&mut self) {
        let previous = &mut self.previous;
        let session = self.source.session();
        session.compat_mode = previous.compatibility;
        session.top_level_await_enabled = previous.top_level_await;
        session.dynamic_eval_enabled = previous.dynamic_eval;
        session.active_source_identity = previous.source_identity.take();
        session.active_primary_source = previous.primary.take();
        session.abi_workspace_handle = previous.workspace;
    }
}
