//! Borrowed child-scope evidence and opaque admission, without transport DTOs.
use runmat_execution::console::{ConsoleLimits, ConsoleTranscript};
use runmat_execution::{ExecutionScopeId, PoolId};

pub use runmat_execution::console::ConsoleSourceIdentity as RuntimeConsoleScopeIdentity;

#[derive(Clone, Copy)]
pub struct RuntimeConsoleScopeRef<'a> {
    pub identity: RuntimeConsoleScopeIdentity,
    pub transcript: &'a ConsoleTranscript,
}

/// One authenticated root ledger is shared across every nested driver. The
/// runtime carries this port; native hosts supply its existing resource owner.
pub trait RuntimeConsoleAdmission: Send + Sync {
    fn max_scopes(&self) -> u32;
    fn reserve(
        &self,
        identity: RuntimeConsoleScopeIdentity,
        limits: ConsoleLimits,
    ) -> Result<(), String>;
    fn admit_record(
        &self,
        identity: RuntimeConsoleScopeIdentity,
        stream: runmat_execution::console::ConsoleStream,
        text: &str,
    ) -> Result<(), String>;
    fn settle(
        &self,
        identity: RuntimeConsoleScopeIdentity,
        transcript: &ConsoleTranscript,
    ) -> Result<(), String>;
    /// A worker receives only this finite subtree share, reserved by the
    /// original root before it starts another process capture.
    fn quote_delegate(
        &self,
        _limits: ConsoleLimits,
        _max_in_flight: u32,
    ) -> Result<runmat_execution::console::ConsoleBudget, String> {
        Err("console admission owner cannot quote a worker subtree".into())
    }
    fn delegate(
        &self,
        _identity: RuntimeConsoleScopeIdentity,
        _limits: ConsoleLimits,
        _max_in_flight: u32,
    ) -> Result<runmat_execution::console::ConsoleBudget, String> {
        Err("console admission owner cannot delegate a worker subtree".into())
    }
    fn settle_subtree(
        &self,
        _identity: RuntimeConsoleScopeIdentity,
        _root: Option<&ConsoleTranscript>,
        _scopes: &[runmat_execution::console::ScopedConsoleTranscript],
    ) -> Result<(), String> {
        Err("console admission owner cannot settle a worker subtree".into())
    }
}

/// Sources retain original child-driver owners, never copied transcripts.
/// Each callback runs while its original evidence owner is borrowed/locked;
/// serializers must consume the reference before the callback returns.
pub trait RuntimeRetainedConsoleSource: Send + Sync {
    fn source_identity(&self) -> Option<(ExecutionScopeId, PoolId)>;
    fn visit(
        &self,
        visitor: &mut dyn FnMut(RuntimeConsoleScopeRef<'_>) -> Result<(), String>,
    ) -> Result<(), String>;
}

#[derive(Default)]
pub(super) struct RetainedConsoleSources {
    sources: std::sync::Mutex<Vec<std::sync::Arc<dyn RuntimeRetainedConsoleSource>>>,
}

impl RetainedConsoleSources {
    pub(super) fn retain(
        &self,
        source: std::sync::Arc<dyn RuntimeRetainedConsoleSource>,
        max_sources: u32,
    ) -> Result<(), String> {
        let identity = source
            .source_identity()
            .ok_or("child console source has no driver identity")?;
        let mut sources = self
            .sources
            .try_lock()
            .map_err(|_| "child console inventory is being serialized")?;
        if sources
            .iter()
            .any(|saved| saved.source_identity() == Some(identity))
        {
            return Ok(());
        }
        if sources.len() >= max_sources as usize {
            return Err(
                "child console source inventory exceeds the admitted root scope limit".into(),
            );
        }
        sources
            .try_reserve(1)
            .map_err(|_| "child console source inventory allocation failed")?;
        sources.push(source);
        Ok(())
    }
}

impl RuntimeRetainedConsoleSource for RetainedConsoleSources {
    fn source_identity(&self) -> Option<(ExecutionScopeId, PoolId)> {
        None
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(RuntimeConsoleScopeRef<'_>) -> Result<(), String>,
    ) -> Result<(), String> {
        let sources = self
            .sources
            .try_lock()
            .map_err(|_| "child console inventory is being changed")?;
        for source in sources.iter() {
            source.visit(visitor)?;
        }
        Ok(())
    }
}
