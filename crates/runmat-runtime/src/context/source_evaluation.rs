//! Executor-owned evaluation of source text in a session's base workspace.
//!
//! No caller-frame pointers are retained by this port. Executors acquire their
//! live, session-owned workspace authority when they evaluate a request.

use super::{RuntimeContext, RuntimeContextLocalId, RuntimeLanguagePolicy, RuntimeServiceFuture};
use crate::RuntimeError;
use runmat_types::SourceId;
use runmat_value::Value;
use std::rc::Rc;
use std::sync::Arc;

/// Retains an executor-issued source record. The executor retires the record
/// when the last lease drops; a callback must not leak permanent catalog rows.
pub trait SourceEvaluationProvenance: std::fmt::Debug {
    fn source_id(&self) -> SourceId;

    /// Authenticated runtime identity for compiler/executor admission. Test or
    /// external provenance that only reports a display ID cannot authorize a
    /// production source evaluation.
    fn runtime_source_ref(&self) -> Option<crate::source_context::RuntimeSourceRef> {
        None
    }

    /// Retain the exact issued source for selected-product admission. An
    /// external display-only provenance cannot mint a portable source owner.
    fn runtime_source_lease(&self) -> Option<crate::source_context::RuntimeSourceLease> {
        None
    }
}

/// Language settings captured when a source callback is admitted.
pub type SourceEvaluationPolicy = RuntimeLanguagePolicy;

/// An owned source invocation bound to one session, not its foreground stack.
/// Source identity is supplied by the executor's source catalog authority.
#[derive(Clone)]
pub struct SessionSourceEvaluation {
    session: RuntimeContextLocalId,
    source: Arc<str>,
    provenance: Rc<dyn SourceEvaluationProvenance>,
    executor: Option<Rc<dyn RuntimeSourceEvaluationService>>,
    authority: Option<Rc<dyn RuntimeSourceEvaluationAuthority>>,
    source_label: Arc<str>,
    requested_outputs: usize,
    policy: SourceEvaluationPolicy,
}

impl SessionSourceEvaluation {
    /// Ask the installed executor to allocate and retain source provenance.
    /// Callback consumers must use this entry point instead of inventing IDs.
    pub fn prepare(
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
    ) -> Result<Self, RuntimeError> {
        if !runtime.dynamic_eval_enabled() {
            return Err(crate::build_runtime_error(
                "dynamic source evaluation is disabled by host policy",
            )
            .with_identifier("RunMat:DynamicEvalDisabled")
            .build());
        }
        if runtime.is_cancelled() {
            return Err(crate::execution::ExecutionServiceError::Cancelled.into_runtime_error());
        }
        if let Some(workspace) = runtime.service_ports().workspace() {
            workspace.ensure_effect_authority()?;
        }
        let service = runtime
            .service_ports()
            .require_source_evaluation("capture session source")
            .map_err(super::RuntimeCapabilityError::into_runtime_error)?;
        let mut request =
            service.capture_session_source(runtime, source, source_label, requested_outputs)?;
        request.validate_session(runtime)?;
        request.executor = Some(service.clone());
        Ok(request)
    }

    /// Executor construction after allocating source identity in its catalog.
    pub fn capture(
        runtime: &RuntimeContext,
        source: Arc<str>,
        provenance: Rc<dyn SourceEvaluationProvenance>,
        source_label: Arc<str>,
        requested_outputs: usize,
    ) -> Self {
        Self {
            session: runtime.local_identity(),
            source,
            provenance,
            executor: None,
            authority: None,
            source_label,
            requested_outputs,
            policy: SourceEvaluationPolicy {
                language_mode: runtime.language_mode(),
                extensions_enabled: runtime.runmat_extensions_enabled(),
                top_level_await_enabled: runtime.top_level_await_enabled(),
                dynamic_eval_enabled: runtime.dynamic_eval_enabled(),
            },
        }
    }

    /// Executor construction with an immutable authority captured at source
    /// admission. Pending work must not consult a later ambient generation.
    pub fn capture_with_authority(
        runtime: &RuntimeContext,
        source: Arc<str>,
        provenance: Rc<dyn SourceEvaluationProvenance>,
        authority: Rc<dyn RuntimeSourceEvaluationAuthority>,
        source_label: Arc<str>,
        requested_outputs: usize,
    ) -> Self {
        let mut request =
            Self::capture(runtime, source, provenance, source_label, requested_outputs);
        request.authority = Some(authority);
        request
    }

    pub fn session(&self) -> RuntimeContextLocalId {
        self.session
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn source_id(&self) -> SourceId {
        self.provenance.source_id()
    }

    pub fn runtime_source_ref(&self) -> Option<crate::source_context::RuntimeSourceRef> {
        self.provenance.runtime_source_ref()
    }

    pub fn runtime_source_lease(&self) -> Option<crate::source_context::RuntimeSourceLease> {
        self.provenance.runtime_source_lease()
    }

    pub fn source_label(&self) -> &str {
        &self.source_label
    }

    pub fn requested_outputs(&self) -> usize {
        self.requested_outputs
    }

    pub fn policy(&self) -> SourceEvaluationPolicy {
        self.policy
    }

    pub fn evaluation_authority(&self) -> Option<Rc<dyn RuntimeSourceEvaluationAuthority>> {
        self.authority.clone()
    }

    /// A cold invocation: validation and executor entry happen on first poll.
    /// The shared cancellation boundary owns abandonment of pending work.
    pub fn evaluate(
        self,
        runtime: RuntimeContext,
    ) -> RuntimeServiceFuture<Result<crate::execution::RootedValueSequence, RuntimeError>> {
        let owner = runtime.clone();
        Box::pin(owner.scope(async move {
            self.validate_session(&runtime)?;
            if !self.policy.dynamic_eval_enabled || !runtime.dynamic_eval_enabled() {
                return Err(crate::build_runtime_error(
                    "dynamic source evaluation is disabled by host policy",
                )
                .with_identifier("RunMat:DynamicEvalDisabled")
                .build());
            }
            if runtime.is_cancelled() {
                return Err(crate::execution::ExecutionServiceError::Cancelled.into_runtime_error());
            }
            let service = runtime
                .service_ports()
                .require_source_evaluation("evaluate session source")
                .map_err(super::RuntimeCapabilityError::into_runtime_error)?
                .clone();
            if !self
                .executor
                .as_ref()
                .is_some_and(|owner| Rc::ptr_eq(owner, &service))
            {
                return Err(crate::build_runtime_error(
                    "source evaluation executor changed after capture",
                )
                .with_identifier("RunMat:SourceEvaluation:ExecutorMismatch")
                .build());
            }
            let cooperation = runtime.cpu_cooperation();
            let _provenance = self.provenance.clone();
            cooperation
                .await_cancellable(service.evaluate_session_source(runtime, self))
                .await
                .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?
        }))
    }

    /// Reject a cross-session request before compilation or workspace effects.
    pub fn validate_session(&self, runtime: &RuntimeContext) -> Result<(), RuntimeError> {
        if self.session == runtime.local_identity() {
            return Ok(());
        }
        Err(
            crate::build_runtime_error("source evaluation belongs to a different runtime session")
                .with_identifier("RunMat:SourceEvaluation:SessionMismatch")
                .build(),
        )
    }
}

/// Immutable executor state selected when source text is captured.
/// Implementations own exact compiler/function authority through suspension.
pub trait RuntimeSourceEvaluationAuthority: std::fmt::Debug {
    fn evaluate(
        &self,
        runtime: RuntimeContext,
        request: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<Result<crate::execution::RootedValueSequence, RuntimeError>>;
}

impl std::fmt::Debug for SessionSourceEvaluation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionSourceEvaluation")
            .field("session", &self.session)
            .field("source_id", &self.source_id())
            .field("source_label", &self.source_label)
            .field("requested_outputs", &self.requested_outputs)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

/// Implemented by the host executor, using its existing compiler and workspace
/// owner. This is deliberately local: browser callback state need not be Send.
pub trait RuntimeSourceEvaluationService {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }

    /// Bind background work to the current executable generation. The bound
    /// service may outlive the foreground frame, but not its admitted session.
    fn bind_background_authority(
        &self,
    ) -> Result<Rc<dyn RuntimeSourceEvaluationService>, RuntimeError>;

    /// Allocate source identity through the executor's source catalog and
    /// retain provenance in its session owner. Never synthesize a timer-local ID.
    fn capture_session_source(
        &self,
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
    ) -> Result<SessionSourceEvaluation, RuntimeError>;

    /// Resolve a stable base-workspace owner for `request.session()`, never an
    /// ambient caller frame. Compile under the captured language/extension/
    /// await policy, subject to current host restrictions. Validate before effects; preserve
    /// mutations, cancellation and original errors through suspension/drop.
    fn evaluate_session_source(
        &self,
        runtime: RuntimeContext,
        request: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<Result<crate::execution::RootedValueSequence, RuntimeError>>;
}

#[cfg(test)]
#[path = "source_evaluation/tests.rs"]
mod tests;
