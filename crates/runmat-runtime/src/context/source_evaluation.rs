//! Executor-owned evaluation of session source text.
//!
//! No caller-frame pointers are retained by this port. Executors acquire their
//! captured session authority when they evaluate a request. Workspace
//! callbacks additionally require the session's base-workspace authority.

use super::{RuntimeContext, RuntimeContextLocalId, RuntimeLanguagePolicy, RuntimeServiceFuture};
use crate::RuntimeError;
use runmat_types::SourceId;
use runmat_value::Value;
use std::rc::Rc;
use std::sync::Arc;

#[path = "source_evaluation/assignin_args.rs"]
mod assignin_args;

/// One already-resolved dynamic-workspace builtin invocation. Arguments stay
/// borrowed from the selected call owner through completion; the service must
/// not rediscover or reselect a callable by name.
pub struct DynamicWorkspaceInvocation<'a> {
    pub builtin: runmat_builtins::DynamicWorkspaceBuiltin,
    pub arguments: crate::call::FunctionArgumentView<'a>,
    pub requested_outputs: usize,
    pub callsite: Option<crate::callsite::CallsiteInfo>,
}

/// Execute the mutation-only `assignin` contract. This path deliberately
/// depends only on the selected workspace/session authorities; it does not
/// require a source evaluator, parser, or compiler service.
pub fn execute_assignin_effect(
    runtime: &RuntimeContext,
    arguments: &[Value],
    requested_outputs: usize,
) -> Result<(), RuntimeError> {
    execute_assignin_effect_view(
        runtime,
        crate::call::FunctionArgumentView::from_slice(arguments),
        requested_outputs,
    )
}

pub fn execute_assignin_effect_view(
    runtime: &RuntimeContext,
    arguments: crate::call::FunctionArgumentView<'_>,
    requested_outputs: usize,
) -> Result<(), RuntimeError> {
    if arguments.len() < 3 {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:NotEnoughInputs",
            "assignin takes three arguments",
        ));
    }
    if arguments.len() > 3 {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:TooManyInputs",
            "assignin takes three arguments",
        ));
    }
    if requested_outputs > 0 {
        return Err(
            crate::builtins::introspection::dynamic_workspace::assignin_too_many_outputs_error(),
        );
    }
    // Keep name validation before the unknown-selector diagnostic, matching
    // the established argument-error order without materializing either text.
    let selector = assignin_args::selector(&arguments[0])?;
    let mut character_row = [0u8; crate::builtins::common::identifiers::MATLAB_NAME_LENGTH_MAX];
    let name = assignin_args::name(&arguments[1], &mut character_row)?;
    match selector {
        assignin_args::AssigninSelector::Caller => {
            let current = runtime
                .service_ports()
                .require_workspace("assignin caller workspace")
                .map_err(super::RuntimeCapabilityError::into_runtime_error)?
                .clone();
            let caller = current
                .caller_workspace_with_context(runtime)?
                .ok_or_else(|| {
                    crate::runtime_error::semantic_error(
                        "RunMat:DynamicWorkspaceUnavailable",
                        "assignin caller workspace is unavailable",
                    )
                })?;
            caller.assign_ref_with_context(runtime, name, &arguments[2])?;
        }
        assignin_args::AssigninSelector::Base => {
            let base = runtime
                .service_ports()
                .require_base_workspace("assignin base workspace")
                .map_err(super::RuntimeCapabilityError::into_runtime_error)?;
            base.assign_logical_ref_with_context(
                &runtime.session_variable_authority(),
                runtime,
                name,
                &arguments[2],
            )?;
        }
        assignin_args::AssigninSelector::Unknown => {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:DynamicWorkspaceSelector",
                "assignin workspace selector must be 'base' or 'caller'",
            ));
        }
    }
    Ok(())
}

/// Execute one catalog-selected dynamic workspace builtin. Registration,
/// scoped replacement, direct-call observation, and implementation admission
/// have already selected the canonical builtin body before this function is
/// entered.
fn required_dynamic_runtime(
    runtime: Option<RuntimeContext>,
    builtin: runmat_builtins::DynamicWorkspaceBuiltin,
) -> Result<RuntimeContext, RuntimeError> {
    runtime.ok_or_else(|| {
        let descriptor = match builtin {
            runmat_builtins::DynamicWorkspaceBuiltin::Run => {
                &runmat_builtins::RUN_ERROR_REQUIRES_VM
            }
            runmat_builtins::DynamicWorkspaceBuiltin::Eval
            | runmat_builtins::DynamicWorkspaceBuiltin::Evalc
            | runmat_builtins::DynamicWorkspaceBuiltin::Evalin
            | runmat_builtins::DynamicWorkspaceBuiltin::Assignin => {
                &runmat_builtins::DYNAMIC_WORKSPACE_ERROR_REQUIRES_VM
            }
        };
        crate::runtime_descriptor_error(builtin.name(), descriptor)
    })
}

pub async fn execute_selected_dynamic_workspace(
    runtime: Option<RuntimeContext>,
    builtin: runmat_builtins::DynamicWorkspaceBuiltin,
    arguments: Vec<Value>,
    requested_outputs: Option<usize>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let runtime = required_dynamic_runtime(runtime, builtin)?;
    let arguments = crate::call::BuiltinArguments::from_borrowed(&arguments, Some(&runtime))?;
    execute_selected_dynamic_workspace_retained(
        Some(runtime),
        builtin,
        arguments,
        requested_outputs,
    )
    .await
}

pub async fn execute_selected_dynamic_workspace_retained(
    runtime: Option<RuntimeContext>,
    builtin: runmat_builtins::DynamicWorkspaceBuiltin,
    arguments: crate::call::BuiltinArguments,
    requested_outputs: Option<usize>,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let runtime = required_dynamic_runtime(runtime, builtin)?;
    let requested_outputs = requested_outputs.unwrap_or(match builtin {
        runmat_builtins::DynamicWorkspaceBuiltin::Assignin
        | runmat_builtins::DynamicWorkspaceBuiltin::Run => 0,
        runmat_builtins::DynamicWorkspaceBuiltin::Eval
        | runmat_builtins::DynamicWorkspaceBuiltin::Evalc
        | runmat_builtins::DynamicWorkspaceBuiltin::Evalin => 1,
    });
    if builtin == runmat_builtins::DynamicWorkspaceBuiltin::Assignin {
        execute_assignin_effect_view(&runtime, arguments.view(), requested_outputs)?;
        return crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
            .map_err(|error| error.into_runtime_error());
    }
    let service = runtime
        .service_ports()
        .require_source_evaluation("execute dynamic workspace builtin")
        .map_err(super::RuntimeCapabilityError::into_runtime_error)?;
    let invocation = DynamicWorkspaceInvocation {
        builtin,
        arguments: arguments.view(),
        requested_outputs,
        callsite: crate::callsite::current_callsite(),
    };
    let result = service
        .evaluate_dynamic_workspace(&runtime, invocation)?
        .await;
    result
}

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

/// A source request's execution domain is fixed before evaluation begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionSourcePurpose {
    WorkspaceCallback,
    NumericText(NumericTextEvaluation),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericTextEvaluation {
    All,
    Restricted,
}

/// An owned source invocation bound to one session, not its foreground stack.
/// Source identity is supplied by the executor's source catalog authority.
#[derive(Clone)]
pub struct SessionSourceEvaluation {
    session: RuntimeContextLocalId,
    source: Arc<str>,
    provenance: Rc<dyn SourceEvaluationProvenance>,
    executor: Option<Rc<dyn RuntimeSourceEvaluationService>>,
    authority: Option<Rc<dyn RuntimeSourceEvaluationAuthority>>,
    /// Exact host-granted base port captured with this source request. Pure
    /// numeric text never reads it; explicit All-mode workspace operations
    /// may use it under the base owner's ordinary at-use effect checks.
    explicit_base_workspace: Option<Rc<dyn super::RuntimeBaseWorkspaceService>>,
    source_label: Arc<str>,
    requested_outputs: usize,
    policy: SourceEvaluationPolicy,
    purpose: SessionSourcePurpose,
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
        Self::prepare_for_purpose(
            runtime,
            source,
            source_label,
            requested_outputs,
            SessionSourcePurpose::WorkspaceCallback,
        )
    }

    fn prepare_for_purpose(
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
        purpose: SessionSourcePurpose,
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
        if purpose == SessionSourcePurpose::WorkspaceCallback {
            if let Some(workspace) = runtime.service_ports().workspace() {
                workspace.ensure_effect_authority_with_context(runtime)?;
            }
        }
        let service = runtime
            .service_ports()
            .require_source_evaluation("capture session source")
            .map_err(super::RuntimeCapabilityError::into_runtime_error)?;
        let mut request =
            service.capture_session_source(runtime, source, source_label, requested_outputs)?;
        request.purpose = purpose;
        request.validate_session(runtime)?;
        request.executor = Some(service.clone());
        Ok(request)
    }

    /// Capture matrix-expression text through the same executor/source owner as
    /// other dynamic requests, but execute it in an isolated value frame.
    pub fn prepare_numeric_text(
        runtime: &RuntimeContext,
        text: &str,
        method: NumericTextEvaluation,
    ) -> Result<Self, RuntimeError> {
        let source: Arc<str> = Arc::from(format!("[{text}]"));
        Self::prepare_for_purpose(
            runtime,
            source,
            Arc::from("<str2num>"),
            1,
            SessionSourcePurpose::NumericText(method),
        )
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
            explicit_base_workspace: runtime.service_ports().base_workspace().cloned(),
            source_label,
            requested_outputs,
            policy: SourceEvaluationPolicy {
                language_mode: runtime.language_mode(),
                extensions_enabled: runtime.runmat_extensions_enabled(),
                top_level_await_enabled: runtime.top_level_await_enabled(),
                dynamic_eval_enabled: runtime.dynamic_eval_enabled(),
            },
            purpose: SessionSourcePurpose::WorkspaceCallback,
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

    pub fn purpose(&self) -> SessionSourcePurpose {
        self.purpose
    }

    pub fn evaluation_authority(&self) -> Option<Rc<dyn RuntimeSourceEvaluationAuthority>> {
        self.authority.clone()
    }

    pub fn explicit_base_workspace(&self) -> Option<Rc<dyn super::RuntimeBaseWorkspaceService>> {
        self.explicit_base_workspace.clone()
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
            .field("purpose", &self.purpose)
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

    /// Execute in the request's typed domain, never an ambient caller frame.
    /// Workspace callbacks resolve a stable base-workspace owner for
    /// `request.session()`. Compile under the captured language/extension/
    /// await policy, subject to current host restrictions. Validate before effects; preserve
    /// mutations, cancellation and original errors through suspension/drop.
    fn evaluate_session_source(
        &self,
        runtime: RuntimeContext,
        request: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<Result<crate::execution::RootedValueSequence, RuntimeError>>;

    /// Execute an exact dynamic-workspace builtin after callable selection.
    /// External source services remain source compatible and fail before any
    /// workspace effect until they implement this stronger authority.
    fn evaluate_dynamic_workspace<'a>(
        &'a self,
        _runtime: &'a RuntimeContext,
        invocation: DynamicWorkspaceInvocation<'a>,
    ) -> Result<
        crate::execution::RuntimeContinuationFuture<'a, crate::execution::RootedValueSequence>,
        RuntimeError,
    > {
        Err(crate::build_runtime_error(format!(
            "{} requires a workspace-aware execution service",
            invocation.builtin.name()
        ))
        .with_identifier("RunMat:DynamicWorkspaceRequiresVm")
        .build())
    }
}

#[cfg(test)]
#[path = "source_evaluation/tests.rs"]
mod tests;
