use super::*;
use crate::context::RuntimeLanguageMode;
use crate::execution::RuntimeExecutionService;
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

#[derive(Debug)]
struct TestProvenance(SourceId);
impl SourceEvaluationProvenance for TestProvenance {
    fn source_id(&self) -> SourceId {
        self.0
    }
}

struct PendingExecutor {
    entries: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
    retirements: Rc<Cell<usize>>,
}

#[derive(Debug)]
struct CountingProvenance(SourceId, Rc<Cell<usize>>);
impl SourceEvaluationProvenance for CountingProvenance {
    fn source_id(&self) -> SourceId {
        self.0
    }
}
impl Drop for CountingProvenance {
    fn drop(&mut self) {
        self.1.set(self.1.get() + 1);
    }
}

struct PendingEvaluation(Rc<Cell<usize>>);

impl Future for PendingEvaluation {
    type Output = Result<crate::execution::RootedValueSequence, RuntimeError>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

struct ObserveInvocationPolicy {
    expected: RuntimeLanguagePolicy,
    polls: Rc<Cell<usize>>,
}

impl Future for ObserveInvocationPolicy {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        let active = crate::context::legacy::active().expect("invocation context is active");
        assert_eq!(active.language_mode(), self.expected.language_mode);
        assert_eq!(
            active.runmat_extensions_enabled(),
            self.expected.extensions_enabled
        );
        assert_eq!(
            active.top_level_await_enabled(),
            self.expected.top_level_await_enabled
        );
        assert_eq!(
            active.dynamic_eval_enabled(),
            self.expected.dynamic_eval_enabled
        );
        assert_eq!(
            crate::compatibility::runmat_extensions_enabled(),
            self.expected.extensions_enabled
        );
        self.polls.set(self.polls.get() + 1);
        Poll::Pending
    }
}

impl Drop for PendingEvaluation {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

impl RuntimeSourceEvaluationService for PendingExecutor {
    fn bind_background_authority(
        &self,
    ) -> Result<Rc<dyn RuntimeSourceEvaluationService>, RuntimeError> {
        Ok(Rc::new(Self {
            entries: self.entries.clone(),
            drops: self.drops.clone(),
            retirements: self.retirements.clone(),
        }))
    }

    fn capture_session_source(
        &self,
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
    ) -> Result<SessionSourceEvaluation, RuntimeError> {
        Ok(SessionSourceEvaluation::capture(
            runtime,
            source,
            Rc::new(CountingProvenance(SourceId(41), self.retirements.clone())),
            source_label,
            requested_outputs,
        ))
    }
    fn evaluate_session_source(
        &self,
        _: RuntimeContext,
        _: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<Result<crate::execution::RootedValueSequence, RuntimeError>> {
        self.entries.set(self.entries.get() + 1);
        Box::pin(PendingEvaluation(self.drops.clone()))
    }
}

fn request(runtime: &RuntimeContext) -> SessionSourceEvaluation {
    if runtime.service_ports().source_evaluation().is_some() {
        return SessionSourceEvaluation::prepare(
            runtime,
            Arc::from("answer = 42;"),
            Arc::from("<timer>"),
            0,
        )
        .unwrap();
    }
    SessionSourceEvaluation::capture(
        runtime,
        Arc::from("answer = 42;"),
        Rc::new(TestProvenance(SourceId(41))),
        Arc::from("<timer>"),
        0,
    )
}

#[test]
fn numeric_text_capture_does_not_request_workspace_effect_authority() {
    let owner = crate::workspace::BaseWorkspaceOwner::new();
    let workspace = Rc::new(crate::workspace::BaseWorkspaceService::new(owner.clone()));
    owner.close().unwrap();
    let service = Rc::new(PendingExecutor {
        entries: Rc::new(Cell::new(0)),
        drops: Rc::new(Cell::new(0)),
        retirements: Rc::new(Cell::new(0)),
    });
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        super::super::RuntimeServicePorts::default()
            .with_workspace(workspace)
            .with_source_evaluation(service),
    );
    let callback_error =
        SessionSourceEvaluation::prepare(&runtime, Arc::from("x = 1"), Arc::from("<callback>"), 0)
            .expect_err("workspace callback needs an open workspace");
    assert_eq!(callback_error.identifier(), Some("RunMat:Workspace:Closed"));
    let numeric = SessionSourceEvaluation::prepare_numeric_text(
        &runtime,
        "1 + 2",
        NumericTextEvaluation::Restricted,
    )
    .expect("isolated numeric capture retains source authority without workspace effects");
    assert_eq!(
        numeric.purpose(),
        SessionSourcePurpose::NumericText(NumericTextEvaluation::Restricted)
    );
}

#[test]
fn executor_entry_is_cold_and_pending_drop_releases_exact_child_once() {
    let entries = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let retirements = Rc::new(Cell::new(0));
    let service = Rc::new(PendingExecutor {
        entries: entries.clone(),
        drops: drops.clone(),
        retirements: retirements.clone(),
    });
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        super::super::RuntimeServicePorts::default().with_source_evaluation(service),
    );
    drop(request(&runtime).evaluate(runtime.clone()));
    assert_eq!(entries.get(), 0);
    assert_eq!(drops.get(), 0);
    assert_eq!(retirements.get(), 1);
    let mut future = request(&runtime).evaluate(runtime);
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(future.as_mut().poll(&mut context).is_pending());
    assert_eq!(entries.get(), 1);
    assert_eq!(retirements.get(), 1);
    drop(future);
    assert_eq!(drops.get(), 1);
    assert_eq!(retirements.get(), 2);
}

#[test]
fn disabled_policy_and_foreign_session_reject_before_executor_entry() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let foreign = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let error = futures::executor::block_on(request(&runtime).evaluate(foreign)).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:SourceEvaluation:SessionMismatch")
    );
    runtime.set_dynamic_eval_enabled(false);
    let captured = request(&runtime);
    runtime.set_dynamic_eval_enabled(true);
    let error = futures::executor::block_on(captured.evaluate(runtime)).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:DynamicEvalDisabled"));
}

#[test]
fn replacing_executor_within_same_session_rejects_before_entry() {
    let entries = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let retirements = Rc::new(Cell::new(0));
    let first = Rc::new(PendingExecutor {
        entries: entries.clone(),
        drops: drops.clone(),
        retirements: retirements.clone(),
    });
    let second = Rc::new(PendingExecutor {
        entries: entries.clone(),
        drops,
        retirements: retirements.clone(),
    });
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        super::super::RuntimeServicePorts::default().with_source_evaluation(first),
    );
    let captured = request(&runtime);
    let replacement = runtime.clone().with_service_ports(
        super::super::RuntimeServicePorts::default().with_source_evaluation(second),
    );
    assert_eq!(runtime.local_identity(), replacement.local_identity());
    let error = futures::executor::block_on(captured.evaluate(replacement)).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:SourceEvaluation:ExecutorMismatch")
    );
    assert_eq!(entries.get(), 0);
    assert_eq!(retirements.get(), 1);
}

#[test]
fn cancelling_pending_evaluation_releases_child_and_source_record() {
    let source = runmat_execution::cancellation::CancellationSource::new();
    let entries = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let retirements = Rc::new(Cell::new(0));
    let service = Rc::new(PendingExecutor {
        entries: entries.clone(),
        drops: drops.clone(),
        retirements: retirements.clone(),
    });
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token())
            .with_service_ports(
                super::super::RuntimeServicePorts::default().with_source_evaluation(service),
            );
    let mut future = request(&runtime).evaluate(runtime);
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(future.as_mut().poll(&mut context).is_pending());
    source.cancel(runmat_execution::CancellationReason::User);
    assert!(matches!(
        future.as_mut().poll(&mut context),
        Poll::Ready(Err(_))
    ));
    drop(future);
    assert_eq!(entries.get(), 1);
    assert_eq!(drops.get(), 1);
    assert_eq!(retirements.get(), 1);
}

#[test]
fn source_request_captures_session_and_policy_without_retaining_a_frame() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    runtime.set_language_mode(RuntimeLanguageMode::RunMat);
    let source: Arc<str> = Arc::from("answer = 42;");
    let request = SessionSourceEvaluation::capture(
        &runtime,
        Arc::clone(&source),
        Rc::new(TestProvenance(SourceId(41))),
        Arc::from("<timer>"),
        0,
    );
    runtime.set_language_mode(RuntimeLanguageMode::Matlab);
    assert_eq!(request.policy().language_mode, RuntimeLanguageMode::RunMat);
    assert_eq!(request.source(), "answer = 42;");
    assert_eq!(request.source().as_ptr(), source.as_ptr());
    assert_eq!(request.source_id(), SourceId(41));
    assert_eq!(request.requested_outputs(), 0);
    assert!(request.validate_session(&runtime).is_ok());
    let other = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    assert_eq!(
        request.validate_session(&other).unwrap_err().identifier(),
        Some("RunMat:SourceEvaluation:SessionMismatch")
    );
}

#[test]
fn pending_sibling_invocations_retain_independent_language_policy() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    runtime.set_language_mode(RuntimeLanguageMode::Matlab);
    runtime.set_runmat_extensions_enabled(false);
    let first_policy = RuntimeLanguagePolicy {
        language_mode: RuntimeLanguageMode::RunMat,
        extensions_enabled: true,
        top_level_await_enabled: false,
        dynamic_eval_enabled: true,
    };
    let second_policy = RuntimeLanguagePolicy {
        language_mode: RuntimeLanguageMode::Strict,
        extensions_enabled: false,
        top_level_await_enabled: true,
        dynamic_eval_enabled: false,
    };
    let first_polls = Rc::new(Cell::new(0));
    let second_polls = Rc::new(Cell::new(0));
    let first_runtime = runtime.clone().with_language_policy(first_policy);
    let second_runtime = runtime.clone().with_language_policy(second_policy);
    let mut first = Box::pin(first_runtime.scope(ObserveInvocationPolicy {
        expected: first_policy,
        polls: first_polls.clone(),
    }));
    let mut second = Box::pin(second_runtime.scope(ObserveInvocationPolicy {
        expected: second_policy,
        polls: second_polls.clone(),
    }));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());

    assert!(first.as_mut().poll(&mut context).is_pending());
    assert!(second.as_mut().poll(&mut context).is_pending());
    drop(second);
    assert!(first.as_mut().poll(&mut context).is_pending());
    drop(first);

    assert_eq!(first_polls.get(), 2);
    assert_eq!(second_polls.get(), 1);
    assert_eq!(runtime.language_mode(), RuntimeLanguageMode::Matlab);
    assert!(!runtime.runmat_extensions_enabled());
    assert!(runtime.top_level_await_enabled());
    assert!(runtime.dynamic_eval_enabled());
}

#[test]
fn assignin_arguments_keep_borrowed_names_and_decode_character_rows_without_heap_text() {
    use crate::builtins::common::identifiers::MATLAB_NAME_LENGTH_MAX;
    use runmat_value::CharArray;

    let selector = Value::CharArray(CharArray::new_row("CaLlEr"));
    assert_eq!(
        assignin_args::selector(&selector).unwrap(),
        assignin_args::AssigninSelector::Caller
    );
    let name = Value::String("shared_name".into());
    let mut scratch = [0u8; MATLAB_NAME_LENGTH_MAX];
    let borrowed = assignin_args::name(&name, &mut scratch).unwrap();
    assert_eq!(borrowed, "shared_name");
    let Value::String(runmat_value::StringElement::Text(original)) = &name else {
        unreachable!("constructed a string scalar")
    };
    assert_eq!(borrowed.as_ptr(), original.as_ptr());

    let character_name = Value::CharArray(CharArray::new_row("shared_name"));
    assert_eq!(
        assignin_args::name(&character_name, &mut scratch).unwrap(),
        "shared_name"
    );
    for invalid in ["for", "éclair", "_leading"] {
        assert_eq!(
            assignin_args::name(&Value::CharArray(CharArray::new_row(invalid)), &mut scratch)
                .unwrap_err()
                .identifier(),
            Some("RunMat:DynamicWorkspaceName")
        );
    }
    let overlong = Value::CharArray(CharArray::new_row(&"a".repeat(MATLAB_NAME_LENGTH_MAX + 1)));
    assert_eq!(
        assignin_args::name(&overlong, &mut scratch)
            .unwrap_err()
            .identifier(),
        Some("RunMat:DynamicWorkspaceName")
    );
}
