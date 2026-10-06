use super::*;
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
    type Output = Result<ValueSequence, RuntimeError>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingEvaluation {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

impl RuntimeSourceEvaluationService for PendingExecutor {
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
    ) -> RuntimeServiceFuture<Result<ValueSequence, RuntimeError>> {
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
