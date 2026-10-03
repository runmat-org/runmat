use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};
use std::time::Duration;

use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use url::Url;

use super::*;
use crate::context::{RuntimeHttpService, RuntimeServiceFuture, RuntimeServicePorts};
use crate::execution::RuntimeExecutionService;

#[derive(Default)]
struct Counts {
    sends: Cell<usize>,
    polls: Cell<usize>,
    aborts: Cell<usize>,
}

struct PendingService(Rc<Counts>);

impl RuntimeHttpService for PendingService {
    fn send(
        &self,
        _request: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        self.0.sends.set(self.0.sends.get() + 1);
        Box::pin(PendingRequest(self.0.clone()))
    }
}

struct PendingRequest(Rc<Counts>);

impl Future for PendingRequest {
    type Output = Result<HttpResponse, TransportError>;

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.polls.set(self.0.polls.get() + 1);
        Poll::Pending
    }
}

impl Drop for PendingRequest {
    fn drop(&mut self) {
        self.0.aborts.set(self.0.aborts.get() + 1);
    }
}

fn request() -> HttpRequest {
    HttpRequest {
        url: Url::parse("https://example.invalid/resource").unwrap(),
        method: HttpMethod::Get,
        headers: Vec::new(),
        body: None,
        timeout: Duration::from_secs(5),
        user_agent: "RunMat test".into(),
    }
}

fn context(counts: Rc<Counts>, cancellation: CancellationSource) -> RuntimeContext {
    RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(RuntimeServicePorts::default().with_http(Rc::new(PendingService(counts))))
}

fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
    let waker = futures::task::noop_waker();
    future.poll(&mut Context::from_waker(&waker))
}

#[test]
fn unpolled_drop_does_not_submit_or_abort() {
    let counts = Rc::new(Counts::default());
    let runtime = context(counts.clone(), CancellationSource::new());
    drop(send_request(request(), Some(&runtime)));
    assert_eq!(
        (counts.sends.get(), counts.polls.get(), counts.aborts.get()),
        (0, 0, 0)
    );
}

#[test]
fn missing_port_and_precancelled_invocation_do_not_submit() {
    let missing = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let error = futures::executor::block_on(send_request(request(), Some(&missing)))
        .expect_err("missing host HTTP service");
    let TransportCallError::Runtime(error) = error else {
        panic!("missing HTTP port is a runtime capability failure");
    };
    assert_eq!(
        error.identifier(),
        Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
    );

    let counts = Rc::new(Counts::default());
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let runtime = context(counts.clone(), cancellation);
    let error = futures::executor::block_on(send_request(request(), Some(&runtime)))
        .expect_err("precancelled invocation");
    let TransportCallError::Runtime(error) = error else {
        panic!("cancellation must retain its runtime diagnostic");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(counts.sends.get(), 0);
}

#[test]
fn pending_repoll_submits_once_and_drop_aborts_once() {
    let counts = Rc::new(Counts::default());
    let runtime = context(counts.clone(), CancellationSource::new());
    let mut future = Box::pin(send_request(request(), Some(&runtime)));
    assert!(poll_once(future.as_mut()).is_pending());
    assert!(poll_once(future.as_mut()).is_pending());
    assert_eq!((counts.sends.get(), counts.polls.get()), (1, 2));
    drop(future);
    assert_eq!(counts.aborts.get(), 1);
}

#[test]
fn cancellation_wakes_pending_request_and_keeps_transport_cause_distinct() {
    let counts = Rc::new(Counts::default());
    let cancellation = CancellationSource::new();
    let runtime = context(counts.clone(), cancellation.clone());
    let mut future = Box::pin(send_request(request(), Some(&runtime)));
    assert!(poll_once(future.as_mut()).is_pending());
    cancellation.cancel(CancellationReason::User);
    let Poll::Ready(Err(TransportCallError::Runtime(error))) = poll_once(future.as_mut()) else {
        panic!("cancelled request must return a runtime cancellation cause");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(counts.aborts.get(), 1);
}

#[test]
fn independently_polled_contexts_keep_request_ownership_separate() {
    let first = Rc::new(Counts::default());
    let second = Rc::new(Counts::default());
    let a = context(first.clone(), CancellationSource::new());
    let b = context(second.clone(), CancellationSource::new());
    let mut a_future = Box::pin(send_request(request(), Some(&a)));
    let mut b_future = Box::pin(send_request(request(), Some(&b)));
    assert!(poll_once(a_future.as_mut()).is_pending());
    assert!(poll_once(b_future.as_mut()).is_pending());
    drop(a_future);
    assert_eq!((first.aborts.get(), second.aborts.get()), (1, 0));
    drop(b_future);
    assert_eq!(second.aborts.get(), 1);
}
