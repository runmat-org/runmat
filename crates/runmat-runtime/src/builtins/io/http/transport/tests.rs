use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
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
        _runtime: RuntimeContext,
        _request: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        self.0.sends.set(self.0.sends.get() + 1);
        Box::pin(PendingRequest(self.0.clone()))
    }
}

struct PendingRequest(Rc<Counts>);

struct CountWakes(AtomicUsize);

impl futures::task::ArcWake for CountWakes {
    fn wake_by_ref(waker: &Arc<Self>) {
        waker.0.fetch_add(1, Ordering::SeqCst);
    }
}

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
    let wakes = Arc::new(CountWakes(AtomicUsize::new(0)));
    let waker = futures::task::waker(wakes.clone());
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert_eq!(wakes.0.load(Ordering::SeqCst), 0);
    cancellation.cancel(CancellationReason::User);
    assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
    let Poll::Ready(Err(TransportCallError::Runtime(error))) = poll_once(future.as_mut()) else {
        panic!("cancelled request must return a runtime cancellation cause");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(counts.aborts.get(), 1);
}

struct ReadyCancellingService(CancellationSource);

impl RuntimeHttpService for ReadyCancellingService {
    fn send(
        &self,
        _runtime: RuntimeContext,
        _request: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        let cancellation = self.0.clone();
        Box::pin(async move {
            cancellation.cancel(CancellationReason::User);
            Ok(HttpResponse {
                headers: Vec::new(),
                body: b"must not publish".to_vec(),
            })
        })
    }
}

#[test]
fn completed_service_cannot_publish_after_concurrent_cancellation() {
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default().with_http(Rc::new(ReadyCancellingService(cancellation))),
    );
    let error = futures::executor::block_on(send_request(request(), Some(&runtime)))
        .expect_err("cancelled response cannot publish");
    let TransportCallError::Runtime(error) = error else {
        panic!("cancellation must remain a runtime cause");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
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

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_live_response_cancels_while_body_is_incomplete() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let driver = tokio::runtime::Runtime::new().expect("HTTP test driver");
    let listener = TcpListener::bind("127.0.0.1:0").expect("local listener");
    let url = Url::parse(&format!("http://{}/body", listener.local_addr().unwrap())).unwrap();
    let cancellation = CancellationSource::new();
    let server_cancellation = cancellation.clone();
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().expect("accepted request");
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut buffer = [0u8; 1024];
        let _ = socket.read(&mut buffer).expect("request read");
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\nConnection: close\r\n\r\na")
            .expect("partial response");
        std::thread::sleep(Duration::from_millis(40));
        server_cancellation.cancel(CancellationReason::User);
        std::thread::sleep(Duration::from_millis(100));
    });
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_http(Rc::new(NativeHttpService::new(driver.handle().clone()))),
    );
    let error = futures::executor::block_on(send_request(request_for(url), Some(&runtime)))
        .expect_err("cancelled live body must not publish");
    let TransportCallError::Runtime(error) = error else {
        panic!("live cancellation must retain its runtime cause");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    server.join().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
fn request_for(url: Url) -> HttpRequest {
    HttpRequest { url, ..request() }
}
