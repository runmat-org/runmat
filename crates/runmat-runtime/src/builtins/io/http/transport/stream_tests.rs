use std::cell::Cell;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;

use super::*;
use crate::execution::RuntimeExecutionService;
use crate::http::HttpResponseHead;

struct CountWakes(AtomicUsize);

impl futures::task::ArcWake for CountWakes {
    fn wake_by_ref(waker: &Arc<Self>) {
        waker.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct PendingBody {
    polls: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
}

impl Stream for PendingBody {
    type Item = Result<Vec<u8>, TransportError>;

    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.polls.set(self.polls.get() + 1);
        Poll::Pending
    }
}

impl Drop for PendingBody {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

fn body(polls: Rc<Cell<usize>>, drops: Rc<Cell<usize>>) -> HttpBodyStream {
    HttpBodyStream::new(
        HttpResponseHead {
            headers: Vec::new(),
            content_length: None,
        },
        Box::pin(PendingBody { polls, drops }),
    )
}

#[test]
fn cancelled_pending_chunk_aborts_body_owner_once() {
    use std::future::Future;
    let polls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let mut body = body(polls.clone(), drops.clone());
    let mut chunk = Box::pin(next_body_chunk(&mut body, Some(&runtime)));
    let wakes = Arc::new(CountWakes(AtomicUsize::new(0)));
    let waker = futures::task::waker(wakes.clone());
    let mut context = Context::from_waker(&waker);
    assert!(chunk.as_mut().poll(&mut context).is_pending());
    assert_eq!(polls.get(), 1);
    assert_eq!(wakes.0.load(Ordering::SeqCst), 0);
    cancellation.cancel(CancellationReason::User);
    assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
    let Poll::Ready(Err(TransportCallError::Runtime(error))) = chunk.as_mut().poll(&mut context)
    else {
        panic!("pending body must return runtime cancellation");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(drops.get(), 1);
    drop(chunk);
    drop(body);
    assert_eq!(drops.get(), 1);
}

#[test]
fn unpolled_chunk_does_not_poll_or_drop_borrowed_owner() {
    let polls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let mut body = body(polls.clone(), drops.clone());
    drop(next_body_chunk(&mut body, Some(&runtime)));
    assert_eq!((polls.get(), drops.get()), (0, 0));
    drop(body);
    assert_eq!(drops.get(), 1);
}

#[test]
fn missing_context_aborts_body_without_polling() {
    let polls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let mut body = body(polls.clone(), drops.clone());
    assert!(matches!(
        futures::executor::block_on(next_body_chunk(&mut body, None)),
        Err(TransportCallError::Runtime(_))
    ));
    assert_eq!((polls.get(), drops.get()), (0, 1));
}

#[test]
fn ready_body_cannot_publish_after_concurrent_cancellation() {
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let mut body = HttpBodyStream::new(
        HttpResponseHead {
            headers: Vec::new(),
            content_length: None,
        },
        Box::pin(futures::stream::once(async move {
            cancellation.cancel(CancellationReason::User);
            Ok(b"must not publish".to_vec())
        })),
    );
    let result = futures::executor::block_on(next_body_chunk(&mut body, Some(&runtime)));
    let Err(TransportCallError::Runtime(error)) = result else {
        panic!("ready body must not publish cancelled bytes");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(futures::executor::block_on(body.next_chunk())
        .unwrap()
        .is_none());
}

#[test]
fn cancelled_ready_error_keeps_runtime_cancellation_priority() {
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let mut body = HttpBodyStream::new(
        HttpResponseHead {
            headers: Vec::new(),
            content_length: None,
        },
        Box::pin(futures::stream::once(async move {
            cancellation.cancel(CancellationReason::User);
            Err(TransportError::host_service_unavailable(
                &"http://127.0.0.1/".parse().unwrap(),
                std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "connection failed"),
            ))
        })),
    );
    let result = futures::executor::block_on(next_body_chunk(&mut body, Some(&runtime)));
    let Err(TransportCallError::Runtime(error)) = result else {
        panic!("concurrent cancellation must retain its runtime diagnostic");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
}

#[test]
fn transport_failure_aborts_custom_body_owner() {
    let polls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let stream = futures::stream::once(async {
        Err(TransportError::host_service_unavailable(
            &"http://127.0.0.1/".parse().unwrap(),
            std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "connection failed"),
        ))
    })
    .chain(PendingBody {
        polls: polls.clone(),
        drops: drops.clone(),
    });
    use futures::StreamExt;
    let mut body = HttpBodyStream::new(
        HttpResponseHead {
            headers: Vec::new(),
            content_length: None,
        },
        Box::pin(stream),
    );
    let result = futures::executor::block_on(next_body_chunk(&mut body, Some(&runtime)));
    let Err(TransportCallError::Transport(error)) = result else {
        panic!("ordinary transport error must remain distinct");
    };
    assert!(matches!(error.kind, TransportErrorKind::Other));
    assert_eq!((polls.get(), drops.get()), (0, 1));
}
