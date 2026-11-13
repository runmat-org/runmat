use super::{
    HostDocumentRequest, RuntimeContext, RuntimeHostService, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::{interaction::InteractionResponse, RuntimeError};
use futures::channel::oneshot;
use futures::FutureExt;
use std::{
    cell::{Cell, RefCell},
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
};

#[derive(Default)]
pub(crate) struct DocumentRecorder {
    pub requests: Rc<RefCell<Vec<HostDocumentRequest>>>,
}
impl RuntimeHostService for DocumentRecorder {
    fn console(&self, _: crate::console::ConsoleStream, _: crate::console::ConsoleText) {}
    fn warning(&self, _: crate::warning_store::RuntimeWarning) {}
    fn interact(
        &self,
        _: super::HostInteraction,
    ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
        Box::pin(async { Err(super::host_document::unavailable()) })
    }
    fn open_document(
        &self,
        request: HostDocumentRequest,
    ) -> RuntimeServiceFuture<Result<(), RuntimeError>> {
        let requests = self.requests.clone();
        Box::pin(async move {
            requests.borrow_mut().push(request);
            Ok(())
        })
    }
}
pub(crate) fn recording_context() -> (RuntimeContext, Rc<DocumentRecorder>) {
    let recorder = Rc::new(DocumentRecorder::default());
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_host(recorder.clone()));
    context.set_runmat_extensions_enabled(crate::compatibility::runmat_extensions_enabled());
    crate::user_functions::inherit_legacy_call_environment(&context);
    (context, recorder)
}

pub(crate) struct PendingDocumentHost {
    pub request: RefCell<Option<HostDocumentRequest>>,
    release: RefCell<Option<oneshot::Receiver<Result<(), RuntimeError>>>>,
    pub started: Cell<usize>,
    pub dropped: Rc<Cell<usize>>,
    pub completed: Rc<Cell<usize>>,
}

struct PendingTurn {
    dropped: Rc<Cell<usize>>,
    completed: bool,
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        if !self.completed {
            self.dropped.set(self.dropped.get() + 1);
        }
    }
}

impl RuntimeHostService for PendingDocumentHost {
    fn console(&self, _: crate::console::ConsoleStream, _: crate::console::ConsoleText) {}

    fn warning(&self, _: crate::warning_store::RuntimeWarning) {}

    fn interact(
        &self,
        _: super::HostInteraction,
    ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
        Box::pin(async {
            Err(
                crate::build_runtime_error("unexpected terminal interaction")
                    .with_identifier("RunMat:test:UnexpectedInteraction")
                    .build(),
            )
        })
    }

    fn open_document(
        &self,
        request: HostDocumentRequest,
    ) -> RuntimeServiceFuture<Result<(), RuntimeError>> {
        self.started.set(self.started.get() + 1);
        *self.request.borrow_mut() = Some(request);
        let mut release = self
            .release
            .borrow_mut()
            .take()
            .expect("one selected request per host fixture");
        let dropped = Rc::clone(&self.dropped);
        let completed = Rc::clone(&self.completed);
        Box::pin(async move {
            let mut turn = PendingTurn {
                dropped,
                completed: false,
            };
            let result = match (&mut release).await {
                Ok(result) => result,
                Err(_) => Err(crate::build_runtime_error("selected document host retired")
                    .with_identifier("RunMat:host:SelectedDocumentRetired")
                    .build()),
            };
            turn.completed = true;
            completed.set(completed.get() + 1);
            result
        })
    }
}

pub(crate) fn controlled_context() -> (
    RuntimeContext,
    Rc<PendingDocumentHost>,
    oneshot::Sender<Result<(), RuntimeError>>,
) {
    let (release, receiver) = oneshot::channel();
    let host = Rc::new(PendingDocumentHost {
        request: RefCell::new(None),
        release: RefCell::new(Some(receiver)),
        started: Cell::new(0),
        dropped: Rc::new(Cell::new(0)),
        completed: Rc::new(Cell::new(0)),
    });
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_host(host.clone()));
    (context, host, release)
}

pub(crate) fn poll_pending_until_host_starts<F: Future>(
    future: &mut Pin<Box<F>>,
    host: &PendingDocumentHost,
) {
    for _ in 0..32 {
        assert!(future.as_mut().now_or_never().is_none());
        if host.started.get() == 1 {
            return;
        }
    }
    panic!("call did not reach the selected document host");
}

pub(crate) fn poll_ready<F: Future>(future: &mut Pin<Box<F>>) -> F::Output {
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    for _ in 0..32 {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
    }
    panic!("released document interaction did not complete");
}
