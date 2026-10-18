use crate::context::{
    HostInteraction, RuntimeContext, RuntimeHostService, RuntimeServiceFuture, RuntimeServicePorts,
};
use crate::interaction::InteractionResponse;
use crate::RuntimeError;
use futures::channel::oneshot;
use futures::FutureExt;
use runmat_filesystem::{
    DirectoryDialogRequest, DirectoryDialogSelection, OpenFileDialogRequest,
    OpenFileDialogSelection, SaveFileDialogRequest, SaveFileDialogSelection,
};
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DialogKind {
    Open,
    Save,
    Directory,
}

#[derive(Debug)]
pub(super) enum DialogRequest {
    Open(OpenFileDialogRequest),
    Save(SaveFileDialogRequest),
    Directory(DirectoryDialogRequest),
}

pub(super) struct PendingDialogHost {
    kind: DialogKind,
    pub request: RefCell<Option<DialogRequest>>,
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

impl PendingDialogHost {
    fn turn<T: 'static>(
        &self,
        kind: DialogKind,
        request: DialogRequest,
        selection: T,
    ) -> RuntimeServiceFuture<Result<T, RuntimeError>> {
        assert_eq!(self.kind, kind);
        self.started.set(self.started.get() + 1);
        *self.request.borrow_mut() = Some(request);
        let mut release = self.release.borrow_mut().take().expect("one request");
        let dropped = self.dropped.clone();
        let completed = self.completed.clone();
        Box::pin(async move {
            let mut turn = PendingTurn {
                dropped,
                completed: false,
            };
            match (&mut release).await {
                Ok(result) => result?,
                Err(_) => {
                    return Err(crate::build_runtime_error("selected dialog host retired")
                        .with_identifier("RunMat:host:SelectedDialogRetired")
                        .build());
                }
            }
            turn.completed = true;
            completed.set(completed.get() + 1);
            Ok(selection)
        })
    }
}

impl RuntimeHostService for PendingDialogHost {
    fn console(&self, _: crate::console::ConsoleStream, _: crate::console::ConsoleText) {}
    fn warning(&self, _: crate::warning_store::RuntimeWarning) {}
    fn interact(
        &self,
        _: HostInteraction,
    ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
        panic!("dialog fixture must not request terminal input")
    }
    fn select_file_open(
        &self,
        request: OpenFileDialogRequest,
    ) -> RuntimeServiceFuture<Result<Option<OpenFileDialogSelection>, RuntimeError>> {
        self.turn(
            DialogKind::Open,
            DialogRequest::Open(request),
            Some(OpenFileDialogSelection {
                paths: vec![PathBuf::from("/selected/input.dat")],
                filter_index: Some(1),
            }),
        )
    }
    fn select_file_save(
        &self,
        request: SaveFileDialogRequest,
    ) -> RuntimeServiceFuture<Result<Option<SaveFileDialogSelection>, RuntimeError>> {
        self.turn(
            DialogKind::Save,
            DialogRequest::Save(request),
            Some(SaveFileDialogSelection {
                path: PathBuf::from("/selected/output.dat"),
                filter_index: Some(1),
            }),
        )
    }
    fn select_directory(
        &self,
        request: DirectoryDialogRequest,
    ) -> RuntimeServiceFuture<Result<Option<DirectoryDialogSelection>, RuntimeError>> {
        self.turn(
            DialogKind::Directory,
            DialogRequest::Directory(request),
            Some(DirectoryDialogSelection {
                path: PathBuf::from("/selected"),
            }),
        )
    }
}

pub(super) fn controlled_context(
    kind: DialogKind,
) -> (
    RuntimeContext,
    Rc<PendingDialogHost>,
    oneshot::Sender<Result<(), RuntimeError>>,
) {
    let (sender, receiver) = oneshot::channel();
    let host = Rc::new(PendingDialogHost {
        kind,
        request: RefCell::new(None),
        release: RefCell::new(Some(receiver)),
        started: Cell::new(0),
        dropped: Rc::new(Cell::new(0)),
        completed: Rc::new(Cell::new(0)),
    });
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_host(host.clone()));
    (context, host, sender)
}

pub(super) fn poll_pending<F: Future>(future: &mut Pin<Box<F>>, host: &PendingDialogHost) {
    for _ in 0..32 {
        assert!(future.as_mut().now_or_never().is_none());
        if host.started.get() == 1 {
            return;
        }
    }
    panic!("dialog call did not reach selected host")
}

pub(super) fn poll_ready<F: Future>(future: &mut Pin<Box<F>>) -> F::Output {
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    for _ in 0..32 {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
    }
    panic!("released dialog did not complete")
}
