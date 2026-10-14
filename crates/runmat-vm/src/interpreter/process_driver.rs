//! Same-thread, invocation-owned continuation of nested semantic VM calls.

use crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort;
use futures::channel::oneshot;
use runmat_runtime::execution::{RootedValueSequence, RootedValues};
use runmat_runtime::RuntimeError;
use std::cell::RefCell;
use std::cmp::Ordering as SortOrder;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::rc::{Rc, Weak as RcWeak};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak as SyncWeak};
use std::task::{Context, Poll, Wake, Waker};

#[path = "process_driver/typed_request.rs"]
mod typed_request;
use typed_request::TypedQueuedRequest;

pub(crate) type SemanticReply = Result<(RootedValueSequence, RootedValues), RuntimeError>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CallId(u64);

struct QueuedCall {
    parent: Option<CallId>,
    depth: u32,
    future: QueuedFuture,
    reply: QueuedReply,
}

enum QueuedFuture {
    Semantic(Pin<Box<dyn Future<Output = SemanticReply>>>),
    Owned(Pin<Box<dyn Future<Output = ()>>>),
}

enum QueuedReply {
    Semantic(oneshot::Sender<SemanticReply>),
    Owned(Rc<dyn OwnedCancellation>),
}

trait OwnedCancellation {
    fn is_canceled(&self) -> bool;
    fn publish(&self);
}

struct OwnedCompletion<T> {
    reply: RefCell<Option<oneshot::Sender<Result<T, RuntimeError>>>>,
    result: RefCell<Option<Result<T, RuntimeError>>>,
}

impl<T> OwnedCancellation for OwnedCompletion<T> {
    fn is_canceled(&self) -> bool {
        self.reply
            .borrow()
            .as_ref()
            .is_none_or(oneshot::Sender::is_canceled)
    }

    fn publish(&self) {
        let result = self.result.borrow_mut().take();
        let reply = self.reply.borrow_mut().take();
        if let (Some(result), Some(reply)) = (result, reply) {
            let _ = reply.send(result);
        }
    }
}

// `None` is an owned typed request. The semantic result remains inline in its
// existing driver-owned completion slot, without a large enum arm or Box.
struct CompletedReply(Option<SemanticReply>);

impl QueuedCall {
    fn is_cancelled(&self) -> bool {
        match &self.reply {
            QueuedReply::Semantic(reply) => reply.is_canceled(),
            QueuedReply::Owned(completion) => completion.is_canceled(),
        }
    }

    fn poll(&mut self, context: &mut Context<'_>) -> Poll<CompletedReply> {
        match &mut self.future {
            QueuedFuture::Semantic(future) => future
                .as_mut()
                .poll(context)
                .map(|result| CompletedReply(Some(result))),
            QueuedFuture::Owned(future) => {
                future.as_mut().poll(context).map(|()| CompletedReply(None))
            }
        }
    }
}

struct DriverState {
    dynamic_admission: Option<DynamicEvalAdmissionPort>,
    tasks: HashMap<CallId, QueuedCall>,
    current: Option<(CallId, u32)>,
    next_id: u64,
    accepting: bool,
    retiring: bool,
    wake_hub: Arc<WakeHub>,
}

#[derive(Default)]
struct WakeHub {
    state: Mutex<WakeHubState>,
}

#[derive(Default)]
struct WakeHubState {
    ready: BinaryHeap<ReadyEntry>,
    host: Option<Waker>,
}

struct ReadyEntry(Arc<TaskWake>);

impl PartialEq for ReadyEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.depth == other.0.depth && self.0.id == other.0.id
    }
}

impl Eq for ReadyEntry {}

impl PartialOrd for ReadyEntry {
    fn partial_cmp(&self, other: &Self) -> Option<SortOrder> {
        Some(self.cmp(other))
    }
}

impl Ord for ReadyEntry {
    fn cmp(&self, other: &Self) -> SortOrder {
        (self.0.depth, self.0.id.0).cmp(&(other.0.depth, other.0.id.0))
    }
}

struct TaskWake {
    id: CallId,
    depth: u32,
    ready: AtomicBool,
    hub: SyncWeak<WakeHub>,
}

impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        Self::wake_by_ref(&self);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        if self.ready.swap(true, Ordering::AcqRel) {
            return;
        }
        let Some(hub) = self.hub.upgrade() else {
            return;
        };
        let host = {
            let mut state = hub
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.ready.push(ReadyEntry(Arc::clone(self)));
            state.host.take()
        };
        if let Some(host) = host {
            host.wake();
        }
    }
}

impl WakeHub {
    fn take_ready(&self) -> Option<Arc<TaskWake>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let ready = state.ready.pop()?.0;
        ready.ready.store(false, Ordering::Release);
        Some(ready)
    }

    fn clear_host(&self) {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .host = None;
    }

    fn park(&self, host: &Waker) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.host = Some(host.clone());
        !state.ready.is_empty()
    }
}

/// A weak capability: VM invoker closures cannot keep the invocation alive.
#[derive(Clone)]
pub(crate) struct ProcessDriverHandle(RcWeak<RefCell<DriverState>>);

impl std::fmt::Debug for ProcessDriverHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProcessDriverHandle")
            .finish_non_exhaustive()
    }
}

impl ProcessDriverHandle {
    /// The owning invocation keeps this port alive; a queued nested call does
    /// not create another authority or extend the process driver's lifetime.
    pub(crate) fn dynamic_admission(&self) -> Option<DynamicEvalAdmissionPort> {
        self.0
            .upgrade()
            .and_then(|owner| owner.borrow().dynamic_admission.clone())
    }

    pub(crate) fn enqueue(
        &self,
        future: impl Future<Output = SemanticReply> + 'static,
    ) -> Result<ProcessReplyTicket<(RootedValueSequence, RootedValues)>, RuntimeError> {
        let (reply, receiver) = oneshot::channel();
        let wake = self.register(
            QueuedFuture::Semantic(Box::pin(future)),
            QueuedReply::Semantic(reply),
        )?;
        Ok(ProcessReplyTicket {
            receiver: Some(receiver),
            wake,
        })
    }

    /// One exact invocation-owned request, with a typed result rather than a
    /// second driver or an untyped payload. Dropping its ticket retires the
    /// request and descendants through the existing queue owner.
    pub(crate) fn enqueue_owned<T: 'static>(
        &self,
        future: impl Future<Output = Result<T, RuntimeError>> + 'static,
    ) -> Result<ProcessReplyTicket<T>, RuntimeError> {
        let (reply, receiver) = oneshot::channel();
        // The same reply channel is the cancellation authority for this cold
        // typed request. Its original future is inline in the one queued Box.
        let completion = Rc::new(OwnedCompletion {
            reply: RefCell::new(Some(reply)),
            result: RefCell::new(None),
        });
        // The pinned envelope keeps `future` alive after Poll::Ready. Its
        // captured roots and nested-call state retire only with the queued
        // parent, after descendants and before the reply is observable.
        let queued = TypedQueuedRequest::new(future, Rc::clone(&completion));
        let wake = self.register(
            QueuedFuture::Owned(Box::pin(queued)),
            QueuedReply::Owned(completion),
        )?;
        Ok(ProcessReplyTicket {
            receiver: Some(receiver),
            wake,
        })
    }

    fn register(
        &self,
        future: QueuedFuture,
        reply: QueuedReply,
    ) -> Result<Arc<TaskWake>, RuntimeError> {
        let owner = self.0.upgrade().ok_or_else(driver_unavailable)?;
        let wake = {
            let mut state = owner.borrow_mut();
            if !state.accepting || state.retiring {
                return Err(driver_unavailable());
            }
            let id = CallId(state.next_id);
            state.next_id = state
                .next_id
                .checked_add(1)
                .ok_or_else(driver_unavailable)?;
            let (parent, depth) = state.current.map_or((None, 1), |(parent, depth)| {
                (Some(parent), depth.saturating_add(1))
            });
            let wake = Arc::new(TaskWake {
                id,
                depth,
                ready: AtomicBool::new(false),
                hub: Arc::downgrade(&state.wake_hub),
            });
            state.tasks.insert(
                id,
                QueuedCall {
                    parent,
                    depth,
                    future,
                    reply,
                },
            );
            wake
        };
        TaskWake::wake_by_ref(&wake);
        Ok(wake)
    }
}

fn driver_unavailable() -> RuntimeError {
    crate::interpreter::errors::mex(
        "ProcessContinuationUnavailable",
        "the selected VM invocation has retired",
    )
}

pub(crate) struct ProcessReplyTicket<T> {
    receiver: Option<oneshot::Receiver<Result<T, RuntimeError>>>,
    wake: Arc<TaskWake>,
}

impl<T> Future for ProcessReplyTicket<T> {
    type Output = Result<T, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let result = Pin::new(
            self.receiver
                .as_mut()
                .expect("VM reply ticket polled after completion"),
        )
        .poll(context);
        if result.is_ready() {
            drop(self.receiver.take());
        }
        result.map(|result| result.unwrap_or_else(|_| Err(driver_unavailable())))
    }
}

impl<T> Drop for ProcessReplyTicket<T> {
    fn drop(&mut self) {
        if let Some(receiver) = self.receiver.take() {
            drop(receiver);
            TaskWake::wake_by_ref(&self.wake);
        }
    }
}

pub(crate) struct ProcessContinuationDriver {
    state: Rc<RefCell<DriverState>>,
}

impl ProcessContinuationDriver {
    pub(crate) fn new() -> Self {
        Self::new_with_dynamic_admission(None)
    }

    pub(crate) fn new_with_dynamic_admission(
        dynamic_admission: Option<DynamicEvalAdmissionPort>,
    ) -> Self {
        let wake_hub = Arc::new(WakeHub::default());
        Self {
            state: Rc::new(RefCell::new(DriverState {
                dynamic_admission,
                tasks: HashMap::new(),
                current: None,
                next_id: 1,
                accepting: true,
                retiring: false,
                wake_hub,
            })),
        }
    }

    pub(crate) fn handle(&self) -> ProcessDriverHandle {
        ProcessDriverHandle(Rc::downgrade(&self.state))
    }

    pub(crate) fn drive<F: Future>(&self, root: F) -> DrivenInvocation<F> {
        let hub = Arc::clone(&self.state.borrow().wake_hub);
        let root_wake = Arc::new(TaskWake {
            id: CallId(0),
            depth: 0,
            ready: AtomicBool::new(false),
            hub: Arc::downgrade(&hub),
        });
        TaskWake::wake_by_ref(&root_wake);
        DrivenInvocation {
            state: Rc::clone(&self.state),
            hub,
            root_wake,
            root: Some(Box::pin(root)),
        }
    }
}

pub(crate) struct DrivenInvocation<F: Future> {
    state: Rc<RefCell<DriverState>>,
    hub: Arc<WakeHub>,
    root_wake: Arc<TaskWake>,
    root: Option<Pin<Box<F>>>,
}

impl<F: Future> Future for DrivenInvocation<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.hub.clear_host();
        const POLL_BUDGET: usize = 1024;
        let mut polls = 0;
        loop {
            let Some(ready) = this.hub.take_ready() else {
                if this.hub.park(context.waker()) {
                    this.hub.clear_host();
                    continue;
                }
                return Poll::Pending;
            };
            polls += 1;
            if polls > POLL_BUDGET {
                TaskWake::wake_by_ref(&ready);
                context.waker().wake_by_ref();
                return Poll::Pending;
            }
            if ready.id == CallId(0) {
                let root_waker = Waker::from(Arc::clone(&this.root_wake));
                let mut root_context = Context::from_waker(&root_waker);
                match this
                    .root
                    .as_mut()
                    .expect("VM invocation polled after completion")
                    .as_mut()
                    .poll(&mut root_context)
                {
                    Poll::Ready(result) => {
                        this.retire_all();
                        return Poll::Ready(result);
                    }
                    Poll::Pending => continue,
                }
            }
            let id = ready.id;
            let Some(mut task) = this.state.borrow_mut().tasks.remove(&id) else {
                continue;
            };
            if task.is_cancelled() {
                this.retire_subtree(id, task, None);
                continue;
            }
            this.state.borrow_mut().current = Some((id, task.depth));
            let task_waker = Waker::from(ready);
            let mut task_context = Context::from_waker(&task_waker);
            let result = task.poll(&mut task_context);
            this.state.borrow_mut().current = None;
            match result {
                Poll::Ready(result) => {
                    this.retire_subtree(id, task, Some(result));
                }
                Poll::Pending => {
                    this.state.borrow_mut().tasks.insert(id, task);
                }
            }
        }
    }
}

impl<F: Future> DrivenInvocation<F> {
    fn retire_subtree(&mut self, id: CallId, task: QueuedCall, result: Option<CompletedReply>) {
        let descendants = {
            let mut state = self.state.borrow_mut();
            state.retiring = true;
            let mut ids = HashSet::from([id]);
            loop {
                let next = state
                    .tasks
                    .iter()
                    .filter_map(|(candidate, task)| {
                        task.parent.filter(|parent| ids.contains(parent))?;
                        (!ids.contains(candidate)).then_some(*candidate)
                    })
                    .collect::<Vec<_>>();
                if next.is_empty() {
                    break;
                }
                ids.extend(next);
            }
            ids.remove(&id);
            ids.into_iter()
                .filter_map(|descendant| state.tasks.remove(&descendant))
                .collect::<Vec<_>>()
        };
        let mut retired = descendants;
        retired.sort_by_key(|task| task.depth);
        while retired.pop().is_some() {}
        let QueuedCall { future, reply, .. } = task;
        drop(future);
        self.state.borrow_mut().retiring = false;
        match (reply, result) {
            (QueuedReply::Semantic(reply), Some(CompletedReply(Some(result)))) => {
                let _ = reply.send(result);
            }
            (QueuedReply::Owned(completion), Some(CompletedReply(None))) => completion.publish(),
            _ => {}
        }
    }

    fn retire_all(&mut self) {
        let tasks = {
            let mut state = self.state.borrow_mut();
            state.accepting = false;
            state.retiring = true;
            std::mem::take(&mut state.tasks)
        };
        self.hub.clear_host();
        let mut tasks = tasks.into_values().collect::<Vec<_>>();
        tasks.sort_by_key(|task| task.depth);
        while tasks.pop().is_some() {}
        drop(self.root.take());
    }
}

impl<F: Future> Drop for DrivenInvocation<F> {
    fn drop(&mut self) {
        self.retire_all();
    }
}

// Moving the owner never moves either boxed future.
impl<F: Future> Unpin for DrivenInvocation<F> {}

#[cfg(test)]
#[path = "process_driver/tests.rs"]
mod tests;
