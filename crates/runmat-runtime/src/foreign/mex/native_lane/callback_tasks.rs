use super::*;
use crate::foreign::mex::services::{OriginMexBoundaryServices, OriginMexOperation};
use runmat_execution::cancellation::CancellationSource;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};

type CallbackFuture = Pin<Box<dyn Future<Output = ()> + 'static>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CallbackTaskId(u64);

struct ReadyQueue {
    ids: Mutex<VecDeque<CallbackTaskId>>,
    host_wakers: Mutex<HashMap<u64, Waker>>,
}

impl ReadyQueue {
    fn pop(&self) -> Option<CallbackTaskId> {
        self.ids
            .lock()
            .expect("MEX ready queue is not poisoned")
            .pop_front()
    }

    fn register(&self, ticket: u64, waker: &Waker) {
        self.host_wakers
            .lock()
            .expect("MEX host wakers are not poisoned")
            .insert(ticket, waker.clone());
    }
}

struct TaskWake {
    id: CallbackTaskId,
    queued: AtomicBool,
    queue: Arc<ReadyQueue>,
}

impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        if !self.queued.swap(true, Ordering::AcqRel) {
            self.queue
                .ids
                .lock()
                .expect("MEX ready queue is not poisoned")
                .push_back(self.id);
        }
        let hosts = self
            .queue
            .host_wakers
            .lock()
            .expect("MEX host wakers are not poisoned")
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for host in hosts {
            host.wake();
        }
    }
}

pub(super) struct CallbackTask {
    owner: NativeLaneOwner,
    service_id: u64,
    future: RefCell<Option<CallbackFuture>>,
    cancelled: Cell<bool>,
    wake: Arc<TaskWake>,
}

impl CallbackTask {
    fn new(
        owner: NativeLaneOwner,
        service_id: u64,
        future: CallbackFuture,
        wake: Arc<TaskWake>,
    ) -> Self {
        Self {
            owner,
            service_id,
            future: RefCell::new(Some(future)),
            cancelled: Cell::new(false),
            wake,
        }
    }

    pub(super) fn cancel(&self) {
        self.cancelled.set(true);
        let future = self.future.borrow_mut().take();
        drop(future);
    }
}

pub(super) struct CallbackTasks {
    next_id: Cell<u64>,
    next_poll: Cell<u64>,
    tasks: RefCell<HashMap<CallbackTaskId, Rc<CallbackTask>>>,
    ready: Arc<ReadyQueue>,
    active_owners: RefCell<Vec<NativeLaneOwner>>,
}

impl CallbackTasks {
    pub(super) fn new() -> Self {
        Self {
            next_id: Cell::new(1),
            next_poll: Cell::new(1),
            tasks: RefCell::new(HashMap::new()),
            ready: Arc::new(ReadyQueue {
                ids: Mutex::new(VecDeque::new()),
                host_wakers: Mutex::new(HashMap::new()),
            }),
            active_owners: RefCell::new(Vec::new()),
        }
    }

    pub(super) fn active_for(&self, owner: &NativeLaneOwner) -> bool {
        self.active_owners
            .borrow()
            .iter()
            .any(|active| active.same(owner))
    }

    pub(super) fn cancel_for(&self, owner: &NativeLaneOwner) {
        let selected = self
            .tasks
            .borrow()
            .iter()
            .filter(|(_, task)| task.owner.same(owner))
            .map(|(id, task)| (*id, Rc::clone(task)))
            .collect::<Vec<_>>();
        for (id, task) in selected {
            // An absent future is genuinely InPoll: its outer poll owns the
            // future and will observe cancellation on return. An idle task
            // can retire now, without leaving a missing-future ready ID that
            // requeues itself forever.
            let in_poll = task.future.borrow().is_none();
            task.cancel();
            if in_poll {
                task.wake.wake_by_ref();
            } else {
                let retired = self.tasks.borrow_mut().remove(&id);
                drop(retired);
            }
        }
    }

    pub(super) fn admit(
        &self,
        owner: &NativeLaneOwner,
        service_id: u64,
        request: BoundaryRequest,
        services: Rc<dyn OriginMexBoundaryServices>,
    ) -> bool {
        let Some((operation, cancellation, reply)) = split_request(request) else {
            return false;
        };
        let prepared = match services.prepare(operation, cancellation) {
            Ok(prepared) => prepared,
            Err(error) => {
                reply.send(MexEngineCompletion::from_result(Err(error)));
                return false;
            }
        };
        let id = CallbackTaskId(self.next_id.get());
        let Some(next) = id.0.checked_add(1) else {
            reply.send(MexEngineCompletion::from_result(Err(lane_diagnostic(
                "native callback task identity exhausted",
            ))));
            return false;
        };
        self.next_id.set(next);
        let wake = Arc::new(TaskWake {
            id,
            queued: AtomicBool::new(false),
            queue: Arc::clone(&self.ready),
        });
        let task = Rc::new(CallbackTask::new(
            owner.clone(),
            service_id,
            Box::pin(async move { reply.send(prepared.await) }),
            Arc::clone(&wake),
        ));
        self.tasks.borrow_mut().insert(id, task);
        wake.wake_by_ref();
        true
    }

    /// Poll only ready IDs. A callback may reenter the lane while InPoll; its
    /// future is absent from its slot and cannot be polled recursively.
    pub(super) fn next_completed<'a>(
        &'a self,
        callback_depth: &'a Cell<usize>,
    ) -> CallbackPollTicket<'a> {
        let id = self.next_poll.get();
        self.next_poll.set(
            id.checked_add(1)
                .expect("MEX callback poll identity exhausted"),
        );
        CallbackPollTicket {
            tasks: self,
            callback_depth,
            id,
        }
    }

    fn poll_one(
        &self,
        cx: &mut Context<'_>,
        callback_depth: &Cell<usize>,
        ticket: u64,
    ) -> Poll<u64> {
        self.ready.register(ticket, cx.waker());
        let budget = self
            .ready
            .ids
            .lock()
            .expect("MEX ready queue is not poisoned")
            .len();
        for _ in 0..budget {
            let Some(id) = self.ready.pop() else { break };
            let task = self.tasks.borrow().get(&id).cloned();
            let Some(task) = task else { continue };
            task.wake.queued.store(false, Ordering::Release);
            let Some(mut future) = task.future.borrow_mut().take() else {
                // A reentrant lane poll cannot run this InPoll task, but its
                // readiness must survive until the outer poll restores it.
                task.wake.wake_by_ref();
                continue;
            };
            if task.cancelled.get() {
                drop(future);
                let retired = self.tasks.borrow_mut().remove(&id);
                drop(retired);
                return Poll::Ready(task.service_id);
            }
            struct CallbackDepth<'a> {
                depth: &'a Cell<usize>,
                active: &'a RefCell<Vec<NativeLaneOwner>>,
            }
            impl Drop for CallbackDepth<'_> {
                fn drop(&mut self) {
                    self.depth.set(self.depth.get() - 1);
                    let _ = self.active.borrow_mut().pop();
                }
            }
            callback_depth.set(callback_depth.get() + 1);
            self.active_owners.borrow_mut().push(task.owner.clone());
            let polled = {
                let _depth = CallbackDepth {
                    depth: callback_depth,
                    active: &self.active_owners,
                };
                let waker = Waker::from(Arc::clone(&task.wake));
                future.as_mut().poll(&mut Context::from_waker(&waker))
            };
            match polled {
                Poll::Ready(()) => {
                    drop(future);
                    let retired = self.tasks.borrow_mut().remove(&id);
                    drop(retired);
                    return Poll::Ready(task.service_id);
                }
                Poll::Pending if task.cancelled.get() => {
                    drop(future);
                    let retired = self.tasks.borrow_mut().remove(&id);
                    drop(retired);
                    return Poll::Ready(task.service_id);
                }
                Poll::Pending => *task.future.borrow_mut() = Some(future),
            }
        }
        Poll::Pending
    }

    pub(super) fn cancel_all(&self) {
        let tasks = std::mem::take(&mut *self.tasks.borrow_mut());
        for task in tasks.into_values() {
            task.cancel();
        }
        self.ready
            .ids
            .lock()
            .expect("MEX ready queue is not poisoned")
            .clear();
        self.ready
            .host_wakers
            .lock()
            .expect("MEX host wakers are not poisoned")
            .clear();
    }
}

pub(super) struct CallbackPollTicket<'a> {
    tasks: &'a CallbackTasks,
    callback_depth: &'a Cell<usize>,
    id: u64,
}

impl Future for CallbackPollTicket<'_> {
    type Output = u64;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.tasks.poll_one(cx, self.callback_depth, self.id)
    }
}

impl Drop for CallbackPollTicket<'_> {
    fn drop(&mut self) {
        let removed = self
            .tasks
            .ready
            .host_wakers
            .lock()
            .expect("MEX host wakers are not poisoned")
            .remove(&self.id);
        drop(removed);
    }
}
enum Reply {
    Unit(mpsc::SyncSender<Result<(), MexDiagnostic>>),
    Values(mpsc::SyncSender<Result<Vec<MxArray>, MexDiagnostic>>),
    Optional(mpsc::SyncSender<Result<Option<MxArray>, MexDiagnostic>>),
    One(mpsc::SyncSender<Result<MxArray, MexDiagnostic>>),
    Async(Arc<NativeAsyncResult>),
}

impl Reply {
    fn send(self, completion: MexEngineCompletion<Vec<MxArray>>) {
        match self {
            Self::Unit(response) => {
                let _ = response.send(completion.result.and_then(|values| {
                    if values.is_empty() {
                        Ok(())
                    } else {
                        Err(lane_diagnostic("unit callback published values"))
                    }
                }));
            }
            Self::Values(response) => {
                let _ = response.send(completion.result);
            }
            Self::Optional(response) => {
                let _ = response.send(completion.result.and_then(|mut values| {
                    if values.len() <= 1 {
                        Ok(values.pop())
                    } else {
                        Err(lane_diagnostic(
                            "optional callback published multiple values",
                        ))
                    }
                }));
            }
            Self::One(response) => {
                let _ = response.send(completion.result.and_then(|mut values| {
                    if values.len() == 1 {
                        Ok(values.remove(0))
                    } else {
                        Err(lane_diagnostic(
                            "single callback published the wrong value count",
                        ))
                    }
                }));
            }
            Self::Async(result) => result.complete(completion),
        }
    }
}

fn split_request(
    request: BoundaryRequest,
) -> Option<(
    OriginMexOperation,
    runmat_execution::cancellation::CancellationToken,
    Reply,
)> {
    let ordinary = CancellationSource::new().token();
    let (operation, cancellation, reply) = match request {
        BoundaryRequest::Eval { command, response } => (
            OriginMexOperation::standard(MexAsyncOperation::Eval {
                command,
                capture_stdout: false,
                capture_stderr: false,
            }),
            ordinary,
            Reply::Unit(response),
        ),
        BoundaryRequest::Call {
            function,
            arguments,
            requested_outputs,
            response,
        } => (
            OriginMexOperation::standard(MexAsyncOperation::Call {
                function,
                arguments,
                requested_outputs,
                capture_stdout: false,
                capture_stderr: false,
            }),
            ordinary,
            Reply::Values(response),
        ),
        BoundaryRequest::GetVariable {
            workspace,
            name,
            response,
        } => (
            OriginMexOperation::OptionalGetVariable { workspace, name },
            ordinary,
            Reply::Optional(response),
        ),
        BoundaryRequest::PutVariable {
            workspace,
            name,
            value,
            response,
        } => (
            OriginMexOperation::standard(MexAsyncOperation::PutVariable {
                workspace,
                name,
                value,
            }),
            ordinary,
            Reply::Unit(response),
        ),
        BoundaryRequest::GetObjectProperty {
            object,
            index,
            name,
            response,
        } => (
            OriginMexOperation::standard(MexAsyncOperation::GetObjectProperty {
                object,
                index,
                name,
            }),
            ordinary,
            Reply::One(response),
        ),
        BoundaryRequest::SetObjectProperty {
            object,
            index,
            name,
            value,
            response,
        } => (
            OriginMexOperation::standard(MexAsyncOperation::SetObjectProperty {
                object,
                index,
                name,
                value,
            }),
            ordinary,
            Reply::One(response),
        ),
        BoundaryRequest::Async { operation, result } => {
            if !result.begin() {
                return None;
            }
            let cancellation = result.cancellation();
            (
                OriginMexOperation::standard(operation),
                cancellation,
                Reply::Async(result),
            )
        }
    };
    Some((operation, cancellation, reply))
}

#[cfg(test)]
mod owner_retirement_tests {
    use super::*;

    #[test]
    fn cancelling_one_idle_owner_retires_only_its_tasks() {
        let tasks = CallbackTasks::new();
        let child = NativeLaneOwner::default();
        let parent = NativeLaneOwner::default();
        for (number, owner) in [(1, &child), (2, &parent)] {
            let id = CallbackTaskId(number);
            let wake = Arc::new(TaskWake {
                id,
                queued: AtomicBool::new(false),
                queue: Arc::clone(&tasks.ready),
            });
            tasks.tasks.borrow_mut().insert(
                id,
                Rc::new(CallbackTask::new(
                    (*owner).clone(),
                    number,
                    Box::pin(futures::future::pending()),
                    wake,
                )),
            );
        }
        tasks.cancel_for(&child);
        assert!(!tasks.tasks.borrow().contains_key(&CallbackTaskId(1)));
        assert!(tasks.tasks.borrow().contains_key(&CallbackTaskId(2)));
        assert!(tasks.ready.pop().is_none());
    }
}
