//! Same-thread, invocation-owned continuation of nested semantic VM calls.

use crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort;
use runmat_execution::resource::ExactShared;
use runmat_runtime::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
    HostAllocationReceipt,
};
use runmat_runtime::context::{PreparedRuntimeScope, RuntimeContext};
use runmat_runtime::execution::{RootedValueSequence, RootedValues};
use runmat_runtime::RuntimeError;
use runmat_value::{ExactHostAllocationClaim, ExactHostNumericAllocationError};
use std::cell::{Cell, RefCell};
use std::cmp::Ordering as SortOrder;
use std::future::Future;
use std::mem::size_of;
use std::pin::Pin;
use std::rc::{Rc, Weak as RcWeak};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak as SyncWeak};
use std::task::{Context, Poll, Wake, Waker};

#[path = "process_driver/exact_waker.rs"]
mod exact_waker;

#[path = "process_driver/control_owner.rs"]
mod control_owner;
use control_owner::{
    managed_driver_control_bytes, managed_request_control_bytes, prepare_managed_driver,
    prepare_managed_request, try_box, CompletionOwner, DriverOwner, DriverStateOwner,
    ManagedDriverPort, ReplyControlOwner,
};

#[path = "process_driver/enqueue.rs"]
mod enqueue;

#[path = "process_driver/typed_request.rs"]
mod typed_request;
use typed_request::TypedQueuedRequest;

#[path = "process_driver/queue_storage.rs"]
mod queue_storage;
use queue_storage::{PreparedQueueGrowth, QueueGrowthGeometry, ReadyStorage, TaskStorage};

#[path = "process_driver/invocation_root.rs"]
mod invocation_root;
use invocation_root::{drain_placement_feedback, InvocationRoot};

pub(crate) type SemanticReply = Result<(RootedValueSequence, RootedValues), RuntimeError>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CallId(u64);

impl CallId {
    fn managed(index: usize, generation: u32) -> Result<Self, HostAllocationError> {
        let slot = u32::try_from(index)
            .map_err(|_| HostAllocationError::Overflow)?
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(Self((u64::from(generation) << 32) | u64::from(slot)))
    }

    fn managed_parts(self) -> Option<(usize, u32)> {
        let slot = u32::try_from(self.0 & u64::from(u32::MAX)).ok()?;
        let index = usize::try_from(slot.checked_sub(1)?).ok()?;
        Some((index, (self.0 >> 32) as u32))
    }
}

struct QueuedCall {
    parent: Option<CallId>,
    depth: u32,
    future: QueuedFuture,
    reply: ReplyControlOwner,
    _wake: WakeOwner,
    _scratch: Option<Arc<runmat_runtime::context::host_effect::ManagedHostEffectStageReservation>>,
}

struct QueuedFuture(Pin<Box<dyn Future<Output = ()>>>);

trait ReplyControl {
    fn is_canceled(&self) -> bool;
    fn publish(&self);
}

struct ReplyCell<T> {
    result: RefCell<Option<Result<T, RuntimeError>>>,
    waker: RefCell<Option<Waker>>,
    canceled: Cell<bool>,
    published: Cell<bool>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl<T> ReplyControl for ReplyCell<T> {
    fn is_canceled(&self) -> bool {
        self.canceled.get()
    }

    fn publish(&self) {
        self.published.set(true);
        let waker = { self.waker.borrow_mut().take() };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

impl QueuedCall {
    fn is_cancelled(&self) -> bool {
        self.reply.is_canceled()
    }

    fn poll(&mut self, context: &mut Context<'_>) -> Poll<()> {
        self.future.0.as_mut().poll(context)
    }

    fn deactivate(&self) {
        self._wake.deactivate();
    }
}

struct DriverState {
    dynamic_admission: Option<DynamicEvalAdmissionPort>,
    tasks: TaskStorage,
    current: Option<(CallId, u32)>,
    next_id: u64,
    accepting: bool,
    retiring: bool,
    wake_hub: WakeHubOwner,
    authority: Option<RuntimeContext>,
    _scratch: Option<Arc<runmat_runtime::context::host_effect::ManagedHostEffectStageReservation>>,
    _claim: Option<ExactHostAllocationClaim>,
}

struct WakeHub {
    state: Mutex<WakeHubState>,
    _claim: Option<ExactHostAllocationClaim>,
}

struct WakeHubState {
    ready: ReadyStorage,
    host: Option<Waker>,
}

#[derive(Clone)]
enum WakeHubOwner {
    Legacy(Arc<WakeHub>),
    Managed(ExactShared<WakeHub>),
}

impl WakeHubOwner {
    fn get(&self) -> &WakeHub {
        match self {
            Self::Legacy(owner) => owner,
            Self::Managed(owner) => owner.get(),
        }
    }
}

enum WakeHubReference {
    Legacy(SyncWeak<WakeHub>),
    Managed(ExactShared<WakeHub>),
}

impl WakeHubReference {
    fn get(&self) -> Option<&WakeHub> {
        match self {
            Self::Legacy(owner) => {
                // Legacy wake dispatch uses `Arc<WakeHub>` through `Wake` and
                // upgrades separately; this borrowed path is Managed-only.
                let _ = owner;
                None
            }
            Self::Managed(owner) => Some(owner.get()),
        }
    }
}

impl DriverState {
    fn ensure_context(&self, context: &RuntimeContext) -> Result<(), RuntimeError> {
        let Some(authority) = &self.authority else {
            return Err(driver_unavailable());
        };
        if authority.local_identity() != context.local_identity()
            || !authority
                .cancellation()
                .same_generation(&context.cancellation())
        {
            return Err(queue_allocation_error(
                context,
                HostAllocationError::StaleAuthority,
            ));
        }
        if context.is_cancelled() {
            return Err(queue_allocation_error(
                context,
                HostAllocationError::Cancelled,
            ));
        }
        Ok(())
    }

    fn register_prepared(&mut self, request: PreparedRegistration, ready: &mut ReadyStorage) {
        let PreparedRegistration {
            id,
            parent,
            depth,
            wake,
            future,
            reply,
            scratch,
        } = request;
        self.tasks.insert(
            id,
            QueuedCall {
                parent,
                depth,
                future,
                reply,
                _wake: wake.clone(),
                _scratch: scratch,
            },
        );
        ready.push(ReadyEntry(wake));
    }
}

struct PreparedRegistration {
    id: CallId,
    parent: Option<CallId>,
    depth: u32,
    wake: WakeOwner,
    future: QueuedFuture,
    reply: ReplyControlOwner,
    scratch: Option<Arc<runmat_runtime::context::host_effect::ManagedHostEffectStageReservation>>,
}

struct ReadyEntry(WakeOwner);

impl PartialEq for ReadyEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.get().depth == other.0.get().depth && self.0.get().id == other.0.get().id
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
        (self.0.get().depth, self.0.get().id.0).cmp(&(other.0.get().depth, other.0.get().id.0))
    }
}

pub(super) struct TaskWake {
    id: CallId,
    depth: u32,
    ready: AtomicBool,
    active: AtomicBool,
    hub: WakeHubReference,
    _claim: Option<ExactHostAllocationClaim>,
}

impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        Self::wake_by_ref(&self);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let Some(hub) = (match &self.hub {
            WakeHubReference::Legacy(owner) => owner.upgrade(),
            WakeHubReference::Managed(_) => None,
        }) else {
            return;
        };
        wake_task(self, &hub);
    }
}

impl TaskWake {
    fn deactivate(&self) {
        let legacy;
        let hub = match &self.hub {
            WakeHubReference::Legacy(owner) => {
                legacy = owner.upgrade();
                let Some(owner) = legacy.as_deref() else {
                    self.active.store(false, Ordering::Release);
                    return;
                };
                owner
            }
            WakeHubReference::Managed(owner) => owner.get(),
        };
        let mut state = hub
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        self.active.store(false, Ordering::Release);
        self.ready.store(false, Ordering::Release);
        state.ready.remove(self.id);
    }
}

#[derive(Clone)]
enum WakeOwner {
    Legacy(Arc<TaskWake>),
    Managed(ExactShared<TaskWake>),
}

impl WakeOwner {
    fn get(&self) -> &TaskWake {
        match self {
            Self::Legacy(owner) => owner,
            Self::Managed(owner) => owner.get(),
        }
    }

    fn wake_by_ref(&self) {
        match self {
            Self::Legacy(owner) => TaskWake::wake_by_ref(owner),
            Self::Managed(owner) => {
                if let Some(hub) = owner.get().hub.get() {
                    wake_task(owner, hub);
                }
            }
        }
    }

    fn waker(&self) -> Waker {
        match self {
            Self::Legacy(owner) => Waker::from(Arc::clone(owner)),
            Self::Managed(owner) => exact_waker::waker(owner.clone()),
        }
    }

    fn deactivate(&self) {
        self.get().deactivate();
    }
}

fn wake_task(owner: impl Into<WakeOwner>, hub: &WakeHub) {
    let owner = owner.into();
    let wake = owner.get();
    if !wake.active.load(Ordering::Acquire) || wake.ready.swap(true, Ordering::AcqRel) {
        return;
    }
    let host = {
        let mut state = hub
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !wake.active.load(Ordering::Acquire) {
            wake.ready.store(false, Ordering::Release);
            return;
        }
        state.ready.push(ReadyEntry(owner));
        state.host.take()
    };
    if let Some(host) = host {
        host.wake();
    }
}

pub(super) fn wake_managed(owner: &ExactShared<TaskWake>) {
    if let Some(hub) = owner.get().hub.get() {
        wake_task(owner, hub);
    }
}

impl From<&Arc<TaskWake>> for WakeOwner {
    fn from(owner: &Arc<TaskWake>) -> Self {
        Self::Legacy(Arc::clone(owner))
    }
}

impl From<&ExactShared<TaskWake>> for WakeOwner {
    fn from(owner: &ExactShared<TaskWake>) -> Self {
        Self::Managed(owner.clone())
    }
}

impl WakeHub {
    fn take_ready(&self) -> Option<WakeOwner> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let ready = state.ready.pop()?.0;
        ready.get().ready.store(false, Ordering::Release);
        Some(ready)
    }

    fn clear_host(&self) {
        let retired = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .host
            .take();
        drop(retired);
    }

    fn park(&self, host: &Waker) -> bool {
        let (retired, ready) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let retired = state.host.replace(host.clone());
            (retired, !state.ready.is_empty())
        };
        drop(retired);
        ready
    }
}

/// A weak capability: VM invoker closures cannot keep the invocation alive.
#[derive(Clone)]
pub(crate) struct ProcessDriverHandle(DriverHandleOwner);

thread_local! {
    // A transient weak capability, never a retained runtime or driver owner.
    static POLLING_DRIVER: RefCell<Option<ProcessDriverHandle>> = const { RefCell::new(None) };
}

struct PollingDriverGuard(Option<ProcessDriverHandle>);
impl PollingDriverGuard {
    fn enter(handle: ProcessDriverHandle) -> Self {
        Self(POLLING_DRIVER.with(|slot| slot.replace(Some(handle))))
    }
}
impl Drop for PollingDriverGuard {
    fn drop(&mut self) {
        POLLING_DRIVER.with(|slot| {
            slot.replace(self.0.take());
        });
    }
}

#[derive(Clone)]
enum DriverHandleOwner {
    Legacy(RcWeak<RefCell<DriverState>>),
    Managed(ExactShared<ManagedDriverPort>),
}

impl std::fmt::Debug for ProcessDriverHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProcessDriverHandle")
            .finish_non_exhaustive()
    }
}

impl ProcessDriverHandle {
    /// Reenter only the driver currently polling this exact invocation. A
    /// different context or cancellation generation is a refusal, not a new
    /// owner fallback. Outside a poll, an independent invocation has no handle.
    pub(crate) fn current_in_context(
        context: &RuntimeContext,
    ) -> Result<Option<Self>, RuntimeError> {
        let handle = POLLING_DRIVER.with(|slot| slot.borrow().clone());
        if let Some(handle) = &handle {
            handle.owner()?.borrow().ensure_context(context)?;
        }
        Ok(handle)
    }
    fn owner(&self) -> Result<DriverStateOwner, RuntimeError> {
        match &self.0 {
            DriverHandleOwner::Legacy(owner) => owner
                .upgrade()
                .map(DriverStateOwner::Legacy)
                .ok_or_else(driver_unavailable),
            DriverHandleOwner::Managed(port) => port
                .get()
                .owner()
                .map(DriverStateOwner::Managed)
                .ok_or_else(driver_unavailable),
        }
    }

    /// The owning invocation keeps this port alive; a queued nested call does
    /// not create another authority or extend the process driver's lifetime.
    pub(crate) fn dynamic_admission(&self) -> Option<DynamicEvalAdmissionPort> {
        self.owner()
            .ok()
            .and_then(|owner| owner.borrow().dynamic_admission.clone())
    }

    #[cfg(test)]
    pub(crate) fn enqueue(
        &self,
        future: impl Future<Output = SemanticReply> + 'static,
    ) -> Result<ProcessReplyTicket<(RootedValueSequence, RootedValues)>, RuntimeError> {
        self.enqueue_typed_unbounded(future)
    }

    /// Queue a future whose polls must run under `scope`. `RuntimeContext::scope`
    /// owns the inner future in a separate box, so this entry admits that
    /// concrete allocation before invoking either factory.
    pub(crate) fn enqueue_scoped_in_context<F, M>(
        &self,
        context: &RuntimeContext,
        scope: PreparedRuntimeScope,
        make_future: M,
    ) -> Result<ProcessReplyTicket<(RootedValueSequence, RootedValues)>, RuntimeError>
    where
        F: Future<Output = SemanticReply> + 'static,
        M: FnOnce() -> F + 'static,
    {
        self.enqueue_owned_catch_scoped_in_context(context, scope, make_future)
    }

    /// A semantic callee inherits the caller's selected catch even when its
    /// first poll is moved to a sibling. Capture before enqueueing; `None`
    /// explicitly masks any other catch active while that sibling is polled.
    pub(crate) fn enqueue_owned_catch_scoped_in_context<T, F, M>(
        &self,
        context: &RuntimeContext,
        scope: PreparedRuntimeScope,
        make_future: M,
    ) -> Result<ProcessReplyTicket<T>, RuntimeError>
    where
        T: 'static,
        F: Future<Output = Result<T, RuntimeError>> + 'static,
        M: FnOnce() -> F + 'static,
    {
        let catch = context.active_catch_exception_shared();
        let catch_runtime = context.clone();
        let scope = scope.prepare_future::<F>()?;
        self.enqueue_typed_in_context(context, 0, move || {
            // Keep the unboxed catch around the deferred owner itself: an
            // unpolled factory's captures and a constructed child's poll/drop
            // must all run under the same selected catch. The inner quote is
            // still exactly F, and the queue admits this concrete wrapper in
            // its original allocation; no extra child Box is introduced.
            Ok(catch_runtime
                .scope_active_catch_unboxed_shared(catch, scope.defer_result(make_future)))
        })
    }

    fn enqueue_typed_unbounded<T: 'static>(
        &self,
        future: impl Future<Output = Result<T, RuntimeError>> + 'static,
    ) -> Result<ProcessReplyTicket<T>, RuntimeError> {
        let completion = Rc::new(ReplyCell {
            result: RefCell::new(None),
            waker: RefCell::new(None),
            canceled: Cell::new(false),
            published: Cell::new(false),
            _claim: None,
        });
        let control: Rc<dyn ReplyControl> = completion.clone();
        let completion = CompletionOwner::Legacy(completion);
        let queued = TypedQueuedRequest::new(future, completion.clone());
        let wake = self.register(
            QueuedFuture(Box::pin(queued)),
            ReplyControlOwner::Legacy(control),
            None,
        )?;
        Ok(ProcessReplyTicket {
            completion,
            wake,
            _scratch: None,
        })
    }

    /// One exact invocation-owned request, with a typed result rather than a
    /// second driver or an untyped payload. Dropping its ticket retires the
    /// request and descendants through the existing queue owner.
    #[cfg(test)]
    pub(crate) fn enqueue_owned<T: 'static>(
        &self,
        future: impl Future<Output = Result<T, RuntimeError>> + 'static,
    ) -> Result<ProcessReplyTicket<T>, RuntimeError> {
        self.enqueue_typed_unbounded(future)
    }

    pub(crate) fn enqueue_owned_in_context<T, F, M>(
        &self,
        context: &RuntimeContext,
        make_future: M,
    ) -> Result<ProcessReplyTicket<T>, RuntimeError>
    where
        T: 'static,
        F: Future<Output = Result<T, RuntimeError>> + 'static,
        M: FnOnce() -> F,
    {
        self.enqueue_typed_in_context(context, 0, move || Ok(make_future()))
    }

    pub(crate) fn enqueue_owned_scoped_in_context<T, F, M>(
        &self,
        context: &RuntimeContext,
        scope: PreparedRuntimeScope,
        make_future: M,
    ) -> Result<ProcessReplyTicket<T>, RuntimeError>
    where
        T: 'static,
        F: Future<Output = Result<T, RuntimeError>> + 'static,
        M: FnOnce() -> F + 'static,
    {
        let scope = scope.prepare_future::<F>()?;
        self.enqueue_typed_in_context(context, 0, move || Ok(scope.defer_result(make_future)))
    }

    fn enqueue_typed_in_context<T, F, M>(
        &self,
        context: &RuntimeContext,
        additional_scratch_bytes: u64,
        make_future: M,
    ) -> Result<ProcessReplyTicket<T>, RuntimeError>
    where
        T: 'static,
        F: Future<Output = Result<T, RuntimeError>> + 'static,
        M: FnOnce() -> Result<F, RuntimeError>,
    {
        let owner = self.owner()?;
        let managed = {
            let state = owner.borrow();
            state.ensure_context(context)?;
            if !state.accepting || state.retiring {
                return Err(driver_unavailable());
            }
            state.tasks.is_managed()
        };
        // The selected branch alone materializes its concrete child. In
        // particular ordinary calls do not retain managed quote/install
        // temporaries while their scoped factory constructs its future.
        if managed {
            enqueue::managed(context, owner, additional_scratch_bytes, make_future)
        } else {
            enqueue::unbounded(self, make_future)
        }
    }

    fn register(
        &self,
        future: QueuedFuture,
        reply: ReplyControlOwner,
        scratch: Option<
            Arc<runmat_runtime::context::host_effect::ManagedHostEffectStageReservation>,
        >,
    ) -> Result<WakeOwner, RuntimeError> {
        let owner = self.owner()?;
        let wake = {
            let mut state = owner.borrow_mut();
            if !state.accepting || state.retiring {
                return Err(driver_unavailable());
            }
            if state.tasks.is_managed() {
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
                active: AtomicBool::new(true),
                hub: match &state.wake_hub {
                    WakeHubOwner::Legacy(hub) => WakeHubReference::Legacy(Arc::downgrade(hub)),
                    WakeHubOwner::Managed(hub) => WakeHubReference::Managed(hub.clone()),
                },
                _claim: None,
            });
            let wake = WakeOwner::Legacy(wake);
            state.tasks.insert(
                id,
                QueuedCall {
                    parent,
                    depth,
                    future,
                    reply,
                    _wake: wake.clone(),
                    _scratch: scratch,
                },
            );
            wake
        };
        wake.wake_by_ref();
        Ok(wake)
    }
}

fn driver_unavailable() -> RuntimeError {
    crate::interpreter::errors::mex(
        "ProcessContinuationUnavailable",
        "the selected VM invocation has retired",
    )
}

fn driver_overflow() -> RuntimeError {
    crate::interpreter::errors::mex(
        "ProcessContinuationOverflow",
        "the selected VM invocation queue capacity overflowed",
    )
}

fn queue_allocation_error(context: &RuntimeContext, error: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(context),
        &format!(
            "RunMat:Interpreter:ProcessContinuation:{}",
            error.runtime_identifier_suffix()
        ),
        "continuation queue admission failed",
        error,
    )
}

fn request_scratch_bytes<F, T>() -> Result<u64, RuntimeError> {
    u64::try_from(size_of::<TypedQueuedRequest<F, T>>()).map_err(|_| driver_overflow())
}

pub(crate) struct ProcessReplyTicket<T> {
    completion: CompletionOwner<T>,
    wake: WakeOwner,
    _scratch: Option<Arc<runmat_runtime::context::host_effect::ManagedHostEffectStageReservation>>,
}

impl<T> Future for ProcessReplyTicket<T> {
    type Output = Result<T, RuntimeError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.completion.get().published.get() {
            return Poll::Ready(
                self.completion
                    .get()
                    .result
                    .borrow_mut()
                    .take()
                    .unwrap_or_else(|| Err(driver_unavailable())),
            );
        }
        *self.completion.get().waker.borrow_mut() = Some(context.waker().clone());
        if self.completion.get().published.get() {
            context.waker().wake_by_ref();
        }
        Poll::Pending
    }
}

impl<T> Drop for ProcessReplyTicket<T> {
    fn drop(&mut self) {
        if !self.completion.get().published.get() {
            self.completion.get().canceled.set(true);
            self.wake.wake_by_ref();
        }
    }
}

pub(crate) struct ProcessContinuationDriver {
    owner: DriverOwner,
    transferred: Cell<bool>,
}

impl ProcessContinuationDriver {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::new_with_dynamic_admission(None)
    }

    pub(crate) fn new_with_dynamic_admission(
        dynamic_admission: Option<DynamicEvalAdmissionPort>,
    ) -> Self {
        let wake_hub = Arc::new(WakeHub {
            state: Mutex::new(WakeHubState {
                ready: ReadyStorage::unbounded(),
                host: None,
            }),
            _claim: None,
        });
        Self {
            owner: DriverOwner::Legacy(Rc::new(RefCell::new(DriverState {
                dynamic_admission,
                tasks: TaskStorage::unbounded(),
                current: None,
                next_id: 1,
                accepting: true,
                retiring: false,
                wake_hub: WakeHubOwner::Legacy(wake_hub),
                authority: None,
                _scratch: None,
                _claim: None,
            }))),
            transferred: Cell::new(false),
        }
    }

    pub(crate) fn new_with_dynamic_admission_in_context(
        context: &RuntimeContext,
        dynamic_admission: Option<DynamicEvalAdmissionPort>,
    ) -> Result<Self, RuntimeError> {
        match HostAllocationReceipt::policy(Some(context))
            .map_err(|error| queue_allocation_error(context, error))?
        {
            HostAllocationPolicy::Unbounded => {
                let driver = Self::new_with_dynamic_admission(dynamic_admission);
                driver.owner.state().borrow_mut().authority = Some(context.clone());
                Ok(driver)
            }
            HostAllocationPolicy::Managed => {
                let raw_bytes = managed_driver_control_bytes()
                    .map_err(|error| queue_allocation_error(context, error))?;
                let retained = HostAllocationPlan::retained_exact_output_bytes(raw_bytes, 3)
                    .map_err(|error| queue_allocation_error(context, error))?;
                let plan = HostAllocationPlan::new(0, retained)
                    .map_err(|error| queue_allocation_error(context, error))?;
                let receipt = HostAllocationReceipt::admit(Some(context), plan)
                    .map_err(|error| queue_allocation_error(context, error))?
                    .ok_or_else(|| {
                        queue_allocation_error(context, HostAllocationError::MissingManagedOwner)
                    })?;
                let allocator = receipt
                    .collection_allocator(context)
                    .map_err(|error| queue_allocation_error(context, error))?;
                let scratch = Some(receipt.retain_scratch_for_cpu_work());
                let owner = receipt
                    .with_output_allocation(context, |permit| {
                        prepare_managed_driver(
                            permit,
                            allocator,
                            context.clone(),
                            dynamic_admission,
                            scratch,
                        )
                    })
                    .map_err(|error| queue_allocation_error(context, error))?;
                Ok(Self {
                    owner,
                    transferred: Cell::new(false),
                })
            }
        }
    }

    pub(crate) fn handle(&self) -> ProcessDriverHandle {
        ProcessDriverHandle(self.owner.handle())
    }

    #[cfg(test)]
    pub(crate) fn drive<F: Future>(&self, root: F) -> DrivenInvocation<F> {
        let state = self.owner.state();
        let root = InvocationRoot::new(state.clone(), Box::pin(root));
        self.drive_root(state, root)
    }

    fn drive_root<F: Future>(
        &self,
        state: DriverStateOwner,
        root: InvocationRoot<F>,
    ) -> DrivenInvocation<F> {
        let hub = state.borrow().wake_hub.clone();
        let WakeHubOwner::Legacy(legacy_hub) = &hub else {
            unreachable!("legacy drive requires legacy hub")
        };
        let root_wake = Arc::new(TaskWake {
            id: CallId(0),
            depth: 0,
            ready: AtomicBool::new(false),
            active: AtomicBool::new(true),
            hub: WakeHubReference::Legacy(Arc::downgrade(legacy_hub)),
            _claim: None,
        });
        TaskWake::wake_by_ref(&root_wake);
        self.transferred.set(true);
        DrivenInvocation {
            root: Some(root),
            state,
            hub,
            root_wake: WakeOwner::Legacy(root_wake),
            _scratch: None,
            managed_port: None,
        }
    }

    pub(crate) fn drive_in_context<F, M>(
        &self,
        context: &RuntimeContext,
        make_root: M,
    ) -> Result<DrivenInvocation<F>, RuntimeError>
    where
        F: Future,
        M: FnOnce() -> F,
    {
        {
            let state = self.owner.state();
            let state = state.borrow();
            state.ensure_context(context)?;
            if !state.accepting || state.retiring {
                return Err(driver_unavailable());
            }
            if !state.tasks.is_managed() {
                drop(state);
                let state = self.owner.state();
                let root = {
                    let _guard = context.enter();
                    InvocationRoot::new(state.clone(), Box::pin(make_root()))
                };
                return Ok(self.drive_root(state, root));
            }
        }
        let state_owner = self.owner.state();
        let (geometry, hub) = {
            let state = state_owner.borrow();
            state.ensure_context(context)?;
            let hub = state
                .wake_hub
                .get()
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let geometry = QueueGrowthGeometry {
                task_capacity: state.tasks.capacity(),
                task_index: None,
                ready_capacity: hub.ready.capacity(),
                ready_len: hub.ready.len(),
            };
            (geometry, state.wake_hub.clone())
        };
        let scratch_bytes = u64::try_from(size_of::<F>()).map_err(|_| driver_overflow())?;
        let mut prepared = PreparedQueueGrowth::prepare(
            context,
            geometry,
            scratch_bytes,
            u64::try_from(ExactShared::<TaskWake>::requested_bytes())
                .map_err(|_| driver_overflow())?,
            1,
        )
        .map_err(|error| queue_allocation_error(context, error))?;
        let scratch = Some(prepared.retained_scratch());
        let (root, root_wake) = prepared
            .with_output_allocation(context, |permit| {
                let root = {
                    let _guard = context.enter();
                    let root = try_box(make_root());
                    if root.is_err() {
                        drain_placement_feedback(context);
                    }
                    root?
                };
                let root = InvocationRoot::new(state_owner.clone(), Box::into_pin(root));
                let claim =
                    permit.reserve_exact_claim(ExactShared::<TaskWake>::requested_bytes())?;
                let managed_hub = match &hub {
                    WakeHubOwner::Managed(hub) => hub.clone(),
                    WakeHubOwner::Legacy(_) => {
                        return Err(ExactHostNumericAllocationError::AllocationFailed)
                    }
                };
                let wake = ExactShared::try_new(TaskWake {
                    id: CallId(0),
                    depth: 0,
                    ready: AtomicBool::new(true),
                    active: AtomicBool::new(true),
                    hub: WakeHubReference::Managed(managed_hub),
                    _claim: Some(claim),
                })
                .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
                Ok((Some(root), WakeOwner::Managed(wake)))
            })
            .map_err(|error| queue_allocation_error(context, error))?;
        prepared
            .checkpoint(context)
            .map_err(|error| queue_allocation_error(context, error))?;
        let retirement = {
            let mut state = state_owner.borrow_mut();
            state.ensure_context(context)?;
            let mut hub_state = hub
                .get()
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let retirement = prepared.install(&mut state.tasks, &mut hub_state.ready);
            hub_state.ready.push(ReadyEntry(root_wake.clone()));
            retirement
        };
        drop(retirement);
        self.transferred.set(true);
        let managed_port = match &self.owner {
            DriverOwner::Managed { port, .. } => Some(port.clone()),
            DriverOwner::Legacy(_) => None,
        };
        Ok(DrivenInvocation {
            state: state_owner,
            hub,
            root_wake,
            root,
            _scratch: scratch,
            managed_port,
        })
    }
}

impl Drop for ProcessContinuationDriver {
    fn drop(&mut self) {
        if !self.transferred.get() {
            self.owner.revoke();
        }
    }
}

pub(crate) struct DrivenInvocation<F: Future> {
    state: DriverStateOwner,
    hub: WakeHubOwner,
    root_wake: WakeOwner,
    root: Option<InvocationRoot<F>>,
    _scratch: Option<Arc<runmat_runtime::context::host_effect::ManagedHostEffectStageReservation>>,
    managed_port: Option<ExactShared<ManagedDriverPort>>,
}

impl<F: Future> Future for DrivenInvocation<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.hub.get().clear_host();
        const POLL_BUDGET: usize = 1024;
        let mut polls = 0;
        loop {
            let Some(ready) = this.hub.get().take_ready() else {
                if this.hub.get().park(context.waker()) {
                    this.hub.get().clear_host();
                    continue;
                }
                return Poll::Pending;
            };
            polls += 1;
            if polls > POLL_BUDGET {
                ready.wake_by_ref();
                context.waker().wake_by_ref();
                return Poll::Pending;
            }
            if ready.get().id == CallId(0) {
                let root_waker = this.root_wake.waker();
                let mut root_context = Context::from_waker(&root_waker);
                let result = {
                    let _driver = PollingDriverGuard::enter(this.poll_handle());
                    this.root
                        .as_mut()
                        .expect("VM invocation polled after completion")
                        .poll(&mut root_context)
                };
                match result {
                    Poll::Ready(result) => {
                        this.retire_all();
                        return Poll::Ready(result);
                    }
                    Poll::Pending => continue,
                }
            }
            let id = ready.get().id;
            let Some(mut task) = this.state.borrow_mut().tasks.take(&id) else {
                continue;
            };
            if task.is_cancelled() {
                this.retire_subtree(id, task, false);
                continue;
            }
            this.state.borrow_mut().current = Some((id, task.depth));
            let task_waker = ready.waker();
            let mut task_context = Context::from_waker(&task_waker);
            let result = {
                let _driver = PollingDriverGuard::enter(this.poll_handle());
                task.poll(&mut task_context)
            };
            this.state.borrow_mut().current = None;
            match result {
                Poll::Ready(()) => {
                    this.retire_subtree(id, task, true);
                }
                Poll::Pending => {
                    this.state.borrow_mut().tasks.insert(id, task);
                }
            }
        }
    }
}

impl<F: Future> DrivenInvocation<F> {
    fn poll_handle(&self) -> ProcessDriverHandle {
        ProcessDriverHandle(match &self.state {
            DriverStateOwner::Legacy(owner) => DriverHandleOwner::Legacy(Rc::downgrade(owner)),
            DriverStateOwner::Managed(_) => DriverHandleOwner::Managed(
                self.managed_port
                    .as_ref()
                    .expect("live managed driver poll")
                    .clone(),
            ),
        })
    }
    fn retire_subtree(&mut self, id: CallId, task: QueuedCall, publish: bool) {
        task.deactivate();
        {
            let mut state = self.state.borrow_mut();
            state.retiring = true;
            state.tasks.release(id);
        }
        // Retire deepest descendants first. The parent chain remains intact
        // while it is needed, and no cancellation-side scratch collection is
        // allocated after effects have begun.
        loop {
            let descendant = {
                let state = self.state.borrow();
                state
                    .tasks
                    .ids()
                    .filter(|candidate| task_descends_from(&state.tasks, *candidate, id))
                    .max_by_key(|candidate| state.tasks.get(candidate).map_or(0, |task| task.depth))
            };
            let Some(descendant) = descendant else {
                break;
            };
            let retired = self.state.borrow_mut().tasks.retire(&descendant);
            if let Some(retired) = retired.as_ref() {
                retired.deactivate();
            }
            drop(retired);
        }
        let QueuedCall {
            future,
            reply,
            _scratch: scratch,
            ..
        } = task;
        drop(future);
        self.state.borrow_mut().retiring = false;
        if publish {
            reply.publish();
        }
        drop(scratch);
    }

    fn retire_all(&mut self) {
        {
            let mut state = self.state.borrow_mut();
            state.accepting = false;
            state.retiring = true;
        }
        self.hub.get().clear_host();
        loop {
            let id = {
                let state = self.state.borrow();
                state
                    .tasks
                    .ids()
                    .max_by_key(|id| state.tasks.get(id).map_or(0, |task| task.depth))
            };
            let Some(id) = id else {
                break;
            };
            let retired = self.state.borrow_mut().tasks.retire(&id);
            if let Some(retired) = retired.as_ref() {
                retired.deactivate();
            }
            drop(retired);
        }
        self.root_wake.deactivate();
        drop(self.root.take());
        if let Some(port) = self.managed_port.take() {
            if let DriverStateOwner::Managed(state) = &self.state {
                port.get().revoke(state);
            }
            drop(port);
        }
    }
}

fn task_descends_from(tasks: &TaskStorage, mut candidate: CallId, ancestor: CallId) -> bool {
    while let Some(parent) = tasks.get(&candidate).and_then(|task| task.parent) {
        if parent == ancestor {
            return true;
        }
        candidate = parent;
    }
    false
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
