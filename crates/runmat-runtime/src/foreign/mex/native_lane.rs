use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use futures::channel::{mpsc as async_mpsc, oneshot};
use futures::FutureExt;
use runmat_mex::{
    MexAsyncHostServices, MexAsyncOperation, MexAsyncResult, MexAsyncResultHandle,
    MexBoundaryHostServices, MexDiagnostic, MexEngineCompletion, MexLoadError, MexModule,
    MexModuleRetirement, MexNativeInvocation, MxApiMode, MxArray, MxBoundaryInterface,
};

mod callback_tasks;
mod inbox;
use super::services::OriginMexBoundaryServices;
use callback_tasks::CallbackTasks;
use inbox::CallbackInbox;

#[derive(Debug, Clone, Copy)]
pub(super) struct NativeModuleMetadata {
    pub mode: MxApiMode,
    pub interface: MxBoundaryInterface,
}

#[derive(Debug)]
pub(super) enum NativeModuleLoadError {
    IsolatedHostRequired,
    Failed {
        message: String,
        dependency: Option<String>,
    },
}

impl std::fmt::Display for NativeModuleLoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IsolatedHostRequired => formatter.write_str(
                "the MEX module already has an in-process owner and requires an isolated host",
            ),
            Self::Failed { message, .. } => formatter.write_str(message),
        }
    }
}

pub(super) struct NativeMexLane {
    commands: mpsc::Sender<NativeCommand>,
    callback_closer: async_mpsc::UnboundedSender<BoundaryEnvelope>,
    callbacks: CallbackInbox,
    services: RefCell<HashMap<u64, OriginServiceEntry>>,
    callback_tasks: CallbackTasks,
    next_service: Cell<u64>,
    callback_depth: Cell<usize>,
    abandoned: Cell<bool>,
    default_owner: NativeLaneOwner,
    session_owners: RefCell<Vec<NativeLaneOwner>>,
}

/// A host-session identity for service settlement on a shared native lane.
/// It does not own the lane or authorize another session's module lifecycle.
#[derive(Clone, Default)]
pub(super) struct NativeLaneOwner(Arc<AtomicBool>);

impl NativeLaneOwner {
    pub(super) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    fn is_live(&self) -> bool {
        !self.0.load(Ordering::Acquire)
    }

    fn retire(&self) {
        self.0.store(true, Ordering::Release);
    }
}

struct NativeLaneState {
    receiver: RefCell<mpsc::Receiver<NativeCommand>>,
    modules: RefCell<HashMap<PathBuf, OwnedNativeModule>>,
    callbacks: async_mpsc::UnboundedSender<BoundaryEnvelope>,
    command_depth: Cell<usize>,
    retirement_pending: Cell<bool>,
    retirement_pruning: Cell<bool>,
    stop_requested: Cell<bool>,
}

struct OwnedNativeModule {
    owner: NativeLaneOwner,
    module: Rc<MexModule>,
}

thread_local! {
    static CURRENT_NATIVE_LANE: RefCell<Option<Rc<NativeLaneState>>> = const { RefCell::new(None) };
}

struct OriginServiceEntry {
    owner: NativeLaneOwner,
    services: Rc<dyn OriginMexBoundaryServices>,
    invocation_active: bool,
    native_lease_active: bool,
    async_requests: usize,
    active_callbacks: usize,
}

enum NativeCommand {
    Load {
        owner: NativeLaneOwner,
        path: PathBuf,
        compatible_isolated: bool,
        response: oneshot::Sender<Result<NativeModuleMetadata, NativeModuleLoadError>>,
    },
    Invoke {
        owner: NativeLaneOwner,
        path: PathBuf,
        inputs: Vec<MxArray>,
        requested_outputs: usize,
        service_id: u64,
        response: oneshot::Sender<Result<MexNativeInvocation, MexLoadError>>,
    },
    Clear {
        owner: NativeLaneOwner,
        path: PathBuf,
        service_id: u64,
        response: oneshot::Sender<Result<bool, MexLoadError>>,
    },
    ShutdownModule {
        owner: NativeLaneOwner,
        path: PathBuf,
        service_id: u64,
        response: oneshot::Sender<Result<MexModuleRetirement, MexLoadError>>,
    },
    RetireOwner {
        owner: NativeLaneOwner,
    },
    Shutdown,
}

enum BoundaryEnvelope {
    Request {
        service_id: u64,
        request: Box<BoundaryRequest>,
    },
    AsyncStarted {
        service_id: u64,
    },
    AsyncFinished {
        service_id: u64,
    },
    ServiceFinished {
        service_id: u64,
    },
    InvocationFinished {
        service_id: u64,
    },
}

pub(super) enum BoundaryRequest {
    Eval {
        command: String,
        response: mpsc::SyncSender<Result<(), MexDiagnostic>>,
    },
    Call {
        function: String,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
        response: mpsc::SyncSender<Result<Vec<MxArray>, MexDiagnostic>>,
    },
    GetVariable {
        workspace: String,
        name: String,
        response: mpsc::SyncSender<Result<Option<MxArray>, MexDiagnostic>>,
    },
    PutVariable {
        workspace: String,
        name: String,
        value: MxArray,
        response: mpsc::SyncSender<Result<(), MexDiagnostic>>,
    },
    GetObjectProperty {
        object: MxArray,
        index: usize,
        name: String,
        response: mpsc::SyncSender<Result<MxArray, MexDiagnostic>>,
    },
    SetObjectProperty {
        object: MxArray,
        index: usize,
        name: String,
        value: MxArray,
        response: mpsc::SyncSender<Result<MxArray, MexDiagnostic>>,
    },
    Async {
        operation: MexAsyncOperation,
        result: Arc<NativeAsyncResult>,
    },
}

enum NativeAsyncState {
    Pending,
    Running,
    Cancelled,
    Complete(MexEngineCompletion<Vec<MxArray>>),
}

pub(super) struct NativeAsyncResult {
    state: Mutex<NativeAsyncState>,
    ready: Condvar,
    cancellation: runmat_execution::cancellation::CancellationSource,
    _origin: Arc<OriginServiceLease>,
}

struct OriginServiceLease {
    callbacks: async_mpsc::UnboundedSender<BoundaryEnvelope>,
    service_id: u64,
    owner: NativeLaneOwner,
}

impl Drop for OriginServiceLease {
    fn drop(&mut self) {
        let _ = self
            .callbacks
            .unbounded_send(BoundaryEnvelope::ServiceFinished {
                service_id: self.service_id,
            });
    }
}

impl NativeAsyncResult {
    fn pending(origin: Arc<OriginServiceLease>) -> Self {
        Self {
            state: Mutex::new(NativeAsyncState::Pending),
            ready: Condvar::new(),
            cancellation: runmat_execution::cancellation::CancellationSource::new(),
            _origin: origin,
        }
    }

    fn begin(&self) -> bool {
        let mut state = self.state.lock().expect("MEX async result is not poisoned");
        if matches!(*state, NativeAsyncState::Pending) {
            *state = NativeAsyncState::Running;
            true
        } else {
            false
        }
    }

    fn complete(&self, result: MexEngineCompletion<Vec<MxArray>>) {
        let mut state = self.state.lock().expect("MEX async result is not poisoned");
        if !matches!(*state, NativeAsyncState::Cancelled) {
            *state = NativeAsyncState::Complete(result);
        }
        self.ready.notify_all();
    }

    fn cancellation(&self) -> runmat_execution::cancellation::CancellationToken {
        self.cancellation.token()
    }
}

impl Drop for NativeAsyncResult {
    fn drop(&mut self) {
        let _ = self
            ._origin
            .callbacks
            .unbounded_send(BoundaryEnvelope::AsyncFinished {
                service_id: self._origin.service_id,
            });
    }
}

impl MexAsyncResult for NativeAsyncResult {
    fn cancel(&self, allow_interrupt: bool) -> bool {
        let requested = {
            let mut state = self.state.lock().expect("MEX async result is not poisoned");
            match *state {
                NativeAsyncState::Pending => {
                    *state = NativeAsyncState::Cancelled;
                    true
                }
                NativeAsyncState::Running if allow_interrupt => true,
                NativeAsyncState::Running
                | NativeAsyncState::Cancelled
                | NativeAsyncState::Complete(_) => false,
            }
        };
        if requested {
            self.cancellation
                .cancel(runmat_execution::CancellationReason::User);
            self.ready.notify_all();
        }
        requested
    }

    fn is_ready(&self) -> bool {
        !matches!(
            *self.state.lock().expect("MEX async result is not poisoned"),
            NativeAsyncState::Pending | NativeAsyncState::Running
        )
    }

    fn wait(&self, timeout: Option<Duration>) -> bool {
        let mut state = self.state.lock().expect("MEX async result is not poisoned");
        match timeout {
            None => {
                while matches!(
                    *state,
                    NativeAsyncState::Pending | NativeAsyncState::Running
                ) {
                    state = self
                        .ready
                        .wait(state)
                        .expect("MEX async result is not poisoned");
                }
            }
            Some(timeout) => {
                let deadline = Instant::now() + timeout;
                while matches!(
                    *state,
                    NativeAsyncState::Pending | NativeAsyncState::Running
                ) {
                    let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                        return false;
                    };
                    let (next, wait) = self
                        .ready
                        .wait_timeout(state, remaining)
                        .expect("MEX async result is not poisoned");
                    state = next;
                    if wait.timed_out()
                        && matches!(
                            *state,
                            NativeAsyncState::Pending | NativeAsyncState::Running
                        )
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn result(&self) -> MexEngineCompletion<Vec<MxArray>> {
        self.wait(None);
        match &*self.state.lock().expect("MEX async result is not poisoned") {
            NativeAsyncState::Complete(result) => result.clone(),
            NativeAsyncState::Cancelled => MexEngineCompletion::from_result(Err(MexDiagnostic {
                identifier: Some("RunMat:MEX:Cancelled".into()),
                message: "asynchronous MEX engine operation was cancelled".into(),
            })),
            NativeAsyncState::Pending | NativeAsyncState::Running => unreachable!(),
        }
    }
}

impl NativeMexLane {
    /// Only runtime sessions register here. The direct lane API's default
    /// owner remains outside this lease set and retains its old Drop behavior.
    pub(super) fn register_session_owner(
        &self,
        owner: &NativeLaneOwner,
    ) -> Result<(), MexLoadError> {
        if self.abandoned.get() || !owner.is_live() {
            return Err(MexLoadError::NativeLane(
                "the MEX native lane has retired".into(),
            ));
        }
        let mut owners = self.session_owners.borrow_mut();
        if !owners.iter().any(|registered| registered.same(owner)) {
            owners.push(owner.clone());
        }
        Ok(())
    }

    pub(super) fn retire_session_owner(&self, owner: &NativeLaneOwner) {
        let mut owners = self.session_owners.borrow_mut();
        let Some(index) = owners.iter().position(|registered| registered.same(owner)) else {
            return;
        };
        owner.retire();
        owners.swap_remove(index);
        let last = owners.is_empty();
        drop(owners);
        if self
            .commands
            .send(NativeCommand::RetireOwner {
                owner: owner.clone(),
            })
            .is_err()
        {
            self.force_abandon();
            return;
        }
        if last {
            self.force_abandon();
        }
    }

    pub(super) fn has_pending_services_for(&self, owner: &NativeLaneOwner) -> bool {
        self.services
            .borrow()
            .values()
            .any(|entry| entry.owner.same(owner))
    }

    pub(super) fn in_callback_for(&self, owner: &NativeLaneOwner) -> bool {
        self.callback_tasks.active_for(owner)
    }

    /// Revoke only this session's callbacks. A testing child may retire while
    /// its parent MEX is still waiting for the test result on this same lane.
    pub(super) fn force_abandon_for(&self, owner: &NativeLaneOwner) {
        self.callback_tasks.cancel_for(owner);
        self.services
            .borrow_mut()
            .retain(|_, entry| !entry.owner.same(owner));
        // Queued envelopes for the retired owner are rejected by the ordinary
        // missing-service path; the shared channel and other owners stay live.
    }

    /// Explicit host forced-disposal boundary. It must run even if an origin
    /// service's RuntimeContext keeps the MEX session Rc alive, so Drop alone
    /// is not an adequate revocation point.
    pub(super) fn force_abandon(&self) {
        if self.abandoned.replace(true) {
            return;
        }
        self.default_owner.retire();
        for owner in std::mem::take(&mut *self.session_owners.borrow_mut()) {
            owner.retire();
        }
        self.callback_closer.close_channel();
        let pending_services = self.services.borrow().len();
        self.callback_tasks.cancel_all();
        let mut rejected_requests = 0usize;
        for envelope in self.callbacks.close_and_drain() {
            if let BoundaryEnvelope::Request { request, .. } = envelope {
                reject_request(*request, "the originating MEX host was disposed");
                rejected_requests += 1;
            }
        }
        // An already-dispatched callback owns a cloned service Rc until it
        // returns. Clearing the table drops all other origin registrations.
        let retired_services = std::mem::take(&mut *self.services.borrow_mut());
        drop(retired_services);
        if pending_services != 0 || rejected_requests != 0 {
            tracing::warn!(
                pending_services,
                rejected_requests,
                "abandoned pending native MEX work during forced host disposal"
            );
        }
        let _ = self.commands.send(NativeCommand::Shutdown);
    }

    pub(super) fn spawn() -> Result<Self, String> {
        let (commands, receiver) = mpsc::channel();
        let (callback_sender, callbacks) = async_mpsc::unbounded();
        let callback_closer = callback_sender.clone();
        std::thread::Builder::new()
            .name("runmat-mex-native".into())
            .spawn(move || lane_main(receiver, callback_sender))
            .map_err(|error| format!("could not start the MEX native lane: {error}"))?;
        Ok(Self {
            commands,
            callback_closer,
            callbacks: CallbackInbox::new(callbacks),
            services: RefCell::new(HashMap::new()),
            callback_tasks: CallbackTasks::new(),
            next_service: Cell::new(1),
            callback_depth: Cell::new(0),
            abandoned: Cell::new(false),
            default_owner: NativeLaneOwner::default(),
            session_owners: RefCell::new(Vec::new()),
        })
    }

    #[cfg(test)]
    pub(super) async fn load(
        &self,
        path: &Path,
    ) -> Result<NativeModuleMetadata, NativeModuleLoadError> {
        self.load_with_policy_for(&self.default_owner, path, false)
            .await
    }

    pub(super) async fn load_with_policy(
        &self,
        path: &Path,
        compatible_isolated: bool,
    ) -> Result<NativeModuleMetadata, NativeModuleLoadError> {
        self.load_with_policy_for(&self.default_owner, path, compatible_isolated)
            .await
    }

    pub(super) async fn load_for(
        &self,
        owner: &NativeLaneOwner,
        path: &Path,
    ) -> Result<NativeModuleMetadata, NativeModuleLoadError> {
        self.load_with_policy_for(owner, path, false).await
    }

    async fn load_with_policy_for(
        &self,
        owner: &NativeLaneOwner,
        path: &Path,
        compatible_isolated: bool,
    ) -> Result<NativeModuleMetadata, NativeModuleLoadError> {
        if self.abandoned.get() {
            return Err(NativeModuleLoadError::Failed {
                message: "the originating MEX host was disposed".into(),
                dependency: None,
            });
        }
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::Load {
                owner: owner.clone(),
                path: path.to_path_buf(),
                compatible_isolated,
                response,
            })
            .map_err(|_| {
                self.force_abandon();
                NativeModuleLoadError::Failed {
                    message: "the MEX native lane stopped".into(),
                    dependency: None,
                }
            })?;
        receiver.await.map_err(|_| {
            self.force_abandon();
            NativeModuleLoadError::Failed {
                message: "the MEX native lane stopped while loading a module".into(),
                dependency: None,
            }
        })?
    }

    pub(super) async fn invoke(
        &self,
        path: &Path,
        inputs: Vec<MxArray>,
        requested_outputs: usize,
        services: Rc<dyn OriginMexBoundaryServices>,
    ) -> Result<MexNativeInvocation, MexLoadError> {
        self.invoke_for(
            &self.default_owner,
            path,
            inputs,
            requested_outputs,
            services,
        )
        .await
    }

    pub(super) async fn invoke_for(
        &self,
        owner: &NativeLaneOwner,
        path: &Path,
        inputs: Vec<MxArray>,
        requested_outputs: usize,
        services: Rc<dyn OriginMexBoundaryServices>,
    ) -> Result<MexNativeInvocation, MexLoadError> {
        if self.abandoned.get() {
            return Err(MexLoadError::NativeLane(
                "the originating MEX host was disposed".into(),
            ));
        }
        let service_id = self.register_services_for(owner, services)?;
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::Invoke {
                owner: owner.clone(),
                path: path.to_path_buf(),
                inputs,
                requested_outputs,
                service_id,
                response,
            })
            .map_err(|_| {
                self.force_abandon();
                MexLoadError::NativeLane("the MEX native lane stopped".into())
            })?;

        self.wait_with_callbacks(receiver, "invocation").await
    }

    pub(super) async fn clear(
        &self,
        path: &Path,
        services: Rc<dyn OriginMexBoundaryServices>,
    ) -> Result<bool, MexLoadError> {
        self.clear_for(&self.default_owner, path, services).await
    }

    pub(super) async fn clear_for(
        &self,
        owner: &NativeLaneOwner,
        path: &Path,
        services: Rc<dyn OriginMexBoundaryServices>,
    ) -> Result<bool, MexLoadError> {
        if self.abandoned.get() {
            return Err(MexLoadError::NativeLane(
                "the originating MEX host was disposed".into(),
            ));
        }
        let service_id = self.register_services_for(owner, services)?;
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::Clear {
                owner: owner.clone(),
                path: path.to_path_buf(),
                service_id,
                response,
            })
            .map_err(|_| {
                self.force_abandon();
                MexLoadError::NativeLane("the MEX native lane stopped".into())
            })?;
        self.wait_with_callbacks(receiver, "clear").await
    }

    pub(super) async fn shutdown_module(
        &self,
        path: &Path,
        services: Rc<dyn OriginMexBoundaryServices>,
    ) -> Result<MexModuleRetirement, MexLoadError> {
        self.shutdown_module_for(&self.default_owner, path, services)
            .await
    }

    pub(super) async fn shutdown_module_for(
        &self,
        owner: &NativeLaneOwner,
        path: &Path,
        services: Rc<dyn OriginMexBoundaryServices>,
    ) -> Result<MexModuleRetirement, MexLoadError> {
        if self.abandoned.get() {
            return Err(MexLoadError::NativeLane(
                "the originating MEX host was disposed".into(),
            ));
        }
        let service_id = self.register_services_for(owner, services)?;
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::ShutdownModule {
                owner: owner.clone(),
                path: path.to_path_buf(),
                service_id,
                response,
            })
            .map_err(|_| {
                self.force_abandon();
                MexLoadError::NativeLane("the MEX native lane stopped".into())
            })?;
        self.wait_with_callbacks(receiver, "shutdown").await
    }

    #[cfg(test)]
    fn register_services(
        &self,
        services: Rc<dyn OriginMexBoundaryServices>,
    ) -> Result<u64, MexLoadError> {
        self.register_services_for(&self.default_owner, services)
    }

    fn register_services_for(
        &self,
        owner: &NativeLaneOwner,
        services: Rc<dyn OriginMexBoundaryServices>,
    ) -> Result<u64, MexLoadError> {
        let id = self.next_service.get();
        let next = id.checked_add(1).ok_or_else(|| {
            MexLoadError::NativeLane("MEX origin service identity exhausted".into())
        })?;
        self.next_service.set(next);
        self.services.borrow_mut().insert(
            id,
            OriginServiceEntry {
                owner: owner.clone(),
                services,
                invocation_active: true,
                native_lease_active: true,
                async_requests: 0,
                active_callbacks: 0,
            },
        );
        Ok(id)
    }

    #[cfg(test)]
    pub(super) fn hold_test_services(&self, services: Rc<dyn OriginMexBoundaryServices>) {
        self.register_services(services)
            .expect("register test origin service");
    }

    async fn wait_with_callbacks<T>(
        &self,
        receiver: oneshot::Receiver<Result<T, MexLoadError>>,
        operation: &str,
    ) -> Result<T, MexLoadError> {
        enum WaitEvent<T> {
            Response(Result<Result<T, MexLoadError>, oneshot::Canceled>),
            Callback(Option<BoundaryEnvelope>),
            Task(u64),
        }

        let mut result = receiver.fuse();
        loop {
            let event = {
                let callback = self.callbacks.next().fuse();
                futures::pin_mut!(callback);
                let task = self
                    .callback_tasks
                    .next_completed(&self.callback_depth)
                    .fuse();
                futures::pin_mut!(task);
                futures::select! {
                    response = result => WaitEvent::Response(response),
                    envelope = callback => WaitEvent::Callback(envelope),
                    service_id = task => WaitEvent::Task(service_id),
                }
            };

            match event {
                WaitEvent::Response(response) => {
                    let response = response.map_err(|_| {
                        self.force_abandon();
                        MexLoadError::NativeLane(format!(
                            "the MEX native lane stopped during {operation}"
                        ))
                    })?;
                    self.drain_ready_callbacks();
                    return response;
                }
                WaitEvent::Callback(envelope) => {
                    let Some(envelope) = envelope else {
                        return result.await.map_err(|_| {
                            self.force_abandon();
                            MexLoadError::NativeLane(format!(
                                "the MEX native lane stopped during {operation}"
                            ))
                        })?;
                    };
                    // Dispatch only after the callback future has left scope and
                    // released the receiver lock. Host services may synchronously
                    // re-enter this lane and must be able to receive callbacks.
                    self.service_envelope(envelope);
                }
                WaitEvent::Task(service_id) => self.finish_callback_task(service_id),
            }
        }
    }

    pub(super) async fn service_background_once(&self) -> bool {
        enum BackgroundEvent {
            Envelope(Option<BoundaryEnvelope>),
            Task(u64),
        }
        let event = {
            let callback = self.callbacks.next().fuse();
            futures::pin_mut!(callback);
            let task = self
                .callback_tasks
                .next_completed(&self.callback_depth)
                .fuse();
            futures::pin_mut!(task);
            futures::select! {
                envelope = callback => BackgroundEvent::Envelope(envelope),
                service_id = task => BackgroundEvent::Task(service_id),
            }
        };
        match event {
            BackgroundEvent::Envelope(Some(envelope)) => {
                self.service_envelope(envelope);
                true
            }
            BackgroundEvent::Envelope(None) => false,
            BackgroundEvent::Task(service_id) => {
                self.finish_callback_task(service_id);
                true
            }
        }
    }

    /// Finish every already-submitted native command and outstanding callback
    /// lease before the origin session can retire its token-table roots.
    /// This remains necessary when the foreground future was cancelled after
    /// command submission. Calling it from within a callback would wait on
    /// the callback currently on the stack, so that phase is rejected.
    pub(super) async fn settle_pending_for(
        &self,
        owner: &NativeLaneOwner,
    ) -> Result<(), MexLoadError> {
        if self.in_callback_for(owner) {
            return Err(MexLoadError::ReentrantModule);
        }
        loop {
            if !self.has_pending_services_for(owner) {
                break;
            }
            if !self.service_background_once().await {
                return Err(MexLoadError::NativeLane(
                    "the MEX callback channel closed before operation retirement".into(),
                ));
            }
        }
        Ok(())
    }

    /// Reach native command completion before module unload. A completed
    /// async-result handle may intentionally remain in the module's request
    /// table until mexAtExit or unload drops it, so it is not a pre-unload
    /// quiescence condition. `settle_pending_for` remains the post-unload gate.
    pub(super) async fn settle_commands_for(
        &self,
        owner: &NativeLaneOwner,
    ) -> Result<(), MexLoadError> {
        if self.in_callback_for(owner) {
            return Err(MexLoadError::ReentrantModule);
        }
        loop {
            let pending_command = self
                .services
                .borrow()
                .values()
                .any(|entry| entry.invocation_active && entry.owner.same(owner));
            if !pending_command {
                return Ok(());
            }
            if !self.service_background_once().await {
                return Err(MexLoadError::NativeLane(
                    "the MEX callback channel closed before command completion".into(),
                ));
            }
        }
    }

    fn drain_ready_callbacks(&self) {
        loop {
            let envelope = self.callbacks.pop_ready();
            match envelope {
                Some(envelope) => self.service_envelope(envelope),
                None => break,
            }
        }
    }

    fn service_envelope(&self, envelope: BoundaryEnvelope) {
        match envelope {
            BoundaryEnvelope::Request {
                service_id,
                request,
            } => {
                if self.abandoned.get() {
                    reject_request(*request, "the originating MEX host was disposed");
                    return;
                }
                let selected = self
                    .services
                    .borrow()
                    .get(&service_id)
                    .map(|entry| (entry.owner.clone(), Rc::clone(&entry.services)));
                if let Some((owner, services)) = selected {
                    if self
                        .callback_tasks
                        .admit(&owner, service_id, *request, services)
                    {
                        if let Some(entry) = self.services.borrow_mut().get_mut(&service_id) {
                            entry.active_callbacks += 1;
                        }
                    }
                } else {
                    reject_request(*request, "the originating runtime service has ended");
                }
            }
            BoundaryEnvelope::AsyncStarted { service_id } => {
                if let Some(entry) = self.services.borrow_mut().get_mut(&service_id) {
                    entry.async_requests += 1;
                }
            }
            BoundaryEnvelope::AsyncFinished { service_id } => {
                if let Some(entry) = self.services.borrow_mut().get_mut(&service_id) {
                    entry.async_requests = entry.async_requests.saturating_sub(1);
                }
                self.remove_completed_service(service_id);
            }
            BoundaryEnvelope::ServiceFinished { service_id } => {
                if let Some(entry) = self.services.borrow_mut().get_mut(&service_id) {
                    entry.native_lease_active = false;
                }
                self.remove_completed_service(service_id);
            }
            BoundaryEnvelope::InvocationFinished { service_id } => {
                if let Some(entry) = self.services.borrow_mut().get_mut(&service_id) {
                    entry.invocation_active = false;
                }
                self.remove_completed_service(service_id);
            }
        }
    }

    fn remove_completed_service(&self, service_id: u64) {
        let remove = self
            .services
            .borrow()
            .get(&service_id)
            .is_some_and(|entry| {
                !entry.invocation_active
                    && !entry.native_lease_active
                    && entry.async_requests == 0
                    && entry.active_callbacks == 0
            });
        if remove {
            let retired = self.services.borrow_mut().remove(&service_id);
            drop(retired);
        }
    }

    fn finish_callback_task(&self, service_id: u64) {
        if let Some(entry) = self.services.borrow_mut().get_mut(&service_id) {
            entry.active_callbacks = entry.active_callbacks.saturating_sub(1);
        }
        self.remove_completed_service(service_id);
    }
}

impl Drop for NativeMexLane {
    fn drop(&mut self) {
        self.force_abandon();
    }
}

fn lane_main(
    receiver: mpsc::Receiver<NativeCommand>,
    callbacks: async_mpsc::UnboundedSender<BoundaryEnvelope>,
) {
    let lane = Rc::new(NativeLaneState {
        receiver: RefCell::new(receiver),
        modules: RefCell::new(HashMap::new()),
        callbacks,
        command_depth: Cell::new(0),
        retirement_pending: Cell::new(false),
        retirement_pruning: Cell::new(false),
        stop_requested: Cell::new(false),
    });
    CURRENT_NATIVE_LANE.with(|current| {
        assert!(current.borrow_mut().replace(Rc::clone(&lane)).is_none());
    });
    while process_next_command(&lane, true) {}
    CURRENT_NATIVE_LANE.with(|current| {
        current.borrow_mut().take();
    });
}

fn process_next_command(lane: &Rc<NativeLaneState>, wait: bool) -> bool {
    let command = if wait {
        lane.receiver.borrow_mut().recv().ok()
    } else {
        lane.receiver.borrow_mut().try_recv().ok()
    };
    let Some(command) = command else {
        return !wait && !lane.stop_requested.get();
    };
    process_command(lane, command)
}

fn process_command(lane: &Rc<NativeLaneState>, command: NativeCommand) -> bool {
    lane.command_depth.set(
        lane.command_depth
            .get()
            .checked_add(1)
            .expect("MEX native command nesting exhausted"),
    );
    let keep_running = process_command_inner(lane, command);
    let remaining = lane.command_depth.get() - 1;
    lane.command_depth.set(remaining);
    if remaining == 0 && lane.retirement_pending.get() && !lane.retirement_pruning.replace(true) {
        while lane.retirement_pending.replace(false) {
            prune_retired_modules(lane);
        }
        lane.retirement_pruning.set(false);
    }
    keep_running && !lane.stop_requested.get()
}

fn process_command_inner(lane: &Rc<NativeLaneState>, command: NativeCommand) -> bool {
    match command {
        NativeCommand::Load {
            owner,
            path,
            compatible_isolated,
            response,
        } => {
            let result = load_module(lane, &owner, &path, compatible_isolated).map(|module| {
                NativeModuleMetadata {
                    mode: module.api_mode(),
                    interface: module.boundary_interface(),
                }
            });
            let _ = response.send(result.map_err(|error| {
                if error.requires_isolated_host() {
                    NativeModuleLoadError::IsolatedHostRequired
                } else {
                    NativeModuleLoadError::Failed {
                        message: error.to_string(),
                        dependency: error.missing_dependency(),
                    }
                }
            }));
        }
        NativeCommand::Invoke {
            owner,
            path,
            inputs,
            requested_outputs,
            service_id,
            response,
        } => {
            let origin = Arc::new(OriginServiceLease {
                callbacks: lane.callbacks.clone(),
                service_id,
                owner: owner.clone(),
            });
            let result = load_module(lane, &owner, &path, false).and_then(|module| {
                module.invoke_native(
                    inputs,
                    requested_outputs,
                    module.api_mode(),
                    Arc::new(NativeBoundaryProxy {
                        origin: Arc::clone(&origin),
                    }),
                )
            });
            drop(origin);
            let _ = lane
                .callbacks
                .unbounded_send(BoundaryEnvelope::InvocationFinished { service_id });
            let _ = response.send(result);
        }
        NativeCommand::Clear {
            owner,
            path,
            service_id,
            response,
        } => {
            let origin = Arc::new(OriginServiceLease {
                callbacks: lane.callbacks.clone(),
                service_id,
                owner: owner.clone(),
            });
            let module = owned_module(lane, &owner, &path);
            let result = match module {
                Ok(Some(module)) => {
                    let cleared =
                        module.clear_with_boundary_services(Arc::new(NativeBoundaryProxy {
                            origin: Arc::clone(&origin),
                        }));
                    if cleared.as_ref().is_ok_and(|cleared| *cleared) {
                        lane.modules.borrow_mut().remove(&path);
                    }
                    cleared
                }
                Ok(None) => Ok(true),
                Err(error) => Err(error),
            };
            drop(origin);
            let _ = lane
                .callbacks
                .unbounded_send(BoundaryEnvelope::InvocationFinished { service_id });
            let _ = response.send(result);
        }
        NativeCommand::ShutdownModule {
            owner,
            path,
            service_id,
            response,
        } => {
            let origin = Arc::new(OriginServiceLease {
                callbacks: lane.callbacks.clone(),
                service_id,
                owner: owner.clone(),
            });
            let services = Arc::new(NativeBoundaryProxy {
                origin: Arc::clone(&origin),
            });
            let outcome = owned_module(lane, &owner, &path).map(|module| {
                module
                    .as_ref()
                    .map_or(MexModuleRetirement::Retired(Ok(())), |module| {
                        module.shutdown_with_boundary_services_retirement(services)
                    })
            });
            if matches!(&outcome, Ok(MexModuleRetirement::Retired(_))) {
                lane.modules.borrow_mut().remove(&path);
            }
            drop(origin);
            let _ = lane
                .callbacks
                .unbounded_send(BoundaryEnvelope::InvocationFinished { service_id });
            let _ = response.send(outcome);
        }
        NativeCommand::RetireOwner { owner } => {
            debug_assert!(!owner.is_live());
            lane.retirement_pending.set(true);
        }
        NativeCommand::Shutdown => {
            lane.stop_requested.set(true);
            return false;
        }
    }
    true
}

/// Forced disposal has no acknowledged mexAtExit. Drop only after the entire
/// native reentry stack unwinds, never while a retired module's code is on it.
/// Remove entries under the map borrow, then run module unloads outside it:
/// unload hooks may synchronously reenter the same physical lane.
fn prune_retired_modules(lane: &NativeLaneState) {
    let paths = lane
        .modules
        .borrow()
        .iter()
        .filter(|(_, entry)| !entry.owner.is_live())
        .map(|(path, _)| path.clone())
        .collect::<Vec<_>>();
    let retired = {
        let mut modules = lane.modules.borrow_mut();
        paths
            .into_iter()
            .filter_map(|path| modules.remove(&path))
            .collect::<Vec<_>>()
    };
    drop(retired);
}

fn load_module(
    lane: &NativeLaneState,
    owner: &NativeLaneOwner,
    path: &Path,
    compatible_isolated: bool,
) -> Result<Rc<MexModule>, MexLoadError> {
    if !owner.is_live() {
        return Err(MexLoadError::NativeLane(
            "the MEX session has retired".into(),
        ));
    }
    if let Some(module) = owned_module(lane, owner, path)? {
        return Ok(module);
    }
    let module = Rc::new(if compatible_isolated {
        MexModule::load_compatible_isolated(path)?
    } else {
        MexModule::load(path)?
    });
    if !owner.is_live() {
        return Err(MexLoadError::NativeLane(
            "the MEX session retired while loading a module".into(),
        ));
    }
    lane.modules.borrow_mut().insert(
        path.to_path_buf(),
        OwnedNativeModule {
            owner: owner.clone(),
            module: Rc::clone(&module),
        },
    );
    Ok(module)
}

fn owned_module(
    lane: &NativeLaneState,
    owner: &NativeLaneOwner,
    path: &Path,
) -> Result<Option<Rc<MexModule>>, MexLoadError> {
    if !owner.is_live() {
        return Err(MexLoadError::NativeLane(
            "the MEX session has retired".into(),
        ));
    }
    lane.modules.borrow().get(path).map_or(Ok(None), |entry| {
        if entry.owner.same(owner) {
            Ok(Some(Rc::clone(&entry.module)))
        } else {
            Err(MexLoadError::InProcessOwnership {
                path: path.display().to_string(),
            })
        }
    })
}

struct NativeBoundaryProxy {
    origin: Arc<OriginServiceLease>,
}

impl NativeBoundaryProxy {
    fn request<T>(
        &self,
        build: impl FnOnce(mpsc::SyncSender<Result<T, MexDiagnostic>>) -> BoundaryRequest,
    ) -> Result<T, MexDiagnostic> {
        if !self.origin.owner.is_live() {
            return Err(lane_diagnostic("the originating MEX session has retired"));
        }
        let (response, receiver) = mpsc::sync_channel(1);
        self.origin
            .callbacks
            .unbounded_send(BoundaryEnvelope::Request {
                service_id: self.origin.service_id,
                request: Box::new(build(response)),
            })
            .map_err(|_| lane_diagnostic("the originating runtime task stopped"))?;
        loop {
            match receiver.recv_timeout(Duration::from_millis(1)) {
                Ok(result) => return result,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(lane_diagnostic("the originating runtime task stopped"));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            CURRENT_NATIVE_LANE.with(|current| {
                if let Some(lane) = current.borrow().as_ref().cloned() {
                    let _ = process_next_command(&lane, false);
                }
            });
            if !self.origin.owner.is_live() {
                return Err(lane_diagnostic("the originating MEX session has retired"));
            }
        }
    }
}

impl MexBoundaryHostServices for NativeBoundaryProxy {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        self.request(|response| BoundaryRequest::Eval {
            command: command.into(),
            response,
        })
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
    ) -> Result<Vec<MxArray>, MexDiagnostic> {
        self.request(|response| BoundaryRequest::Call {
            function: function.into(),
            arguments,
            requested_outputs,
            response,
        })
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<MxArray>, MexDiagnostic> {
        self.request(|response| BoundaryRequest::GetVariable {
            workspace: workspace.into(),
            name: name.into(),
            response,
        })
    }

    fn put_variable(
        &self,
        workspace: &str,
        name: &str,
        value: MxArray,
    ) -> Result<(), MexDiagnostic> {
        self.request(|response| BoundaryRequest::PutVariable {
            workspace: workspace.into(),
            name: name.into(),
            value,
            response,
        })
    }

    fn get_object_property(
        &self,
        object: MxArray,
        index: usize,
        name: &str,
    ) -> Result<MxArray, MexDiagnostic> {
        self.request(|response| BoundaryRequest::GetObjectProperty {
            object,
            index,
            name: name.into(),
            response,
        })
    }

    fn set_object_property(
        &self,
        object: MxArray,
        index: usize,
        name: &str,
        value: MxArray,
    ) -> Result<MxArray, MexDiagnostic> {
        self.request(|response| BoundaryRequest::SetObjectProperty {
            object,
            index,
            name: name.into(),
            value,
            response,
        })
    }
}

impl MexAsyncHostServices for NativeBoundaryProxy {
    fn submit(&self, operation: MexAsyncOperation) -> Result<MexAsyncResultHandle, MexDiagnostic> {
        self.origin
            .callbacks
            .unbounded_send(BoundaryEnvelope::AsyncStarted {
                service_id: self.origin.service_id,
            })
            .map_err(|_| lane_diagnostic("the originating runtime task stopped"))?;
        let result = Arc::new(NativeAsyncResult::pending(Arc::clone(&self.origin)));
        self.origin
            .callbacks
            .unbounded_send(BoundaryEnvelope::Request {
                service_id: self.origin.service_id,
                request: Box::new(BoundaryRequest::Async {
                    operation,
                    result: Arc::clone(&result),
                }),
            })
            .map_err(|_| lane_diagnostic("the originating runtime task stopped"))?;
        Ok(result)
    }
}

fn reject_request(request: BoundaryRequest, message: &str) {
    let diagnostic = || lane_diagnostic(message);
    match request {
        BoundaryRequest::Eval { response, .. } | BoundaryRequest::PutVariable { response, .. } => {
            let _ = response.send(Err(diagnostic()));
        }
        BoundaryRequest::Call { response, .. } => {
            let _ = response.send(Err(diagnostic()));
        }
        BoundaryRequest::GetVariable { response, .. } => {
            let _ = response.send(Err(diagnostic()));
        }
        BoundaryRequest::GetObjectProperty { response, .. }
        | BoundaryRequest::SetObjectProperty { response, .. } => {
            let _ = response.send(Err(diagnostic()));
        }
        BoundaryRequest::Async { result, .. } => {
            if result.begin() {
                result.complete(MexEngineCompletion::from_result(Err(diagnostic())));
            }
        }
    }
}

fn lane_diagnostic(message: &str) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:NativeLane".into()),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::future::Future;

    impl<T: MexBoundaryHostServices + 'static> OriginMexBoundaryServices for T {
        fn prepare(
            self: Rc<Self>,
            operation: super::super::services::OriginMexOperation,
            cancellation: runmat_execution::cancellation::CancellationToken,
        ) -> Result<super::super::services::OriginMexCompletionFuture, MexDiagnostic> {
            let completion = match operation {
                super::super::services::OriginMexOperation::Standard(operation) => {
                    self.execute_async(*operation, cancellation)
                }
                super::super::services::OriginMexOperation::OptionalGetVariable {
                    workspace,
                    name,
                } => MexEngineCompletion::from_result(
                    self.get_variable(&workspace, &name)
                        .map(|value| value.into_iter().collect()),
                ),
            };
            Ok(Box::pin(async move { completion }))
        }
    }

    fn pending_async_result() -> Arc<NativeAsyncResult> {
        let (callbacks, _receiver) = async_mpsc::unbounded();
        Arc::new(NativeAsyncResult::pending(Arc::new(OriginServiceLease {
            callbacks,
            service_id: 1,
            owner: NativeLaneOwner::default(),
        })))
    }

    struct PendingOriginServices {
        release: RefCell<Option<oneshot::Receiver<()>>>,
    }

    impl OriginMexBoundaryServices for PendingOriginServices {
        fn prepare(
            self: Rc<Self>,
            operation: super::super::services::OriginMexOperation,
            _: runmat_execution::cancellation::CancellationToken,
        ) -> Result<super::super::services::OriginMexCompletionFuture, MexDiagnostic> {
            let super::super::services::OriginMexOperation::Standard(operation) = operation else {
                panic!("test only admits Eval callbacks");
            };
            let MexAsyncOperation::Eval { command, .. } = *operation else {
                panic!("test only admits Eval callbacks");
            };
            let pending = if command == "pending" {
                self.release.borrow_mut().take()
            } else {
                None
            };
            Ok(Box::pin(async move {
                if let Some(pending) = pending {
                    pending.await.expect("release first origin callback");
                }
                MexEngineCompletion::from_result(Ok(Vec::new()))
            }))
        }
    }

    #[test]
    fn pending_origin_callback_does_not_starve_following_envelope() {
        let (commands, _worker_commands) = mpsc::channel();
        let (sender, callbacks) = async_mpsc::unbounded();
        let lane = NativeMexLane {
            commands,
            callback_closer: sender.clone(),
            callbacks: CallbackInbox::new(callbacks),
            services: RefCell::new(HashMap::new()),
            callback_tasks: CallbackTasks::new(),
            next_service: Cell::new(1),
            callback_depth: Cell::new(0),
            abandoned: Cell::new(false),
            default_owner: NativeLaneOwner::default(),
            session_owners: RefCell::new(Vec::new()),
        };
        let (release, released) = oneshot::channel();
        let service_id = lane
            .register_services(Rc::new(PendingOriginServices {
                release: RefCell::new(Some(released)),
            }))
            .expect("register pending origin service");
        let (first_reply, first_result) = mpsc::sync_channel(1);
        sender
            .unbounded_send(BoundaryEnvelope::Request {
                service_id,
                request: Box::new(BoundaryRequest::Eval {
                    command: "pending".into(),
                    response: first_reply,
                }),
            })
            .expect("queue pending callback");
        assert!(futures::executor::block_on(lane.service_background_once()));

        let (second_reply, second_result) = mpsc::sync_channel(1);
        sender
            .unbounded_send(BoundaryEnvelope::Request {
                service_id,
                request: Box::new(BoundaryRequest::Eval {
                    command: "ready".into(),
                    response: second_reply,
                }),
            })
            .expect("queue sibling callback");
        assert!(futures::executor::block_on(lane.service_background_once()));
        assert!(futures::executor::block_on(lane.service_background_once()));
        assert!(second_result.recv().expect("sibling settled").is_ok());
        assert!(matches!(
            first_result.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));

        release.send(()).expect("release pending origin callback");
        assert!(futures::executor::block_on(lane.service_background_once()));
        assert!(first_result.recv().expect("first settled").is_ok());
    }

    #[test]
    fn retiring_child_owner_does_not_wait_on_or_revoke_parent_service() {
        let (commands, worker_commands) = mpsc::channel();
        let (sender, callbacks) = async_mpsc::unbounded();
        let lane = NativeMexLane {
            commands,
            callback_closer: sender,
            callbacks: CallbackInbox::new(callbacks),
            services: RefCell::new(HashMap::new()),
            callback_tasks: CallbackTasks::new(),
            next_service: Cell::new(1),
            callback_depth: Cell::new(0),
            abandoned: Cell::new(false),
            default_owner: NativeLaneOwner::default(),
            session_owners: RefCell::new(Vec::new()),
        };
        let parent = NativeLaneOwner::default();
        let child = NativeLaneOwner::default();
        lane.register_session_owner(&parent)
            .expect("parent owns lane");
        lane.register_session_owner(&child)
            .expect("child owns lane");
        lane.register_session_owner(&child)
            .expect("owner registration is idempotent");
        lane.register_services_for(&parent, Rc::new(NoopServices))
            .expect("register parent callback");
        lane.register_services_for(&child, Rc::new(NoopServices))
            .expect("register child callback");
        lane.force_abandon_for(&child);
        lane.retire_session_owner(&child);
        lane.retire_session_owner(&child);
        assert!(lane.has_pending_services_for(&parent));
        assert!(!lane.has_pending_services_for(&child));
        assert!(!lane.abandoned.get());
        futures::executor::block_on(lane.settle_commands_for(&child))
            .expect("child may retire while parent command remains live");
        assert!(matches!(
            worker_commands.try_recv(),
            Ok(NativeCommand::RetireOwner { owner }) if owner.same(&child)
        ));
        assert!(matches!(
            worker_commands.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        lane.force_abandon_for(&parent);
        lane.retire_session_owner(&parent);
        assert!(lane.abandoned.get());
        assert!(matches!(
            worker_commands.try_recv(),
            Ok(NativeCommand::RetireOwner { owner }) if owner.same(&parent)
        ));
        assert!(matches!(
            worker_commands.try_recv(),
            Ok(NativeCommand::Shutdown)
        ));
    }

    #[test]
    fn nested_shutdown_remains_visible_after_its_immediate_result_is_ignored() {
        let (_commands, receiver) = mpsc::channel();
        let (callbacks, _callback_receiver) = async_mpsc::unbounded();
        let lane = Rc::new(NativeLaneState {
            receiver: RefCell::new(receiver),
            modules: RefCell::new(HashMap::new()),
            callbacks,
            command_depth: Cell::new(1),
            retirement_pending: Cell::new(false),
            retirement_pruning: Cell::new(false),
            stop_requested: Cell::new(false),
        });
        assert!(!process_command(&lane, NativeCommand::Shutdown));
        assert_eq!(lane.command_depth.get(), 1);
        assert!(lane.stop_requested.get());
        // NativeBoundaryProxy intentionally ignores the nested poll result;
        // the outer worker still observes the latched terminal state.
        assert!(!process_next_command(&lane, false));
    }

    #[test]
    fn forced_origin_retirement_rejects_late_native_callback() {
        let (callbacks, receiver) = async_mpsc::unbounded();
        let proxy = NativeBoundaryProxy {
            origin: Arc::new(OriginServiceLease {
                callbacks,
                service_id: 1,
                owner: NativeLaneOwner::default(),
            }),
        };
        drop(receiver);

        let error = proxy
            .eval("late_callback")
            .expect_err("a retired origin must not dispatch to managed callback services");
        assert_eq!(error.identifier.as_deref(), Some("RunMat:MEX:NativeLane"));
    }

    #[test]
    fn forced_lane_drop_rejects_queued_callback_before_service_retirement() {
        let (commands, worker_commands) = mpsc::channel();
        let (sender, callbacks) = async_mpsc::unbounded();
        let lane = NativeMexLane {
            commands,
            callback_closer: sender.clone(),
            callbacks: CallbackInbox::new(callbacks),
            services: RefCell::new(HashMap::new()),
            callback_tasks: CallbackTasks::new(),
            next_service: Cell::new(1),
            callback_depth: Cell::new(0),
            abandoned: Cell::new(false),
            default_owner: NativeLaneOwner::default(),
            session_owners: RefCell::new(Vec::new()),
        };
        let service_id = lane
            .register_services(Rc::new(NoopServices))
            .expect("register origin service");
        let (response, result) = mpsc::sync_channel(1);
        sender
            .unbounded_send(BoundaryEnvelope::Request {
                service_id,
                request: Box::new(BoundaryRequest::Eval {
                    command: "queued_after_host_disposal".into(),
                    response,
                }),
            })
            .unwrap();

        drop(lane);

        let diagnostic = result
            .recv()
            .expect("queued request must settle on forced disposal")
            .expect_err("queued request must not enter a retired origin service");
        assert_eq!(
            diagnostic.identifier.as_deref(),
            Some("RunMat:MEX:NativeLane")
        );
        assert!(sender
            .unbounded_send(BoundaryEnvelope::InvocationFinished { service_id })
            .is_err());
        assert!(matches!(
            worker_commands.try_recv(),
            Ok(NativeCommand::Shutdown)
        ));
    }

    #[test]
    fn forced_abandon_rejects_queued_request_while_receive_ticket_is_pending() {
        let (commands, _worker_commands) = mpsc::channel();
        let (sender, callbacks) = async_mpsc::unbounded();
        let lane = NativeMexLane {
            commands,
            callback_closer: sender.clone(),
            callbacks: CallbackInbox::new(callbacks),
            services: RefCell::new(HashMap::new()),
            callback_tasks: CallbackTasks::new(),
            next_service: Cell::new(1),
            callback_depth: Cell::new(0),
            abandoned: Cell::new(false),
            default_owner: NativeLaneOwner::default(),
            session_owners: RefCell::new(Vec::new()),
        };
        let mut pending = Box::pin(lane.callbacks.next());
        let waker = futures::task::noop_waker();
        assert!(matches!(
            pending
                .as_mut()
                .poll(&mut std::task::Context::from_waker(&waker)),
            std::task::Poll::Pending
        ));
        let (reply, result) = mpsc::sync_channel(1);
        sender
            .unbounded_send(BoundaryEnvelope::Request {
                service_id: 1,
                request: Box::new(BoundaryRequest::Eval {
                    command: "queued_while_receiver_waits".into(),
                    response: reply,
                }),
            })
            .expect("queue request after receiver went pending");
        lane.force_abandon();
        let error = result
            .try_recv()
            .expect("forced disposal settles queued request")
            .expect_err("disposed origin must reject callback");
        assert_eq!(error.identifier.as_deref(), Some("RunMat:MEX:NativeLane"));
        drop(pending);
    }

    #[test]
    fn pending_async_cancellation_completes_without_starting_work() {
        let result = pending_async_result();

        assert!(result.cancel(false));
        assert!(result.is_ready());
        assert!(result.cancellation().is_cancelled());
        let error = result
            .result()
            .result
            .expect_err("cancelled work must not produce outputs");
        assert_eq!(error.identifier.as_deref(), Some("RunMat:MEX:Cancelled"));
        assert!(!result.begin());
    }

    #[test]
    fn running_async_cancellation_only_interrupts_when_allowed() {
        let result = pending_async_result();
        assert!(result.begin());

        assert!(!result.cancel(false));
        assert!(!result.cancellation().is_cancelled());
        assert!(!result.is_ready());

        assert!(result.cancel(true));
        assert!(result.cancellation().is_cancelled());
        assert!(!result.is_ready());
        result.complete(MexEngineCompletion::from_result(Err(lane_diagnostic(
            "operation observed cancellation",
        ))));
        assert!(result.is_ready());
    }

    struct EchoServices {
        origin: std::thread::ThreadId,
        calls: Cell<usize>,
        observed_allocation: Cell<usize>,
    }

    struct StreamServices;

    struct CancellationServices;

    struct NoopServices;

    impl MexBoundaryHostServices for NoopServices {
        fn eval(&self, _: &str) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected nested eval callback"))
        }

        fn call(&self, _: &str, _: Vec<MxArray>, _: usize) -> Result<Vec<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected nested function callback"))
        }

        fn get_variable(&self, _: &str, _: &str) -> Result<Option<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected nested workspace read"))
        }

        fn put_variable(&self, _: &str, _: &str, _: MxArray) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected nested workspace write"))
        }

        fn get_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected nested property read"))
        }

        fn set_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
            _: MxArray,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected nested property write"))
        }
    }

    impl MexBoundaryHostServices for StreamServices {
        fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
            assert_eq!(command, "stream_eval");
            Ok(())
        }

        fn eval_captured(&self, command: &str) -> MexEngineCompletion<()> {
            assert_eq!(command, "stream_eval");
            MexEngineCompletion {
                result: Ok(()),
                stdout: "evaluation output\n".into(),
                stderr: "evaluation error\n".into(),
            }
        }

        fn call(
            &self,
            function: &str,
            arguments: Vec<MxArray>,
            requested_outputs: usize,
        ) -> Result<Vec<MxArray>, MexDiagnostic> {
            if function == "stream_fail" {
                return Err(MexDiagnostic {
                    identifier: Some("Fixture:StreamFailure".into()),
                    message: "captured callback failed".into(),
                });
            }
            assert_eq!(function, "stream_echo");
            assert_eq!(requested_outputs, 1);
            Ok(arguments)
        }

        fn call_captured(
            &self,
            function: &str,
            arguments: Vec<MxArray>,
            requested_outputs: usize,
        ) -> MexEngineCompletion<Vec<MxArray>> {
            MexEngineCompletion {
                result: self.call(function, arguments, requested_outputs),
                stdout: "function output: π\n".into(),
                stderr: "function error: λ\n".into(),
            }
        }

        fn get_variable(&self, _: &str, _: &str) -> Result<Option<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected stream workspace read"))
        }

        fn put_variable(&self, _: &str, _: &str, _: MxArray) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected stream workspace write"))
        }

        fn get_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected stream property read"))
        }

        fn set_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
            _: MxArray,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected stream property write"))
        }
    }

    impl MexBoundaryHostServices for CancellationServices {
        fn execute_async(
            &self,
            operation: MexAsyncOperation,
            cancellation: runmat_execution::cancellation::CancellationToken,
        ) -> MexEngineCompletion<Vec<MxArray>> {
            let MexAsyncOperation::Call { function, .. } = operation else {
                return MexEngineCompletion::from_result(Err(lane_diagnostic(
                    "unexpected cancellation operation",
                )));
            };
            assert_eq!(function, "wait_for_cancel");
            let deadline = Instant::now() + Duration::from_secs(2);
            while !cancellation.is_cancelled() && Instant::now() < deadline {
                std::thread::yield_now();
            }
            let diagnostic = if cancellation.is_cancelled() {
                MexDiagnostic {
                    identifier: Some("RunMat:MEX:Cancelled".into()),
                    message: "asynchronous callback observed cancellation".into(),
                }
            } else {
                lane_diagnostic("asynchronous cancellation was not delivered")
            };
            MexEngineCompletion::from_result(Err(diagnostic))
        }

        fn eval(&self, _: &str) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected cancellation eval"))
        }

        fn call(&self, _: &str, _: Vec<MxArray>, _: usize) -> Result<Vec<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic(
                "cancellation call bypassed async execution",
            ))
        }

        fn get_variable(&self, _: &str, _: &str) -> Result<Option<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected cancellation workspace read"))
        }

        fn put_variable(&self, _: &str, _: &str, _: MxArray) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected cancellation workspace write"))
        }

        fn get_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected cancellation property read"))
        }

        fn set_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
            _: MxArray,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected cancellation property write"))
        }
    }

    struct ReentrantServices {
        lane: Rc<NativeMexLane>,
        inner: PathBuf,
        outer: PathBuf,
        calls: Cell<usize>,
        rejected_shutdown: Cell<bool>,
    }

    #[derive(Default)]
    struct WorkspaceServices {
        value: RefCell<Option<MxArray>>,
        observed_allocation: Cell<usize>,
    }

    struct PropertyServices;

    impl MexBoundaryHostServices for PropertyServices {
        fn eval(&self, _: &str) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected property eval callback"))
        }

        fn call(&self, _: &str, _: Vec<MxArray>, _: usize) -> Result<Vec<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected property function callback"))
        }

        fn get_variable(&self, _: &str, _: &str) -> Result<Option<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected property workspace read"))
        }

        fn put_variable(&self, _: &str, _: &str, _: MxArray) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected property workspace write"))
        }

        fn get_object_property(
            &self,
            object: MxArray,
            index: usize,
            name: &str,
        ) -> Result<MxArray, MexDiagnostic> {
            let count = object.numel();
            let runmat_mex::mxarray::MxArrayData::Object {
                properties, values, ..
            } = object.data()
            else {
                return Err(lane_diagnostic("property input is not an object array"));
            };
            let field = properties
                .iter()
                .position(|property| property == name)
                .ok_or_else(|| lane_diagnostic("property does not exist"))?;
            values
                .get(field.saturating_mul(count).saturating_add(index))
                .and_then(Option::as_deref)
                .cloned()
                .ok_or_else(|| lane_diagnostic("property index is out of range"))
        }

        fn set_object_property(
            &self,
            mut object: MxArray,
            index: usize,
            name: &str,
            value: MxArray,
        ) -> Result<MxArray, MexDiagnostic> {
            let count = object.numel();
            let runmat_mex::mxarray::MxArrayData::Object {
                properties, values, ..
            } = object.data_mut()
            else {
                return Err(lane_diagnostic("property input is not an object array"));
            };
            let field = properties
                .iter()
                .position(|property| property == name)
                .ok_or_else(|| lane_diagnostic("property does not exist"))?;
            let slot = values
                .get_mut(field.saturating_mul(count).saturating_add(index))
                .ok_or_else(|| lane_diagnostic("property index is out of range"))?;
            *slot = Some(Box::new(value));
            Ok(object)
        }
    }

    impl MexBoundaryHostServices for WorkspaceServices {
        fn eval(&self, _: &str) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected workspace eval callback"))
        }

        fn call(&self, _: &str, _: Vec<MxArray>, _: usize) -> Result<Vec<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected workspace function callback"))
        }

        fn get_variable(&self, _: &str, name: &str) -> Result<Option<MxArray>, MexDiagnostic> {
            assert_eq!(name, "kept");
            Ok(self.value.borrow().clone())
        }

        fn put_variable(&self, _: &str, name: &str, value: MxArray) -> Result<(), MexDiagnostic> {
            assert_eq!(name, "kept");
            if let runmat_mex::mxarray::MxArrayData::Numeric(numeric) = value.data() {
                // SAFETY: `value` retains the allocation while stored by this
                // service; only its identity is observed.
                self.observed_allocation
                    .set(unsafe { numeric.real.foreign_data_pointer() } as usize);
            }
            self.value.replace(Some(value));
            Ok(())
        }

        fn get_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected workspace property read"))
        }

        fn set_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
            _: MxArray,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected workspace property write"))
        }
    }

    impl MexBoundaryHostServices for ReentrantServices {
        fn eval(&self, _: &str) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected reentrant eval callback"))
        }

        fn call(
            &self,
            function: &str,
            arguments: Vec<MxArray>,
            requested_outputs: usize,
        ) -> Result<Vec<MxArray>, MexDiagnostic> {
            if function != "invoke_inner" {
                return Err(lane_diagnostic("unexpected reentrant function callback"));
            }
            self.calls.set(self.calls.get() + 1);
            let retirement = pollster::block_on(
                self.lane
                    .shutdown_module(&self.outer, Rc::new(NoopServices)),
            )
            .map_err(|error| lane_diagnostic(&error.to_string()))?;
            if !matches!(
                retirement,
                MexModuleRetirement::Retained(MexLoadError::ReentrantModule)
            ) {
                return Err(lane_diagnostic(
                    "reentrant shutdown retired an active module",
                ));
            }
            self.rejected_shutdown.set(true);
            pollster::block_on(self.lane.invoke(
                &self.inner,
                arguments,
                requested_outputs,
                Rc::new(NoopServices),
            ))
            .map(|invocation| invocation.outputs)
            .map_err(|error| lane_diagnostic(&error.to_string()))
        }

        fn get_variable(&self, _: &str, _: &str) -> Result<Option<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected reentrant workspace read"))
        }

        fn put_variable(&self, _: &str, _: &str, _: MxArray) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected reentrant workspace write"))
        }

        fn get_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected reentrant property read"))
        }

        fn set_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
            _: MxArray,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected reentrant property write"))
        }
    }

    impl MexBoundaryHostServices for EchoServices {
        fn eval(&self, _: &str) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected eval callback"))
        }

        fn call(
            &self,
            function: &str,
            arguments: Vec<MxArray>,
            requested_outputs: usize,
        ) -> Result<Vec<MxArray>, MexDiagnostic> {
            assert_eq!(std::thread::current().id(), self.origin);
            assert_eq!(function, "lane_echo");
            assert_eq!(requested_outputs, 1);
            self.calls.set(self.calls.get() + 1);
            if let Some(runmat_mex::mxarray::MxArrayData::Numeric(numeric)) =
                arguments.first().map(MxArray::data)
            {
                // SAFETY: the boundary value owns a lease for the duration of
                // this callback; only allocation identity is observed.
                self.observed_allocation
                    .set(unsafe { numeric.real.foreign_data_pointer() } as usize);
            }
            Ok(arguments)
        }

        fn get_variable(&self, _: &str, _: &str) -> Result<Option<MxArray>, MexDiagnostic> {
            Err(lane_diagnostic("unexpected workspace read"))
        }

        fn put_variable(&self, _: &str, _: &str, _: MxArray) -> Result<(), MexDiagnostic> {
            Err(lane_diagnostic("unexpected workspace write"))
        }

        fn get_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected property read"))
        }

        fn set_object_property(
            &self,
            _: MxArray,
            _: usize,
            _: &str,
            _: MxArray,
        ) -> Result<MxArray, MexDiagnostic> {
            Err(lane_diagnostic("unexpected property write"))
        }
    }

    #[test]
    fn native_lane_runs_module_while_origin_services_callbacks() {
        let directory = tempfile::tempdir().expect("temporary C++ MEX directory");
        let source = directory.path().join("native_lane.cpp");
        std::fs::write(
            &source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        static_assert(std::is_same_v<matlab::engine::RunMatEngine,
                                     matlab::engine::MATLABEngine>);
        auto engine = getEngine();
        outputs[0] = engine->feval(u"lane_echo", inputs[0]);
    }
};
"#,
        )
        .expect("write native-lane fixture");
        let artifact = runmat_mex::MexBuild::new(&source, directory.path())
            .compile()
            .expect("compile native-lane fixture");
        let lane = NativeMexLane::spawn().expect("start native lane");
        let metadata = futures::executor::block_on(lane.load(&artifact.module))
            .expect("load native-lane fixture");
        let values = runmat_mex::MxValueContext::new();
        let input = values
            .encode(
                &runmat_value::Value::Tensor(
                    runmat_value::Tensor::new(vec![2.0, 4.0, 8.0], vec![3, 1])
                        .expect("valid input tensor"),
                ),
                metadata.mode,
                metadata.interface,
            )
            .expect("encode native input");
        let services = std::rc::Rc::new(EchoServices {
            origin: std::thread::current().id(),
            calls: Cell::new(0),
            observed_allocation: Cell::new(0),
        });
        let invocation = futures::executor::block_on(lane.invoke(
            &artifact.module,
            vec![input],
            1,
            services.clone(),
        ))
        .expect("invoke native-lane fixture");
        assert_eq!(services.calls.get(), 1);
        let output = values
            .decode(&invocation.outputs[0])
            .expect("decode native output");
        let runmat_value::Value::Tensor(output) = output else {
            panic!("native lane must preserve the tensor output");
        };
        assert_eq!(output.as_f64_slice(), Some(&[2.0, 4.0, 8.0][..]));
        assert_eq!(output.shape(), &[3, 1]);
    }

    #[test]
    fn native_lane_services_reentrant_calls_into_another_module() {
        let directory = tempfile::tempdir().expect("temporary reentrant MEX directory");
        let inner_source = directory.path().join("inner.cpp");
        std::fs::write(
            &inner_source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        outputs[0] = inputs[0];
    }
};
"#,
        )
        .expect("write inner reentrant fixture");
        let outer_source = directory.path().join("outer.cpp");
        std::fs::write(
            &outer_source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        outputs[0] = getEngine()->feval(u"invoke_inner", inputs[0]);
    }
};
"#,
        )
        .expect("write outer reentrant fixture");
        let inner = runmat_mex::MexBuild::new(&inner_source, directory.path())
            .compile()
            .expect("compile inner reentrant fixture");
        let outer = runmat_mex::MexBuild::new(&outer_source, directory.path())
            .compile()
            .expect("compile outer reentrant fixture");
        let lane = Rc::new(NativeMexLane::spawn().expect("start reentrant native lane"));
        let metadata = futures::executor::block_on(lane.load(&outer.module))
            .expect("load outer reentrant fixture");
        let foreign_owner = NativeLaneOwner::default();
        assert!(matches!(
            futures::executor::block_on(lane.load_for(&foreign_owner, &outer.module)),
            Err(NativeModuleLoadError::IsolatedHostRequired)
        ));
        let values = runmat_mex::MxValueContext::new();
        let input = values
            .encode(
                &runmat_value::Value::Tensor(
                    runmat_value::Tensor::new(vec![7.0, 11.0], vec![2, 1])
                        .expect("valid reentrant input"),
                ),
                metadata.mode,
                metadata.interface,
            )
            .expect("encode reentrant input");
        let services = Rc::new(ReentrantServices {
            lane: Rc::clone(&lane),
            inner: inner.module,
            outer: outer.module.clone(),
            calls: Cell::new(0),
            rejected_shutdown: Cell::new(false),
        });
        let invocation = futures::executor::block_on(lane.invoke(
            &outer.module,
            vec![input],
            1,
            services.clone(),
        ))
        .expect("invoke reentrant fixture");
        assert_eq!(services.calls.get(), 1);
        assert!(services.rejected_shutdown.get());
        assert_eq!(
            values
                .decode(&invocation.outputs[0])
                .expect("reentrant output"),
            runmat_value::Value::Tensor(
                runmat_value::Tensor::new(vec![7.0, 11.0], vec![2, 1])
                    .expect("expected reentrant tensor")
            )
        );
        let retirement =
            futures::executor::block_on(lane.shutdown_module(&outer.module, Rc::new(NoopServices)))
                .expect("shutdown after outer invocation");
        assert!(matches!(retirement, MexModuleRetirement::Retired(Ok(()))));

        // The original direct-lane module has retired. A child-owned reload
        // must leave no path-only cache entry after forced child retirement;
        // the still-live parent owner can then load the same physical module.
        let parent_owner = NativeLaneOwner::default();
        let child_owner = NativeLaneOwner::default();
        lane.register_session_owner(&parent_owner)
            .expect("parent session owns lane");
        lane.register_session_owner(&child_owner)
            .expect("child session owns lane");
        futures::executor::block_on(lane.load_for(&child_owner, &outer.module))
            .expect("child owns reloaded module");
        lane.force_abandon_for(&child_owner);
        lane.retire_session_owner(&child_owner);
        futures::executor::block_on(lane.load_for(&parent_owner, &outer.module))
            .expect("retired child module releases path ownership");
        let retirement = futures::executor::block_on(lane.shutdown_module_for(
            &parent_owner,
            &outer.module,
            Rc::new(NoopServices),
        ))
        .expect("parent module shutdown");
        assert!(matches!(retirement, MexModuleRetirement::Retired(Ok(()))));
        lane.retire_session_owner(&parent_owner);
    }

    #[test]
    fn native_lane_services_async_engine_future_without_copying_payload() {
        let directory = tempfile::tempdir().expect("temporary async MEX directory");
        let source = directory.path().join("native_lane_async.cpp");
        std::fs::write(
            &source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <chrono>

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        auto future = getEngine()->fevalAsync(u"lane_echo", inputs[0]);
        auto shared = future.share();
        if (shared.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
            throw matlab::Exception("asynchronous callback did not make progress");
        }
        outputs[0] = shared.get();
        outputs[1] = shared.get();
    }
};
"#,
        )
        .expect("write async native-lane fixture");
        let artifact = runmat_mex::MexBuild::new(&source, directory.path())
            .compile()
            .expect("compile async native-lane fixture");
        let lane = NativeMexLane::spawn().expect("start native lane");
        let metadata = futures::executor::block_on(lane.load(&artifact.module))
            .expect("load async native-lane fixture");
        let values = runmat_mex::MxValueContext::new();
        let input = values
            .encode(
                &runmat_value::Value::Tensor(
                    runmat_value::Tensor::new(vec![3.0, 6.0], vec![2, 1])
                        .expect("valid input tensor"),
                ),
                metadata.mode,
                metadata.interface,
            )
            .expect("encode async native input");
        let input_allocation = match input.data() {
            runmat_mex::mxarray::MxArrayData::Numeric(numeric) => {
                // SAFETY: `input` owns this allocation throughout the test;
                // only its identity is observed.
                (unsafe { numeric.real.foreign_data_pointer() }) as usize
            }
            _ => panic!("async native input must use numeric host storage"),
        };
        let services = std::rc::Rc::new(EchoServices {
            origin: std::thread::current().id(),
            calls: Cell::new(0),
            observed_allocation: Cell::new(0),
        });
        let invocation = futures::executor::block_on(lane.invoke(
            &artifact.module,
            vec![input],
            2,
            services.clone(),
        ))
        .expect("invoke async native-lane fixture");
        assert_eq!(services.calls.get(), 1);
        assert_eq!(services.observed_allocation.get(), input_allocation);
        let first = values.decode(&invocation.outputs[0]).expect("first output");
        let second = values
            .decode(&invocation.outputs[1])
            .expect("second output");
        assert_eq!(first, second);
    }

    #[test]
    fn native_lane_cpp_engine_routes_sync_and_async_streams_without_value_copies() {
        let directory = tempfile::tempdir().expect("temporary stream MEX directory");
        let source = directory.path().join("native_lane_streams.cpp");
        std::fs::write(
            &source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <sstream>

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        auto engine = getEngine();
        auto syncOut = std::make_shared<std::basic_stringbuf<char16_t>>();
        auto syncErr = std::make_shared<std::basic_stringbuf<char16_t>>();
        outputs[0] = engine->feval(u"stream_echo", inputs[0], syncOut, syncErr);
        if (syncOut->str() != u"function output: π\n" ||
            syncErr->str() != u"function error: λ\n") {
            throw matlab::Exception("synchronous redirected streams were not preserved");
        }

        auto asyncOut = std::make_shared<std::basic_stringbuf<char16_t>>();
        auto asyncErr = std::make_shared<std::basic_stringbuf<char16_t>>();
        auto evaluation = engine->evalAsync(u"stream_eval", asyncOut, asyncErr);
        evaluation.get();
        if (asyncOut->str() != u"evaluation output\n" ||
            asyncErr->str() != u"evaluation error\n") {
            throw matlab::Exception("asynchronous redirected streams were not preserved");
        }

        try {
            engine->fevalAsync(u"stream_fail", inputs[0]).get();
            throw matlab::Exception("failed callback unexpectedly returned");
        } catch (const matlab::engine::MATLABExecutionException &failure) {
            if (std::string(failure.what()) !=
                "Fixture:StreamFailure: captured callback failed") {
                throw matlab::Exception("asynchronous diagnostic lost its identifier or message");
            }
        }

        const double scalar = engine->feval<double>(u"stream_echo", 12.5);
        if (scalar != 12.5) {
            throw matlab::Exception("typed scalar engine conversion changed the value");
        }
        const std::vector<double> vector =
            engine->fevalAsync<std::vector<double>>(
                u"stream_echo", std::vector<double>{3.0, 6.0, 9.0})
                .get();
        if (vector != std::vector<double>{3.0, 6.0, 9.0}) {
            throw matlab::Exception("typed vector engine conversion changed the values");
        }
        const std::int64_t wide =
            engine->feval<std::int64_t>(u"stream_echo",
                                        std::int64_t{9007199254740993LL});
        if (wide != std::int64_t{9007199254740993LL}) {
            throw matlab::Exception("typed integer engine conversion lost precision");
        }
        const std::u16string text =
            engine->feval<std::u16string>(u"stream_echo", u"RunMat π");
        if (text != u"RunMat π") {
            throw matlab::Exception("typed text engine conversion changed the value");
        }
        const std::complex<double> complex =
            engine->feval<std::complex<double>>(
                u"stream_echo", std::complex<double>{2.0, -4.0});
        if (complex != std::complex<double>{2.0, -4.0}) {
            throw matlab::Exception("typed complex engine conversion changed the value");
        }
    }
};
"#,
        )
        .expect("write stream native-lane fixture");
        let artifact = runmat_mex::MexBuild::new(&source, directory.path())
            .compile()
            .expect("compile stream native-lane fixture");
        let lane = NativeMexLane::spawn().expect("start stream native lane");
        let metadata = futures::executor::block_on(lane.load(&artifact.module))
            .expect("load stream native-lane fixture");
        let values = runmat_mex::MxValueContext::new();
        let input = values
            .encode(
                &runmat_value::Value::Tensor(
                    runmat_value::Tensor::new(vec![5.0, 10.0], vec![2, 1])
                        .expect("valid stream input tensor"),
                ),
                metadata.mode,
                metadata.interface,
            )
            .expect("encode stream input");
        let input_allocation = match input.data() {
            runmat_mex::mxarray::MxArrayData::Numeric(numeric) => {
                // SAFETY: `input` retains the allocation while its identity is observed.
                (unsafe { numeric.real.foreign_data_pointer() }) as usize
            }
            _ => panic!("stream input must use numeric host storage"),
        };
        let invocation = futures::executor::block_on(lane.invoke(
            &artifact.module,
            vec![input],
            1,
            Rc::new(StreamServices),
        ))
        .expect("invoke stream native-lane fixture");
        let output_allocation = match invocation.outputs[0].data() {
            runmat_mex::mxarray::MxArrayData::Numeric(numeric) => {
                // SAFETY: the output retains the allocation while its identity is observed.
                (unsafe { numeric.real.foreign_data_pointer() }) as usize
            }
            _ => panic!("stream output must use numeric host storage"),
        };
        assert_eq!(output_allocation, input_allocation);
    }

    #[test]
    fn native_lane_cpp_engine_maps_cooperative_cancellation_to_cancel_exception() {
        let directory = tempfile::tempdir().expect("temporary cancellation MEX directory");
        let source = directory.path().join("native_lane_cancellation.cpp");
        std::fs::write(
            &source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        auto future = getEngine()->fevalAsync(u"wait_for_cancel", inputs[0]);
        if (!future.cancel(true)) {
            throw matlab::Exception("asynchronous cancellation was not accepted");
        }
        try {
            outputs[0] = future.get();
            throw matlab::Exception("cancelled future unexpectedly returned");
        } catch (const matlab::engine::CancelException &) {
            outputs[0] = inputs[0];
        }
    }
};
"#,
        )
        .expect("write cancellation native-lane fixture");
        let artifact = runmat_mex::MexBuild::new(&source, directory.path())
            .compile()
            .expect("compile cancellation native-lane fixture");
        let lane = NativeMexLane::spawn().expect("start cancellation native lane");
        let metadata = futures::executor::block_on(lane.load(&artifact.module))
            .expect("load cancellation native-lane fixture");
        let values = runmat_mex::MxValueContext::new();
        let input = values
            .encode(
                &runmat_value::Value::Tensor(
                    runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1])
                        .expect("valid cancellation input"),
                ),
                metadata.mode,
                metadata.interface,
            )
            .expect("encode cancellation input");
        let invocation = futures::executor::block_on(lane.invoke(
            &artifact.module,
            vec![input],
            1,
            Rc::new(CancellationServices),
        ))
        .expect("invoke cancellation native-lane fixture");
        assert_eq!(
            values
                .decode(&invocation.outputs[0])
                .expect("decode cancellation output"),
            runmat_value::Value::Tensor(
                runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1])
                    .expect("expected cancellation output")
            )
        );
    }

    #[test]
    fn native_lane_services_async_workspace_without_copying_payload() {
        let directory = tempfile::tempdir().expect("temporary async workspace MEX directory");
        let source = directory.path().join("native_lane_async_workspace.cpp");
        std::fs::write(
            &source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        auto engine = getEngine();
        engine->setVariableAsync(u"kept", inputs[0]).get();
        outputs[0] = engine->getVariableAsync(u"kept").get();
    }
};
"#,
        )
        .expect("write async workspace fixture");
        let artifact = runmat_mex::MexBuild::new(&source, directory.path())
            .compile()
            .expect("compile async workspace fixture");
        let lane = NativeMexLane::spawn().expect("start async workspace native lane");
        let metadata = futures::executor::block_on(lane.load(&artifact.module))
            .expect("load async workspace fixture");
        let values = runmat_mex::MxValueContext::new();
        let input = values
            .encode(
                &runmat_value::Value::Tensor(
                    runmat_value::Tensor::new(vec![13.0, 17.0], vec![2, 1])
                        .expect("valid async workspace input"),
                ),
                metadata.mode,
                metadata.interface,
            )
            .expect("encode async workspace input");
        let input_allocation = match input.data() {
            runmat_mex::mxarray::MxArrayData::Numeric(numeric) => {
                // SAFETY: only allocation identity is observed while the
                // boundary value owns its host-storage lease.
                (unsafe { numeric.real.foreign_data_pointer() }) as usize
            }
            _ => panic!("async workspace input must use numeric host storage"),
        };
        let services = Rc::new(WorkspaceServices::default());
        let invocation = futures::executor::block_on(lane.invoke(
            &artifact.module,
            vec![input],
            1,
            services.clone(),
        ))
        .expect("invoke async workspace fixture");
        assert_eq!(services.observed_allocation.get(), input_allocation);
        assert_eq!(
            values
                .decode(&invocation.outputs[0])
                .expect("workspace output"),
            runmat_value::Value::Tensor(
                runmat_value::Tensor::new(vec![13.0, 17.0], vec![2, 1])
                    .expect("expected async workspace tensor")
            )
        );
    }

    #[test]
    fn native_lane_services_async_indexed_properties_update_array_control() {
        let directory = tempfile::tempdir().expect("temporary async property MEX directory");
        let source = directory.path().join("native_lane_async_property.cpp");
        std::fs::write(
            &source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        matlab::data::Array object = inputs[0];
        auto engine = getEngine();
        outputs[0] = engine->getPropertyAsync(object, 1, u"Value").get();
        matlab::data::ArrayFactory factory;
        engine->setPropertyAsync(
            object, 1, u"Value", factory.createScalar<double>(9.0)).get();
        outputs[1] = object;
    }
};
"#,
        )
        .expect("write async property fixture");
        let artifact = runmat_mex::MexBuild::new(&source, directory.path())
            .compile()
            .expect("compile async property fixture");
        let lane = NativeMexLane::spawn().expect("start async property native lane");
        let metadata = futures::executor::block_on(lane.load(&artifact.module))
            .expect("load async property fixture");
        let mut first = runmat_value::ObjectInstance::new("FixtureValue");
        first.replace_property_unbounded("Value".into(), runmat_value::Value::Num(2.0));
        let mut second = runmat_value::ObjectInstance::new("FixtureValue");
        second.replace_property_unbounded("Value".into(), runmat_value::Value::Num(4.0));
        let input_value = runmat_value::Value::ObjectArray(
            runmat_value::ObjectArray::from_objects(
                "FixtureValue",
                vec![first, second],
                vec![1, 2],
            )
            .expect("valid property object array"),
        );
        let values = runmat_mex::MxValueContext::new();
        let input = values
            .encode(&input_value, metadata.mode, metadata.interface)
            .expect("encode async property input");
        let invocation = futures::executor::block_on(lane.invoke(
            &artifact.module,
            vec![input],
            2,
            Rc::new(PropertyServices),
        ))
        .expect("invoke async property fixture");
        assert_eq!(
            values
                .decode(&invocation.outputs[0])
                .expect("prior property"),
            runmat_value::Value::Num(4.0)
        );
        let updated = values
            .decode(&invocation.outputs[1])
            .expect("updated property object array");
        let runmat_value::Value::ObjectArray(updated) = updated else {
            panic!("property output must remain an object array");
        };
        let runmat_value::Value::Object(second) = &updated.data()[1] else {
            panic!("second property element must remain a value object");
        };
        assert_eq!(
            second.properties().get("Value"),
            Some(&runmat_value::Value::Num(9.0))
        );
    }

    #[test]
    fn native_lane_services_worker_submission_after_gateway_return() {
        let directory = tempfile::tempdir().expect("temporary delayed async MEX directory");
        let source = directory.path().join("native_lane_delayed_async.cpp");
        std::fs::write(
            &source,
            r#"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <chrono>
#include <future>
#include <thread>

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override {
        if (!worker_.valid()) {
            auto engine = getEngine();
            auto input = inputs[0];
            mexLock();
            worker_ = std::async(std::launch::async,
                [engine, input]() mutable {
                    std::this_thread::sleep_for(std::chrono::milliseconds(25));
                    return engine->fevalAsync(u"lane_echo", input).get();
                });
            return;
        }
        outputs[0] = worker_.get();
        mexUnlock();
    }

private:
    std::future<matlab::data::Array> worker_;
};
"#,
        )
        .expect("write delayed async native-lane fixture");
        let artifact = runmat_mex::MexBuild::new(&source, directory.path())
            .compile()
            .expect("compile delayed async native-lane fixture");
        let lane = NativeMexLane::spawn().expect("start native lane");
        let metadata = futures::executor::block_on(lane.load(&artifact.module))
            .expect("load delayed async native-lane fixture");
        let values = runmat_mex::MxValueContext::new();
        let input = values
            .encode(
                &runmat_value::Value::Tensor(
                    runmat_value::Tensor::new(vec![5.0, 10.0], vec![2, 1])
                        .expect("valid delayed async input"),
                ),
                metadata.mode,
                metadata.interface,
            )
            .expect("encode delayed async input");
        let input_allocation = match input.data() {
            runmat_mex::mxarray::MxArrayData::Numeric(numeric) => {
                // SAFETY: `input` owns this allocation until it is transferred
                // to the native lane; only its identity is observed.
                (unsafe { numeric.real.foreign_data_pointer() }) as usize
            }
            _ => panic!("delayed async input must use numeric host storage"),
        };
        let services = std::rc::Rc::new(EchoServices {
            origin: std::thread::current().id(),
            calls: Cell::new(0),
            observed_allocation: Cell::new(0),
        });

        let started = futures::executor::block_on(lane.invoke(
            &artifact.module,
            vec![input],
            0,
            services.clone(),
        ))
        .expect("start delayed async native-lane work");
        assert!(started.outputs.is_empty());
        assert_eq!(services.calls.get(), 0);

        futures::executor::block_on(async {
            while services.calls.get() == 0 {
                assert!(lane.service_background_once().await);
            }
        });
        assert_eq!(services.observed_allocation.get(), input_allocation);

        let completed = futures::executor::block_on(lane.invoke(
            &artifact.module,
            Vec::new(),
            1,
            services.clone(),
        ))
        .expect("collect delayed async native-lane result");
        let output = values
            .decode(&completed.outputs[0])
            .expect("delayed output");
        assert_eq!(
            output,
            runmat_value::Value::Tensor(
                runmat_value::Tensor::new(vec![5.0, 10.0], vec![2, 1])
                    .expect("expected delayed async tensor")
            )
        );
    }
}
