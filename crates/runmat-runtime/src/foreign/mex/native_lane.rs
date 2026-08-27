use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Condvar, Mutex,
};
use std::time::{Duration, Instant};

use futures::channel::{mpsc as async_mpsc, oneshot};
use futures::{FutureExt, StreamExt};
use runmat_mex::{
    MexAsyncHostServices, MexAsyncOperation, MexAsyncResult, MexBoundaryHostServices,
    MexDiagnostic, MexEngineCompletion, MexLoadError, MexModule, MexNativeInvocation, MxApiMode,
    MxArray, MxBoundaryInterface,
};

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
    callbacks: futures::lock::Mutex<async_mpsc::UnboundedReceiver<BoundaryEnvelope>>,
    services: RefCell<HashMap<u64, OriginServiceEntry>>,
    next_service: Cell<u64>,
}

struct NativeLaneState {
    receiver: RefCell<mpsc::Receiver<NativeCommand>>,
    modules: RefCell<HashMap<PathBuf, Rc<MexModule>>>,
    callbacks: async_mpsc::UnboundedSender<BoundaryEnvelope>,
}

thread_local! {
    static CURRENT_NATIVE_LANE: RefCell<Option<Rc<NativeLaneState>>> = const { RefCell::new(None) };
}

struct OriginServiceEntry {
    services: std::rc::Rc<dyn MexBoundaryHostServices>,
    invocation_active: bool,
    native_lease_active: bool,
    async_requests: usize,
}

enum NativeCommand {
    Load {
        path: PathBuf,
        compatible_isolated: bool,
        response: oneshot::Sender<Result<NativeModuleMetadata, NativeModuleLoadError>>,
    },
    Invoke {
        path: PathBuf,
        inputs: Vec<MxArray>,
        requested_outputs: usize,
        service_id: u64,
        response: oneshot::Sender<Result<MexNativeInvocation, String>>,
    },
    Clear {
        path: PathBuf,
        service_id: u64,
        response: oneshot::Sender<Result<bool, String>>,
    },
    ShutdownModule {
        path: PathBuf,
        service_id: u64,
        response: oneshot::Sender<Result<(), String>>,
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
    cancellation: Arc<AtomicBool>,
    _origin: Arc<OriginServiceLease>,
}

struct OriginServiceLease {
    callbacks: async_mpsc::UnboundedSender<BoundaryEnvelope>,
    service_id: u64,
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
            cancellation: Arc::new(AtomicBool::new(false)),
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

    fn cancellation(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancellation)
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
        let mut state = self.state.lock().expect("MEX async result is not poisoned");
        match *state {
            NativeAsyncState::Pending => {
                self.cancellation.store(true, Ordering::Release);
                *state = NativeAsyncState::Cancelled;
                self.ready.notify_all();
                true
            }
            NativeAsyncState::Running if allow_interrupt => {
                self.cancellation.store(true, Ordering::Release);
                true
            }
            NativeAsyncState::Running
            | NativeAsyncState::Cancelled
            | NativeAsyncState::Complete(_) => false,
        }
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
    pub(super) fn spawn() -> Result<Self, String> {
        let (commands, receiver) = mpsc::channel();
        let (callback_sender, callbacks) = async_mpsc::unbounded();
        std::thread::Builder::new()
            .name("runmat-mex-native".into())
            .spawn(move || lane_main(receiver, callback_sender))
            .map_err(|error| format!("could not start the MEX native lane: {error}"))?;
        Ok(Self {
            commands,
            callbacks: futures::lock::Mutex::new(callbacks),
            services: RefCell::new(HashMap::new()),
            next_service: Cell::new(1),
        })
    }

    pub(super) async fn load(
        &self,
        path: &Path,
    ) -> Result<NativeModuleMetadata, NativeModuleLoadError> {
        self.load_with_policy(path, false).await
    }

    pub(super) async fn load_with_policy(
        &self,
        path: &Path,
        compatible_isolated: bool,
    ) -> Result<NativeModuleMetadata, NativeModuleLoadError> {
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::Load {
                path: path.to_path_buf(),
                compatible_isolated,
                response,
            })
            .map_err(|_| NativeModuleLoadError::Failed {
                message: "the MEX native lane stopped".into(),
                dependency: None,
            })?;
        receiver.await.map_err(|_| NativeModuleLoadError::Failed {
            message: "the MEX native lane stopped while loading a module".into(),
            dependency: None,
        })?
    }

    pub(super) async fn invoke(
        &self,
        path: &Path,
        inputs: Vec<MxArray>,
        requested_outputs: usize,
        services: std::rc::Rc<dyn MexBoundaryHostServices>,
    ) -> Result<MexNativeInvocation, String> {
        let service_id = self.register_services(services);
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::Invoke {
                path: path.to_path_buf(),
                inputs,
                requested_outputs,
                service_id,
                response,
            })
            .map_err(|_| "the MEX native lane stopped".to_string())?;

        self.wait_with_callbacks(receiver, "invocation").await
    }

    pub(super) async fn clear(
        &self,
        path: &Path,
        services: std::rc::Rc<dyn MexBoundaryHostServices>,
    ) -> Result<bool, String> {
        let service_id = self.register_services(services);
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::Clear {
                path: path.to_path_buf(),
                service_id,
                response,
            })
            .map_err(|_| "the MEX native lane stopped".to_string())?;
        self.wait_with_callbacks(receiver, "clear").await
    }

    pub(super) async fn shutdown_module(
        &self,
        path: &Path,
        services: std::rc::Rc<dyn MexBoundaryHostServices>,
    ) -> Result<(), String> {
        let service_id = self.register_services(services);
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(NativeCommand::ShutdownModule {
                path: path.to_path_buf(),
                service_id,
                response,
            })
            .map_err(|_| "the MEX native lane stopped".to_string())?;
        self.wait_with_callbacks(receiver, "shutdown").await
    }

    fn register_services(&self, services: std::rc::Rc<dyn MexBoundaryHostServices>) -> u64 {
        let id = self.next_service.get().max(1);
        self.next_service.set(id.wrapping_add(1).max(1));
        self.services.borrow_mut().insert(
            id,
            OriginServiceEntry {
                services,
                invocation_active: true,
                native_lease_active: true,
                async_requests: 0,
            },
        );
        id
    }

    async fn wait_with_callbacks<T>(
        &self,
        receiver: oneshot::Receiver<Result<T, String>>,
        operation: &str,
    ) -> Result<T, String> {
        let mut result = receiver.fuse();
        loop {
            let callback = async {
                let mut callbacks = self.callbacks.lock().await;
                callbacks.next().await
            }
            .fuse();
            futures::pin_mut!(callback);
            futures::select! {
                response = result => {
                    let response = response.map_err(|_| {
                        format!("the MEX native lane stopped during {operation}")
                    })?;
                    self.drain_ready_callbacks();
                    return response;
                }
                envelope = callback => {
                    let Some(envelope) = envelope else {
                        return result.await.map_err(|_| {
                            format!("the MEX native lane stopped during {operation}")
                        })?;
                    };
                    self.service_envelope(envelope);
                }
            }
        }
    }

    pub(super) async fn service_background_once(&self) -> bool {
        let envelope = {
            let mut callbacks = self.callbacks.lock().await;
            callbacks.next().await
        };
        if let Some(envelope) = envelope {
            self.service_envelope(envelope);
            true
        } else {
            false
        }
    }

    fn drain_ready_callbacks(&self) {
        loop {
            let Some(mut callbacks) = self.callbacks.try_lock() else {
                break;
            };
            let envelope = callbacks.next().now_or_never().flatten();
            drop(callbacks);
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
                let services = self
                    .services
                    .borrow()
                    .get(&service_id)
                    .map(|entry| std::rc::Rc::clone(&entry.services));
                if let Some(services) = services {
                    service_request(*request, services.as_ref());
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
                !entry.invocation_active && !entry.native_lease_active && entry.async_requests == 0
            });
        if remove {
            self.services.borrow_mut().remove(&service_id);
        }
    }
}

impl Drop for NativeMexLane {
    fn drop(&mut self) {
        let _ = self.commands.send(NativeCommand::Shutdown);
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
        return !wait;
    };
    process_command(lane, command)
}

fn process_command(lane: &Rc<NativeLaneState>, command: NativeCommand) -> bool {
    match command {
        NativeCommand::Load {
            path,
            compatible_isolated,
            response,
        } => {
            let result =
                load_module(lane, &path, compatible_isolated).map(|module| NativeModuleMetadata {
                    mode: module.api_mode(),
                    interface: module.boundary_interface(),
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
            path,
            inputs,
            requested_outputs,
            service_id,
            response,
        } => {
            let origin = Arc::new(OriginServiceLease {
                callbacks: lane.callbacks.clone(),
                service_id,
            });
            let result = load_module(lane, &path, false).and_then(|module| {
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
            let _ = response.send(result.map_err(|error| error.to_string()));
        }
        NativeCommand::Clear {
            path,
            service_id,
            response,
        } => {
            let origin = Arc::new(OriginServiceLease {
                callbacks: lane.callbacks.clone(),
                service_id,
            });
            let module = lane.modules.borrow().get(&path).cloned();
            let result = if let Some(module) = module {
                let cleared = module.clear_with_boundary_services(Arc::new(NativeBoundaryProxy {
                    origin: Arc::clone(&origin),
                }));
                if cleared.as_ref().is_ok_and(|cleared| *cleared) {
                    lane.modules.borrow_mut().remove(&path);
                }
                cleared
            } else {
                Ok(true)
            };
            drop(origin);
            let _ = lane
                .callbacks
                .unbounded_send(BoundaryEnvelope::InvocationFinished { service_id });
            let _ = response.send(result.map_err(|error| error.to_string()));
        }
        NativeCommand::ShutdownModule {
            path,
            service_id,
            response,
        } => {
            let origin = Arc::new(OriginServiceLease {
                callbacks: lane.callbacks.clone(),
                service_id,
            });
            let services = Arc::new(NativeBoundaryProxy {
                origin: Arc::clone(&origin),
            });
            let module = lane.modules.borrow().get(&path).cloned();
            let result = module.as_ref().map_or(Ok(()), |module| {
                module.shutdown_with_boundary_services(services)
            });
            lane.modules.borrow_mut().remove(&path);
            drop(origin);
            let _ = lane
                .callbacks
                .unbounded_send(BoundaryEnvelope::InvocationFinished { service_id });
            let _ = response.send(result.map_err(|error| error.to_string()));
        }
        NativeCommand::Shutdown => return false,
    }
    true
}

fn load_module(
    lane: &NativeLaneState,
    path: &Path,
    compatible_isolated: bool,
) -> Result<Rc<MexModule>, MexLoadError> {
    if let Some(module) = lane.modules.borrow().get(path) {
        return Ok(Rc::clone(module));
    }
    let module = Rc::new(if compatible_isolated {
        MexModule::load_compatible_isolated(path)?
    } else {
        MexModule::load(path)?
    });
    lane.modules
        .borrow_mut()
        .insert(path.to_path_buf(), Rc::clone(&module));
    Ok(module)
}

struct NativeBoundaryProxy {
    origin: Arc<OriginServiceLease>,
}

impl NativeBoundaryProxy {
    fn request<T>(
        &self,
        build: impl FnOnce(mpsc::SyncSender<Result<T, MexDiagnostic>>) -> BoundaryRequest,
    ) -> Result<T, MexDiagnostic> {
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
    fn submit(
        &self,
        operation: MexAsyncOperation,
    ) -> Result<Arc<dyn MexAsyncResult>, MexDiagnostic> {
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

fn service_request(request: BoundaryRequest, services: &dyn MexBoundaryHostServices) {
    match request {
        BoundaryRequest::Eval { command, response } => {
            let _ = response.send(services.eval(&command));
        }
        BoundaryRequest::Call {
            function,
            arguments,
            requested_outputs,
            response,
        } => {
            let _ = response.send(services.call(&function, arguments, requested_outputs));
        }
        BoundaryRequest::GetVariable {
            workspace,
            name,
            response,
        } => {
            let _ = response.send(services.get_variable(&workspace, &name));
        }
        BoundaryRequest::PutVariable {
            workspace,
            name,
            value,
            response,
        } => {
            let _ = response.send(services.put_variable(&workspace, &name, value));
        }
        BoundaryRequest::GetObjectProperty {
            object,
            index,
            name,
            response,
        } => {
            let _ = response.send(services.get_object_property(object, index, &name));
        }
        BoundaryRequest::SetObjectProperty {
            object,
            index,
            name,
            value,
            response,
        } => {
            let _ = response.send(services.set_object_property(object, index, &name, value));
        }
        BoundaryRequest::Async { operation, result } => {
            if !result.begin() {
                return;
            }
            let completed = services.execute_async(operation, result.cancellation());
            result.complete(completed);
        }
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

    fn pending_async_result() -> Arc<NativeAsyncResult> {
        let (callbacks, _receiver) = async_mpsc::unbounded();
        Arc::new(NativeAsyncResult::pending(Arc::new(OriginServiceLease {
            callbacks,
            service_id: 1,
        })))
    }

    #[test]
    fn pending_async_cancellation_completes_without_starting_work() {
        let result = pending_async_result();

        assert!(result.cancel(false));
        assert!(result.is_ready());
        assert!(result.cancellation().load(Ordering::Acquire));
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
        assert!(!result.cancellation().load(Ordering::Acquire));
        assert!(!result.is_ready());

        assert!(result.cancel(true));
        assert!(result.cancellation().load(Ordering::Acquire));
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
            cancellation: Arc<AtomicBool>,
        ) -> MexEngineCompletion<Vec<MxArray>> {
            let MexAsyncOperation::Call { function, .. } = operation else {
                return MexEngineCompletion::from_result(Err(lane_diagnostic(
                    "unexpected cancellation operation",
                )));
            };
            assert_eq!(function, "wait_for_cancel");
            let deadline = Instant::now() + Duration::from_secs(2);
            while !cancellation.load(Ordering::Acquire) && Instant::now() < deadline {
                std::thread::yield_now();
            }
            let diagnostic = if cancellation.load(Ordering::Acquire) {
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
        calls: Cell<usize>,
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
            pollster::block_on(self.lane.invoke(
                &self.inner,
                arguments,
                requested_outputs,
                Rc::new(NoopServices),
            ))
            .map(|invocation| invocation.outputs)
            .map_err(|message| lane_diagnostic(&message))
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
        assert_eq!(output.shape, vec![3, 1]);
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
            calls: Cell::new(0),
        });
        let invocation = futures::executor::block_on(lane.invoke(
            &outer.module,
            vec![input],
            1,
            services.clone(),
        ))
        .expect("invoke reentrant fixture");
        assert_eq!(services.calls.get(), 1);
        assert_eq!(
            values
                .decode(&invocation.outputs[0])
                .expect("reentrant output"),
            runmat_value::Value::Tensor(
                runmat_value::Tensor::new(vec![7.0, 11.0], vec![2, 1])
                    .expect("expected reentrant tensor")
            )
        );
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
        let mut first = runmat_value::ObjectInstance::new("FixtureValue".into());
        first
            .properties
            .insert("Value".into(), runmat_value::Value::Num(2.0));
        let mut second = runmat_value::ObjectInstance::new("FixtureValue".into());
        second
            .properties
            .insert("Value".into(), runmat_value::Value::Num(4.0));
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
            second.properties.get("Value"),
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
