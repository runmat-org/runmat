use runmat_process_host::private_bytes::PrivateByteLease;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::cpython::Interpreter;
use crate::session::import_owner::ProcessImportReservation;
use crate::session::protocol::{
    PythonCallbackCommand, PythonCallbackMailbox, PythonLaneCompletion, PythonLaneEvent,
    PythonLaneEventSink, PythonLaneReply,
};
use crate::{
    InstalledPythonArtifacts, PythonCallbackInvocation, PythonError, PythonInstallation,
    PythonObjectMetadata, PythonValue,
};

static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct ActiveReentry {
    session: u64,
    mailbox: PythonCallbackMailbox,
}

thread_local! {
    static ACTIVE_REENTRY: RefCell<Option<ActiveReentry>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone)]
pub struct PythonSessionConfig {
    pub installation: PythonInstallation,
    pub module_paths: Vec<std::path::PathBuf>,
}

#[derive(Debug, Clone)]
pub enum PythonCall {
    InvokeQualified {
        name: String,
        arguments: Vec<PythonValue>,
    },
    GetMember {
        receiver: crate::PythonObjectHandle,
        name: String,
    },
    SetMember {
        receiver: crate::PythonObjectHandle,
        name: String,
        value: PythonValue,
    },
    InvokeMember {
        receiver: crate::PythonObjectHandle,
        name: String,
        arguments: Vec<PythonValue>,
    },
    GetItem {
        receiver: crate::PythonObjectHandle,
        index: PythonValue,
    },
    SetItem {
        receiver: crate::PythonObjectHandle,
        index: PythonValue,
        value: PythonValue,
    },
    Iterate {
        receiver: crate::PythonObjectHandle,
    },
    ExecutePersistent {
        code: String,
        inputs: Vec<(String, PythonValue)>,
        outputs: Vec<String>,
    },
    ExecuteFile {
        path: std::path::PathBuf,
        arguments: Vec<String>,
        inputs: Vec<(String, PythonValue)>,
        outputs: Vec<String>,
    },
    Release {
        handle: crate::PythonObjectHandle,
    },
}

enum LaneOperation {
    Invoke(PythonCall),
    Metadata(crate::PythonObjectHandle),
}

enum LaneRequest {
    Operation {
        operation: LaneOperation,
        events: PythonLaneEventSink,
        admitted: Option<OwnedActiveCall>,
        observer: Option<Box<dyn super::PythonLaneObserver>>,
    },
    Shutdown,
}

struct SessionInner {
    id: u64,
    requests: mpsc::Sender<LaneRequest>,
    lane: Mutex<Option<JoinHandle<bool>>>,
    interrupt: crate::cpython::PythonInterrupt,
    // Kept by every cloned session until the interpreter lane has joined.
    _byte_leases: Vec<PrivateByteLease>,
    admission: Arc<Mutex<SessionAdmission>>,
}

#[derive(Default)]
struct SessionAdmission {
    retirement_requested: bool,
    active_calls: usize,
    joined_cleanly: Option<bool>,
}

struct ActiveCall<'a>(&'a SessionInner);

struct OwnedActiveCall(Arc<Mutex<SessionAdmission>>);

struct LaneByteGuard {
    leases: Vec<PrivateByteLease>,
    cleaned: bool,
}

impl Drop for LaneByteGuard {
    fn drop(&mut self) {
        if !self.cleaned {
            for bytes in &self.leases {
                let _ =
                    bytes.persist_unverified(None, "Python execution lane did not verify cleanup");
            }
        }
    }
}

impl Drop for OwnedActiveCall {
    fn drop(&mut self) {
        let mut admission = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        admission.active_calls -= 1;
    }
}

impl Drop for ActiveCall<'_> {
    fn drop(&mut self) {
        let mut admission = self
            .0
            .admission
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        admission.active_calls -= 1;
    }
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        let _ = self.requests.send(LaneRequest::Shutdown);
        if let Some(lane) = self.lane.get_mut().ok().and_then(Option::take) {
            if self
                .admission
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .active_calls
                != 0
            {
                // The lane still owns its selected byte guard and will clean
                // or persist it after the admitted call actually completes.
                // Joining here would block a cancelled Runtime future.
                drop(lane);
                return;
            }
            if !matches!(lane.join(), Ok(true)) {
                for bytes in &self._byte_leases {
                    let _ = bytes.persist_unverified(None, "Python interpreter lane panicked");
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct PythonSession {
    inner: Arc<SessionInner>,
}

impl PythonSession {
    /// Identity only: stale matching authority remains an error at the
    /// reentry submission boundary, not a reason to ordinary-submit.
    pub fn owns_reentry_authority(&self, authority: &super::PythonReentryAuthority) -> bool {
        authority.same_session(self.inner.id)
    }

    /// A selected owner may retire only when no invocation clone can still
    /// submit work. This never blocks on a potentially uninterruptible lane.
    pub fn request_retirement(&self) -> bool {
        {
            let mut admission = self
                .inner
                .admission
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !admission.retirement_requested {
                if Arc::strong_count(&self.inner) != 1 || admission.active_calls != 0 {
                    return false;
                }
                admission.retirement_requested = true;
                let _ = self.inner.requests.send(LaneRequest::Shutdown);
            }
        }
        self.retirement_complete()
    }

    pub fn retirement_complete(&self) -> bool {
        let mut admission = self
            .inner
            .admission
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !admission.retirement_requested {
            return false;
        }
        if let Some(joined_cleanly) = admission.joined_cleanly {
            return joined_cleanly;
        }
        let mut lane = self
            .inner
            .lane
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if lane.as_ref().is_some_and(|lane| !lane.is_finished()) {
            return false;
        }
        let joined_cleanly = lane
            .take()
            .is_none_or(|lane| matches!(lane.join(), Ok(true)));
        if !joined_cleanly {
            for bytes in &self.inner._byte_leases {
                let _ = bytes.persist_unverified(None, "Python interpreter lane panicked");
            }
        }
        admission.joined_cleanly = Some(joined_cleanly);
        joined_cleanly
    }

    fn admit_call(&self) -> Result<ActiveCall<'_>, PythonError> {
        let mut admission = self
            .inner
            .admission
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if admission.retirement_requested {
            return Err(PythonError::host(
                "PythonHostError",
                "the Python execution lane is retiring",
            ));
        }
        admission.active_calls += 1;
        Ok(ActiveCall(&self.inner))
    }

    fn admit_owned_call(&self) -> Result<OwnedActiveCall, PythonError> {
        let mut admission = self
            .inner
            .admission
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if admission.retirement_requested {
            return Err(PythonError::host(
                "PythonHostError",
                "the Python execution lane is retiring",
            ));
        }
        admission.active_calls += 1;
        Ok(OwnedActiveCall(Arc::clone(&self.inner.admission)))
    }

    pub fn start(config: PythonSessionConfig) -> Result<Self, PythonError> {
        Self::start_with_lease(config, None)
    }

    pub fn start_with_lease(
        config: PythonSessionConfig,
        selected_bytes: Option<PrivateByteLease>,
    ) -> Result<Self, PythonError> {
        Self::start_with_leases(config, selected_bytes.into_iter().collect())
    }

    pub fn start_with_leases(
        config: PythonSessionConfig,
        byte_leases: Vec<PrivateByteLease>,
    ) -> Result<Self, PythonError> {
        Self::start_with_installed_artifacts(config, byte_leases, None)
    }

    /// The installed owner, not caller-supplied module paths, certifies which
    /// private wheel imports this lane may remove when it retires.
    pub fn start_with_installed_artifacts(
        config: PythonSessionConfig,
        mut byte_leases: Vec<PrivateByteLease>,
        installed: Option<&InstalledPythonArtifacts>,
    ) -> Result<Self, PythonError> {
        if let Some(installed) = installed {
            let installed_lease = installed.byte_lease();
            if !byte_leases
                .iter()
                .any(|lease| lease.path() == installed_lease.path())
            {
                byte_leases.push(installed_lease);
            }
        }
        if installed.is_some_and(|installed| {
            !installed
                .module_paths()
                .iter()
                .all(|path| config.module_paths.contains(path))
        }) {
            return Err(PythonError::host(
                "PythonImportOwnerError",
                "installed wheel roots differ from the active Python configuration",
            ));
        }
        let import_owner = installed
            .map(ProcessImportReservation::reserve)
            .transpose()?;
        let (requests, receiver) = mpsc::channel();
        let (started, startup) = mpsc::sync_channel(1);
        let lane_bytes = byte_leases.clone();
        let lane = thread::Builder::new()
            .name("runmat-python".to_owned())
            .spawn(move || {
                let mut lane_guard = LaneByteGuard {
                    leases: lane_bytes,
                    cleaned: false,
                };
                let cleaned = (|| {
                    let mut import_owner = import_owner;
                    let empty_names = BTreeSet::new();
                    let interpreter = match Interpreter::start(&config.installation) {
                        Ok(interpreter) => {
                            if let Some(owner) = import_owner.as_ref() {
                                if let Err(error) =
                                    interpreter.admit_managed_imports(&owner.imports().names)
                                {
                                    let _ = started.send(Err(error));
                                    return interpreter
                                        .clear_with_import_roots(&[], &owner.imports().names);
                                }
                            }
                            let managed = import_owner.as_ref().map(|owner| {
                                (
                                    owner.imports().roots.as_slice(),
                                    &owner.imports().byte_lease,
                                )
                            });
                            if let Err(error) =
                                interpreter.prepend_module_paths(&config.module_paths, managed)
                            {
                                let _ = started.send(Err(error));
                                let cleaned = interpreter.clear_with_import_roots(
                                    import_owner
                                        .as_ref()
                                        .map_or(&[][..], |owner| owner.imports().roots.as_slice()),
                                    import_owner
                                        .as_ref()
                                        .map_or(&empty_names, |owner| &owner.imports().names),
                                ) && !interpreter.managed_exposure_unverified();
                                if !cleaned {
                                    if let Some(owner) = import_owner.as_mut() {
                                        owner.retain_conflict();
                                    }
                                }
                                return cleaned;
                            }
                            let _ = started.send(Ok(interpreter.interrupt_handle()));
                            interpreter
                        }
                        Err(error) => {
                            let _ = started.send(Err(error));
                            return true;
                        }
                    };
                    while let Ok(request) = receiver.recv() {
                        match request {
                            LaneRequest::Operation {
                                operation,
                                events,
                                admitted,
                                observer,
                            } => {
                                let result = super::observation::observe(observer, || {
                                    interpreter.with_callback_events(events.clone(), || {
                                        match operation {
                                            LaneOperation::Invoke(call) => interpreter
                                                .invoke(
                                                    call,
                                                    import_owner
                                                        .as_ref()
                                                        .map_or(&[][..], |owner| {
                                                            owner.imports().roots.as_slice()
                                                        }),
                                                )
                                                .map(PythonLaneCompletion::Values),
                                            LaneOperation::Metadata(handle) => interpreter
                                                .metadata(handle)
                                                .map(PythonLaneCompletion::Metadata),
                                        }
                                    })
                                });
                                let _ = events.send(PythonLaneEvent::Complete(result));
                                drop(admitted);
                            }
                            LaneRequest::Shutdown => break,
                        }
                    }
                    let cleaned = interpreter.clear_with_import_roots(
                        import_owner
                            .as_ref()
                            .map_or(&[][..], |owner| owner.imports().roots.as_slice()),
                        import_owner
                            .as_ref()
                            .map_or(&empty_names, |owner| &owner.imports().names),
                    );
                    if !cleaned {
                        if let Some(owner) = import_owner.as_mut() {
                            owner.retain_conflict();
                        }
                    }
                    cleaned
                })();
                lane_guard.cleaned = cleaned;
                cleaned
            })
            .map_err(|error| {
                PythonError::host(
                    "PythonHostError",
                    format!("could not start the Python execution lane: {error}"),
                )
            })?;
        match startup.recv() {
            Ok(Ok(interrupt)) => Ok(Self {
                inner: Arc::new(SessionInner {
                    id: NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed),
                    requests,
                    lane: Mutex::new(Some(lane)),
                    interrupt,
                    _byte_leases: byte_leases,
                    admission: Arc::new(Mutex::new(SessionAdmission::default())),
                }),
            }),
            Ok(Err(error)) => {
                if !matches!(lane.join(), Ok(true)) {
                    for bytes in &byte_leases {
                        let _ = bytes.persist_unverified(None, "Python startup lane panicked");
                    }
                }
                Err(error)
            }
            Err(error) => {
                if !matches!(lane.join(), Ok(true)) {
                    for bytes in &byte_leases {
                        let _ = bytes.persist_unverified(None, "Python startup lane panicked");
                    }
                }
                Err(PythonError::host(
                    "PythonHostError",
                    format!("Python execution lane ended during startup: {error}"),
                ))
            }
        }
    }

    pub fn invoke(&self, call: PythonCall) -> Result<Vec<PythonValue>, PythonError> {
        self.invoke_with_callback(call, |_| {
            Err(PythonError::host(
                "PythonCallbackError",
                "a Python call attempted to invoke a RunMat callback where no callback handler was installed",
            ))
        })
    }

    pub fn invoke_with_callback(
        &self,
        call: PythonCall,
        callback: impl FnMut(PythonCallbackInvocation) -> Result<PythonValue, PythonError>,
    ) -> Result<Vec<PythonValue>, PythonError> {
        self.invoke_with_callback_and_cancellation(call, None, callback)
    }

    pub fn invoke_with_callback_and_cancellation(
        &self,
        call: PythonCall,
        cancellation: Option<&dyn Fn() -> bool>,
        mut callback: impl FnMut(PythonCallbackInvocation) -> Result<PythonValue, PythonError>,
    ) -> Result<Vec<PythonValue>, PythonError> {
        let _active_call = self.admit_call()?;
        if let Some(active) = ACTIVE_REENTRY.with(|active| active.borrow().clone()) {
            if active.session == self.inner.id {
                let (events, response) = mpsc::sync_channel(1);
                active.mailbox.command(PythonCallbackCommand::Reenter {
                    call,
                    reply: PythonLaneReply::Events {
                        sink: PythonLaneEventSink::Sync(events),
                        completion: PythonLaneCompletion::Values,
                    },
                    observer: None,
                })?;
                // A synchronous callback handler can service nested callbacks
                // on the same typed event protocol. It cannot safely request
                // an interrupt while the callback lane holds the GIL.
                return self.drive_sync_events(response, None, &mut callback);
            }
        }
        let (events, response) = mpsc::sync_channel(1);
        self.inner
            .requests
            .send(LaneRequest::Operation {
                operation: LaneOperation::Invoke(call),
                events: PythonLaneEventSink::Sync(events),
                admitted: None,
                observer: None,
            })
            .map_err(lane_ended)?;
        self.drive_sync_events(response, cancellation, &mut callback)
    }

    fn drive_sync_events(
        &self,
        response: mpsc::Receiver<PythonLaneEvent>,
        cancellation: Option<&dyn Fn() -> bool>,
        callback: &mut impl FnMut(PythonCallbackInvocation) -> Result<PythonValue, PythonError>,
    ) -> Result<Vec<PythonValue>, PythonError> {
        let mut interrupted = false;
        loop {
            let event = match response.recv_timeout(std::time::Duration::from_millis(10)) {
                Ok(event) => event,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if !interrupted && cancellation.is_some_and(|is_cancelled| is_cancelled()) {
                        if !self.inner.interrupt.request() {
                            return Err(PythonError::host(
                                "PythonCancellationError",
                                "the Python execution lane could not be interrupted",
                            ));
                        }
                        interrupted = true;
                    }
                    continue;
                }
                Err(error) => return Err(lane_ended(error)),
            };
            match event {
                PythonLaneEvent::Complete(result) => return result?.into_values(),
                PythonLaneEvent::Callback(mut event) => {
                    let (invocation, completion) = event.take_parts();
                    let previous = ACTIVE_REENTRY.with(|active| {
                        active.replace(Some(ActiveReentry {
                            session: self.inner.id,
                            mailbox: completion.mailbox(),
                        }))
                    });
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        callback(invocation)
                    }))
                    .unwrap_or_else(|_| {
                        Err(PythonError::host(
                            "PythonCallbackError",
                            "RunMat callback panicked",
                        ))
                    });
                    ACTIVE_REENTRY.with(|active| {
                        active.replace(previous);
                    });
                    completion.finish(result);
                }
            }
        }
    }

    /// The Runtime path yields to its VM continuation while the single
    /// interpreter lane services callback and exact-session reentry events.
    pub async fn invoke_with_async_callback<F, Fut>(
        &self,
        call: PythonCall,
        reentry: Option<&super::PythonReentryAuthority>,
        observer: Option<Box<dyn super::PythonLaneObserver>>,
        callback: F,
    ) -> Result<Vec<PythonValue>, PythonError>
    where
        F: FnMut(PythonCallbackInvocation, super::PythonReentryAuthority) -> Fut,
        Fut: Future<Output = Result<PythonValue, PythonError>> + 'static,
    {
        if let Some(reentry) = reentry {
            if !reentry.matches(self.inner.id) {
                return Err(PythonError::host(
                    "PythonCallbackError",
                    "Python callback reentry authority is stale or belongs to another session",
                ));
            }
            let admitted = self.admit_call()?;
            let result = reentry.invoke(call, observer, callback).await;
            drop(admitted);
            return result;
        }
        let admitted = self.admit_owned_call()?;
        let (events, response) = futures::channel::mpsc::unbounded();
        self.inner
            .requests
            .send(LaneRequest::Operation {
                operation: LaneOperation::Invoke(call),
                events: PythonLaneEventSink::Async(events),
                admitted: Some(admitted),
                observer,
            })
            .map_err(lane_ended)?;
        super::event_pump::drive(self.inner.id, response, callback)
            .await?
            .into_values()
    }

    pub fn metadata(
        &self,
        handle: crate::PythonObjectHandle,
    ) -> Result<PythonObjectMetadata, PythonError> {
        self.metadata_with_reentry(handle, None)
    }

    pub fn metadata_with_reentry(
        &self,
        handle: crate::PythonObjectHandle,
        reentry: Option<&super::PythonReentryAuthority>,
    ) -> Result<PythonObjectMetadata, PythonError> {
        let _active_call = self.admit_call()?;
        if let Some(reentry) = reentry {
            if !reentry.matches(self.inner.id) {
                return Err(PythonError::host(
                    "PythonCallbackError",
                    "Python callback reentry authority is stale or belongs to another session",
                ));
            }
            return reentry.metadata(handle);
        }
        if let Some(active) = ACTIVE_REENTRY.with(|active| active.borrow().clone()) {
            if active.session == self.inner.id {
                let (reply, response) = mpsc::sync_channel(1);
                active.mailbox.command(PythonCallbackCommand::Metadata {
                    handle,
                    reply: PythonLaneReply::Sync(reply),
                    observer: None,
                })?;
                return response.recv().map_err(lane_ended)?;
            }
        }
        let (events, response) = mpsc::sync_channel(1);
        self.inner
            .requests
            .send(LaneRequest::Operation {
                operation: LaneOperation::Metadata(handle),
                events: PythonLaneEventSink::Sync(events),
                admitted: None,
                observer: None,
            })
            .map_err(lane_ended)?;
        loop {
            match response.recv().map_err(lane_ended)? {
                PythonLaneEvent::Complete(result) => return result?.into_metadata(),
                PythonLaneEvent::Callback(mut event) => {
                    let (_, completion) = event.take_parts();
                    completion.finish(Err(PythonError::host(
                        "PythonCallbackError",
                        "metadata has no RunMat callback handler",
                    )));
                }
            }
        }
    }

    pub async fn metadata_with_reentry_async<F, Fut>(
        &self,
        handle: crate::PythonObjectHandle,
        reentry: Option<&super::PythonReentryAuthority>,
        observer: Option<Box<dyn super::PythonLaneObserver>>,
        callback: F,
    ) -> Result<PythonObjectMetadata, PythonError>
    where
        F: FnMut(PythonCallbackInvocation, super::PythonReentryAuthority) -> Fut,
        Fut: Future<Output = Result<PythonValue, PythonError>> + 'static,
    {
        if let Some(reentry) = reentry {
            let _active_call = self.admit_call()?;
            if !reentry.matches(self.inner.id) {
                return Err(PythonError::host(
                    "PythonCallbackError",
                    "Python callback reentry authority is stale or belongs to another session",
                ));
            }
            reentry.metadata_async(handle, observer, callback).await
        } else {
            let admitted = self.admit_owned_call()?;
            let (events, response) = futures::channel::mpsc::unbounded();
            self.inner
                .requests
                .send(LaneRequest::Operation {
                    operation: LaneOperation::Metadata(handle),
                    events: PythonLaneEventSink::Async(events),
                    admitted: Some(admitted),
                    observer,
                })
                .map_err(lane_ended)?;
            super::event_pump::drive(self.inner.id, response, callback)
                .await?
                .into_metadata()
        }
    }
}

pub(super) fn lane_ended(error: impl std::fmt::Display) -> PythonError {
    PythonError::host(
        "PythonHostError",
        format!("Python execution lane is unavailable: {error}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        discover_python, PythonArray, PythonBufferOwner, PythonDType, PythonDateTime,
        PythonDiscoveryRequest, PythonObjectHandle, PythonTimeDelta,
    };

    #[derive(Debug)]
    struct TestBuffer(Vec<f64>);

    impl PythonBufferOwner for TestBuffer {
        fn address(&self) -> usize {
            self.0.as_ptr() as usize
        }

        fn byte_length(&self) -> usize {
            self.0.len() * std::mem::size_of::<f64>()
        }

        fn copy_bytes(&self) -> Vec<u8> {
            let length = self.byte_length();
            // SAFETY: the f64 allocation contains exactly length initialized
            // bytes and remains live for this immutable copy.
            unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), length) }.to_vec()
        }
    }

    fn session() -> Option<PythonSession> {
        let installation = discover_python(&PythonDiscoveryRequest::default()).ok()?;
        Some(
            PythonSession::start(PythonSessionConfig {
                installation,
                module_paths: Vec::new(),
            })
            .expect("start selected CPython"),
        )
    }

    #[test]
    fn selected_bytes_outlive_cloned_lane_until_retirement() {
        let installation = match discover_python(&PythonDiscoveryRequest::default()) {
            Ok(installation) => installation,
            Err(error) => {
                eprintln!("Python byte-retirement execution unavailable: {error}");
                return;
            }
        };
        let lease = PrivateByteLease::new_attempt_scratch().expect("private byte owner");
        let private_path = lease.path().to_path_buf();
        let session = PythonSession::start_with_lease(
            PythonSessionConfig {
                installation,
                module_paths: Vec::new(),
            },
            Some(lease.clone()),
        )
        .expect("start selected CPython");
        drop(lease);
        let active = session.clone();
        assert!(!session.request_retirement());
        assert!(private_path.exists());
        drop(active);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !session.request_retirement() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(session.retirement_complete());
        drop(session);
        assert!(!private_path.exists());
    }

    #[test]
    fn invokes_qualified_functions_with_positional_and_keyword_arguments() {
        let Some(session) = session() else {
            return;
        };
        let result = session
            .invoke(PythonCall::InvokeQualified {
                name: "py.math.isclose".to_owned(),
                arguments: vec![
                    PythonValue::Float(1.0),
                    PythonValue::Float(1.000_001),
                    PythonValue::Keywords(vec![(
                        "rel_tol".to_owned(),
                        PythonValue::Float(0.000_01),
                    )]),
                ],
            })
            .expect("invoke math.isclose");
        assert!(matches!(result.as_slice(), [PythonValue::Bool(true)]));
    }

    #[test]
    fn persistent_code_keeps_workspace_and_file_code_does_not_share_it() {
        let Some(session) = session() else {
            return;
        };
        let first = session
            .invoke(PythonCall::ExecutePersistent {
                code: "counter = 40".to_owned(),
                inputs: Vec::new(),
                outputs: vec!["counter".to_owned()],
            })
            .expect("initialize persistent workspace");
        assert!(matches!(first.as_slice(), [PythonValue::Signed(40)]));
        let second = session
            .invoke(PythonCall::ExecutePersistent {
                code: "counter += 2".to_owned(),
                inputs: Vec::new(),
                outputs: vec!["counter".to_owned()],
            })
            .expect("reuse persistent workspace");
        assert!(matches!(second.as_slice(), [PythonValue::Signed(42)]));
    }

    #[test]
    fn opaque_objects_keep_identity_until_explicit_release() {
        let Some(session) = session() else {
            return;
        };
        let object = session
            .invoke(PythonCall::InvokeQualified {
                name: "py.object".to_owned(),
                arguments: Vec::new(),
            })
            .expect("construct Python object");
        let [PythonValue::Object(handle)] = object.as_slice() else {
            panic!("expected opaque object handle");
        };
        let metadata = session.metadata(*handle).expect("inspect live object");
        assert_eq!(metadata.type_name, "object");
        let probe = session
            .invoke_with_callback(
                PythonCall::ExecutePersistent {
                    code: "class Meta(type):\n    def __getattribute__(cls, name):\n        if name == '__name__':\n            return callback()\n        return super().__getattribute__(name)\nclass Probe(metaclass=Meta):\n    def __iter__(self):\n        raise TypeError('metadata must not iterate')\nresult = Probe()".into(),
                    inputs: vec![("callback".into(), PythonValue::Callback(17))],
                    outputs: vec!["result".into()],
                },
                |_| Ok(PythonValue::String("Probe".into())),
            )
            .expect("construct metadata callback probe");
        let [PythonValue::Object(probe_handle)] = probe.as_slice() else {
            panic!("expected metadata probe object");
        };
        let sync_error = session
            .metadata(*probe_handle)
            .expect_err("metadata without a callback service must fail");
        assert_eq!(sync_error.type_name, "RuntimeError");
        let probed = futures::executor::block_on(session.metadata_with_reentry_async(
            *probe_handle,
            None,
            None,
            |invocation, _| async move {
                assert_eq!(invocation.callback, 17);
                Ok(PythonValue::String("Probe".into()))
            },
        ))
        .expect("metadata callback runs through the origin event pump");
        assert_eq!(probed.type_name, "Probe");
        assert_eq!(probed.iterable, Some(true));
        session
            .invoke(PythonCall::Release { handle: *handle })
            .expect("release object");
        let error = session
            .metadata(PythonObjectHandle(handle.0))
            .expect_err("released object must be stale");
        assert_eq!(error.type_name, "PythonStaleObject");
    }

    #[test]
    fn numeric_arrays_alias_runmat_storage_and_return_with_dtype_and_shape() {
        let Some(session) = session() else {
            return;
        };
        let owner = Arc::new(TestBuffer(vec![1.0, 2.0, 3.0, 4.0]));
        let address = owner.address();
        let result = session
            .invoke(PythonCall::ExecutePersistent {
                code: "address = values.__array_interface__['data'][0]\nreturned = values".into(),
                inputs: vec![(
                    "values".into(),
                    PythonValue::Array(PythonArray {
                        dtype: PythonDType::Float64,
                        shape: vec![2, 2],
                        column_major: true,
                        read_only: true,
                        owner,
                    }),
                )],
                outputs: vec!["address".into(), "returned".into()],
            })
            .expect("round-trip shared array");
        assert!(matches!(
            result.first(),
            Some(PythonValue::Signed(value)) if *value == address as i64
        ));
        let Some(PythonValue::Array(array)) = result.get(1) else {
            panic!("expected typed Python array");
        };
        assert_eq!(array.dtype, PythonDType::Float64);
        assert_eq!(array.shape, vec![2, 2]);
        assert!(array.column_major);
        assert_eq!(
            array.owner.copy_bytes(),
            TestBuffer(vec![1.0, 2.0, 3.0, 4.0]).copy_bytes()
        );
    }

    #[test]
    fn temporal_scalars_and_arrays_use_explicit_python_types() {
        let Some(session) = session() else {
            return;
        };
        let result = session
            .invoke(PythonCall::ExecutePersistent {
                code: "import datetime\nimport numpy as np\nscalar_date_type = type(date).__name__\nscalar_delta_type = type(delta).__name__\ndates = np.array(['1969-12-31T23:59:59.123456', '2026-08-27T12:34:56.654321'], dtype='datetime64[us]')\ndeltas = np.array([-1001, 2002], dtype='timedelta64[us]')".into(),
                inputs: vec![
                    (
                        "date".into(),
                        PythonValue::DateTime(PythonDateTime {
                            year: 2026,
                            month: 8,
                            day: 27,
                            hour: 12,
                            minute: 34,
                            second: 56,
                            microsecond: 654_321,
                        }),
                    ),
                    (
                        "delta".into(),
                        PythonValue::TimeDelta(PythonTimeDelta {
                            days: -1,
                            seconds: 86_399,
                            microseconds: 998_999,
                        }),
                    ),
                ],
                outputs: vec![
                    "scalar_date_type".into(),
                    "scalar_delta_type".into(),
                    "dates".into(),
                    "deltas".into(),
                ],
            })
            .expect("round-trip Python temporal values");
        assert!(matches!(result.first(), Some(PythonValue::String(value)) if value == "datetime"));
        assert!(matches!(result.get(1), Some(PythonValue::String(value)) if value == "timedelta"));
        let Some(PythonValue::Array(dates)) = result.get(2) else {
            panic!("expected datetime64 array");
        };
        assert_eq!(dates.dtype, PythonDType::DateTime64Micros);
        assert_eq!(dates.shape, vec![2]);
        assert_eq!(
            dates.owner.copy_bytes(),
            [-876_544_i64, 1_787_834_096_654_321_i64]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        let Some(PythonValue::Array(deltas)) = result.get(3) else {
            panic!("expected timedelta64 array");
        };
        assert_eq!(deltas.dtype, PythonDType::TimeDelta64Micros);
        assert_eq!(
            deltas.owner.copy_bytes(),
            [-1_001_i64, 2_002_i64]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn naive_datetime_converts_and_aware_datetime_keeps_python_identity() {
        let Some(session) = session() else {
            return;
        };
        let result = session
            .invoke(PythonCall::ExecutePersistent {
                code: "import datetime\nnaive = datetime.datetime(2026, 8, 27, 1, 2, 3, 456789)\naware = datetime.datetime(2026, 8, 27, tzinfo=datetime.timezone.utc)".into(),
                inputs: Vec::new(),
                outputs: vec!["naive".into(), "aware".into()],
            })
            .expect("read Python datetime values");
        assert!(matches!(
            result.first(),
            Some(PythonValue::DateTime(PythonDateTime {
                year: 2026,
                month: 8,
                day: 27,
                hour: 1,
                minute: 2,
                second: 3,
                microsecond: 456_789,
            }))
        ));
        assert!(matches!(result.get(1), Some(PythonValue::Object(_))));
    }

    #[test]
    fn exceptions_retain_python_type_message_and_traceback_frames() {
        let Some(session) = session() else {
            return;
        };
        let error = session
            .invoke(PythonCall::ExecutePersistent {
                code: "def fail_here():\n    return 1 / 0\nfail_here()".into(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            })
            .expect_err("Python failure must cross the host boundary");
        assert_eq!(error.type_name, "ZeroDivisionError");
        assert!(error.message.contains("division by zero"));
        assert!(error
            .traceback
            .iter()
            .any(|frame| frame.function.as_deref() == Some("fail_here") && frame.line == Some(2)));
        assert!(error.formatted_traceback.contains("ZeroDivisionError"));
    }

    #[test]
    fn exceptions_retain_explicit_python_causes() {
        let Some(session) = session() else {
            return;
        };
        let error = session
            .invoke(PythonCall::ExecutePersistent {
                code: "try:\n    int('not-an-integer')\nexcept ValueError as cause:\n    raise RuntimeError('outer failure') from cause".into(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            })
            .expect_err("chained Python failure must cross the host boundary");
        assert_eq!(error.type_name, "RuntimeError");
        let cause = error.cause.expect("explicit Python cause");
        assert_eq!(cause.type_name, "ValueError");
        assert!(cause.message.contains("not-an-integer"));
    }

    #[test]
    fn callbacks_return_on_the_origin_thread_and_can_reenter_python() {
        let Some(session) = session() else {
            return;
        };
        let nested = session.clone();
        let result = session
            .invoke_with_callback(
                PythonCall::ExecutePersistent {
                    code: "result = callback(40)".into(),
                    inputs: vec![("callback".into(), PythonValue::Callback(7))],
                    outputs: vec!["result".into()],
                },
                move |invocation| {
                    assert_eq!(invocation.callback, 7);
                    assert!(matches!(
                        invocation.arguments.as_slice(),
                        [PythonValue::Signed(40)]
                    ));
                    let nested_result = nested.invoke(PythonCall::InvokeQualified {
                        name: "py.operator.add".into(),
                        arguments: vec![PythonValue::Signed(40), PythonValue::Signed(2)],
                    })?;
                    nested_result.into_iter().next().ok_or_else(|| {
                        PythonError::host("PythonCallbackError", "nested call returned no value")
                    })
                },
            )
            .expect("callback and nested Python call");
        assert!(matches!(result.as_slice(), [PythonValue::Signed(42)]));

        let nested = session.clone();
        let result = session
            .invoke_with_callback(
                PythonCall::ExecutePersistent {
                    code: "result = callback(40)".into(),
                    inputs: vec![("callback".into(), PythonValue::Callback(7))],
                    outputs: vec!["result".into()],
                },
                move |invocation| {
                    assert_eq!(invocation.callback, 7);
                    nested
                        .invoke_with_callback(
                            PythonCall::ExecutePersistent {
                                code: "result = callback(41)".into(),
                                inputs: vec![("callback".into(), PythonValue::Callback(8))],
                                outputs: vec!["result".into()],
                            },
                            |nested_invocation| {
                                assert_eq!(nested_invocation.callback, 8);
                                assert!(matches!(
                                    nested_invocation.arguments.as_slice(),
                                    [PythonValue::Signed(41)]
                                ));
                                Ok(PythonValue::Signed(42))
                            },
                        )?
                        .into_iter()
                        .next()
                        .ok_or_else(|| {
                            PythonError::host(
                                "PythonCallbackError",
                                "nested callback returned no value",
                            )
                        })
                },
            )
            .expect("nested callback uses its own synchronous event pump");
        assert!(matches!(result.as_slice(), [PythonValue::Signed(42)]));
    }

    #[test]
    fn cancellation_interrupts_python_at_a_bytecode_boundary() {
        let Some(session) = session() else {
            return;
        };
        let cancellation = std::sync::atomic::AtomicBool::new(true);
        let error = session
            .invoke_with_callback_and_cancellation(
                PythonCall::ExecutePersistent {
                    code: "while True:\n    pass".into(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                },
                Some(&|| cancellation.load(Ordering::SeqCst)),
                |_| {
                    Err(PythonError::host(
                        "PythonCallbackError",
                        "unexpected callback",
                    ))
                },
            )
            .expect_err("cancelled Python execution must stop");
        assert_eq!(error.type_name, "KeyboardInterrupt");
    }
}
