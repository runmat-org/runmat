//! One thread owns each Runtime Java session, JNI object registry, and callback
//! registrations. The standalone `JavaSession` synchronous API is unchanged.

mod protocol;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use futures::channel::{mpsc as async_mpsc, oneshot};
use runmat_process_host::private_bytes::PrivateByteLease;

use super::{JavaInvocationError, JavaSession, JavaValue};
use crate::{JavaObjectHandle, JvmInstallation, JvmProcess, SessionClasspath};
use protocol::{wait_callback, JavaLaneCommand};
pub use protocol::{JavaCallbackCompletion, JavaCallbackEvent, JavaReentryAuthority};

static NEXT_LANE_ID: AtomicU64 = AtomicU64::new(1);

struct Admission {
    active: usize,
    retiring: bool,
}

struct ActiveCall(Arc<Mutex<Admission>>);

impl Drop for ActiveCall {
    fn drop(&mut self) {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active -= 1;
    }
}

enum LaneRequest {
    Run(JavaLaneCommand),
    Shutdown,
}

pub(super) struct JavaLaneOwner {
    id: u64,
    session: Rc<JavaSession>,
    callbacks: RefCell<BTreeMap<u64, JavaValue>>,
    released: Arc<Mutex<BTreeSet<JavaObjectHandle>>>,
    events: RefCell<Option<async_mpsc::UnboundedSender<JavaCallbackEvent>>>,
}

/// Restore the suspended operation's event stream even if a reentrant Java
/// callback command unwinds into its existing JNI panic boundary.
struct CallbackEventScope<'a> {
    events: &'a RefCell<Option<async_mpsc::UnboundedSender<JavaCallbackEvent>>>,
    previous: Option<async_mpsc::UnboundedSender<JavaCallbackEvent>>,
}

impl<'a> CallbackEventScope<'a> {
    fn install(
        events: &'a RefCell<Option<async_mpsc::UnboundedSender<JavaCallbackEvent>>>,
        selected: async_mpsc::UnboundedSender<JavaCallbackEvent>,
    ) -> Self {
        let previous = events.replace(Some(selected));
        Self { events, previous }
    }
}

impl Drop for CallbackEventScope<'_> {
    fn drop(&mut self) {
        self.events.replace(self.previous.take());
    }
}

/// A borrow into the exact thread-owned JNI session. The closure passed to
/// `JavaLane::submit` is `Send`; this owner and `JavaSession` never are.
pub struct JavaLaneSession<'a> {
    owner: &'a Rc<JavaLaneOwner>,
}

impl JavaLaneSession<'_> {
    pub fn session(&self) -> &JavaSession {
        &self.owner.session
    }

    /// Acknowledged releases may be retried after their caller is cancelled.
    /// Only this lane's own prior successful release is idempotent; an unknown
    /// or stale handle that was never released still fails normally.
    pub fn release_registered(&self, handle: JavaObjectHandle) -> Result<(), JavaInvocationError> {
        if self
            .owner
            .released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(&handle)
        {
            return Ok(());
        }
        self.owner.session.release(handle)?;
        self.owner
            .released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(handle);
        Ok(())
    }

    /// Resolve one origin-owned callback token to a stable Java proxy identity.
    /// Repeated use of that token preserves listener removal semantics.
    pub fn callback_value(&self, token: u64) -> Result<JavaValue, JavaInvocationError> {
        if let Some(value) = self.owner.callbacks.borrow().get(&token) {
            return Ok(value.clone());
        }
        let owner = Rc::downgrade(self.owner);
        let value = self.owner.session.register_callback(move |invocation| {
            let owner = owner.upgrade().ok_or_else(|| {
                JavaInvocationError::Callback("Java callback session has ended".into())
            })?;
            let events = owner.events.borrow().as_ref().cloned().ok_or_else(|| {
                JavaInvocationError::Callback(
                    "Java callback ran outside an admitted Runtime operation".into(),
                )
            })?;
            let (event, mailbox) = JavaCallbackEvent::new(owner.id, token, invocation);
            events.unbounded_send(event).map_err(|_| {
                JavaInvocationError::Callback("Java callback Runtime owner has ended".into())
            })?;
            wait_callback(&mailbox, &owner)
        })?;
        self.owner
            .callbacks
            .borrow_mut()
            .insert(token, value.clone());
        Ok(value)
    }

    pub fn materialize_values(
        &self,
        values: Vec<JavaValue>,
    ) -> Result<Vec<JavaValue>, JavaInvocationError> {
        values
            .into_iter()
            .map(|value| self.materialize_value(value))
            .collect()
    }

    fn materialize_value(&self, value: JavaValue) -> Result<JavaValue, JavaInvocationError> {
        match value {
            JavaValue::Callback(token) => self.callback_value(token),
            JavaValue::Array {
                component,
                elements,
            } => Ok(JavaValue::Array {
                component,
                elements: self.materialize_values(elements)?,
            }),
            value => Ok(value),
        }
    }
}

pub struct JavaLaneOperation<T> {
    result: oneshot::Receiver<Result<T, JavaInvocationError>>,
    events: async_mpsc::UnboundedReceiver<JavaCallbackEvent>,
}

impl<T> JavaLaneOperation<T> {
    pub fn into_parts(
        self,
    ) -> (
        oneshot::Receiver<Result<T, JavaInvocationError>>,
        async_mpsc::UnboundedReceiver<JavaCallbackEvent>,
    ) {
        (self.result, self.events)
    }
}

/// Transport handle only. The JNI session and its selected private-byte lease
/// remain on the lane through completion and verified classloader closure.
pub struct JavaLane {
    id: u64,
    thread_id: std::thread::ThreadId,
    installation: JvmInstallation,
    requests: mpsc::Sender<LaneRequest>,
    admission: Arc<Mutex<Admission>>,
    released: Arc<Mutex<BTreeSet<JavaObjectHandle>>>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl JavaLane {
    /// Identity only: a matching stale callback is still passed to
    /// `submit_reentry`, which rejects it instead of ordinary-submitting.
    pub fn owns_reentry_authority(&self, authority: &JavaReentryAuthority) -> bool {
        authority.same_session(self.id)
    }

    pub fn start(
        process: JvmProcess,
        classpath: SessionClasspath,
        lease: Option<PrivateByteLease>,
    ) -> Result<Self, JavaInvocationError> {
        let id = NEXT_LANE_ID.fetch_add(1, Ordering::Relaxed);
        if id == 0 {
            return Err(JavaInvocationError::Callback(
                "Java lane identity space is exhausted".into(),
            ));
        }
        let installation = process.installation().clone();
        let released = Arc::new(Mutex::new(BTreeSet::new()));
        let lane_released = Arc::clone(&released);
        let (requests, receiver) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("runmat-java-session".into())
            .spawn(move || {
                let owner = Rc::new(JavaLaneOwner {
                    id,
                    session: Rc::new(JavaSession::with_classpath_and_lease(
                        process, classpath, lease,
                    )),
                    callbacks: RefCell::new(BTreeMap::new()),
                    released: lane_released,
                    events: RefCell::new(None),
                });
                while let Ok(request) = receiver.recv() {
                    match request {
                        LaneRequest::Run(command) => command(&owner),
                        LaneRequest::Shutdown => break,
                    }
                }
            })
            .map_err(|error| {
                JavaInvocationError::Callback(format!("could not start Java session lane: {error}"))
            })?;
        Ok(Self {
            id,
            thread_id: thread.thread().id(),
            installation,
            requests,
            admission: Arc::new(Mutex::new(Admission {
                active: 0,
                retiring: false,
            })),
            released,
            thread: Mutex::new(Some(thread)),
        })
    }

    pub fn installation(&self) -> &JvmInstallation {
        &self.installation
    }

    /// The origin removes this metadata only after observing a successful
    /// release and removing its own foreign-resource mapping. Cancelled calls
    /// leave the receipt available for one exact retry.
    pub fn acknowledge_release(&self, handle: JavaObjectHandle) {
        self.released
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&handle);
    }

    fn admit(&self) -> Result<ActiveCall, JavaInvocationError> {
        let mut state = self
            .admission
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.retiring {
            return Err(JavaInvocationError::Callback(
                "Java classloader is retiring".into(),
            ));
        }
        state.active = state
            .active
            .checked_add(1)
            .ok_or_else(|| JavaInvocationError::Callback("Java lane admission exhausted".into()))?;
        Ok(ActiveCall(Arc::clone(&self.admission)))
    }

    pub fn submit<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&JavaLaneSession<'_>) -> Result<T, JavaInvocationError> + Send + 'static,
    ) -> Result<JavaLaneOperation<T>, JavaInvocationError> {
        let admitted = self.admit()?;
        let (reply, response) = oneshot::channel();
        let (events, callbacks) = async_mpsc::unbounded();
        self.requests
            .send(LaneRequest::Run(Box::new(move |owner| {
                let result = {
                    let _events = CallbackEventScope::install(&owner.events, events);
                    operation(&JavaLaneSession { owner })
                };
                let _ = reply.send(result);
                drop(admitted);
            })))
            .map_err(|_| JavaInvocationError::Callback("Java session lane ended".into()))?;
        Ok(JavaLaneOperation {
            result: response,
            events: callbacks,
        })
    }

    pub fn submit_reentry<T: Send + 'static>(
        &self,
        authority: &JavaReentryAuthority,
        operation: impl FnOnce(&JavaLaneSession<'_>) -> Result<T, JavaInvocationError> + Send + 'static,
    ) -> Result<JavaLaneOperation<T>, JavaInvocationError> {
        if !authority.matches(self.id) {
            return Err(JavaInvocationError::Callback(
                "Java reentry belongs to a stale or different callback".into(),
            ));
        }
        let admitted = self.admit()?;
        let (reply, response) = oneshot::channel();
        let (events, callbacks) = async_mpsc::unbounded();
        authority.command(Box::new(move |owner| {
            // A nested operation gets its own callback stream. The outer
            // operation remains suspended with its stream retained until this
            // exact reentry finishes; its pump must not label nested events
            // with the outer operation's RuntimeContext.
            let result = {
                let _events = CallbackEventScope::install(&owner.events, events);
                operation(&JavaLaneSession { owner })
            };
            let _ = reply.send(result);
            drop(admitted);
        }))?;
        Ok(JavaLaneOperation {
            result: response,
            events: callbacks,
        })
    }

    /// Return success only after the exact lane has closed its classloader and
    /// removed JNI callback registrations. A busy or failed close is Retained.
    pub fn retire_classloader(&self) -> Result<(), JavaInvocationError> {
        if std::thread::current().id() == self.thread_id {
            return Err(JavaInvocationError::Callback(
                "Java classloader retirement cannot wait on its own lane".into(),
            ));
        }
        {
            let mut state = self
                .admission
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.retiring || state.active != 0 {
                return Err(JavaInvocationError::Callback(
                    "Java session still has an admitted operation".into(),
                ));
            }
            state.retiring = true;
        }
        let (reply, response) = mpsc::sync_channel(1);
        if self
            .requests
            .send(LaneRequest::Run(Box::new(move |owner| {
                let _ = reply.send(owner.session.retire_classloader());
            })))
            .is_err()
        {
            self.admission
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retiring = false;
            return Err(JavaInvocationError::Callback(
                "Java session lane ended".into(),
            ));
        }
        let result = response.recv().map_err(|_| {
            JavaInvocationError::Callback("Java classloader retirement was not verified".into())
        });
        if !matches!(result.as_ref(), Ok(Ok(()))) {
            self.admission
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retiring = false;
        }
        result??;
        let _ = self.requests.send(LaneRequest::Shutdown);
        let thread = self
            .thread
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .ok_or_else(|| {
                JavaInvocationError::Callback("Java lane join owner is missing".into())
            })?;
        thread.join().map_err(|_| {
            JavaInvocationError::Callback("Java lane ended without verified teardown".into())
        })
    }
}

impl Drop for JavaLane {
    fn drop(&mut self) {
        let _ = self.requests.send(LaneRequest::Shutdown);
        // A dropped caller is not a retirement certificate. The lane retains
        // its JavaSession and lease until physical completion; JavaSession's
        // destructor persists bytes if classloader closure is unverified.
        self.thread
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
    }
}
