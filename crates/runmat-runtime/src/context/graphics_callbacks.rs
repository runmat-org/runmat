//! Session-local graphics callback ingress and ordered drain.
//!
//! GUI/browser producers receive only a `Send` selection port. Callable values,
//! selected invocation context, and workspaces remain on the session lane.
//! The Core session must own this queue strongly; an installed presentation
//! service may refer to it only weakly, because registrations retain selected
//! contexts whose service ports can point back to presentation.

use super::{RuntimeContext, RuntimeContextLocalId, RuntimeServiceFuture};
use crate::{build_runtime_error, RuntimeError};
pub use runmat_value::GraphicsCallbackIngressFailure;
pub use runmat_value::GraphicsCallbackRolePayload;
pub use runmat_value::{GraphicsCallbackKind, GraphicsCallbackResult};
use runmat_value::{GraphicsReceiverKind, GraphicsReceiverRef};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use tokio::sync::Notify;

#[path = "graphics_callbacks/claim.rs"]
mod claim;
use claim::CallbackClaim;
#[path = "graphics_callbacks/attachment.rs"]
mod attachment;
#[path = "graphics_callbacks/mutation.rs"]
mod mutation;
#[path = "graphics_callbacks/property_registration.rs"]
mod property_registration;
pub(crate) use mutation::GraphicsCallbackMutationScope;
#[path = "graphics_callbacks/host_request.rs"]
mod host_request;
pub use host_request::GraphicsCallbackHostTicket;
use host_request::HostReply;
#[path = "graphics_callbacks/drive.rs"]
mod drive;
#[path = "graphics_callbacks/retirement.rs"]
mod retirement;
#[path = "graphics_callbacks/wake.rs"]
mod wake;
pub use runmat_value::{GraphicsCallbackDisposition, GraphicsCallbackHostFailure};

/// Host metadata only. The session-side adapter constructs callable arguments
/// under the registration's semantic context; a numeric figure is never
/// treated as a session capability or a native Figure/Axes object. Host
/// gestures do not carry a command-profiling origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphicsCallbackHostEvent {
    pub figure: u32,
    pub kind: GraphicsCallbackKind,
    pub position: Option<(f64, f64)>,
}

struct Envelope {
    registration: u64,
    /// An ID may be reused by a property replacement. Old queued gestures
    /// must never execute the new callable at that same registration slot.
    claim: Arc<CallbackClaim>,
    event: GraphicsCallbackHostEvent,
    payload: Option<GraphicsCallbackRolePayload>,
    reply: Option<Arc<HostReply>>,
}

/// This is the sole part that may cross the GUI/worker boundary.
#[derive(Clone)]
pub struct GraphicsCallbackIngress {
    representation: IngressRepresentation,
    registration: u64,
    figure: u32,
    kind: GraphicsCallbackKind,
    live: Arc<CallbackClaim>,
    closed: Arc<AtomicBool>,
    sender: SyncSender<Envelope>,
    admission_wake: Arc<Notify>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum IngressRepresentation {
    Metadata,
    Bound,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphicsCallbackIngressError {
    pub event: GraphicsCallbackHostEvent,
    pub failure: GraphicsCallbackIngressFailure,
}

impl GraphicsCallbackIngress {
    pub fn figure(&self) -> u32 {
        self.figure
    }

    pub fn kind(&self) -> GraphicsCallbackKind {
        self.kind
    }

    pub fn revoke(&self) -> GraphicsCallbackDisposition {
        self.live.retire()
    }

    pub fn is_live(&self) -> bool {
        !self.closed.load(Ordering::Acquire) && self.live.is_live()
    }

    /// Never block a GUI/DOM event lane. Full is a typed backpressure result;
    /// the host must retain or report this event, never silently discard it.
    pub fn send(
        &self,
        event: GraphicsCallbackHostEvent,
    ) -> Result<(), GraphicsCallbackIngressError> {
        if self.representation != IngressRepresentation::Metadata {
            return Err(GraphicsCallbackIngressError {
                event,
                failure: GraphicsCallbackIngressFailure::MismatchedScope,
            });
        }
        if event.figure != self.figure || event.kind != self.kind {
            return Err(GraphicsCallbackIngressError {
                event,
                failure: GraphicsCallbackIngressFailure::MismatchedScope,
            });
        }
        if !self.is_live() {
            return Err(GraphicsCallbackIngressError {
                event,
                failure: GraphicsCallbackIngressFailure::Retired,
            });
        }
        self.sender
            .try_send(Envelope {
                registration: self.registration,
                claim: Arc::clone(&self.live),
                event,
                payload: None,
                reply: None,
            })
            .map_err(|error| match error {
                TrySendError::Full(envelope) => GraphicsCallbackIngressError {
                    event: envelope.event,
                    failure: GraphicsCallbackIngressFailure::Full,
                },
                TrySendError::Disconnected(envelope) => GraphicsCallbackIngressError {
                    event: envelope.event,
                    failure: GraphicsCallbackIngressFailure::Closed,
                },
            })?;
        // The envelope is physically queued even if its exact claim retires
        // in the following race check. Wake the session owner to discard it
        // and release bounded ingress capacity.
        self.admission_wake.notify_one();
        // A concurrent host admission can race session close between the
        // first liveness check and try_send. Closed queues never execute it.
        if !self.is_live() {
            return Err(GraphicsCallbackIngressError {
                event,
                failure: GraphicsCallbackIngressFailure::Retired,
            });
        }
        Ok(())
    }

    /// Submit a result-bearing host gesture without blocking the UI lane.
    /// The returned ticket is lazy only with respect to its acknowledgement:
    /// the request itself is admitted here, and Drop can retire it before the
    /// session callback begins.
    pub fn request(
        &self,
        event: GraphicsCallbackHostEvent,
    ) -> Result<GraphicsCallbackHostTicket, GraphicsCallbackIngressError> {
        self.request_inner(event, None)
    }

    pub fn request_bound(
        &self,
        event: GraphicsCallbackHostEvent,
        payload: GraphicsCallbackRolePayload,
    ) -> Result<GraphicsCallbackHostTicket, GraphicsCallbackIngressError> {
        if !payload.matches_role(event.kind) {
            return Err(GraphicsCallbackIngressError {
                event,
                failure: GraphicsCallbackIngressFailure::MismatchedScope,
            });
        }
        self.request_inner(event, Some(payload))
    }

    fn request_inner(
        &self,
        event: GraphicsCallbackHostEvent,
        payload: Option<GraphicsCallbackRolePayload>,
    ) -> Result<GraphicsCallbackHostTicket, GraphicsCallbackIngressError> {
        if (self.representation == IngressRepresentation::Bound) != payload.is_some() {
            return Err(GraphicsCallbackIngressError {
                event,
                failure: GraphicsCallbackIngressFailure::MismatchedScope,
            });
        }
        if event.figure != self.figure || event.kind != self.kind {
            return Err(GraphicsCallbackIngressError {
                event,
                failure: GraphicsCallbackIngressFailure::MismatchedScope,
            });
        }
        if !self.is_live() {
            return Err(GraphicsCallbackIngressError {
                event,
                failure: GraphicsCallbackIngressFailure::Retired,
            });
        }
        let (reply, ticket) = HostReply::new();
        self.sender
            .try_send(Envelope {
                registration: self.registration,
                claim: Arc::clone(&self.live),
                event,
                payload,
                reply: Some(Arc::clone(&reply)),
            })
            .map_err(|error| match error {
                TrySendError::Full(envelope) => GraphicsCallbackIngressError {
                    event: envelope.event,
                    failure: GraphicsCallbackIngressFailure::Full,
                },
                TrySendError::Disconnected(envelope) => GraphicsCallbackIngressError {
                    event: envelope.event,
                    failure: GraphicsCallbackIngressFailure::Closed,
                },
            })?;
        if !self.is_live() {
            reply.reject_queued(GraphicsCallbackHostFailure::Retired);
        }
        self.admission_wake.notify_one();
        Ok(ticket)
    }
}

/// The registration adapter owns the exact callable and constructs its
/// documented `(source,event)` arguments on the session lane. This queue does
/// not invent Figure/Axes objects from numeric handles. An opaque handler may
/// capture its own runtime context; the queue cannot certify that capture as
/// an event's command-profiling origin.
pub type GraphicsCallbackHandler =
    dyn Fn(GraphicsCallbackHostEvent) -> RuntimeServiceFuture<Result<(), RuntimeError>>;

pub type GraphicsCallbackResponder =
    dyn Fn(
        GraphicsCallbackHostEvent,
    ) -> RuntimeServiceFuture<Result<GraphicsCallbackResult, RuntimeError>>;

pub type GraphicsBoundCallbackResponder =
    dyn Fn(
        GraphicsCallbackHostEvent,
        GraphicsCallbackRolePayload,
    ) -> RuntimeServiceFuture<Result<GraphicsCallbackResult, RuntimeError>>;

/// The invocation selection is supplied by the queue after event admission.
/// Persistent responders must not substitute their registration-time context.
pub type GraphicsSelectedBoundCallbackResponder =
    dyn Fn(
        RuntimeContext,
        GraphicsCallbackHostEvent,
        GraphicsCallbackRolePayload,
    ) -> RuntimeServiceFuture<Result<GraphicsCallbackResult, RuntimeError>>;

#[derive(Clone)]
enum RegisteredCallback {
    Metadata(Rc<GraphicsCallbackResponder>),
    Bound(Rc<GraphicsSelectedBoundCallbackResponder>),
}

impl From<Rc<GraphicsCallbackResponder>> for RegisteredCallback {
    fn from(value: Rc<GraphicsCallbackResponder>) -> Self {
        Self::Metadata(value)
    }
}

impl RegisteredCallback {
    fn representation(&self) -> IngressRepresentation {
        match self {
            Self::Metadata(_) => IngressRepresentation::Metadata,
            Self::Bound(_) => IngressRepresentation::Bound,
        }
    }
}

struct Registration {
    selected: RuntimeContext,
    callback: RegisteredCallback,
    figure: u32,
    kind: GraphicsCallbackKind,
    receiver: Option<GraphicsReceiverRef>,
    live: Arc<CallbackClaim>,
}

struct PropertyRegistration {
    receiver: GraphicsReceiverRef,
    kind: GraphicsCallbackKind,
    owner: GraphicsCallbackRegistration,
}

#[derive(Clone)]
enum InFlight {
    /// A checkpoint may suspend before the callback has been invoked. An
    /// abandoned drain leaves this event at the logical FIFO head for retry.
    Ready(Arc<Envelope>),
    /// The callback may already have produced effects. Abandonment must not
    /// invoke it again without an explicit host reconciliation decision.
    Invoking(Arc<Envelope>),
    Indeterminate(Arc<Envelope>),
}

impl InFlight {
    fn envelope(&self) -> Arc<Envelope> {
        match self {
            Self::Ready(envelope) | Self::Invoking(envelope) | Self::Indeterminate(envelope) => {
                Arc::clone(envelope)
            }
        }
    }
}

struct Inner {
    owner: Option<RuntimeContextLocalId>,
    receiver: RefCell<Receiver<Envelope>>,
    sender: SyncSender<Envelope>,
    closed: Arc<AtomicBool>,
    admission_wake: Arc<Notify>,
    registrations: RefCell<HashMap<u64, Registration>>,
    /// The session queue, not a process-global scene or a RuntimeContext
    /// service port, owns the selected callable's registration lifetime.
    property_registrations: RefCell<Vec<PropertyRegistration>>,
    maximum_registrations: NonZeroUsize,
    next_registration: Cell<u64>,
    driving: Cell<bool>,
    draining: Cell<bool>,
    in_flight: RefCell<Option<InFlight>>,
}

/// Strong owner belongs to the session, not a service port or GUI window.
#[derive(Clone)]
pub struct RuntimeGraphicsCallbackQueue(Rc<Inner>);

#[derive(Clone)]
pub struct WeakRuntimeGraphicsCallbackQueue(Weak<Inner>);

/// Recovery belongs to the callback producer, never diagnostic spelling.
/// Callback means that exact envelope was consumed with live authority;
/// it does not authorize invoking it again or replaying its effects.
#[derive(Debug, thiserror::Error)]
pub enum GraphicsCallbackDriveError {
    #[error("{0}")]
    Callback(#[source] RuntimeError),
    #[error("{0}")]
    Service(#[source] RuntimeError),
}

impl GraphicsCallbackDriveError {
    pub fn into_runtime_error(self) -> RuntimeError {
        match self {
            Self::Callback(error) | Self::Service(error) => error,
        }
    }
}

impl From<RuntimeError> for GraphicsCallbackDriveError {
    fn from(error: RuntimeError) -> Self {
        Self::Service(error)
    }
}

impl WeakRuntimeGraphicsCallbackQueue {
    pub fn upgrade(&self) -> Option<RuntimeGraphicsCallbackQueue> {
        self.0.upgrade().map(RuntimeGraphicsCallbackQueue)
    }

    pub(crate) fn points_to(&self, queue: &RuntimeGraphicsCallbackQueue) -> bool {
        self.0
            .upgrade()
            .is_some_and(|owner| Rc::ptr_eq(&owner, &queue.0))
    }
}

pub struct GraphicsCallbackRegistration {
    owner: Weak<Inner>,
    id: u64,
    live: Arc<CallbackClaim>,
}

impl Drop for GraphicsCallbackRegistration {
    fn drop(&mut self) {
        self.live.retire();
        if let Some(owner) = self.owner.upgrade() {
            let removed = { owner.registrations.borrow_mut().remove(&self.id) };
            // Callback/context destructors may reenter this session queue.
            drop(removed);
        }
    }
}

impl RuntimeGraphicsCallbackQueue {
    pub(crate) fn belongs_to(&self, selected: RuntimeContextLocalId) -> bool {
        self.0.owner == Some(selected) && !self.0.closed.load(Ordering::Acquire)
    }

    pub fn downgrade(&self) -> WeakRuntimeGraphicsCallbackQueue {
        WeakRuntimeGraphicsCallbackQueue(Rc::downgrade(&self.0))
    }

    pub fn new_for_selected(
        selected: &RuntimeContext,
        maximum_pending_events: NonZeroUsize,
        maximum_registrations: NonZeroUsize,
    ) -> Self {
        Self::new_with_owner(
            Some(selected.local_identity()),
            maximum_pending_events,
            maximum_registrations,
        )
    }

    #[cfg(test)]
    pub fn new(maximum_pending_events: NonZeroUsize, maximum_registrations: NonZeroUsize) -> Self {
        Self::new_with_owner(None, maximum_pending_events, maximum_registrations)
    }

    fn new_with_owner(
        owner: Option<RuntimeContextLocalId>,
        maximum_pending_events: NonZeroUsize,
        maximum_registrations: NonZeroUsize,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel(maximum_pending_events.get());
        Self(Rc::new(Inner {
            owner,
            receiver: RefCell::new(receiver),
            sender,
            closed: Arc::new(AtomicBool::new(false)),
            admission_wake: Arc::new(Notify::new()),
            registrations: RefCell::new(HashMap::new()),
            property_registrations: RefCell::new(Vec::new()),
            maximum_registrations,
            next_registration: Cell::new(1),
            driving: Cell::new(false),
            draining: Cell::new(false),
            in_flight: RefCell::new(None),
        }))
    }

    #[cfg(test)]
    pub fn register(
        &self,
        selected: RuntimeContext,
        figure: u32,
        kind: GraphicsCallbackKind,
        callback: Rc<GraphicsCallbackHandler>,
    ) -> Result<(GraphicsCallbackRegistration, GraphicsCallbackIngress), RuntimeError> {
        let responder: Rc<GraphicsCallbackResponder> = Rc::new(move |event| {
            let future = callback(event);
            Box::pin(async move { future.await.map(|()| GraphicsCallbackResult::Completed) })
        });
        self.register_inner(selected, figure, kind, None, responder)
    }

    #[cfg(test)]
    pub fn register_response(
        &self,
        selected: RuntimeContext,
        figure: u32,
        kind: GraphicsCallbackKind,
        callback: Rc<GraphicsCallbackResponder>,
    ) -> Result<(GraphicsCallbackRegistration, GraphicsCallbackIngress), RuntimeError> {
        self.register_inner(selected, figure, kind, None, callback)
    }

    /// Admit a production callback only for an exact live Figure lease and
    /// the selected session's own callback queue. The shared plot registry
    /// may be used by other sessions, but their callable roots may not be
    /// installed through this queue's selected execution context.
    pub fn register_graphics_response(
        &self,
        selected: RuntimeContext,
        receiver: GraphicsReceiverRef,
        kind: GraphicsCallbackKind,
        callback: Rc<GraphicsCallbackResponder>,
    ) -> Result<(GraphicsCallbackRegistration, GraphicsCallbackIngress), RuntimeError> {
        let figure = self.validate_graphics_source(&selected, &receiver)?;
        self.register_inner(selected, figure, kind, Some(receiver), callback)
    }

    fn validate_graphics_source(
        &self,
        selected: &RuntimeContext,
        receiver: &GraphicsReceiverRef,
    ) -> Result<u32, RuntimeError> {
        if self.is_closed() {
            return Err(build_runtime_error("graphics callback session has retired")
                .with_identifier("RunMat:graphics:CallbackSessionRetired")
                .build());
        }
        let GraphicsReceiverKind::Figure { figure } = receiver.kind() else {
            return Err(
                build_runtime_error("graphics callback source must be a Figure")
                    .with_identifier("RunMat:graphics:CallbackSource")
                    .build(),
            );
        };
        if !crate::plotting_hooks::graphics_receiver_valid(receiver) {
            return Err(build_runtime_error("graphics callback source is retired")
                .with_identifier("RunMat:graphics:CallbackSourceRetired")
                .build());
        }
        if self.0.owner != Some(selected.local_identity())
            || !selected
                .service_ports()
                .graphics_callback_queue()
                .is_some_and(|queue| queue.points_to(self))
        {
            return Err(build_runtime_error(
                "graphics callback selected context does not own this queue",
            )
            .with_identifier("RunMat:graphics:CallbackSessionMismatch")
            .build());
        }
        Ok(figure)
    }

    /// Replace exactly one live Figure property registration. The queue's
    /// bounded slot remains owned by the session; no old ingress can dispatch
    /// the replacement because queued envelopes carry the old claim identity.
    pub fn replace_graphics_property_response(
        &self,
        selected: RuntimeContext,
        receiver: GraphicsReceiverRef,
        kind: GraphicsCallbackKind,
        callback: Rc<GraphicsCallbackResponder>,
    ) -> Result<GraphicsCallbackIngress, RuntimeError> {
        self.replace_property(
            selected,
            receiver,
            kind,
            RegisteredCallback::Metadata(callback),
        )
    }

    pub fn replace_graphics_bound_property_response(
        &self,
        selected: RuntimeContext,
        receiver: GraphicsReceiverRef,
        kind: GraphicsCallbackKind,
        callback: Rc<GraphicsBoundCallbackResponder>,
    ) -> Result<GraphicsCallbackIngress, RuntimeError> {
        self.replace_selected_graphics_bound_property_response(
            selected,
            receiver,
            kind,
            Rc::new(move |_, event, payload| callback(event, payload)),
        )
    }

    pub fn replace_selected_graphics_bound_property_response(
        &self,
        selected: RuntimeContext,
        receiver: GraphicsReceiverRef,
        kind: GraphicsCallbackKind,
        callback: Rc<GraphicsSelectedBoundCallbackResponder>,
    ) -> Result<GraphicsCallbackIngress, RuntimeError> {
        self.replace_property(
            selected,
            receiver,
            kind,
            RegisteredCallback::Bound(callback),
        )
    }

    fn replace_property(
        &self,
        selected: RuntimeContext,
        receiver: GraphicsReceiverRef,
        kind: GraphicsCallbackKind,
        callback: RegisteredCallback,
    ) -> Result<GraphicsCallbackIngress, RuntimeError> {
        self.replace_property_transaction(selected, receiver, kind, callback, Ok)
    }

    /// The sealed publisher may only replace the prepared Figure property slot.
    /// It cannot await, invoke user code, or touch this queue.
    #[cfg(feature = "plot-core")]
    pub(crate) fn replace_selected_property_transaction<T>(
        &self,
        selected: RuntimeContext,
        receiver: GraphicsReceiverRef,
        kind: GraphicsCallbackKind,
        callback: Rc<GraphicsSelectedBoundCallbackResponder>,
        publish: impl FnOnce(GraphicsCallbackIngress) -> Result<T, RuntimeError>,
    ) -> Result<T, RuntimeError> {
        self.replace_property_transaction(
            selected,
            receiver,
            kind,
            RegisteredCallback::Bound(callback),
            publish,
        )
    }

    /// Clearing a property revokes its exact claim and releases callable roots
    /// without affecting a different Figure whose number was later reused.
    pub fn clear_graphics_property_response(
        &self,
        receiver: &GraphicsReceiverRef,
        kind: GraphicsCallbackKind,
    ) -> bool {
        let old = {
            let mut properties = self.0.property_registrations.borrow_mut();
            properties
                .iter()
                .position(|property| {
                    property.kind == kind && property.receiver.shares_lease(receiver)
                })
                .map(|index| properties.swap_remove(index))
        };
        old.is_some()
    }

    /// A global Figure can retire without borrowing a session-local queue.
    /// The next session-lane admission/drain releases those exact old roots;
    /// a reused numeric figure never inherits their registration slot.
    fn prune_retired_graphics_properties(&self) {
        let candidates = {
            let properties = self.0.property_registrations.borrow();
            properties
                .iter()
                .map(|property| {
                    (
                        property.receiver.clone(),
                        property.kind,
                        property.owner.live.is_live(),
                    )
                })
                .collect::<Vec<_>>()
        };
        for (receiver, kind, live) in candidates {
            if !live || !crate::plotting_hooks::graphics_receiver_valid(&receiver) {
                self.clear_graphics_property_response(&receiver, kind);
            }
        }
    }

    fn register_inner(
        &self,
        selected: RuntimeContext,
        figure: u32,
        kind: GraphicsCallbackKind,
        receiver: Option<GraphicsReceiverRef>,
        callback: impl Into<RegisteredCallback>,
    ) -> Result<(GraphicsCallbackRegistration, GraphicsCallbackIngress), RuntimeError> {
        let callback = callback.into();
        let representation = callback.representation();
        if self.is_closed() {
            return Err(build_runtime_error("graphics callback session has retired")
                .with_identifier("RunMat:graphics:CallbackSessionRetired")
                .build());
        }
        if self.0.registrations.borrow().len() >= self.0.maximum_registrations.get() {
            return Err(
                build_runtime_error("graphics callback registration capacity exhausted")
                    .with_identifier("RunMat:graphics:CallbackRegistrationCapacity")
                    .build(),
            );
        }
        let id = self.0.next_registration.get();
        let next = id.checked_add(1).ok_or_else(|| {
            build_runtime_error("graphics callback identity space exhausted")
                .with_identifier("RunMat:graphics:CallbackIdentityExhausted")
                .build()
        })?;
        self.0.next_registration.set(next);
        let live = Arc::new(CallbackClaim::default());
        self.0.registrations.borrow_mut().insert(
            id,
            Registration {
                // Registration-time command accounting must not be replayed
                // by an unrelated host gesture or a later drain command.
                selected: selected
                    .with_cleanup_command_origin(None)
                    .with_graphics_callback_mutation(None),
                callback,
                figure,
                kind,
                receiver,
                live: live.clone(),
            },
        );
        Ok((
            GraphicsCallbackRegistration {
                owner: Rc::downgrade(&self.0),
                id,
                live: live.clone(),
            },
            GraphicsCallbackIngress {
                representation,
                registration: id,
                figure,
                kind,
                live,
                closed: Arc::clone(&self.0.closed),
                sender: self.0.sender.clone(),
                admission_wake: Arc::clone(&self.0.admission_wake),
            },
        ))
    }

    /// `nocallbacks`/`update` leave received ingress queued.
    /// A later Process drains all work admitted before its terminal empty check,
    /// including callbacks added by another callback. The host must order its
    /// frame receipt after all input events that the flush promises to drain.
    pub fn defer(&self) {}

    pub async fn process(&self) -> Result<(), RuntimeError> {
        self.process_classified()
            .await
            .map_err(GraphicsCallbackDriveError::into_runtime_error)
    }

    pub(crate) async fn process_classified(&self) -> Result<(), GraphicsCallbackDriveError> {
        if self.is_closed() {
            return Err(build_runtime_error("graphics callback session has retired")
                .with_identifier("RunMat:graphics:CallbackSessionRetired")
                .build()
                .into());
        }
        if self.0.draining.replace(true) {
            return Err(
                build_runtime_error("graphics callback queue is already draining")
                    .with_identifier("RunMat:graphics:CallbackDrainReentrant")
                    .build()
                    .into(),
            );
        }
        struct DrainGuard<'a>(&'a Inner);
        impl Drop for DrainGuard<'_> {
            fn drop(&mut self) {
                let mut in_flight = self.0.in_flight.borrow_mut();
                if let Some(InFlight::Invoking(envelope)) = in_flight.as_ref() {
                    *in_flight = Some(InFlight::Indeterminate(Arc::clone(envelope)));
                }
                self.0.draining.set(false);
            }
        }
        let _guard = DrainGuard(&self.0);
        self.prune_retired_graphics_properties();
        loop {
            if self.is_closed() {
                return Err(build_runtime_error("graphics callback session has retired")
                    .with_identifier("RunMat:graphics:CallbackSessionRetired")
                    .build()
                    .into());
            }
            let current = self.0.in_flight.borrow().clone();
            let envelope = match current {
                Some(InFlight::Indeterminate(envelope)) => {
                    return Err(build_runtime_error(format!(
                        "graphics callback may have run before drain abandonment (registration {}, figure {}, kind {:?})",
                        envelope.registration, envelope.event.figure, envelope.event.kind
                    ))
                    .with_identifier("RunMat:graphics:CallbackIndeterminate")
                    .build().into());
                }
                Some(in_flight) => in_flight.envelope(),
                None => {
                    let received = self.0.receiver.borrow().try_recv();
                    match received {
                        Ok(envelope) => {
                            let envelope = Arc::new(envelope);
                            *self.0.in_flight.borrow_mut() =
                                Some(InFlight::Ready(Arc::clone(&envelope)));
                            envelope
                        }
                        Err(TryRecvError::Empty | TryRecvError::Disconnected) => return Ok(()),
                    }
                }
            };
            let selected = {
                let registrations = self.0.registrations.borrow();
                registrations
                    .get(&envelope.registration)
                    .and_then(|registration| {
                        (registration.figure == envelope.event.figure
                            && registration.kind == envelope.event.kind
                            && Arc::ptr_eq(&registration.live, &envelope.claim)
                            && registration.live.is_live())
                        .then_some(registration)
                        .filter(|registration| {
                            registration.receiver.as_ref().is_none_or(|receiver| {
                                crate::plotting_hooks::graphics_receiver_valid(receiver)
                            })
                        })
                        .map(|registration| {
                            (
                                registration.selected.clone(),
                                registration.callback.clone(),
                                Arc::clone(&registration.live),
                                registration.receiver.clone(),
                            )
                        })
                    })
            };
            let Some((selected, callback, claim, receiver)) = selected else {
                // A retired registration cannot be resurrected by a late host
                // event, even when its numeric figure handle has been reused.
                if let Some(reply) = &envelope.reply {
                    reply.reject_queued(GraphicsCallbackHostFailure::Retired);
                }
                *self.0.in_flight.borrow_mut() = None;
                continue;
            };
            if envelope
                .reply
                .as_ref()
                .is_some_and(|reply| reply.is_retired())
            {
                *self.0.in_flight.borrow_mut() = None;
                continue;
            }
            let mut cooperation = selected.cpu_cooperation();
            cooperation
                .checkpoint(1)
                .await
                .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
            let payload_valid = || match &callback {
                RegisteredCallback::Metadata(_) => envelope.payload.is_none(),
                RegisteredCallback::Bound(_) => receiver
                    .as_ref()
                    .zip(envelope.payload.as_ref())
                    .is_some_and(|(receiver, payload)| {
                        crate::plotting_hooks::graphics_callback_payload_valid(
                            receiver,
                            envelope.event.kind,
                            payload,
                        )
                    }),
            };
            if !payload_valid() {
                if let Some(reply) = &envelope.reply {
                    reply.reject_queued(GraphicsCallbackHostFailure::Retired);
                }
                *self.0.in_flight.borrow_mut() = None;
                continue;
            }
            // The owner may replace or retire this registration while the
            // cooperation turn is pending. Starting an invocation and
            // retiring its claim have one atomic winner.
            let Some(mut invocation) = claim.begin() else {
                if let Some(reply) = &envelope.reply {
                    reply.reject_queued(GraphicsCallbackHostFailure::Retired);
                }
                *self.0.in_flight.borrow_mut() = None;
                continue;
            };
            if let Some(reply) = envelope.reply.as_ref() {
                if !reply.begin() {
                    invocation.abort_before_invocation();
                    *self.0.in_flight.borrow_mut() = None;
                    continue;
                }
            }
            *self.0.in_flight.borrow_mut() = Some(InFlight::Invoking(Arc::clone(&envelope)));
            let event = envelope.event;
            let mutation =
                receiver
                    .as_ref()
                    .zip(envelope.payload.as_ref())
                    .map(|(receiver, payload)| {
                        GraphicsCallbackMutationScope::begin(
                            receiver.clone(),
                            payload.figure_and_scene().1.clone(),
                        )
                    });
            let callback_selected = selected
                .clone()
                .with_graphics_callback_mutation(mutation.as_ref().map(|(scope, _)| scope.clone()));
            let result = callback_selected
                .scope(async {
                    match &callback {
                        RegisteredCallback::Metadata(callback) => callback(event).await,
                        RegisteredCallback::Bound(callback) => {
                            callback(
                                callback_selected.clone(),
                                event,
                                envelope
                                    .payload
                                    .as_ref()
                                    .expect("bound payload admitted")
                                    .clone(),
                            )
                            .await
                        }
                    }
                })
                .await;
            *self.0.in_flight.borrow_mut() = None;
            let payload_after_valid = mutation
                .as_ref()
                .zip(receiver.as_ref())
                .zip(envelope.payload.as_ref())
                .is_some_and(|(((scope, _), receiver), payload)| {
                    scope
                        .final_scene(receiver, payload.figure_and_scene().1)
                        .is_some_and(|scene| {
                            crate::plotting_hooks::graphics_callback_payload_valid_at_scene(
                                receiver,
                                envelope.event.kind,
                                payload,
                                &scene,
                            )
                        })
                });
            if callback.representation() == IngressRepresentation::Bound
                && (!payload_after_valid
                    || !claim.is_live()
                    || self.is_closed()
                    || selected.is_cancelled())
            {
                // Effects may already have occurred. Retire the ticket without
                // publishing success or replacing the callback's primary error.
                if let Some(reply) = &envelope.reply {
                    reply.close_started();
                }
                *self.0.in_flight.borrow_mut() =
                    Some(InFlight::Indeterminate(Arc::clone(&envelope)));
                return match result {
                    Err(error) => Err(error.into()),
                    Ok(_) => Err(build_runtime_error(
                        "graphics callback authority changed after invocation",
                    )
                    .with_identifier("RunMat:graphics:CallbackIndeterminate")
                    .build()
                    .into()),
                };
            }
            invocation.terminal();
            drop(mutation);
            match result {
                Ok(value) => {
                    let valid = matches!(
                        (envelope.event.kind, &value),
                        (
                            GraphicsCallbackKind::PanButtonDownFilter
                                | GraphicsCallbackKind::ZoomButtonDownFilter,
                            GraphicsCallbackResult::Suppress(_)
                        ) | (
                            GraphicsCallbackKind::DataCursorUpdate,
                            GraphicsCallbackResult::DataTipText(_)
                        ) | (
                            GraphicsCallbackKind::PanActionPre
                                | GraphicsCallbackKind::PanActionPost
                                | GraphicsCallbackKind::ZoomActionPre
                                | GraphicsCallbackKind::ZoomActionPost,
                            GraphicsCallbackResult::Completed
                        )
                    );
                    if envelope.reply.is_some() && !valid {
                        if let Some(reply) = &envelope.reply {
                            reply.complete(Err(GraphicsCallbackHostFailure::InvalidResult));
                        }
                        let error = build_runtime_error(
                            "graphics callback returned a result incompatible with its event kind",
                        )
                        .with_identifier("RunMat:graphics:CallbackResult")
                        .build();
                        return Err(
                            if self.callback_scope_live(&selected, &claim, receiver.as_ref()) {
                                GraphicsCallbackDriveError::Callback(error)
                            } else {
                                GraphicsCallbackDriveError::Service(error)
                            },
                        );
                    }
                    if let Some(reply) = &envelope.reply {
                        reply.complete(Ok(value));
                    }
                }
                Err(error) => {
                    if let Some(reply) = &envelope.reply {
                        reply.complete(Err(GraphicsCallbackHostFailure::CallbackFailed));
                    }
                    let local = error.failure_origin()
                        == crate::runtime_error::RuntimeFailureOrigin::LanguageEvaluation
                        && self.callback_scope_live(&selected, &claim, receiver.as_ref());
                    return Err(if local {
                        GraphicsCallbackDriveError::Callback(error)
                    } else {
                        GraphicsCallbackDriveError::Service(error)
                    });
                }
            }
        }
    }

    fn callback_scope_live(
        &self,
        selected: &RuntimeContext,
        claim: &CallbackClaim,
        receiver: Option<&GraphicsReceiverRef>,
    ) -> bool {
        !self.is_closed()
            && !selected.is_cancelled()
            && claim.is_live()
            && receiver.is_none_or(crate::plotting_hooks::graphics_receiver_valid)
    }
}

#[cfg(test)]
#[path = "graphics_callbacks/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "graphics_callbacks/profiling_tests.rs"]
mod profiling_tests;

#[cfg(all(test, feature = "plot-core"))]
#[path = "graphics_callbacks/mutation_tests.rs"]
mod mutation_tests;
