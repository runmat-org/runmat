//! Session-local graphics callback ingress and ordered drain.
//!
//! GUI/browser producers receive only a `Send` metadata port. Callable values,
//! selected invocation context, and workspaces remain on the session lane.
//! The Core session must own this queue strongly; an installed presentation
//! service may refer to it only weakly, because registrations retain selected
//! contexts whose service ports can point back to presentation.

use super::{RuntimeContext, RuntimeContextLocalId, RuntimeServiceFuture};
use crate::{build_runtime_error, RuntimeError};
pub use runmat_value::GraphicsCallbackIngressFailure;
pub use runmat_value::{GraphicsCallbackKind, GraphicsCallbackResult};
use runmat_value::{GraphicsReceiverKind, GraphicsReceiverRef};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::rc::{Rc, Weak};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;

#[path = "graphics_callbacks/claim.rs"]
mod claim;
use claim::CallbackClaim;
#[path = "graphics_callbacks/host_request.rs"]
mod host_request;
pub use host_request::GraphicsCallbackHostTicket;
use host_request::HostReply;
pub use runmat_value::{GraphicsCallbackDisposition, GraphicsCallbackHostFailure};

/// Host metadata only. The session-side adapter constructs callable arguments
/// under the originating selected context; a numeric figure is never treated
/// as a session capability or a native Figure/Axes object.
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
    reply: Option<Arc<HostReply>>,
}

/// This is the sole part that may cross the GUI/worker boundary.
#[derive(Clone)]
pub struct GraphicsCallbackIngress {
    registration: u64,
    figure: u32,
    kind: GraphicsCallbackKind,
    live: Arc<CallbackClaim>,
    sender: SyncSender<Envelope>,
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
        self.live.is_live()
    }

    /// Never block a GUI/DOM event lane. Full is a typed backpressure result;
    /// the host must retain or report this event, never silently discard it.
    pub fn send(
        &self,
        event: GraphicsCallbackHostEvent,
    ) -> Result<(), GraphicsCallbackIngressError> {
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
            })
    }

    /// Submit a result-bearing host gesture without blocking the UI lane.
    /// The returned ticket is lazy only with respect to its acknowledgement:
    /// the request itself is admitted here, and Drop can retire it before the
    /// session callback begins.
    pub fn request(
        &self,
        event: GraphicsCallbackHostEvent,
    ) -> Result<GraphicsCallbackHostTicket, GraphicsCallbackIngressError> {
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
        Ok(ticket)
    }
}

/// The registration adapter owns the exact callable and constructs its
/// documented `(source,event)` arguments on the session lane. This queue does
/// not invent Figure/Axes objects from numeric handles.
pub type GraphicsCallbackHandler =
    dyn Fn(GraphicsCallbackHostEvent) -> RuntimeServiceFuture<Result<(), RuntimeError>>;

pub type GraphicsCallbackResponder =
    dyn Fn(
        GraphicsCallbackHostEvent,
    ) -> RuntimeServiceFuture<Result<GraphicsCallbackResult, RuntimeError>>;

struct Registration {
    selected: RuntimeContext,
    callback: Rc<GraphicsCallbackResponder>,
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
    registrations: RefCell<HashMap<u64, Registration>>,
    /// The session queue, not a process-global scene or a RuntimeContext
    /// service port, owns the selected callable's registration lifetime.
    property_registrations: RefCell<Vec<PropertyRegistration>>,
    maximum_registrations: NonZeroUsize,
    next_registration: Cell<u64>,
    draining: Cell<bool>,
    in_flight: RefCell<Option<InFlight>>,
}

/// Strong owner belongs to the session, not a service port or GUI window.
#[derive(Clone)]
pub struct RuntimeGraphicsCallbackQueue(Rc<Inner>);

#[derive(Clone)]
pub struct WeakRuntimeGraphicsCallbackQueue(Weak<Inner>);

impl WeakRuntimeGraphicsCallbackQueue {
    pub fn upgrade(&self) -> Option<RuntimeGraphicsCallbackQueue> {
        self.0.upgrade().map(RuntimeGraphicsCallbackQueue)
    }

    fn points_to(&self, queue: &RuntimeGraphicsCallbackQueue) -> bool {
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
            registrations: RefCell::new(HashMap::new()),
            property_registrations: RefCell::new(Vec::new()),
            maximum_registrations,
            next_registration: Cell::new(1),
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
        let GraphicsReceiverKind::Figure { figure } = receiver.kind() else {
            return Err(
                build_runtime_error("graphics callback source must be a Figure")
                    .with_identifier("RunMat:graphics:CallbackSource")
                    .build(),
            );
        };
        if !crate::builtins::plotting::state::graphics_receiver_valid(receiver) {
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
        let figure = self.validate_graphics_source(&selected, &receiver)?;
        self.prune_retired_graphics_properties();
        let mut properties = self.0.property_registrations.borrow_mut();
        if let Some(property) = properties
            .iter_mut()
            .find(|property| property.kind == kind && property.receiver.shares_lease(&receiver))
        {
            let id = property.owner.id;
            if !self.0.registrations.borrow().contains_key(&id) {
                return Err(
                    build_runtime_error("graphics callback property owner is retired")
                        .with_identifier("RunMat:graphics:CallbackSourceRetired")
                        .build(),
                );
            }
            let new_live = Arc::new(CallbackClaim::default());
            let old_live = std::mem::replace(&mut property.owner.live, Arc::clone(&new_live));
            let old = self.0.registrations.borrow_mut().insert(
                id,
                Registration {
                    selected,
                    callback,
                    figure,
                    kind,
                    receiver: Some(receiver),
                    live: Arc::clone(&new_live),
                },
            );
            debug_assert!(old.is_some(), "property slot has one registration owner");
            // Dropping the old callback or selected context can reenter queue
            // ownership. Neither RefCell guard may span that destruction.
            drop(properties);
            old_live.retire();
            drop(old);
            return Ok(GraphicsCallbackIngress {
                registration: id,
                figure,
                kind,
                live: new_live,
                sender: self.0.sender.clone(),
            });
        }
        drop(properties);
        let (owner, ingress) =
            self.register_graphics_response(selected, receiver.clone(), kind, callback)?;
        self.0
            .property_registrations
            .borrow_mut()
            .push(PropertyRegistration {
                receiver,
                kind,
                owner,
            });
        Ok(ingress)
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
                .map(|property| (property.receiver.clone(), property.kind))
                .collect::<Vec<_>>()
        };
        for (receiver, kind) in candidates {
            if !crate::builtins::plotting::state::graphics_receiver_valid(&receiver) {
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
        callback: Rc<GraphicsCallbackResponder>,
    ) -> Result<(GraphicsCallbackRegistration, GraphicsCallbackIngress), RuntimeError> {
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
                selected,
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
                registration: id,
                figure,
                kind,
                live,
                sender: self.0.sender.clone(),
            },
        ))
    }

    /// `nocallbacks`/`update` leave received ingress queued.
    /// A later Process drains all work admitted before its terminal empty check,
    /// including callbacks added by another callback. The host must order its
    /// frame receipt after all input events that the flush promises to drain.
    pub fn defer(&self) {}

    pub async fn process(&self) -> Result<(), RuntimeError> {
        if self.0.draining.replace(true) {
            return Err(
                build_runtime_error("graphics callback queue is already draining")
                    .with_identifier("RunMat:graphics:CallbackDrainReentrant")
                    .build(),
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
            let current = self.0.in_flight.borrow().clone();
            let envelope = match current {
                Some(InFlight::Indeterminate(envelope)) => {
                    return Err(build_runtime_error(format!(
                        "graphics callback may have run before drain abandonment (registration {}, figure {}, kind {:?})",
                        envelope.registration, envelope.event.figure, envelope.event.kind
                    ))
                    .with_identifier("RunMat:graphics:CallbackIndeterminate")
                    .build());
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
                                crate::builtins::plotting::state::graphics_receiver_valid(receiver)
                            })
                        })
                        .map(|registration| {
                            (
                                registration.selected.clone(),
                                registration.callback.clone(),
                                Arc::clone(&registration.live),
                            )
                        })
                    })
            };
            let Some((selected, callback, claim)) = selected else {
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
                    invocation.terminal();
                    *self.0.in_flight.borrow_mut() = None;
                    continue;
                }
            }
            *self.0.in_flight.borrow_mut() = Some(InFlight::Invoking(Arc::clone(&envelope)));
            let event = envelope.event;
            let result = selected.scope(async move { callback(event).await }).await;
            invocation.terminal();
            *self.0.in_flight.borrow_mut() = None;
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
                        return Err(build_runtime_error(
                            "graphics callback returned a result incompatible with its event kind",
                        )
                        .with_identifier("RunMat:graphics:CallbackResult")
                        .build());
                    }
                    if let Some(reply) = &envelope.reply {
                        reply.complete(Ok(value));
                    }
                }
                Err(error) => {
                    if let Some(reply) = &envelope.reply {
                        reply.complete(Err(GraphicsCallbackHostFailure::CallbackFailed));
                    }
                    return Err(error);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "graphics_callbacks/tests.rs"]
mod tests;
