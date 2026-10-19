//! Lazy flush state machine; tickets remain inspectable during cancellation.

use super::{error, Inner, OwnedWindow, Scene};
use crate::builtins::plotting::state;
use crate::context::{
    GraphicsCallbackDriveError, GraphicsCallbackPolicy, GraphicsCallbackReceipt,
    GraphicsFlushDisposition, GraphicsFlushRate, GraphicsFlushReceipt, GraphicsFlushRequest,
    GraphicsPresentation, RuntimeGraphicsFlushOperation,
};
use crate::RuntimeError;
use runmat_plot::gui::frame_command::{FrameDisposition, FrameTicket};
use runmat_plot::gui::host_loop::{WindowOpenDisposition, WindowOpenTicket};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

enum Phase {
    Initial,
    Next,
    Opening(Scene, WindowOpenTicket),
    Presenting(Scene, FrameTicket),
    Callbacks(Pin<Box<dyn Future<Output = Result<(), GraphicsCallbackDriveError>>>>),
    Done,
}

pub(super) struct Flush {
    owner: Rc<Inner>,
    request: GraphicsFlushRequest,
    phase: Phase,
    scenes: VecDeque<Scene>,
    disposition: GraphicsFlushDisposition,
    presentation: GraphicsPresentation,
    owns_busy: bool,
    selected: bool,
}

impl Flush {
    pub(super) fn new(owner: Rc<Inner>, request: GraphicsFlushRequest) -> Self {
        Self {
            owner,
            request,
            phase: Phase::Initial,
            scenes: VecDeque::new(),
            disposition: GraphicsFlushDisposition::NotPublished,
            presentation: GraphicsPresentation::NoFigure,
            owns_busy: false,
            selected: false,
        }
    }

    pub(super) fn selected(owner: Rc<Inner>, scene: Scene) -> Self {
        let mut operation = Self::new(
            owner,
            GraphicsFlushRequest {
                rate: GraphicsFlushRate::Immediate,
                callbacks: GraphicsCallbackPolicy::Defer,
            },
        );
        operation.scenes.push_back(scene);
        operation.selected = true;
        operation
    }

    fn observe(&mut self, disposition: GraphicsFlushDisposition) {
        self.disposition = match (self.disposition, disposition) {
            (GraphicsFlushDisposition::Published, _) | (_, GraphicsFlushDisposition::Published) => {
                GraphicsFlushDisposition::Published
            }
            (GraphicsFlushDisposition::Indeterminate, _)
            | (_, GraphicsFlushDisposition::Indeterminate) => {
                GraphicsFlushDisposition::Indeterminate
            }
            _ => GraphicsFlushDisposition::NotPublished,
        };
    }

    fn next_scene(&mut self, scene: Scene) -> Result<Phase, RuntimeError> {
        if !state::figure_scene_is_current(&scene.receiver, &scene.snapshot.scene_lease) {
            if self.selected {
                return Err(error("selected presentation scene has retired"));
            }
            return Ok(Phase::Next);
        }
        let handle = scene.snapshot.handle.as_u32();
        let mut windows = self.owner.windows.borrow_mut();
        let replaced = windows
            .get(&handle)
            .is_some_and(|window| !window.receiver.shares_lease(&scene.receiver));
        let removed = if replaced || !scene.snapshot.figure.visible {
            windows.remove(&handle)
        } else {
            None
        };
        if !scene.snapshot.figure.visible {
            drop(windows);
            if removed.is_some() {
                self.observe(GraphicsFlushDisposition::Indeterminate);
            }
            drop(removed);
            return Ok(Phase::Next);
        }
        if let Some(window) = windows.get(&handle) {
            let frame = window
                .receipt
                .frame_port()
                .present(scene.snapshot.figure.as_ref().clone());
            return Ok(Phase::Presenting(scene, frame));
        }
        drop(windows);
        if removed.is_some() {
            self.observe(GraphicsFlushDisposition::Indeterminate);
        }
        drop(removed);
        let mut config = self.owner.window_config.clone();
        config.title = format!("RunMat Figure {handle}");
        let ticket = self.owner.opener.open(handle, config);
        Ok(Phase::Opening(scene, ticket))
    }

    fn callbacks(&mut self) -> Result<Phase, RuntimeError> {
        // Frames have settled; callbacks may present their own admitted scene.
        // The callback queue separately owns drain/reentrancy authority.
        if self.owns_busy {
            self.owner.busy.set(false);
            self.owns_busy = false;
        }
        match self.request.callbacks {
            GraphicsCallbackPolicy::Defer => Ok(Phase::Done),
            GraphicsCallbackPolicy::Process => {
                let queue = self
                    .owner
                    .callbacks
                    .upgrade()
                    .ok_or_else(|| error("graphics callback owner has retired"))?;
                Ok(Phase::Callbacks(Box::pin(async move {
                    queue.process_classified().await
                })))
            }
        }
    }
}

impl Future for Flush {
    type Output = Result<GraphicsFlushReceipt, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            if self.owner.closed.get() {
                return Poll::Ready(Err(error("native graphics attachment has retired")));
            }
            let phase = std::mem::replace(&mut self.phase, Phase::Done);
            self.phase = match phase {
                Phase::Initial => {
                    if self.request.callbacks == GraphicsCallbackPolicy::Process
                        && self
                            .owner
                            .callbacks
                            .upgrade()
                            .is_none_or(|queue| queue.is_closed())
                    {
                        return Poll::Ready(Err(error("graphics callback owner has retired")));
                    }
                    if self.owner.busy.get() {
                        if self.request.rate != GraphicsFlushRate::WhenIdle {
                            return Poll::Ready(Err(error(
                                "native presentation is already flushing",
                            )));
                        }
                        self.presentation = GraphicsPresentation::DeferredByBusy;
                        self.callbacks()?
                    } else if self.request.rate == GraphicsFlushRate::LimitRate
                        && self
                            .owner
                            .last_presented
                            .get()
                            .is_some_and(|last| last.elapsed() < Duration::from_millis(50))
                    {
                        self.presentation = GraphicsPresentation::DeferredByRate;
                        self.callbacks()?
                    } else {
                        self.owner.busy.set(true);
                        self.owns_busy = true;
                        if !self.selected {
                            self.scenes = self.owner.capture_scenes()?;
                        }
                        if self.owner.prune_retired_windows() {
                            self.observe(GraphicsFlushDisposition::Indeterminate);
                        }
                        Phase::Next
                    }
                }
                Phase::Next => match self.scenes.pop_front() {
                    Some(scene) => self.next_scene(scene)?,
                    None => self.callbacks()?,
                },
                Phase::Opening(scene, mut ticket) => {
                    let result = Pin::new(&mut ticket).poll(context);
                    if ticket.disposition() != WindowOpenDisposition::NotOpened {
                        self.observe(GraphicsFlushDisposition::Indeterminate);
                    }
                    match result {
                        Poll::Pending => {
                            self.phase = Phase::Opening(scene, ticket);
                            return Poll::Pending;
                        }
                        Poll::Ready(Err(source)) => {
                            return Poll::Ready(Err(super::source_error(
                                "native window open failed",
                                source,
                            )))
                        }
                        Poll::Ready(Ok(receipt)) => {
                            if !state::figure_scene_is_current(
                                &scene.receiver,
                                &scene.snapshot.scene_lease,
                            ) {
                                drop(receipt);
                                if self.selected {
                                    return Poll::Ready(Err(error(
                                        "selected presentation scene retired while opening",
                                    )));
                                }
                                Phase::Next
                            } else {
                                let frame = receipt
                                    .frame_port()
                                    .present(scene.snapshot.figure.as_ref().clone());
                                self.owner.windows.borrow_mut().insert(
                                    scene.snapshot.handle.as_u32(),
                                    OwnedWindow {
                                        receiver: scene.receiver.clone(),
                                        receipt,
                                    },
                                );
                                Phase::Presenting(scene, frame)
                            }
                        }
                    }
                }
                Phase::Presenting(scene, mut ticket) => {
                    let result = Pin::new(&mut ticket).poll(context);
                    let disposition = match ticket.disposition() {
                        FrameDisposition::NotPublished => GraphicsFlushDisposition::NotPublished,
                        FrameDisposition::Indeterminate => GraphicsFlushDisposition::Indeterminate,
                        FrameDisposition::Published => GraphicsFlushDisposition::Published,
                    };
                    self.observe(disposition);
                    match result {
                        Poll::Pending => {
                            self.phase = Phase::Presenting(scene, ticket);
                            return Poll::Pending;
                        }
                        Poll::Ready(Err(source)) => {
                            self.owner
                                .windows
                                .borrow_mut()
                                .remove(&scene.snapshot.handle.as_u32());
                            return Poll::Ready(Err(super::source_error(
                                "native frame presentation failed",
                                source,
                            )));
                        }
                        Poll::Ready(Ok(())) => {
                            if self.selected
                                && !state::figure_scene_is_current(
                                    &scene.receiver,
                                    &scene.snapshot.scene_lease,
                                )
                            {
                                return Poll::Ready(Err(error(
                                    "selected presentation scene retired after publication",
                                )));
                            }
                            self.presentation = GraphicsPresentation::Presented;
                            self.owner.last_presented.set(Some(Instant::now()));
                            Phase::Next
                        }
                    }
                }
                Phase::Callbacks(mut future) => {
                    self.observe(GraphicsFlushDisposition::Indeterminate);
                    match future.as_mut().poll(context) {
                        Poll::Pending => {
                            self.phase = Phase::Callbacks(future);
                            return Poll::Pending;
                        }
                        Poll::Ready(Err(source)) => {
                            return Poll::Ready(Err(super::source_error(
                                "native graphics callback pump failed",
                                source,
                            )))
                        }
                        Poll::Ready(Ok(())) => {
                            self.observe(GraphicsFlushDisposition::Published);
                            Phase::Done
                        }
                    }
                }
                Phase::Done => {
                    return Poll::Ready(Ok(GraphicsFlushReceipt {
                        presentation: self.presentation,
                        callbacks: match self.request.callbacks {
                            GraphicsCallbackPolicy::Process => GraphicsCallbackReceipt::Processed,
                            GraphicsCallbackPolicy::Defer => GraphicsCallbackReceipt::Deferred,
                        },
                    }))
                }
            };
        }
    }
}

impl RuntimeGraphicsFlushOperation for Flush {
    fn disposition(&self) -> GraphicsFlushDisposition {
        self.disposition
    }

    fn abandon(mut self: Pin<&mut Self>) -> GraphicsFlushDisposition {
        match &self.phase {
            Phase::Opening(_, ticket) => {
                if ticket.abandon() != WindowOpenDisposition::NotOpened {
                    self.observe(GraphicsFlushDisposition::Indeterminate);
                }
            }
            Phase::Presenting(_, ticket) => {
                let next = match ticket.abandon() {
                    FrameDisposition::NotPublished => GraphicsFlushDisposition::NotPublished,
                    FrameDisposition::Indeterminate => GraphicsFlushDisposition::Indeterminate,
                    FrameDisposition::Published => GraphicsFlushDisposition::Published,
                };
                self.observe(next);
            }
            _ => {}
        }
        self.phase = Phase::Done;
        self.scenes.clear();
        self.disposition
    }
}

impl Drop for Flush {
    fn drop(&mut self) {
        if self.owns_busy {
            self.owner.busy.set(false);
        }
    }
}
