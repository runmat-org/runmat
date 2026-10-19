//! Session-lane bridge to the OS-main-thread plot host. The existing plot
//! domain is explicitly SharedGlobalRegistry, not a session-private registry.

use super::{
    GraphicsFlushRequest, GraphicsSceneStaging, RuntimeGraphicsFlushOperation,
    RuntimeGraphicsPresentationService, WeakRuntimeGraphicsCallbackQueue,
};
use crate::builtins::plotting::state::{self, FigureSnapshotEntry};
use crate::{build_runtime_error, RuntimeError};
use runmat_plot::gui::host_loop::{WindowOpenPort, WindowOpenReceipt};
use runmat_value::{GraphicsReceiverKind, GraphicsReceiverRef};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::rc::Rc;
use std::time::Instant;

mod operation;

pub(super) struct Scene {
    receiver: GraphicsReceiverRef,
    snapshot: FigureSnapshotEntry,
}

pub(super) struct OwnedWindow {
    receiver: GraphicsReceiverRef,
    receipt: WindowOpenReceipt,
}

pub(super) struct Inner {
    opener: WindowOpenPort,
    callbacks: WeakRuntimeGraphicsCallbackQueue,
    pending: RefCell<BTreeMap<u32, GraphicsReceiverRef>>,
    windows: RefCell<BTreeMap<u32, OwnedWindow>>,
    maximum_figures: NonZeroUsize,
    maximum_snapshot_bytes: NonZeroUsize,
    window_config: runmat_plot::gui::WindowConfig,
    busy: Cell<bool>,
    closed: Cell<bool>,
    last_presented: Cell<Option<Instant>>,
}

/// Explicit local host attachment. This cannot be serialized into a worker
/// request; the owning session must install it and retire it on shutdown.
pub struct NativeGraphicsAttachment(Rc<Inner>);

impl NativeGraphicsAttachment {
    pub fn new(
        opener: WindowOpenPort,
        callbacks: WeakRuntimeGraphicsCallbackQueue,
        maximum_figures: NonZeroUsize,
        maximum_snapshot_bytes: NonZeroUsize,
        window_config: runmat_plot::gui::WindowConfig,
    ) -> Result<Self, RuntimeError> {
        if opener.is_closed() {
            return Err(source_error(
                "native window host has retired",
                runmat_plot::gui::host_loop::WindowOpenError::Closed,
            ));
        }
        Ok(Self(Rc::new(Inner {
            opener,
            callbacks,
            pending: RefCell::new(BTreeMap::new()),
            windows: RefCell::new(BTreeMap::new()),
            maximum_figures,
            maximum_snapshot_bytes,
            window_config,
            busy: Cell::new(false),
            closed: Cell::new(false),
            last_presented: Cell::new(None),
        })))
    }
}

pub(super) fn error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:graphics:NativePresentation")
        .build()
}

pub(super) fn source_error(
    message: &str,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:graphics:NativePresentation")
        .with_source(source)
        .build()
}

#[derive(Debug, thiserror::Error)]
#[error("native scene identity capacity exhausted (maximum {maximum})")]
struct SceneCapacity {
    maximum: usize,
}

impl Inner {
    fn admit_receiver(&self, receiver: &GraphicsReceiverRef) -> Result<u32, RuntimeError> {
        if self.opener.is_closed() {
            return Err(error("native window host has retired"));
        }
        if self.closed.get() {
            return Err(error("native graphics attachment has retired"));
        }
        let GraphicsReceiverKind::Figure { figure } = receiver.kind() else {
            return Err(error(
                "native presentation requires an exact Figure receiver",
            ));
        };
        if !state::graphics_receiver_valid(receiver) {
            return Err(error("native scene receiver has retired"));
        }
        let pending = self.pending.borrow();
        let windows = self.windows.borrow();
        let already = pending
            .get(&figure)
            .is_some_and(|owned| owned.shares_lease(receiver))
            || windows
                .get(&figure)
                .is_some_and(|owned| owned.receiver.shares_lease(receiver));
        let windows_count = windows
            .values()
            .filter(|owned| state::graphics_receiver_valid(&owned.receiver))
            .count();
        let pending_only = pending
            .iter()
            .filter(|(id, owned)| {
                state::graphics_receiver_valid(owned)
                    && !windows
                        .get(id)
                        .is_some_and(|window| window.receiver.shares_lease(owned))
            })
            .count();
        if !already
            && windows_count
                .checked_add(pending_only)
                .is_none_or(|count| count >= self.maximum_figures.get())
        {
            return Err(source_error(
                "native scene retention capacity exhausted",
                SceneCapacity {
                    maximum: self.maximum_figures.get(),
                },
            ));
        }
        Ok(figure)
    }
    fn prune_retired_windows(&self) -> bool {
        let mut windows = self.windows.borrow_mut();
        let before = windows.len();
        windows.retain(|_, window| state::graphics_receiver_valid(&window.receiver));
        before != windows.len()
    }
    /// Freeze the latest scenes at flush admission, never refresh their lease
    /// after an open/frame has become Pending. Staging retains object identity,
    /// not an obsolete source-data copy or a numeric revision as authority.
    fn capture_scenes(&self) -> Result<std::collections::VecDeque<Scene>, RuntimeError> {
        let pending = self.pending.borrow();
        let handles: Vec<_> = pending
            .iter()
            .filter(|(_, receiver)| state::graphics_receiver_valid(receiver))
            .map(|(handle, _)| state::FigureHandle::from(*handle))
            .collect();
        let snapshot = state::capture_figure_persistence_snapshot(
            &handles,
            self.maximum_figures.get(),
            self.maximum_snapshot_bytes.get(),
        )
        .map_err(|source| source_error("native flush scene capture failed", source))?;
        let mut scenes = std::collections::VecDeque::new();
        for entry in snapshot.entries {
            let receiver = pending
                .get(&entry.handle.as_u32())
                .expect("captured pending identity");
            if state::figure_scene_is_current(receiver, &entry.scene_lease) {
                scenes.push_back(Scene {
                    receiver: receiver.clone(),
                    snapshot: entry,
                });
            }
        }
        Ok(scenes)
    }
}

impl RuntimeGraphicsPresentationService for NativeGraphicsAttachment {
    fn flush(&self, request: GraphicsFlushRequest) -> Pin<Box<dyn RuntimeGraphicsFlushOperation>> {
        Box::pin(operation::Flush::new(Rc::clone(&self.0), request))
    }

    fn admit_scene_update(
        &self,
        proposal: &super::RuntimeGraphicsSceneProposal,
    ) -> Result<(), RuntimeError> {
        self.0.admit_receiver(proposal.receiver())?;
        if !state::figure_scene_is_current(proposal.receiver(), proposal.scene()) {
            return Err(error("original property scene has retired"));
        }
        // Refuse known immediate-presentation conflicts before the property
        // plan writes; the operation still rechecks when it is polled.
        if self.0.busy.get() {
            return Err(error("native presentation is already flushing"));
        }
        proposal
            .validate_retained_budget(self.0.maximum_snapshot_bytes)
            .map_err(|source| {
                source_error("proposed native scene retention budget exhausted", source)
            })
    }

    fn present_scene_update(
        &self,
        scene: super::RuntimeGraphicsScene,
    ) -> Result<Pin<Box<dyn RuntimeGraphicsFlushOperation>>, RuntimeError> {
        self.0.admit_receiver(&scene.receiver)?;
        if !state::figure_scene_is_current(&scene.receiver, &scene.snapshot.scene_lease) {
            return Err(error("selected presentation scene has retired"));
        }
        let bytes = scene.retained_byte_upper_bound().ok_or_else(|| {
            source_error(
                "native scene size overflowed",
                state::FigurePersistenceSnapshotError::SizeOverflow,
            )
        })?;
        if bytes > self.0.maximum_snapshot_bytes.get() {
            return Err(source_error(
                "native scene retention budget exhausted",
                state::FigurePersistenceSnapshotError::BudgetExceeded {
                    retained_bytes: bytes,
                    maximum_bytes: self.0.maximum_snapshot_bytes.get(),
                },
            ));
        }
        Ok(Box::pin(operation::Flush::selected(
            Rc::clone(&self.0),
            Scene {
                receiver: scene.receiver,
                snapshot: scene.snapshot,
            },
        )))
    }

    fn stage_scene(
        &self,
        receiver: GraphicsReceiverRef,
    ) -> Result<GraphicsSceneStaging, RuntimeError> {
        let figure = self.0.admit_receiver(&receiver)?;
        let mut pending = self.0.pending.borrow_mut();
        pending.retain(|_, receiver| state::graphics_receiver_valid(receiver));
        pending.insert(figure, receiver);
        Ok(GraphicsSceneStaging::DeferredToFlush)
    }

    fn retire(&self) {
        self.0.closed.set(true);
        self.0.pending.borrow_mut().clear();
        // Linear receipts close exact windows even if routing handles are reused.
        self.0.windows.borrow_mut().clear();
    }

    fn retire_scene(&self, receiver: &GraphicsReceiverRef) {
        let GraphicsReceiverKind::Figure { figure } = receiver.kind() else {
            return;
        };
        let mut pending = self.0.pending.borrow_mut();
        if pending
            .get(&figure)
            .is_some_and(|owned| owned.shares_lease(receiver))
        {
            pending.remove(&figure);
        }
        drop(pending);
        let mut windows = self.0.windows.borrow_mut();
        if windows
            .get(&figure)
            .is_some_and(|window| window.receiver.shares_lease(receiver))
        {
            windows.remove(&figure);
        }
    }
}

impl Drop for NativeGraphicsAttachment {
    fn drop(&mut self) {
        self.retire();
    }
}
