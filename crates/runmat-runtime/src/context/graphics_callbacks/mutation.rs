//! Invocation-local scene successors issued only by admitted mutation guards.

use runmat_value::{GraphicsReceiverRef, GraphicsSceneLease};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct GraphicsCallbackMutationScope(Rc<Inner>);

struct Inner {
    receiver: GraphicsReceiverRef,
    original: GraphicsSceneLease,
    current: RefCell<GraphicsSceneLease>,
    live: Cell<bool>,
}

/// Callback exit revokes even clones retained by suspended work.
pub(super) struct MutationInvocation(GraphicsCallbackMutationScope);
impl Drop for MutationInvocation {
    fn drop(&mut self) {
        self.0 .0.live.set(false);
    }
}

impl GraphicsCallbackMutationScope {
    #[cfg(feature = "plot-core")]
    pub(crate) fn figure(&self) -> u32 {
        self.0.original.figure()
    }
    pub(super) fn begin(
        receiver: GraphicsReceiverRef,
        scene: GraphicsSceneLease,
    ) -> (Self, MutationInvocation) {
        let scope = Self(Rc::new(Inner {
            receiver,
            original: scene.clone(),
            current: RefCell::new(scene),
            live: Cell::new(true),
        }));
        let invocation = MutationInvocation(scope.clone());
        (scope, invocation)
    }

    pub(crate) fn selects(&self, receiver: &GraphicsReceiverRef) -> bool {
        self.0.receiver.shares_lease(receiver)
    }

    /// Called under the actual guard, before effects.
    #[cfg(feature = "plot-core")]
    pub(crate) fn admits(
        &self,
        receiver: &GraphicsReceiverRef,
        predecessor: &GraphicsSceneLease,
    ) -> bool {
        self.0.live.get()
            && self.selects(receiver)
            && self.0.current.borrow().shares_lease(predecessor)
    }

    #[cfg(feature = "plot-core")]
    pub(crate) fn record_successor(
        &self,
        receiver: &GraphicsReceiverRef,
        predecessor: &GraphicsSceneLease,
        successor: &GraphicsSceneLease,
    ) -> bool {
        if !self.admits(receiver, predecessor) {
            self.0.live.set(false);
            return false;
        }
        *self.0.current.borrow_mut() = successor.clone();
        true
    }

    pub(super) fn final_scene(
        &self,
        receiver: &GraphicsReceiverRef,
        original: &GraphicsSceneLease,
    ) -> Option<GraphicsSceneLease> {
        (self.0.live.get() && self.selects(receiver) && self.0.original.shares_lease(original))
            .then(|| self.0.current.borrow().clone())
    }
}
