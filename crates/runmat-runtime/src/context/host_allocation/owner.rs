//! The captured context facts needed by one host-allocation receipt.

use std::cell::RefCell;
use std::rc::Rc;

use runmat_execution::cancellation::CancellationToken;

use crate::context::{RuntimeContext, RuntimeContextState, RuntimeParallelService};

/// Retain the original session and live cancellation owner, without retaining
/// unrelated invocation projections or cloning their metadata. This is not an
/// allocation grant; the receipt's ledger and fence still authorize all work.
#[derive(Clone)]
pub(in crate::context) struct HostAllocationContext {
    state: Rc<RuntimeContextState>,
    cancellation: Rc<RefCell<CancellationToken>>,
    parallel: Rc<dyn RuntimeParallelService>,
}

impl HostAllocationContext {
    pub(super) fn same_owner(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.state, &other.state)
            && self
                .cancellation
                .borrow()
                .same_generation(&other.cancellation.borrow())
            && Rc::ptr_eq(&self.parallel, &other.parallel)
    }
    pub(in crate::context) fn new(
        state: Rc<RuntimeContextState>,
        cancellation: Rc<RefCell<CancellationToken>>,
        parallel: Rc<dyn RuntimeParallelService>,
    ) -> Self {
        Self {
            state,
            cancellation,
            parallel,
        }
    }

    pub(super) fn matches_runtime(&self, runtime: &RuntimeContext) -> bool {
        self.state.local_identity == runtime.local_identity()
            && self
                .cancellation
                .borrow()
                .same_generation(&runtime.cancellation())
    }

    pub(super) fn matches_cancellation(&self, original: &CancellationToken) -> bool {
        self.cancellation.borrow().same_generation(original)
    }

    pub(super) fn parallel(&self) -> &dyn RuntimeParallelService {
        self.parallel.as_ref()
    }
}
