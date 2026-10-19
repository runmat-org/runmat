//! Move-only completion handoff for a closed VM frame.
//!
//! The node is admitted with the frame, before execution. Frame retirement
//! moves already-owned maps and vectors into it and links it without growing
//! a thread-local collection. Taking a result unlinks the same node.

use super::{WorkspaceAssignedReport, WorkspaceFrameId, WorkspaceOwner, WorkspaceValueSnapshot};
use runmat_execution::resource::ExactShared;
use runmat_runtime::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use runmat_runtime::context::RuntimeContext;
use runmat_thread_local::runmat_thread_local;
use runmat_value::ExactHostAllocationClaim;
use std::cell::{Cell, RefCell};

pub(super) struct CompletionNode {
    owner: WorkspaceOwner,
    frame: Cell<WorkspaceFrameId>,
    legacy_visible: Cell<bool>,
    state: RefCell<Option<WorkspaceValueSnapshot>>,
    assigned: RefCell<Option<WorkspaceAssignedReport>>,
    next: RefCell<Option<CompletionHandle>>,
    // ExactShared frees its header before dropping this claim-bearing payload.
    _claim: Option<ExactHostAllocationClaim>,
}

pub(super) struct CompletionHandle(ExactShared<CompletionNode>);

runmat_thread_local! {
    static COMPLETED: RefCell<Option<CompletionHandle>> = const { RefCell::new(None) };
}

impl CompletionHandle {
    pub(super) fn retained_bytes() -> Result<u64, HostAllocationError> {
        HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(ExactShared::<CompletionNode>::requested_bytes())
                .map_err(|_| HostAllocationError::Overflow)?,
            1,
        )
    }

    pub(super) fn prepare(
        owner: WorkspaceOwner,
        context: Option<&RuntimeContext>,
        receipt: Option<&HostAllocationReceipt>,
    ) -> Result<Self, HostAllocationError> {
        let prepared = match (context, receipt) {
            (Some(context), Some(receipt)) => {
                receipt.with_output_allocation(context, |permit| {
                    let claim = permit
                        .reserve_exact_claim(ExactShared::<CompletionNode>::requested_bytes())?;
                    ExactShared::try_new(CompletionNode::new(owner, Some(claim))).map_err(|_| {
                        runmat_value::ExactHostNumericAllocationError::AllocationFailed
                    })
                })?
            }
            _ => ExactShared::try_new(CompletionNode::new(owner, None)).map_err(|_| {
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                )
            })?,
        };
        Ok(Self(prepared))
    }

    pub(super) fn bind_frame(&self, frame: WorkspaceFrameId) {
        assert_eq!(self.0.get().frame.replace(frame), WorkspaceFrameId(0));
    }

    pub(super) fn publish(self, state: WorkspaceValueSnapshot, assigned: WorkspaceAssignedReport) {
        *self.0.get().state.borrow_mut() = Some(state);
        *self.0.get().assigned.borrow_mut() = Some(assigned);
        COMPLETED.with(|completed| {
            let mut completed = completed.borrow_mut();
            let mut previous = completed.as_ref().map(|node| node.0.clone());
            while let Some(node) = previous {
                if node.get().owner == self.0.get().owner {
                    node.get().legacy_visible.set(false);
                    break;
                }
                previous = node.get().next.borrow().as_ref().map(|next| next.0.clone());
            }
            *self.0.get().next.borrow_mut() = completed.take();
            *completed = Some(self);
        });
    }
}

impl CompletionNode {
    fn new(owner: WorkspaceOwner, claim: Option<ExactHostAllocationClaim>) -> Self {
        Self {
            owner,
            frame: Cell::new(WorkspaceFrameId(0)),
            legacy_visible: Cell::new(true),
            state: RefCell::new(None),
            assigned: RefCell::new(None),
            next: RefCell::new(None),
            _claim: claim,
        }
    }
}

impl Drop for CompletionNode {
    fn drop(&mut self) {
        // A long chain is retired iteratively. Each uniquely owned successor
        // has its link detached before its own payload is dropped.
        let mut next = self.next.get_mut().take();
        while let Some(mut node) = next {
            if !node.0.is_unique() {
                break;
            }
            next = node.0.get_mut_unique().next.get_mut().take();
        }
    }
}

fn with_matching<R>(
    mut matches: impl FnMut(&CompletionNode) -> bool,
    take: impl FnOnce(&CompletionNode) -> R,
) -> Option<R> {
    COMPLETED.with(|completed| {
        let mut head = completed.borrow_mut();
        let mut cursor = head.take();
        let mut prefix: Option<CompletionHandle> = None;
        let mut result = None;
        while let Some(node) = cursor {
            let next = node.0.get().next.borrow_mut().take();
            if matches(node.0.get()) {
                result = Some(take(node.0.get()));
                let keep = node.0.get().state.borrow().is_some()
                    || node.0.get().assigned.borrow().is_some();
                cursor = if keep {
                    *node.0.get().next.borrow_mut() = next;
                    Some(node)
                } else {
                    next
                };
                break;
            }
            *node.0.get().next.borrow_mut() = prefix;
            prefix = Some(node);
            cursor = next;
        }
        while let Some(node) = prefix {
            prefix = node.0.get().next.borrow_mut().take();
            *node.0.get().next.borrow_mut() = cursor;
            cursor = Some(node);
        }
        *head = cursor;
        result
    })
}

pub(super) fn take_state_for_owner(owner: WorkspaceOwner) -> Option<WorkspaceValueSnapshot> {
    with_matching(
        |node| node.owner == owner && node.legacy_visible.get(),
        |node| node.state.borrow_mut().take(),
    )
    .flatten()
}

pub(super) fn take_assigned_for_owner(owner: WorkspaceOwner) -> Option<WorkspaceAssignedReport> {
    with_matching(
        |node| node.owner == owner && node.legacy_visible.get(),
        |node| node.assigned.borrow_mut().take(),
    )
    .flatten()
}

pub(super) fn take_for_frame(
    frame: WorkspaceFrameId,
) -> Option<(
    Option<WorkspaceValueSnapshot>,
    Option<WorkspaceAssignedReport>,
)> {
    with_matching(
        |node| node.frame.get() == frame,
        |node| {
            (
                node.state.borrow_mut().take(),
                node.assigned.borrow_mut().take(),
            )
        },
    )
}

pub(super) fn reset_for_tests() {
    COMPLETED.with(|completed| {
        let mut next = completed.borrow_mut().take();
        while let Some(node) = next {
            next = node.0.get().next.borrow_mut().take();
        }
    });
}
