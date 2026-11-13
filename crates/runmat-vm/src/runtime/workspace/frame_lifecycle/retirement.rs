use super::*;

#[derive(Default)]
pub(super) struct LifecycleReplacements {
    pub(super) removed_slots: Option<FrameMap<usize, WorkspaceName>>,
    pub(super) removed_ids: Option<FrameSet<usize>>,
    pub(super) removed_names: Option<FrameSet<WorkspaceName>>,
    pub(super) lifecycle: Option<FrameMap<usize, SlotLifecycle>>,
    pub(super) reconciliation: Option<FrameSet<usize>>,
    pub(super) cleared_names: Option<FrameMap<WorkspaceName, usize>>,
    pub(super) cleared_assigned: Option<FrameSet<WorkspaceName>>,
}

pub(super) struct FrameLifecycleRetirement {
    pub(super) _replacements: LifecycleReplacements,
    pub(super) _names: FrameVec<WorkspaceName>,
    pub(super) _values: FrameVec<InterpreterSlot>,
    pub(super) _aliases: FrameVec<FrameGlobalBindingsRetirement>,
}
