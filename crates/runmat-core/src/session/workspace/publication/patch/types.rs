use runmat_runtime::workspace::WorkspaceName;

use super::super::super::binding_key::RetainedBindingKey;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::session::workspace::publication) enum PendingBindingSlot {
    Preserve,
    Explicit(usize),
    Dynamic,
}

#[derive(Debug)]
pub(in crate::session::workspace::publication) enum PendingBindingKey {
    Preserve,
    Replace(RetainedBindingKey),
}

#[derive(Debug)]
pub(in crate::session::workspace::publication) enum PendingBindingAction {
    Upsert {
        slot: PendingBindingSlot,
        key: PendingBindingKey,
        introduces_binding: bool,
    },
    Retire,
}

pub(in crate::session::workspace::publication) struct PreparedPatchAction {
    pub(in crate::session::workspace::publication) name: WorkspaceName,
    pub(in crate::session::workspace::publication) action: PendingBindingAction,
}

pub(in crate::session::workspace::publication) enum PreparedPatchUpdate {
    Noop,
    Action(PreparedPatchAction),
    Clear { next_epoch: u64 },
}
