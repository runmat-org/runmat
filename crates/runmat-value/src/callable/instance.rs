use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

/// A process-local identity for one constructed callable value. Cloning that
/// value retains the identity; reconstructing an independent value does not.
///
/// This is deliberately not serialized or used as execution authority.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CallableInstanceId(NonZeroU64);

/// The physical instance together with its callable form. A public named
/// payload can be cloned into different Value variants; those variants must
/// never share graph nodes or foreign callback proxies.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CallableInstanceKey {
    kind: CallableInstanceKind,
    instance: CallableInstanceId,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CallableInstanceKind {
    Workspace,
    External,
    Method,
    Bound,
    Closure,
    Scoped,
}

static NEXT_CALLABLE_INSTANCE: AtomicU64 = AtomicU64::new(1);

impl CallableInstanceId {
    pub(super) fn mint() -> Self {
        let next = NEXT_CALLABLE_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("process-local callable instance IDs exhausted");
        Self(NonZeroU64::new(next).expect("callable instance counter starts at one"))
    }
}

impl CallableInstanceKey {
    pub(crate) fn new(kind: CallableInstanceKind, instance: CallableInstanceId) -> Self {
        Self { kind, instance }
    }
}
