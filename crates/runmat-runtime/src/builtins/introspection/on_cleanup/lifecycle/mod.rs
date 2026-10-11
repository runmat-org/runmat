//! Session-owned cleanup admission and dispatch. The graph policy is kept
//! separate from callback execution and host shutdown authority.

pub(crate) mod dependencies;
pub(crate) mod dispatch;
pub(crate) mod registry;
