//! Host-owned, single-threaded background execution for session-local work.
//!
//! A runtime holds only a weak submission port. The host owns the supervisor
//! and must poll `drive_until` while its session is idle as well as active.

use super::RuntimeContextLocalId;
use crate::RuntimeError;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::{Rc, Weak};
#[cfg(not(target_arch = "wasm32"))]
mod driver;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::RuntimeLocalTaskSupervisor;

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
pub use browser::BrowserLocalTaskSupervisor;

pub type RuntimeLocalTaskFuture = Pin<Box<dyn Future<Output = ()> + 'static>>;

/// Opaque identity shared by one host supervisor and only its submission
/// ports; context clones cannot manufacture ownership of a foreign driver.
#[derive(Clone)]
pub struct LocalTaskOwnerToken(Rc<()>);

impl LocalTaskOwnerToken {
    fn new() -> Self {
        Self(Rc::new(()))
    }

    pub(crate) fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

mod lease;
mod slot;
#[cfg(target_arch = "wasm32")]
use lease::CompletionOnDrop;
use lease::TaskState;
pub use lease::{RuntimeLocalTaskLease, RuntimeLocalTaskOutcome};
use slot::TaskSlot;

#[derive(Clone, Copy)]
enum LocalTaskFailure {
    Closed,
    ForeignSession,
    IdentityExhausted,
    ReentrantDrive,
}

impl LocalTaskFailure {
    fn runtime_error(self) -> RuntimeError {
        let (identifier, message) = match self {
            Self::Closed => (
                "RunMat:RuntimeContext:LocalTaskClosed",
                "local task session has ended",
            ),
            Self::ForeignSession => (
                "RunMat:RuntimeContext:LocalTaskForeignSession",
                "local task belongs to another runtime session",
            ),
            Self::IdentityExhausted => (
                "RunMat:RuntimeContext:LocalTaskIdentityExhausted",
                "local task identity exhausted",
            ),
            Self::ReentrantDrive => (
                "RunMat:RuntimeContext:LocalTaskDriverBusy",
                "local task supervisor is closed or already driving",
            ),
        };
        crate::build_runtime_error(message)
            .with_identifier(identifier)
            .build()
    }
}

pub trait RuntimeLocalTaskService {
    /// Existing embedding ports remain source-compatible but cannot claim
    /// canonical lifecycle-host ownership without an exact supervisor token.
    fn owner_token(&self) -> Option<LocalTaskOwnerToken> {
        None
    }

    fn spawn_local(
        &self,
        owner: RuntimeContextLocalId,
        future: RuntimeLocalTaskFuture,
    ) -> Result<RuntimeLocalTaskLease, RuntimeError>;
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
