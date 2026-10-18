//! Host-owned, single-threaded background execution for session-local work.
//!
//! A runtime holds only a weak submission port. The host owns the supervisor
//! and must poll `drive_until` while its session is idle as well as active.

use super::RuntimeContextLocalId;
use crate::RuntimeError;
use runmat_execution::profiling::{CommandDeferredWork, DeferredWorkTerminal};
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
pub use native::{RuntimeLocalTaskChildFactory, RuntimeLocalTaskSupervisor};

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
pub use browser::BrowserLocalTaskSupervisor;

pub type RuntimeLocalTaskFuture = Pin<Box<dyn Future<Output = ()> + 'static>>;

/// An opaque submission authority created only by RunMat's native/browser
/// supervisors. A custom task service may forward this capability, but
/// cannot use a copied owner token to certify its own scheduling.
#[derive(Clone)]
pub struct CanonicalLocalTaskAdmission {
    #[cfg(not(target_arch = "wasm32"))]
    port: native::SubmissionPort,
    #[cfg(target_arch = "wasm32")]
    port: browser::BrowserPort,
}

impl CanonicalLocalTaskAdmission {
    pub(crate) fn is_active_driver(&self) -> bool {
        self.port.is_active_driver()
    }

    pub(crate) fn spawn_admitted(
        &self,
        owner: RuntimeContextLocalId,
        future: RuntimeLocalTaskFuture,
        deferred: Option<CommandDeferredWork>,
    ) -> Result<RuntimeLocalTaskLease, RuntimeError> {
        self.port.submit(owner, future, deferred)
    }
}

fn reject_admitted(
    error: RuntimeError,
    future: RuntimeLocalTaskFuture,
    deferred: Option<CommandDeferredWork>,
) -> Result<RuntimeLocalTaskLease, RuntimeError> {
    drop(future);
    if let Some(deferred) = deferred {
        deferred.complete(DeferredWorkTerminal::Failed);
    }
    Err(error)
}

/// Opaque identity shared by one host supervisor and only its submission
/// ports; context clones cannot manufacture ownership of a foreign driver.
#[derive(Clone)]
pub struct LocalTaskOwnerToken(Rc<()>);

impl LocalTaskOwnerToken {
    fn new() -> Self {
        Self(Rc::new(()))
    }

    /// Whether two host guards bind the same local-task supervisor authority.
    pub fn same(&self, other: &Self) -> bool {
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
    /// Existing embedding ports remain source-compatible. This token is a
    /// lifecycle identity, not proof of canonical task scheduling.
    fn owner_token(&self) -> Option<LocalTaskOwnerToken> {
        None
    }

    /// Only a real supervisor port can construct the returned capability.
    /// Callers submit through it, never through the implementer's raw method.
    fn canonical_admission(&self) -> Option<CanonicalLocalTaskAdmission> {
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
