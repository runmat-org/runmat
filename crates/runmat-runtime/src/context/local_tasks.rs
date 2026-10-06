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
#[cfg(target_arch = "wasm32")]
use std::task::{Poll, Waker};
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

mod lease;
#[cfg(target_arch = "wasm32")]
use lease::CompletionOnDrop;
use lease::TaskState;
pub use lease::{RuntimeLocalTaskLease, RuntimeLocalTaskOutcome};

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
    fn spawn_local(
        &self,
        owner: RuntimeContextLocalId,
        future: RuntimeLocalTaskFuture,
    ) -> Result<RuntimeLocalTaskLease, RuntimeError>;
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
