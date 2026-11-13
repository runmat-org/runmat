//! Selected original issuer, or admitted original absence: never ambient choice.
use super::*;
use crate::context::{ContextFuture, PreparedUnselectedScope};
use std::rc::Rc;

pub(super) enum Origin {
    Selected {
        runtime: Box<RuntimeContext>,
        token: CancellationToken,
    },
    Unselected(PreparedUnselectedScope),
}
impl Origin {
    pub fn new(
        runtime: Option<&RuntimeContext>,
        token: Option<&CancellationToken>,
    ) -> BuiltinResult<Self> {
        match (runtime, token) {
            (Some(runtime), Some(token)) if token.same_generation(&runtime.cancellation()) => {}
            (None, None) => {}
            _ => return Err(failure(Failure::Foreign)),
        }
        check_original_cancellation(runtime, token)?;
        crate::context::host_allocation::require_unbounded_workspace(
            runtime,
            "scoped callable",
            "finite host-memory scoped-callable queues are unsupported",
        )?;
        Ok(match (runtime, token) {
            (Some(runtime), Some(token)) => Self::Selected {
                runtime: Box::new(runtime.clone().with_cancellation_token(token.clone())),
                token: token.clone(),
            },
            (None, None) => Self::Unselected(PreparedUnselectedScope::prepare(None)?),
            _ => return Err(failure(Failure::Foreign)),
        })
    }
    pub fn runtime(&self) -> Option<&RuntimeContext> {
        match self {
            Self::Selected { runtime, .. } => Some(runtime),
            Self::Unselected(_) => None,
        }
    }
    pub fn token(&self) -> Option<&CancellationToken> {
        match self {
            Self::Selected { token, .. } => Some(token),
            Self::Unselected(_) => None,
        }
    }
    pub fn matches(&self, caller: Option<&RuntimeContext>) -> bool {
        match (self, caller) {
            (Self::Unselected(_), None) => true,
            (
                Self::Selected {
                    runtime: original,
                    token,
                },
                Some(caller),
            ) => {
                original.local_identity() == caller.local_identity()
                    && Rc::ptr_eq(original.execution(), caller.execution())
                    && original
                        .service_ports()
                        .same_device_route_owners(caller.service_ports())
                    && match (original.program_revision(), caller.program_revision()) {
                        // Normal invocation projections share the immutable
                        // revision; avoid walking its domain facts per call.
                        (Some(left), Some(right)) => std::ptr::eq(left, right) || left == right,
                        (None, None) => true,
                        _ => false,
                    }
                    && token.same_generation(&caller.cancellation())
            }
            _ => false,
        }
    }
    pub fn construct<T>(&self, factory: impl FnOnce() -> T) -> T {
        let _guard = self.enter();
        factory()
    }
    pub fn enter(&self) -> crate::context::RuntimeContextGuard {
        match self {
            Self::Selected { runtime, .. } => runtime.enter(),
            Self::Unselected(scope) => scope.enter(),
        }
    }
    pub fn scope<F: Future>(&self, future: F) -> ContextFuture<F> {
        match self {
            Self::Selected { runtime, .. } => runtime.scope(future),
            Self::Unselected(scope) => scope.scope(future),
        }
    }
}
