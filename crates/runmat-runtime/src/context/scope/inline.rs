//! Logical poll scope inside an already admitted future. This owner neither
//! allocates a future Box nor claims a physical receipt. The caller quotes
//! the complete ScopedFuture<F> at its existing concrete continuation owner.
use super::{PreparedRuntimeScope, PreparedUnselectedScope, RuntimeContext, RuntimeContextGuard};
use crate::RuntimeError;
use runmat_execution::cancellation::CancellationToken;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
enum Authority {
    Pending {
        runtime: Option<RuntimeContext>,
        cancellation: Option<CancellationToken>,
    },
    Selected(PreparedRuntimeScope),
    Unselected(PreparedUnselectedScope),
}
impl Authority {
    fn prepare(&mut self) -> Result<(), RuntimeError> {
        let Self::Pending {
            runtime,
            cancellation,
        } = self
        else {
            return Ok(());
        };
        let selected = match runtime {
            Some(runtime) => {
                check_captured(runtime, cancellation.as_ref())?;
                let scope = runtime.prepare_poll_scope()?;
                if let Err(error) = check_captured(runtime, cancellation.as_ref()) {
                    with_captured_scope(Some(runtime), cancellation.as_ref(), || drop(scope));
                    return Err(error);
                }
                Self::Selected(scope)
            }
            None => Self::Unselected(PreparedUnselectedScope::prepare(None)?),
        };
        *self = selected;
        Ok(())
    }
    fn enter(&self) -> CapturedScopeGuard {
        match self {
            Self::Pending {
                runtime,
                cancellation,
            } => CapturedScopeGuard::enter(runtime.as_ref(), cancellation.as_ref()),
            Self::Selected(scope) => CapturedScopeGuard {
                _runtime: scope.enter(),
                _cancellation: None,
            },
            Self::Unselected(scope) => CapturedScopeGuard {
                _runtime: scope.enter(),
                _cancellation: None,
            },
        }
    }
    fn check_pending(&self) -> Result<(), RuntimeError> {
        match self {
            Self::Pending {
                runtime: Some(runtime),
                cancellation,
            } => check_captured(runtime, cancellation.as_ref()),
            _ => Ok(()),
        }
    }
    fn runtime(&self) -> Option<&RuntimeContext> {
        match self {
            Self::Pending { runtime, .. } => runtime.as_ref(),
            _ => None,
        }
    }
}
// Only a synchronous scope: no fresh Runtime, Rc token slot, projection
// snapshot or selected-route clearing. The existing token guard restores the
// caller's current slot after construction/retirement has finished.
struct CapturedScopeGuard {
    _runtime: RuntimeContextGuard,
    _cancellation: Option<crate::context::RuntimeCancellationScope>,
}
impl CapturedScopeGuard {
    fn enter(runtime: Option<&RuntimeContext>, cancellation: Option<&CancellationToken>) -> Self {
        let token = runtime
            .zip(cancellation)
            .map(|(runtime, token)| runtime.scoped_cancellation_token(token.clone()));
        let selected = match runtime {
            Some(runtime) => RuntimeContextGuard::enter(runtime.clone()),
            None => RuntimeContextGuard::enter_unselected(),
        };
        Self {
            _runtime: selected,
            _cancellation: token,
        }
    }
}
pub(super) fn with_captured_scope<O>(
    runtime: Option<&RuntimeContext>,
    cancellation: Option<&CancellationToken>,
    work: impl FnOnce() -> O,
) -> O {
    let _guard = CapturedScopeGuard::enter(runtime, cancellation);
    work()
}
pub(super) fn check_captured(
    runtime: &RuntimeContext,
    cancellation: Option<&CancellationToken>,
) -> Result<(), RuntimeError> {
    let source = match cancellation {
        Some(token) if token.same_generation(&runtime.cancellation()) => token
            .is_cancelled()
            .then_some(super::HostAllocationError::Cancelled),
        _ => Some(super::HostAllocationError::StaleAuthority),
    };
    match source {
        Some(source) => with_captured_scope(Some(runtime), cancellation, || {
            Err(super::scope_allocation_error(
                runtime,
                "poll scope cancellation authority changed",
                source,
            ))
        }),
        None => Ok(()),
    }
}
/// Select the original owner only for synchronous work. This does not snapshot
/// instruction-local projections reserved for a future's first poll.
pub(crate) fn with_original_scope<O>(
    runtime: Option<&RuntimeContext>,
    work: impl FnOnce() -> O,
) -> O {
    let _guard = match runtime {
        Some(runtime) => RuntimeContextGuard::enter(runtime.clone()),
        None => RuntimeContextGuard::enter_unselected(),
    };
    work()
}
pub(crate) struct RuntimeScopeFactory<M> {
    authority: Option<Authority>,
    make_work: Option<M>,
}
impl<M> RuntimeScopeFactory<M> {
    pub(crate) fn new(runtime: Option<RuntimeContext>, make_work: M) -> Self {
        let cancellation = runtime.as_ref().map(RuntimeContext::cancellation);
        Self {
            authority: Some(Authority::Pending {
                runtime,
                cancellation,
            }),
            make_work: Some(make_work),
        }
    }
    pub(crate) fn runtime(&self) -> Option<&RuntimeContext> {
        self.authority.as_ref().and_then(Authority::runtime)
    }
    pub(crate) fn construct<F>(mut self) -> ScopedFuture<F>
    where
        M: FnOnce() -> F,
    {
        let authority = self.authority.take().expect("unconstructed scope");
        let failure = authority.check_pending().err();
        let _guard = authority.enter();
        let future = if failure.is_some() {
            drop(self.make_work.take());
            None
        } else {
            Some(self.make_work.take().expect("unconstructed future")())
        };
        ScopedFuture {
            authority,
            future,
            failure,
            finished: false,
        }
    }
}
impl<M> Drop for RuntimeScopeFactory<M> {
    fn drop(&mut self) {
        if let Some(authority) = &self.authority {
            let _guard = authority.enter();
            drop(self.make_work.take());
        }
    }
}
pin_project_lite::pin_project! {
    pub(crate) struct ScopedFuture<F> {
        authority:Authority,
        #[pin]
        future:Option<F>,
        failure:Option<RuntimeError>,
        finished:bool,
    }
    impl<F> PinnedDrop for ScopedFuture<F> {
        fn drop(this:Pin<&mut Self>){
            let this=this.project();
            let _guard=this.authority.enter();
            retire_child(this.future);
            drop(this.failure.take());
        }
    }
}
impl<F> ScopedFuture<F> {
    pub(super) fn from_prepared(scope: PreparedRuntimeScope, future: F) -> Self {
        Self {
            authority: Authority::Selected(scope),
            future: Some(future),
            failure: None,
            finished: false,
        }
    }
}
// Replacement can require a concrete-F-sized temporary in an unoptimized
// build. Keep it outside the frame that polls the still-active child.
fn retire_child<F>(mut future: Pin<&mut Option<F>>) {
    future.set(None);
}
impl<F, O> Future for ScopedFuture<F>
where
    F: Future<Output = Result<O, RuntimeError>>,
{
    type Output = Result<O, RuntimeError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        assert!(!*this.finished, "scope polled after completion");
        if let Some(error) = this.failure.take() {
            *this.finished = true;
            return Poll::Ready(Err(error));
        }
        if let Err(source) = this.authority.prepare() {
            *this.finished = true;
            let _guard = this.authority.enter();
            retire_child(this.future);
            return Poll::Ready(Err(source));
        }
        let _guard = this.authority.enter();
        let result = this
            .future
            .as_mut()
            .as_pin_mut()
            .expect("scope polled after completion")
            .poll(cx);
        if result.is_ready() {
            *this.finished = true;
            retire_child(this.future)
        }
        result
    }
}
