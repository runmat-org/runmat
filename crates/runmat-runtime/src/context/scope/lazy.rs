//! First-poll preparation at the existing physical poll-future owner.
//! Work remains unpolled and movable until it enters that owner's single Box.
use super::{ContextFuture, PreparedUnselectedScope, RuntimeContext};
use crate::RuntimeError;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

enum State<F> {
    Pending {
        runtime: Option<RuntimeContext>,
        cancellation: Option<runmat_execution::CancellationToken>,
        work: F,
    },
    Running(ContextFuture<F>),
    Finished,
}
pub(crate) struct LazyRuntimeScope<F> {
    state: State<F>,
}
impl<F> LazyRuntimeScope<F> {
    pub(crate) fn new(runtime: Option<RuntimeContext>, work: F) -> Self {
        Self {
            state: State::Pending {
                cancellation: runtime.as_ref().map(RuntimeContext::cancellation),
                runtime,
                work,
            },
        }
    }
}
impl<F, O> LazyRuntimeScope<F>
where
    F: Future<Output = Result<O, RuntimeError>>,
{
    // Moving the unpolled concrete child into its existing Box is a distinct
    // phase. Its construction temporaries must retire before polling it.
    fn prepare(&mut self) -> Result<(), RuntimeError> {
        if matches!(self.state, State::Running(_)) {
            return Ok(());
        }
        let State::Pending {
            runtime,
            cancellation,
            work,
        } = std::mem::replace(&mut self.state, State::Finished)
        else {
            panic!("runtime scope polled after completion")
        };
        let running = match runtime.as_ref() {
            Some(runtime) => match super::inline::check_captured(runtime, cancellation.as_ref())
                .and_then(|()| runtime.prepare_poll_scope())
                .and_then(|scope| scope.prepare_future::<F>())
            {
                Ok(prepared) => {
                    if let Err(error) =
                        super::inline::check_captured(runtime, cancellation.as_ref())
                    {
                        super::inline::with_captured_scope(
                            Some(runtime),
                            cancellation.as_ref(),
                            || drop((prepared, work)),
                        );
                        Err(error)
                    } else {
                        let running = prepared.try_scope_with(|| work);
                        match super::inline::check_captured(runtime, cancellation.as_ref()) {
                            Ok(()) => running,
                            Err(error) => {
                                super::inline::with_captured_scope(
                                    Some(runtime),
                                    cancellation.as_ref(),
                                    || drop(running),
                                );
                                Err(error)
                            }
                        }
                    }
                }
                Err(source) => {
                    super::inline::with_captured_scope(
                        Some(runtime),
                        cancellation.as_ref(),
                        || drop(work),
                    );
                    Err(source)
                }
            },
            None => match PreparedUnselectedScope::prepare(None) {
                Ok(scope) => Ok(scope.scope(work)),
                Err(source) => {
                    super::with_original_scope(None, || drop(work));
                    Err(source)
                }
            },
        };
        self.state = State::Running(running?);
        Ok(())
    }
}
impl<F, O> Future for LazyRuntimeScope<F>
where
    F: Future<Output = Result<O, RuntimeError>>,
{
    type Output = Result<O, RuntimeError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        if let Err(source) = this.prepare() {
            return Poll::Ready(Err(source));
        }
        let State::Running(running) = &mut this.state else {
            unreachable!("successful scope preparation retains its child")
        };
        Pin::new(running).poll(cx)
    }
}
impl<F> Drop for LazyRuntimeScope<F> {
    fn drop(&mut self) {
        match std::mem::replace(&mut self.state, State::Finished) {
            State::Pending {
                runtime,
                cancellation,
                work,
            } => {
                super::inline::with_captured_scope(runtime.as_ref(), cancellation.as_ref(), || {
                    drop(work)
                })
            }
            State::Running(running) => drop(running),
            State::Finished => {}
        }
    }
}
// F is never polled inline; only the unchanged ContextFuture's Box pins it.
impl<F> Unpin for LazyRuntimeScope<F> {}
