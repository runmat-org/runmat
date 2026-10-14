//! Account for a submitted native host task until its owned future retires.
//! Returned outputs retain their own independent publication authority.

use runmat_execution::profiling::{
    with_command_profiling, CommandDeferredWork, CommandProfilingReceipt, DeferredWorkTerminal,
};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub(crate) fn spawn<F>(
    driver: &tokio::runtime::Handle,
    future: F,
    profiling: Option<CommandProfilingReceipt>,
) -> tokio::task::JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    if let Some((origin, deferred)) = profiling.and_then(|origin| {
        origin
            .begin_deferred_work()
            .map(|deferred| (origin, deferred))
    }) {
        driver.spawn(RetiringNativeTask::new(future, origin, deferred))
    } else {
        driver.spawn(future)
    }
}

struct RetiringNativeTask<F> {
    future: Option<Pin<Box<F>>>,
    origin: CommandProfilingReceipt,
    deferred: Option<CommandDeferredWork>,
}

// The inner future alone is pinned, in its Box. Moving the outer owner does
// not move that allocation or permit the inner future to be moved after poll.
impl<F> Unpin for RetiringNativeTask<F> {}

impl<F> RetiringNativeTask<F> {
    fn new(future: F, origin: CommandProfilingReceipt, deferred: CommandDeferredWork) -> Self {
        Self {
            future: Some(Box::pin(future)),
            origin,
            deferred: Some(deferred),
        }
    }
}

impl<F: Future> Future for RetiringNativeTask<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let polled = with_command_profiling(Some(this.origin.clone()), || {
            let result = this
                .future
                .as_mut()
                .expect("native task is polled before retirement")
                .as_mut()
                .poll(context);
            if result.is_ready() {
                drop(this.future.take());
            }
            result
        });
        match polled {
            Poll::Pending => Poll::Pending,
            Poll::Ready(output) => {
                if let Some(deferred) = this.deferred.take() {
                    deferred.complete(DeferredWorkTerminal::Completed);
                }
                Poll::Ready(output)
            }
        }
    }
}

impl<F> Drop for RetiringNativeTask<F> {
    fn drop(&mut self) {
        if let Some(inner) = self.future.take() {
            with_command_profiling(Some(self.origin.clone()), || drop(inner));
        }
        drop(self.deferred.take());
    }
}

#[cfg(test)]
mod tests;
