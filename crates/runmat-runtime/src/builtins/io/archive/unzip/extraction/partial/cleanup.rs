//! The browser's detached rollback owns the provider/path until its removal
//! future is destroyed. A command token, when admitted, retires afterward.

use super::*;
use runmat_execution::profiling::{CommandDeferredWork, DeferredWorkTerminal};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub(super) fn spawn(
    provider: Arc<dyn vfs::FsProvider>,
    path: PathBuf,
    token: Option<CommandDeferredWork>,
) {
    if let Some(token) = token {
        wasm_bindgen_futures::spawn_local(DetachedPartialCleanup::new(provider, path, token));
    } else {
        // An unattributed host rollback needs no boxed accounting future.
        wasm_bindgen_futures::spawn_local(async move {
            let _ = provider.remove_file(&path).await;
        });
    }
}

pub(super) struct DetachedPartialCleanup {
    inner: Option<Pin<Box<dyn Future<Output = std::io::Result<()>>>>>,
    token: Option<CommandDeferredWork>,
}

impl DetachedPartialCleanup {
    pub(super) fn new(
        provider: Arc<dyn vfs::FsProvider>,
        path: PathBuf,
        token: CommandDeferredWork,
    ) -> Self {
        Self {
            inner: Some(Box::pin(async move { provider.remove_file(&path).await })),
            token: Some(token),
        }
    }
}

impl Future for DetachedPartialCleanup {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let result = std::task::ready!(this
            .inner
            .as_mut()
            .expect("one rollback")
            .as_mut()
            .poll(context));
        // A resumed command may seal immediately after Poll::Ready. The
        // captured provider/path must be gone before its terminal fact.
        drop(this.inner.take());
        if let Some(token) = this.token.take() {
            token.complete(if result.is_ok() {
                DeferredWorkTerminal::Completed
            } else {
                DeferredWorkTerminal::Failed
            });
        }
        Poll::Ready(())
    }
}

impl Drop for DetachedPartialCleanup {
    fn drop(&mut self) {
        // An abandoned/panicking inner future must retire first. The token's
        // own Drop records Missing only after its captured resources are gone.
        drop(self.inner.take());
    }
}

#[cfg(test)]
#[path = "cleanup/tests.rs"]
mod tests;
