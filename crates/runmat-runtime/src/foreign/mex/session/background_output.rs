//! Wait on the actual application pipes as well as protocol notifications.
use std::future::Future;
use std::path::PathBuf;
use std::task::Poll;

use super::{MexRuntimeSession, RuntimeError};

impl MexRuntimeSession {
    pub(super) async fn wait_for_isolated_activity(
        &self,
        paths: &[PathBuf],
    ) -> Result<bool, RuntimeError> {
        let notified = self.isolated_ready.notified();
        tokio::pin!(notified);
        std::future::poll_fn(|context| {
            for path in paths {
                if self.active_isolated.borrow().contains(path) {
                    continue;
                }
                let Some(slot) = self.isolated.borrow().get(path).cloned() else {
                    continue;
                };
                let Ok(mut operation) = slot.acquire() else {
                    continue;
                };
                match operation.client().poll_application_output(context) {
                    Ok(true) => return Poll::Ready(Ok(true)),
                    Ok(false) => {}
                    Err(error) => return Poll::Ready(Err(error)),
                }
            }
            notified.as_mut().poll(context).map(|()| Ok(false))
        })
        .await
    }
}
