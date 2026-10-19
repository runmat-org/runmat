use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use futures::channel::mpsc::UnboundedReceiver;
use futures::future::LocalBoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};

use crate::session::protocol::{
    PythonCallbackCommand, PythonCallbackMailbox, PythonCallbackReturn, PythonLaneCompletion,
    PythonLaneEvent, PythonLaneEventSink, PythonLaneReply,
};
use crate::{
    PythonCall, PythonCallbackInvocation, PythonError, PythonObjectHandle, PythonObjectMetadata,
    PythonValue,
};

/// An exact callback-generation capability, not a session-global reentry flag.
/// Cloned runtime contexts retain it only until the callback returns.
#[derive(Clone)]
pub struct PythonReentryAuthority {
    session: u64,
    active: Arc<AtomicBool>,
    mailbox: PythonCallbackMailbox,
}

impl PythonReentryAuthority {
    pub(crate) fn same_session(&self, session: u64) -> bool {
        self.session == session
    }

    fn new(session: u64, mailbox: PythonCallbackMailbox) -> Self {
        Self {
            session,
            active: Arc::new(AtomicBool::new(true)),
            mailbox,
        }
    }

    pub(crate) fn matches(&self, session: u64) -> bool {
        self.same_session(session) && self.active.load(Ordering::Acquire)
    }

    pub(crate) async fn invoke<F, Fut>(
        &self,
        call: PythonCall,
        observer: Option<Box<dyn super::PythonLaneObserver>>,
        callback: F,
    ) -> Result<Vec<PythonValue>, PythonError>
    where
        F: FnMut(PythonCallbackInvocation, PythonReentryAuthority) -> Fut,
        Fut: Future<Output = Result<PythonValue, PythonError>> + 'static,
    {
        if !self.active.load(Ordering::Acquire) {
            return Err(stale_reentry());
        }
        let (events, response) = futures::channel::mpsc::unbounded();
        self.mailbox.command(PythonCallbackCommand::Reenter {
            call,
            reply: PythonLaneReply::Events {
                sink: PythonLaneEventSink::Async(events),
                completion: PythonLaneCompletion::Values,
            },
            observer,
        })?;
        drive(self.session, response, callback).await?.into_values()
    }

    pub(crate) fn metadata(
        &self,
        handle: PythonObjectHandle,
    ) -> Result<PythonObjectMetadata, PythonError> {
        if !self.active.load(Ordering::Acquire) {
            return Err(stale_reentry());
        }
        let (reply, response) = mpsc::sync_channel(1);
        self.mailbox.command(PythonCallbackCommand::Metadata {
            handle,
            reply: PythonLaneReply::Sync(reply),
            observer: None,
        })?;
        response.recv().map_err(super::model::lane_ended)?
    }

    pub(crate) async fn metadata_async<F, Fut>(
        &self,
        handle: PythonObjectHandle,
        observer: Option<Box<dyn super::PythonLaneObserver>>,
        callback: F,
    ) -> Result<PythonObjectMetadata, PythonError>
    where
        F: FnMut(PythonCallbackInvocation, PythonReentryAuthority) -> Fut,
        Fut: Future<Output = Result<PythonValue, PythonError>> + 'static,
    {
        if !self.active.load(Ordering::Acquire) {
            return Err(stale_reentry());
        }
        let (events, response) = futures::channel::mpsc::unbounded();
        self.mailbox.command(PythonCallbackCommand::Metadata {
            handle,
            reply: PythonLaneReply::Events {
                sink: PythonLaneEventSink::Async(events),
                completion: PythonLaneCompletion::Metadata,
            },
            observer,
        })?;
        drive(self.session, response, callback)
            .await?
            .into_metadata()
    }
}

fn stale_reentry() -> PythonError {
    PythonError::host(
        "PythonCallbackError",
        "Python callback reentry authority has ended",
    )
}

struct CallbackCompletion {
    completion: Option<PythonCallbackReturn>,
    active: Arc<AtomicBool>,
}

impl CallbackCompletion {
    fn finish(mut self, result: Result<PythonValue, PythonError>) {
        self.active.store(false, Ordering::Release);
        if let Some(completion) = self.completion.take() {
            completion.finish(result);
        }
    }
}

impl Drop for CallbackCompletion {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
        drop(self.completion.take());
    }
}

pub(super) async fn drive<F, Fut>(
    session: u64,
    mut events: UnboundedReceiver<PythonLaneEvent>,
    mut callback: F,
) -> Result<PythonLaneCompletion, PythonError>
where
    F: FnMut(PythonCallbackInvocation, PythonReentryAuthority) -> Fut,
    Fut: Future<Output = Result<PythonValue, PythonError>> + 'static,
{
    let mut callbacks: FuturesUnordered<LocalBoxFuture<'static, ()>> = FuturesUnordered::new();
    loop {
        let event = if callbacks.is_empty() {
            events.next().await
        } else {
            use futures::FutureExt;
            let next_event = events.next().fuse();
            let next_callback = callbacks.next().fuse();
            futures::pin_mut!(next_event, next_callback);
            futures::select! {
                event = next_event => event,
                _ = next_callback => continue,
            }
        };
        match event {
            Some(PythonLaneEvent::Complete(result)) => return result,
            Some(PythonLaneEvent::Callback(mut event)) => {
                let (invocation, completion) = event.take_parts();
                let authority = PythonReentryAuthority::new(session, completion.mailbox());
                let guard = CallbackCompletion {
                    completion: Some(completion),
                    active: Arc::clone(&authority.active),
                };
                let future = callback(invocation, authority);
                callbacks.push(Box::pin(async move {
                    guard.finish(future.await);
                }));
            }
            None => {
                return Err(PythonError::host(
                    "PythonHostError",
                    "Python execution lane ended without a completion",
                ))
            }
        }
    }
}
