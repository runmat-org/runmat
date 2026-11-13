use crate::{PythonCallbackInvocation, PythonObjectHandle, PythonObjectMetadata};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum PythonCallbackCommand {
    Reenter {
        call: crate::PythonCall,
        reply: PythonLaneReply<Vec<crate::PythonValue>>,
        observer: Option<Box<dyn super::PythonLaneObserver>>,
    },
    Metadata {
        handle: PythonObjectHandle,
        reply: PythonLaneReply<PythonObjectMetadata>,
        observer: Option<Box<dyn super::PythonLaneObserver>>,
    },
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum PythonLaneReply<T> {
    Sync(std::sync::mpsc::SyncSender<Result<T, crate::PythonError>>),
    Events {
        sink: PythonLaneEventSink,
        completion: fn(T) -> PythonLaneCompletion,
    },
}

#[cfg(not(target_arch = "wasm32"))]
impl<T> PythonLaneReply<T> {
    pub(crate) fn send(self, result: Result<T, crate::PythonError>) {
        match self {
            Self::Sync(reply) => {
                let _ = reply.send(result);
            }
            Self::Events { sink, completion } => {
                let _ = sink.send(PythonLaneEvent::Complete(result.map(completion)));
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum PythonLaneEvent {
    Complete(Result<PythonLaneCompletion, crate::PythonError>),
    Callback(PythonCallbackEvent),
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum PythonLaneCompletion {
    Values(Vec<crate::PythonValue>),
    Metadata(PythonObjectMetadata),
}

#[cfg(not(target_arch = "wasm32"))]
impl PythonLaneCompletion {
    pub(crate) fn into_values(self) -> Result<Vec<crate::PythonValue>, crate::PythonError> {
        match self {
            Self::Values(values) => Ok(values),
            Self::Metadata(_) => Err(wrong_completion()),
        }
    }

    pub(crate) fn into_metadata(self) -> Result<PythonObjectMetadata, crate::PythonError> {
        match self {
            Self::Metadata(metadata) => Ok(metadata),
            Self::Values(_) => Err(wrong_completion()),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn wrong_completion() -> crate::PythonError {
    crate::PythonError::host(
        "PythonHostError",
        "Python lane returned a response for a different operation",
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct PythonCallbackEvent {
    invocation: Option<PythonCallbackInvocation>,
    completion: Option<PythonCallbackReturn>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct PythonCallbackReturn {
    mailbox: Option<PythonCallbackMailbox>,
}

#[cfg(not(target_arch = "wasm32"))]
impl PythonCallbackReturn {
    pub(crate) fn mailbox(&self) -> PythonCallbackMailbox {
        self.mailbox
            .as_ref()
            .expect("callback return is pending")
            .clone()
    }

    pub(crate) fn finish(mut self, result: Result<crate::PythonValue, crate::PythonError>) {
        if let Some(mailbox) = self.mailbox.take() {
            mailbox.complete(result);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for PythonCallbackReturn {
    fn drop(&mut self) {
        if let Some(mailbox) = self.mailbox.take() {
            mailbox.complete(Err(crate::PythonError::host(
                "PythonCallbackError",
                "RunMat callback was cancelled",
            )));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl PythonCallbackEvent {
    pub(crate) fn new(
        invocation: PythonCallbackInvocation,
        mailbox: PythonCallbackMailbox,
    ) -> Self {
        Self {
            invocation: Some(invocation),
            completion: Some(PythonCallbackReturn {
                mailbox: Some(mailbox),
            }),
        }
    }

    pub(crate) fn take_parts(&mut self) -> (PythonCallbackInvocation, PythonCallbackReturn) {
        (
            self.invocation
                .take()
                .expect("callback invocation is transferred once"),
            self.completion
                .take()
                .expect("callback return is transferred once"),
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for PythonCallbackEvent {
    fn drop(&mut self) {
        drop(self.completion.take());
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) struct PythonCallbackMailbox {
    inner: std::sync::Arc<(
        std::sync::Mutex<PythonCallbackMailboxState>,
        std::sync::Condvar,
    )>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct PythonCallbackMailboxState {
    commands: std::collections::VecDeque<PythonCallbackCommand>,
    completion: Option<Result<crate::PythonValue, crate::PythonError>>,
    closed: bool,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) enum PythonCallbackMessage {
    Command(PythonCallbackCommand),
    Complete(Result<crate::PythonValue, crate::PythonError>),
}

#[cfg(not(target_arch = "wasm32"))]
impl PythonCallbackMailbox {
    pub(crate) fn new() -> Self {
        Self {
            inner: std::sync::Arc::new((
                std::sync::Mutex::new(PythonCallbackMailboxState::default()),
                std::sync::Condvar::new(),
            )),
        }
    }

    pub(crate) fn command(&self, command: PythonCallbackCommand) -> Result<(), crate::PythonError> {
        let (state, ready) = &*self.inner;
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Every producer holds the exact SessionAdmission active-call lease
        // until its reply. The queue therefore tracks admitted reentries,
        // not a second caller-controlled payload budget or magic cap.
        if state.closed || state.completion.is_some() {
            return Err(crate::PythonError::host(
                "PythonCallbackError",
                "Python callback command mailbox is unavailable",
            ));
        }
        state.commands.push_back(command);
        ready.notify_one();
        Ok(())
    }

    pub(crate) fn complete(&self, result: Result<crate::PythonValue, crate::PythonError>) {
        let (state, ready) = &*self.inner;
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.closed && state.completion.is_none() {
            state.completion = Some(result);
            ready.notify_one();
        }
    }

    pub(crate) fn recv(&self) -> PythonCallbackMessage {
        let (state, ready) = &*self.inner;
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(result) = state.completion.take() {
                state.closed = true;
                state.commands.clear();
                return PythonCallbackMessage::Complete(result);
            }
            if let Some(command) = state.commands.pop_front() {
                return PythonCallbackMessage::Command(command);
            }
            state = ready
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

/// The interpreter lane publishes one canonical event protocol to either a
/// synchronous standalone caller or an async RunMat invocation owner.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) enum PythonLaneEventSink {
    Sync(std::sync::mpsc::SyncSender<PythonLaneEvent>),
    Async(futures::channel::mpsc::UnboundedSender<PythonLaneEvent>),
}

#[cfg(not(target_arch = "wasm32"))]
impl PythonLaneEventSink {
    pub(crate) fn send(&self, event: PythonLaneEvent) -> Result<(), ()> {
        match self {
            Self::Sync(sender) => sender.send(event).map_err(|_| ()),
            Self::Async(sender) => sender.unbounded_send(event).map_err(|_| ()),
        }
    }
}
