use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::JavaLaneOwner;
use crate::{JavaCallbackInvocation, JavaInvocationError, JavaValue};

pub(super) type JavaLaneCommand = Box<dyn FnOnce(&Rc<JavaLaneOwner>) + Send>;

#[derive(Clone)]
pub(super) struct CallbackMailbox(Arc<(Mutex<MailboxState>, Condvar)>);

#[derive(Default)]
struct MailboxState {
    commands: VecDeque<JavaLaneCommand>,
    completion: Option<Result<JavaValue, JavaInvocationError>>,
    closed: bool,
}

enum MailboxMessage {
    Command(JavaLaneCommand),
    Complete(Result<JavaValue, JavaInvocationError>),
}

impl CallbackMailbox {
    pub(super) fn new() -> Self {
        Self(Arc::new((
            Mutex::new(MailboxState::default()),
            Condvar::new(),
        )))
    }

    fn command(&self, command: JavaLaneCommand) -> Result<(), JavaInvocationError> {
        let (state, ready) = &*self.0;
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.closed || state.completion.is_some() {
            return Err(JavaInvocationError::Callback(
                "Java callback reentry has ended".into(),
            ));
        }
        state.commands.push_back(command);
        ready.notify_one();
        Ok(())
    }

    fn complete(&self, result: Result<JavaValue, JavaInvocationError>) {
        let (state, ready) = &*self.0;
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.closed && state.completion.is_none() {
            state.completion = Some(result);
            ready.notify_one();
        }
    }

    fn recv(&self) -> MailboxMessage {
        let (state, ready) = &*self.0;
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            // Return has priority even if nested calls are queued. Dropping
            // those commands closes their replies and releases admissions.
            if let Some(result) = state.completion.take() {
                state.closed = true;
                state.commands.clear();
                return MailboxMessage::Complete(result);
            }
            if let Some(command) = state.commands.pop_front() {
                return MailboxMessage::Command(command);
            }
            state = ready
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

pub(super) fn wait_callback(
    mailbox: &CallbackMailbox,
    owner: &Rc<JavaLaneOwner>,
) -> Result<JavaValue, JavaInvocationError> {
    loop {
        match mailbox.recv() {
            MailboxMessage::Command(command) => command(owner),
            MailboxMessage::Complete(result) => return result,
        }
    }
}

/// One JNI callback and its exact origin-side completion authority.
pub struct JavaCallbackEvent {
    session: u64,
    token: u64,
    invocation: Option<JavaCallbackInvocation>,
    completion: Option<JavaCallbackCompletion>,
}

impl JavaCallbackEvent {
    pub(super) fn new(
        session: u64,
        token: u64,
        invocation: JavaCallbackInvocation,
    ) -> (Self, CallbackMailbox) {
        let active = Arc::new(AtomicBool::new(true));
        let mailbox = CallbackMailbox::new();
        (
            Self {
                session,
                token,
                invocation: Some(invocation),
                completion: Some(JavaCallbackCompletion {
                    mailbox: Some(mailbox.clone()),
                    active,
                }),
            },
            mailbox,
        )
    }

    pub fn take_parts(
        &mut self,
    ) -> (
        u64,
        JavaCallbackInvocation,
        JavaCallbackCompletion,
        JavaReentryAuthority,
    ) {
        let completion = self.completion.take().expect("Java callback taken once");
        let authority = JavaReentryAuthority {
            session: self.session,
            active: Arc::clone(&completion.active),
            mailbox: completion
                .mailbox
                .as_ref()
                .expect("pending Java callback")
                .clone(),
        };
        (
            self.token,
            self.invocation.take().expect("Java callback taken once"),
            completion,
            authority,
        )
    }
}

impl Drop for JavaCallbackEvent {
    fn drop(&mut self) {
        drop(self.completion.take());
    }
}

pub struct JavaCallbackCompletion {
    mailbox: Option<CallbackMailbox>,
    active: Arc<AtomicBool>,
}

impl JavaCallbackCompletion {
    pub fn finish(mut self, result: Result<JavaValue, JavaInvocationError>) {
        self.active.store(false, Ordering::Release);
        if let Some(mailbox) = self.mailbox.take() {
            mailbox.complete(result);
        }
    }
}

impl Drop for JavaCallbackCompletion {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
        if let Some(mailbox) = self.mailbox.take() {
            mailbox.complete(Err(JavaInvocationError::Callback(
                "RunMat Java callback was cancelled".into(),
            )));
        }
    }
}

/// Exact, revocable callback-generation authority for nested Java calls.
#[derive(Clone)]
pub struct JavaReentryAuthority {
    session: u64,
    active: Arc<AtomicBool>,
    mailbox: CallbackMailbox,
}

impl JavaReentryAuthority {
    pub(super) fn same_session(&self, session: u64) -> bool {
        self.session == session
    }

    pub(super) fn matches(&self, session: u64) -> bool {
        self.same_session(session) && self.active.load(Ordering::Acquire)
    }

    pub(super) fn command(&self, command: JavaLaneCommand) -> Result<(), JavaInvocationError> {
        if !self.active.load(Ordering::Acquire) {
            return Err(JavaInvocationError::Callback(
                "Java callback reentry authority has ended".into(),
            ));
        }
        self.mailbox.command(command)
    }
}
