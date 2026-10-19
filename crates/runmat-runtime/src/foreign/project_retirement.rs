//! Exact, retryable authority for one project-host replacement.

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Retiring,
    Retained,
    Verified,
}

struct State {
    identity: Rc<()>,
    phase: Phase,
}

pub(super) struct ProjectRetirementAuthority {
    state: RefCell<State>,
}

impl Default for ProjectRetirementAuthority {
    fn default() -> Self {
        Self {
            state: RefCell::new(State {
                identity: Rc::new(()),
                phase: Phase::Idle,
            }),
        }
    }
}

impl ProjectRetirementAuthority {
    pub(super) fn snapshot(&self) -> Rc<()> {
        Rc::clone(&self.state.borrow().identity)
    }

    pub(super) fn is_pending(&self) -> bool {
        self.state.borrow().phase != Phase::Idle
    }

    pub(super) fn matches_idle(&self, identity: &Rc<()>) -> bool {
        let state = self.state.borrow();
        Rc::ptr_eq(&state.identity, identity) && state.phase == Phase::Idle
    }

    pub(super) fn matches_verified(&self, identity: &Rc<()>) -> bool {
        let state = self.state.borrow();
        Rc::ptr_eq(&state.identity, identity) && state.phase == Phase::Verified
    }

    /// A successful replacement without a retired host still publishes a new
    /// owner revision. Stale prepared replacements must not reuse its token.
    pub(super) fn publish_fresh(&self, identity: &Rc<()>) {
        let mut state = self.state.borrow_mut();
        assert!(Rc::ptr_eq(&state.identity, identity) && state.phase == Phase::Idle);
        state.identity = Rc::new(());
    }

    pub(super) fn begin(&self, identity: &Rc<()>) -> Option<ProjectRetirementAttempt<'_>> {
        let mut state = self.state.borrow_mut();
        if !Rc::ptr_eq(&state.identity, identity)
            || !matches!(state.phase, Phase::Idle | Phase::Retained)
        {
            return None;
        }
        state.phase = Phase::Retiring;
        Some(ProjectRetirementAttempt {
            owner: self,
            identity: Rc::clone(identity),
            verified: false,
        })
    }
}

pub(super) struct ProjectRetirementAttempt<'a> {
    owner: &'a ProjectRetirementAuthority,
    identity: Rc<()>,
    verified: bool,
}

impl<'a> ProjectRetirementAttempt<'a> {
    pub(super) fn verified(mut self) -> ProjectRetirementTicket<'a> {
        let mut state = self.owner.state.borrow_mut();
        assert!(Rc::ptr_eq(&state.identity, &self.identity) && state.phase == Phase::Retiring);
        state.phase = Phase::Verified;
        self.verified = true;
        ProjectRetirementTicket {
            owner: self.owner,
            identity: Rc::clone(&self.identity),
        }
    }
}

impl Drop for ProjectRetirementAttempt<'_> {
    fn drop(&mut self) {
        if !self.verified {
            let mut state = self.owner.state.borrow_mut();
            if Rc::ptr_eq(&state.identity, &self.identity) && state.phase == Phase::Retiring {
                state.phase = Phase::Retained;
            }
        }
    }
}

pub(super) struct ProjectRetirementTicket<'a> {
    owner: &'a ProjectRetirementAuthority,
    identity: Rc<()>,
}

impl ProjectRetirementTicket<'_> {
    pub(super) fn identity(&self) -> &Rc<()> {
        &self.identity
    }
}

impl Drop for ProjectRetirementTicket<'_> {
    fn drop(&mut self) {
        let mut state = self.owner.state.borrow_mut();
        if Rc::ptr_eq(&state.identity, &self.identity) && state.phase == Phase::Verified {
            state.identity = Rc::new(());
            state.phase = Phase::Idle;
        }
    }
}
