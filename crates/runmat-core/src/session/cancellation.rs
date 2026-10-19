//! Session admission owns generation publication; runtime tokens never reset.

use std::cell::RefCell;
use std::rc::Rc;

use runmat_execution::cancellation::{CancellationSource, CancellationToken};

#[derive(Clone, Default)]
pub(super) struct SessionCancellation(Rc<RefCell<State>>);

#[derive(Default)]
struct State {
    source: CancellationSource,
    depth: usize,
    admitted: Option<CancellationToken>,
}

impl SessionCancellation {
    pub fn handle(&self) -> CancellationSource {
        self.0.borrow().source.clone()
    }

    pub fn admit(&self, external: Option<&CancellationToken>) -> Admission {
        let token = {
            let mut state = self.0.borrow_mut();
            let parent = state
                .admitted
                .clone()
                .unwrap_or_else(|| state.source.token());
            let token = external.map_or(parent.clone(), |external| parent.or(external));
            state.depth += 1;
            if state.admitted.is_none() {
                state.admitted = Some(token.clone());
            }
            token
        };
        Admission {
            owner: self.clone(),
            token,
        }
    }
}

pub(crate) struct Admission {
    owner: SessionCancellation,
    token: CancellationToken,
}

impl super::RunMatSession {
    pub(crate) fn admit_execution(&mut self, external: Option<&CancellationToken>) -> Admission {
        let admission = self.cancellation.admit(external);
        self.runtime_context = self
            .runtime_context
            .clone()
            .with_cancellation_token(admission.token());
        admission
    }
}

impl Admission {
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        let retired = {
            let mut state = self.owner.0.borrow_mut();
            state.depth -= 1;
            if state.depth == 0 {
                let old = std::mem::take(&mut state.source);
                let token = state.admitted.take();
                Some((old, token))
            } else {
                None
            }
        };
        drop(retired);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::CancellationReason;

    #[test]
    fn preadmission_signal_survives_and_old_handle_cannot_cancel_next_generation() {
        let controller = SessionCancellation::default();
        let old = controller.handle();
        old.cancel(CancellationReason::User);
        let admission = controller.admit(None);
        assert!(admission.token().is_cancelled());
        drop(admission);
        let next = controller.admit(None);
        assert!(!next.token().is_cancelled());
        old.cancel(CancellationReason::Shutdown);
        assert!(!next.token().is_cancelled());
    }

    #[test]
    fn nested_fixture_procedures_share_outer_generation_and_external_control() {
        let controller = SessionCancellation::default();
        let external = CancellationSource::new();
        let outer = controller.admit(Some(&external.token()));
        let first = controller.admit(None);
        drop(first);
        let second = controller.admit(None);
        external.cancel(CancellationReason::Deadline);
        assert!(outer.token().is_cancelled());
        assert!(second.token().is_cancelled());
        drop(second);
        drop(outer);
        assert!(!controller.admit(None).token().is_cancelled());
    }
}
