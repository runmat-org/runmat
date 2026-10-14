use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_value::MException;
use std::cell::Cell;
use std::rc::Rc;
use std::task::{Context, Poll};

struct PendingCatchProbe {
    runtime: runmat_runtime::context::RuntimeContext,
    polls: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
}

impl Future for PendingCatchProbe {
    type Output = NativeExecutorResult<CallCompletion>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        assert_eq!(
            self.runtime.active_catch_exception().unwrap().identifier,
            "RunMat:native"
        );
        self.polls.set(self.polls.get() + 1);
        Poll::Pending
    }
}

impl Drop for PendingCatchProbe {
    fn drop(&mut self) {
        assert_eq!(
            self.runtime.active_catch_exception().unwrap().identifier,
            "RunMat:native"
        );
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn pending_native_call_scopes_each_poll_and_drop() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        state.function.blocks[0].catch_scopes = vec![runmat_mir::CatchScopeId(0)];
        let entry = state.function.entry;
        state
            .bind_catch_exception(
                entry,
                MException::new("RunMat:native".into(), "native".into()),
            )
            .unwrap();
        state.enter_site_block(entry);
        assert_eq!(
            state.active_catch_exception().unwrap().identifier,
            "RunMat:native"
        );
        let polls = Rc::new(Cell::new(0));
        let drops = Rc::new(Cell::new(0));
        let future: CompletionFuture = Box::pin(PendingCatchProbe {
            runtime: state.runtime.clone(),
            polls: polls.clone(),
            drops: drops.clone(),
        });
        let mut future = scope_completion(&state, future);
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert!(state.runtime.active_catch_exception().is_none());
        drop(future);
        assert_eq!(polls.get(), 1);
        assert_eq!(drops.get(), 1);
        assert!(state.runtime.active_catch_exception().is_none());
    });
}
