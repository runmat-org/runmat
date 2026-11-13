//! Session admission scopes its root token slot; execution snapshots retain
//! their original generation when the owner publishes the next root token.

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
        let (token, previous) = {
            let mut state = self.0.borrow_mut();
            let parent = state
                .admitted
                .clone()
                .unwrap_or_else(|| state.source.token());
            let token = external.map_or(parent.clone(), |external| parent.or(external));
            state.depth += 1;
            let previous = state.admitted.replace(token.clone());
            (token, previous)
        };
        Admission {
            owner: self.clone(),
            token,
            previous,
            runtime_scope: None,
        }
    }
}

pub(crate) struct Admission {
    owner: SessionCancellation,
    token: CancellationToken,
    previous: Option<CancellationToken>,
    runtime_scope: Option<runmat_runtime::context::RuntimeCancellationScope>,
}

impl super::RunMatSession {
    pub(crate) fn admit_execution(&mut self, external: Option<&CancellationToken>) -> Admission {
        // Capture a live caller before installing the child/root binding. A
        // preflight check cannot inherit later cancellation or deadlines.
        let parent = runmat_runtime::interrupt::current_interrupt();
        let inherited = match (parent, external) {
            (Some(parent), Some(external)) => Some(parent.or(external)),
            (Some(parent), None) => Some(parent),
            (None, external) => external.cloned(),
        };
        let mut admission = self.cancellation.admit(inherited.as_ref());
        admission.runtime_scope = Some(
            self.runtime_context
                .scoped_cancellation_token(admission.token()),
        );
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
        let (successor, retired) = {
            let mut state = self.owner.0.borrow_mut();
            state.depth -= 1;
            if state.depth == 0 {
                let old = std::mem::take(&mut state.source);
                let token = state.admitted.take();
                (state.source.token(), Some((old, token)))
            } else {
                state.admitted = self.previous.take();
                (
                    state
                        .admitted
                        .clone()
                        .expect("nested admission has a parent"),
                    None,
                )
            }
        };
        if let Some(scope) = self.runtime_scope.take() {
            scope.retire_to(successor);
        }
        drop(retired);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutableSource, InvocationControl, ProcedureInvocation, RunMatSession};
    use runmat_execution::CancellationReason;
    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
    use runmat_runtime::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService,
    };
    use runmat_value::Value;

    struct ManagedAllocation(ManagedHostEffectLedger);

    impl RuntimeParallelService for ManagedAllocation {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    #[tokio::test]
    async fn managed_success_publishes_before_retiring_its_invocation_generation() {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let unit = session
            .compile_executable_unit(
                ExecutableSource::new(
                    "finite-success",
                    "finiteSuccess.m",
                    "function y = finiteSuccess(x); y = x + 1; end",
                ),
                None,
            )
            .await
            .unwrap();
        let invocation = || {
            let mut call = ProcedureInvocation::function("finiteSuccess", vec![Value::Num(41.0)]);
            call.requested_outputs = 1;
            call
        };
        // Hydrate the ordinary settings owner before installing the allocation grant.
        drop(
            session
                .invoke_executable(&unit, invocation(), &InvocationControl::default())
                .await
                .unwrap(),
        );
        let ledger = ManagedHostEffectLedger::new(64 * 1024 * 1024);
        session.install_parallel_service(Rc::new(ManagedAllocation(ledger.clone())));
        let external = CancellationToken::default();
        let control = InvocationControl::default().with_cancellation(external);
        let output = session
            .invoke_executable(&unit, invocation(), &control)
            .await
            .unwrap();
        assert_eq!(output.as_slice(), &[Value::Num(42.0)]);
        let snapshot = ledger.snapshot();
        assert!(snapshot.peak_reserved_bytes > 0);
        assert!(snapshot.peak_claimed_bytes <= snapshot.peak_reserved_bytes);
        assert!(snapshot.peak_reserved_bytes <= snapshot.capacity_bytes);
        assert!(!session.runtime_context.is_cancelled());
        drop(output);
        // Deadline arrives while this real managed invocation is queued, so
        // its outer checkpoint (not a language loop) produces the failure.
        let deadline = u64::try_from(runmat_time::unix_timestamp_ms()).unwrap() + 50;
        let control = InvocationControl::default().with_deadline_unix_ms(deadline);
        let pending = session.invoke_executable(&unit, invocation(), &control);
        std::thread::sleep(std::time::Duration::from_millis(60));
        let error = pending.await.unwrap_err();
        let crate::RunError::Runtime(error) = error else {
            panic!("expected runtime deadline");
        };
        assert_eq!(error.identifier(), Some("RunMat:ExecutionDeadline"));
        let original = std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<runmat_runtime::RuntimeError>())
            .expect("original checkpoint diagnostic remains owned");
        assert_eq!(original.identifier(), Some("RunMat:ExecutionCancelled"));
        assert!(
            std::error::Error::source(original)
                .and_then(|source| source
                    .downcast_ref::<runmat_runtime::context::host_allocation::HostAllocationError>(
                ))
                .is_some(),
            "typed host checkpoint cause must survive presentation"
        );
    }

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
        let child_control = CancellationSource::new();
        let child = controller.admit(Some(&child_control.token()));
        let grandchild = controller.admit(None);
        child_control.cancel(CancellationReason::User);
        assert!(grandchild.token().is_cancelled());
        drop(grandchild);
        drop(child);
        let second = controller.admit(None);
        assert!(
            !second.token().is_cancelled(),
            "retired child control leaked"
        );
        external.cancel(CancellationReason::Deadline);
        assert!(outer.token().is_cancelled());
        assert!(second.token().is_cancelled());
        drop(second);
        drop(outer);
        assert!(!controller.admit(None).token().is_cancelled());
    }

    #[tokio::test]
    async fn dropped_pending_invocation_retires_its_generation_without_resetting_snapshots() {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let unit = session
            .compile_executable_unit(
                ExecutableSource::new("drop-control", "dropInput.m", "a = input('pending', 's');"),
                None,
            )
            .await
            .unwrap();
        // The serial CPU service is NeverYield: an infinite language loop
        // cannot establish a pending future. Suspend the actual selected input
        // operation instead and prove that dropping it retires that operation.
        struct PendingInput(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for PendingInput {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::Release);
            }
        }
        let entered = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let retired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let input_entered = entered.clone();
        let input_retired = retired.clone();
        session.install_async_input_handler(move |request| {
            let entered = input_entered.clone();
            let retired = input_retired.clone();
            async move {
                assert_eq!(request.prompt, "pending");
                let _retirement = PendingInput(retired);
                entered.store(true, std::sync::atomic::Ordering::Release);
                futures::future::pending::<Result<crate::session::InputResponse, String>>().await
            }
        });
        let ledger = ManagedHostEffectLedger::new(64 * 1024 * 1024);
        session.install_parallel_service(Rc::new(ManagedAllocation(ledger.clone())));
        let root = session.runtime_context.clone();
        let old = session.interrupt_handle();
        let deadline = u64::try_from(runmat_time::unix_timestamp_ms()).unwrap() + 60_000;
        let external = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let control = InvocationControl::default()
            .with_cancellation(CancellationToken::from_shared(external.clone()))
            .with_deadline_unix_ms(deadline);
        let mut pending =
            Box::pin(session.invoke_executable(&unit, ProcedureInvocation::entrypoint(), &control));
        futures::future::poll_fn(|cx| {
            assert!(std::future::Future::poll(pending.as_mut(), cx).is_pending());
            if entered.load(std::sync::atomic::Ordering::Acquire) {
                assert_eq!(root.cancellation().deadline_unix_ms(), Some(deadline));
                std::task::Poll::Ready(())
            } else {
                std::task::Poll::Pending
            }
        })
        .await;
        assert!(!retired.load(std::sync::atomic::Ordering::Acquire));
        let retained = root.clone().with_cancellation_token(root.cancellation());
        external.store(true, std::sync::atomic::Ordering::Release);
        old.cancel(runmat_execution::CancellationReason::User);
        assert!(retained.is_cancelled());
        drop(pending);
        assert!(retired.load(std::sync::atomic::Ordering::Acquire));
        assert!(
            !root.is_cancelled(),
            "dropped admission poisoned the session"
        );
        assert_eq!(root.cancellation().deadline_unix_ms(), None);
        assert!(
            retained.is_cancelled(),
            "old execution generation was reset"
        );
        old.cancel(runmat_execution::CancellationReason::Shutdown);
        assert!(!root.is_cancelled(), "old handle cancelled the successor");
        let next = session
            .compile_executable_unit(
                ExecutableSource::new("after-drop", "afterDrop.m", "ans = 42;"),
                None,
            )
            .await
            .unwrap();
        let mut invocation = ProcedureInvocation::entrypoint();
        invocation.requested_outputs = 1;
        assert_eq!(
            session
                .invoke_executable(&next, invocation, &InvocationControl::default())
                .await
                .unwrap()
                .as_slice(),
            &[Value::Num(42.0)]
        );
    }
}
