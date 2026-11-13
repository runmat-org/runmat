use super::*;
use crate::execute::call_suspension::{self, PendingCall, TerminatorCompletion};
use runmat_value::HandleRef;
use std::cell::Cell;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

#[test]
fn genuinely_pending_branch_completion_restores_origin_scope_and_resumes_once() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = terminator_request(&mut state);
        set_value(&mut state, NativeValueId(1), Value::Bool(true));
        let mut terminator = state.function.blocks[0].terminator.clone();
        terminator.kind = NativeTerminatorKind::Branch {
            condition: NativeValueId(1),
            then_edge: edge(),
            else_edge: edge(),
        };
        let polls = Rc::new(Cell::new(0));
        let polled = Rc::clone(&polls);
        let runtime = state.runtime.clone();
        let payload = call_suspension::terminator::prepare_payload(&runtime).unwrap();
        let future = call_suspension::admit_completion_with(&state, move || async move {
            runtime
                .scope(futures::future::poll_fn(move |cx| {
                    assert!(runmat_runtime::context::legacy::active().is_some());
                    polled.set(polled.get() + 1);
                    if polled.get() == 1 {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    } else {
                        Poll::Ready(())
                    }
                }))
                .await;
            call_suspension::terminator::completion(
                &runtime,
                payload,
                TerminatorCompletion::BranchTruth(true),
            )
        })
        .unwrap();
        let (continuation, generation) = state.next_suspension_identity().unwrap();
        state.pending_call = Some(PendingCall {
            continuation,
            generation,
            request,
            embedded: None,
            retained_roots: Vec::new(),
            future,
        });
        assert_eq!(polls.get(), 0);
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut completion = Box::pin(call_suspension::complete(
            &mut state,
            continuation,
            generation,
        ));
        assert!(completion.as_mut().poll(&mut context).is_pending());
        assert_eq!(polls.get(), 1);
        assert!(runmat_runtime::context::legacy::active().is_none());
        assert!(matches!(
            completion.as_mut().poll(&mut context),
            Poll::Ready(Ok(completed)) if completed == request
        ));
        drop(completion);
        assert_eq!(polls.get(), 2);
        let outcome = execute_terminator(
            &mut state,
            &mut NativeCall::default(),
            NativeBlockId(0),
            &terminator,
            &mut NativeExit::completed(0),
        )
        .unwrap();
        assert_eq!(outcome.edge, 0);
        assert!(state.completed_call.is_none());
    });
}

#[test]
fn unpolled_managed_branch_input_survives_allocation_retirement_then_reclaims() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::unpolled_managed_branch_input_survives_allocation_retirement_then_reclaims"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        terminator_request(&mut state);
        let allocation = runmat_gc::gc_allocate_rooted(Value::String("condition".into())).unwrap();
        let handle = allocation.handle();
        let target = (handle.addr(), handle.epoch());
        let observed = Arc::new(AtomicBool::new(false));
        let observed_in_callback = Arc::clone(&observed);
        let observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
            let runmat_gc::RootRetirement::Explicit(retired) = event else {
                return;
            };
            if (retired.addr(), retired.epoch()) != target
                || observed_in_callback.swap(true, Ordering::SeqCst)
            {
                return;
            }
            runmat_gc::gc_collect_major().unwrap();
            assert_eq!(
                runmat_gc::gc_clone_value(&retired).unwrap(),
                Value::String("condition".into())
            );
        }))
        .unwrap();
        assert!(matches!(
            call_suspension::begin_branch_truth(
                &mut state,
                Value::HandleObject(HandleRef {
                    class_name: "BranchHandle".into(),
                    target: handle,
                    valid: true,
                }),
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        allocation.unroot().unwrap();
        assert!(observed.load(Ordering::SeqCst));
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        drop(state);
        drop(observer);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
