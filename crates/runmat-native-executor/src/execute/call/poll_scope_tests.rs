use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_runtime::execution::RootedValueSequence;
use runmat_runtime::native::{NativeSitePhase, NativeSiteRequest};
use std::cell::Cell;
use std::rc::Rc;
use std::task::{Context, Poll};

#[test]
fn resumed_call_presents_output_and_class_context_only_during_its_poll() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = NativeSiteRequest {
            function: 0,
            block: 0,
            position: 0,
            phase: NativeSitePhase::RVALUE,
            ordinal: 0,
            reserved: 0,
        };
        state.current_request = Some(request);
        let class = runmat_types::ClassIdentity::new("NativePollClass").unwrap();
        let class_in_future = class.clone();
        let polls = Rc::new(Cell::new(0));
        let polls_in_future = Rc::clone(&polls);
        let outcome = complete_rooted_call(
            &mut state,
            None,
            with_output_count(
                1,
                with_class_access_context(Some(class), async move {
                    futures::future::poll_fn(move |context| {
                        assert_eq!(
                            runmat_runtime::output_context::requested_output_count(),
                            Some(1)
                        );
                        assert_eq!(
                            runmat_runtime::class_access_context(),
                            Some(class_in_future.clone().into())
                        );
                        polls_in_future.set(polls_in_future.get() + 1);
                        if polls_in_future.get() == 1 {
                            context.waker().wake_by_ref();
                            Poll::Pending
                        } else {
                            Poll::Ready(())
                        }
                    })
                    .await;
                    let sequence = runmat_value::ValueSequence::single(Value::Num(3.0)).unwrap();
                    RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
                }),
            ),
            1,
        );
        assert!(matches!(outcome, Err(NativeExecutorError::CallSuspended)));
        assert_eq!(polls.get(), 0);
        let runtime = state.runtime.clone();
        let pending = state.pending_call.as_ref().unwrap();
        let identity = (pending.continuation, pending.generation);
        let outer_class = runmat_types::ClassIdentity::new("OuterPollClass").unwrap();
        futures::executor::block_on(runtime.scope(async {
            let _outer_output = runmat_runtime::output_context::push_output_count(2)
                .expect("outer output-count scope");
            let _outer_class =
                runmat_runtime::push_class_access_context(Some(outer_class.clone())).unwrap();
            let pending = state.pending_call.as_mut().unwrap();
            let mut context = Context::from_waker(futures::task::noop_waker_ref());
            assert!(pending.future.as_mut().poll(&mut context).is_pending());
            assert_eq!(
                runmat_runtime::output_context::requested_output_count(),
                Some(2)
            );
            assert_eq!(
                runmat_runtime::class_access_context(),
                Some(outer_class.into())
            );
        }));
        assert_eq!(polls.get(), 1);
        futures::executor::block_on(runtime.scope(async {
            assert_eq!(
                runmat_runtime::output_context::requested_output_count(),
                None
            );
            assert_eq!(runmat_runtime::class_access_context(), None);
        }));
        assert_eq!(
            futures::executor::block_on(super::super::call_suspension::complete(
                &mut state, identity.0, identity.1,
            ))
            .unwrap(),
            request
        );
        assert_eq!(polls.get(), 2);
        assert_eq!(
            super::super::call_suspension::take_completed(&mut state, None).unwrap(),
            Some(vec![Value::Num(3.0)])
        );
    });
}
