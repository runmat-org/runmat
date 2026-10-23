use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_runtime::native::NativeSitePhase;
use std::cell::Cell;
use std::rc::Rc;
use std::task::Poll;

#[test]
fn expanded_call_inputs_never_become_final_outputs_at_an_unscoped_site() {
    runmat_gc::gc_test_context(|| {
        for first_poll_pending in [false, true] {
            let mut state = observed_state(&Rc::new(Probe::default()));
            let request = request();
            state.current_request = Some(request);
            let arguments = vec![Value::Num(7.0), Value::Num(3.0)];
            let rooted = RootedValues::new(arguments.clone()).expect("root call inputs");
            let polls = Rc::new(Cell::new(0));
            let observed = Rc::clone(&polls);
            assert!(matches!(
                begin_expanded_call_arguments(&mut state, None, || async move {
                    futures::future::poll_fn(move |cx| {
                        observed.set(observed.get() + 1);
                        if first_poll_pending && observed.get() == 1 {
                            cx.waker().wake_by_ref();
                            Poll::Pending
                        } else {
                            Poll::Ready(())
                        }
                    })
                    .await;
                    Ok(rooted)
                }),
                Err(NativeExecutorError::CallSuspended)
            ));
            assert_eq!(polls.get(), 0, "entry must not poll argument expansion");
            let pending = state
                .pending_call
                .as_ref()
                .expect("pending argument expansion");
            let (continuation, generation) = (pending.continuation, pending.generation);
            assert_eq!(
                futures::executor::block_on(complete(&mut state, continuation, generation))
                    .expect("complete argument expansion"),
                request
            );
            assert_eq!(polls.get(), if first_poll_pending { 2 } else { 1 });
            assert_eq!(completed_roots(&state).count(), 2);
            assert_eq!(
                take_completed(&mut state, None).expect("outer call must defer to argument stage"),
                None
            );
            assert_eq!(
                take_completed_expanded_call_arguments(&mut state, None)
                    .expect("consume exact call input stage"),
                Some(arguments)
            );
            assert_eq!(completed_roots(&state).count(), 0);
        }
    });
}

#[test]
fn failed_expansion_retains_its_runtime_identifier_without_publishing_outputs() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = request();
        state.current_request = Some(request);
        assert!(matches!(
            begin_expanded_call_arguments(&mut state, None, || async {
                Err(runmat_runtime::runtime_error::semantic_error(
                    "ArgumentExpansionFailure",
                    "expansion failed before call dispatch",
                ))
            }),
            Err(NativeExecutorError::CallSuspended)
        ));
        let pending = state.pending_call.as_ref().expect("pending failure");
        let (continuation, generation) = (pending.continuation, pending.generation);
        assert!(matches!(
            futures::executor::block_on(complete(&mut state, continuation, generation)),
            Err(NativeExecutorError::Runtime(error))
                if error.identifier() == Some("RunMat:ArgumentExpansionFailure")
        ));
        assert_eq!(completed_roots(&state).count(), 0);
    });
}

fn request() -> NativeSiteRequest {
    NativeSiteRequest {
        function: 0,
        block: 0,
        position: 0,
        phase: NativeSitePhase::RVALUE,
        ordinal: 0,
        reserved: 0,
    }
}
