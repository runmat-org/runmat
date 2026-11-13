use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_runtime::execution::RootedValueLease;
use runmat_runtime::native::{NativeSitePhase, NativeSiteRequest};
use runmat_value::{HandleRef, Value, ValueSequence};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn state() -> HostState {
    let mut state = observed_state(&Rc::new(Probe::default()));
    state.current_request = Some(NativeSiteRequest {
        function: 0,
        block: 0,
        position: 0,
        phase: NativeSitePhase::RVALUE,
        ordinal: 0,
        reserved: 0,
    });
    state
}

#[test]
fn sequence_read_reenters_only_the_exact_typed_site_without_replaying() {
    runmat_gc::gc_test_context(|| {
        let mut state = state();
        let stage = SequenceReadStage::Capture {
            destination: MirSequenceLocalId(3),
        };
        let polls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&polls);
        assert!(matches!(
            begin_sequence_read(&mut state, stage, || async move {
                observed.fetch_add(1, Ordering::SeqCst);
                RootedValueSequence::new(ValueSequence::single(Value::Num(9.0)).unwrap())
                    .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))
            },),
            Err(NativeExecutorError::CallSuspended)
        ));
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        futures::executor::block_on(super::super::complete(&mut state, continuation, generation))
            .unwrap();
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        assert!(matches!(
            take_completed_sequence_read(
                &mut state,
                SequenceReadStage::Capture {
                    destination: MirSequenceLocalId(4),
                },
            ),
            Err(NativeExecutorError::Host(_))
        ));
        assert!(state.completed_call.is_some());
        state.current_request.as_mut().unwrap().ordinal = 1;
        assert!(matches!(
            take_completed_sequence_read(&mut state, stage),
            Err(NativeExecutorError::Host(_))
        ));
        state.current_request.as_mut().unwrap().ordinal = 0;
        let value = take_completed_sequence_read(&mut state, stage)
            .unwrap()
            .unwrap();
        assert_eq!(value.value().as_slice(), &[Value::Num(9.0)]);
        assert_eq!(polls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn unpolled_sequence_input_lease_retires_only_after_pending_future_drop() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::unpolled_sequence_input_lease_retires_only_after_pending_future_drop"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let mut state = state();
        let allocation = runmat_gc::gc_allocate_rooted(Value::String("sequence".into())).unwrap();
        let handle = allocation.handle();
        let input = Value::HandleObject(HandleRef {
            class_name: "SequenceInput".into(),
            target: handle,
            valid: true,
        });
        let roots = RootedValueLease::for_values(std::iter::once(&input)).unwrap();
        assert!(matches!(
            begin_sequence_read(
                &mut state,
                SequenceReadStage::Capture {
                    destination: MirSequenceLocalId(0),
                },
                || async move {
                    let _retained = (input, roots);
                    std::future::pending::<NativeExecutorResult<RootedValueSequence>>().await
                },
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        allocation.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(
            runmat_gc::gc_clone_value(&handle).unwrap(),
            Value::String("sequence".into())
        );
        drop(state.pending_call.take());
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
