use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_mir::{MirConstant, MirOperand};
use runmat_runtime::native::{NativeSitePhase, NativeSiteRequest};
use std::rc::Rc;

#[test]
fn capture_materialization_suspends_before_poll_then_resumes_once() {
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
        let destination = MirSequenceLocalId(4);
        let source = MirExpansionSource::ReturnedOutputs(MirOperand::Constant(
            MirConstant::Number("8".into()),
        ));
        assert!(matches!(
            capture_sequence(&mut state, destination, &source),
            Err(NativeExecutorError::CallSuspended)
        ));
        assert!(!state.captured_sequences.contains_key(&destination));
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        assert_eq!(
            futures::executor::block_on(super::super::call_suspension::complete(
                &mut state,
                continuation,
                generation,
            ))
            .unwrap(),
            request
        );
        capture_sequence(&mut state, destination, &source).unwrap();
        assert_eq!(
            take_captured_sequence(&mut state, destination).unwrap(),
            vec![Value::Num(8.0)]
        );
        assert!(state.completed_call.is_none());
    });
}
