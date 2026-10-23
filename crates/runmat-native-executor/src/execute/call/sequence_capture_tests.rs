use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_mir::{MirConstant, MirOperand};
use runmat_runtime::native::{NativeSitePhase, NativeSiteRequest};
use std::rc::Rc;

#[test]
fn consumed_selector_keeps_original_registration_until_successor_or_abandonment() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let id = MirSequenceLocalId(9);
        for publish_successor in [true, false] {
            let initial = runmat_gc::gc_allocate_rooted(Value::Num(29.)).unwrap();
            let handle = initial.handle();
            let value = Value::HandleObject(runmat_value::HandleRef {
                class_name: "SelectorLeaseFixture".into(),
                target: handle,
                valid: true,
            });
            let sequence = runmat_runtime::sequence::ValueSequence::single(value).unwrap();
            let sequence = runmat_runtime::execution::RootedValueSequence::in_context(
                sequence,
                &state.runtime,
            )
            .unwrap();
            state.prepare_captured_sequence_slot(id).unwrap();
            state.publish_captured_sequence(id, sequence).unwrap();
            initial.unroot().unwrap();
            let mut values = state.take_selector_sequence(id).unwrap();
            let raw_selector = values.next().unwrap();
            drop(values);
            // This raw selector is not itself a registered root. Later
            // component allocation/GC must still see the original authority.
            assert!(state.captured_sequence_width(id).is_err());
            assert!(state.take_selector_sequence(id).is_err());
            assert!(state.prepare_captured_sequence_slot(id).is_err());
            assert!(state.captured_sequences_live());
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            if publish_successor {
                let successor = RootedValueLease::for_values_iterator_in_context(
                    [&raw_selector].into_iter(),
                    &state.runtime,
                )
                .unwrap();
                state.retire_selector_sequence_lease(id);
                assert!(!state.captured_sequences_live());
                runmat_gc::gc_collect_major().unwrap();
                assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                drop(successor);
            } else {
                state.abandon_site_continuations().unwrap();
                assert!(!state.captured_sequences_live());
            }
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
            drop(raw_selector);
        }
    });
}

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
        assert!(!state.captured_sequence_live(destination));
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
            take_captured_sequence(&mut state, destination)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![Value::Num(8.0)]
        );
        assert!(state.completed_call.is_none());
    });
}

#[test]
fn phase_sequence_move_preserves_backing_and_preflights_destination() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let from = MirSequenceLocalId(3);
        let to = MirSequenceLocalId(4);
        let values = vec![Value::Num(2.0), Value::Num(5.0)];
        let pointer = values.as_ptr();
        let sequence = runmat_runtime::sequence::ValueSequence::comma_separated(values).unwrap();
        let sequence =
            runmat_runtime::execution::RootedValueSequence::in_context(sequence, &state.runtime)
                .unwrap();
        state.prepare_captured_sequence_slot(from).unwrap();
        state.publish_captured_sequence(from, sequence).unwrap();
        state.move_captured_sequence(from, to).unwrap();
        assert!(!state.captured_sequence_live(from));
        assert_eq!(state.captured_sequence_width(to).unwrap(), 2);
        // Normal native site retirement must retain a live frame receipt.
        state.finish_contextual_site().unwrap();
        assert!(state.captured_sequence_live(to));
        let owner = state.take_captured_sequence_owner(to).unwrap();
        assert_eq!(owner.as_slice().as_ptr(), pointer);
        assert!(!state.captured_sequences_live());
        state.prepare_captured_sequence_slot(from).unwrap();
        state.publish_captured_sequence(from, owner).unwrap();
        let occupied =
            runmat_runtime::sequence::ValueSequence::comma_separated(vec![Value::Num(9.0)])
                .unwrap();
        let occupied =
            runmat_runtime::execution::RootedValueSequence::in_context(occupied, &state.runtime)
                .unwrap();
        state.publish_captured_sequence(to, occupied).unwrap();
        assert!(state.move_captured_sequence(from, to).is_err());
        assert_eq!(state.captured_sequence_width(from).unwrap(), 2);
        state.abandon_site_continuations().unwrap();
        assert!(!state.captured_sequences_live());

        // Shared canonical output aliases remain supported: projection keeps
        // the original backing/roots, rather than refusing or recollecting.
        let shared = runmat_value::OutputListStorage::shared_unbounded(vec![
            Value::Num(7.0),
            Value::Num(8.0),
        ]);
        let alias = shared.clone();
        let sequence =
            runmat_runtime::sequence::ValueSequence::from_output_storage(shared).unwrap();
        let sequence =
            runmat_runtime::execution::RootedValueSequence::in_context(sequence, &state.runtime)
                .unwrap();
        state.prepare_captured_sequence_slot(from).unwrap();
        state.publish_captured_sequence(from, sequence).unwrap();
        let mut values = take_captured_sequence(&mut state, from).unwrap();
        assert_eq!(values.next(), Some(Value::Num(7.0)));
        assert_eq!(values.next_back(), Some(Value::Num(8.0)));
        assert_eq!(alias.as_slice(), &[Value::Num(7.0), Value::Num(8.0)]);
        assert!(values.next().is_none());
    });
}
