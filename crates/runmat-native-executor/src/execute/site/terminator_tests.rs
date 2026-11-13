use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_native_codegen::{NativeBlockId, NativeValueId};
use runmat_value::Value;
use std::rc::Rc;

#[path = "terminator_tests/witnesses.rs"]
mod witnesses;

fn edge() -> NativeEdge {
    NativeEdge {
        target: NativeBlockId(0),
        arguments: vec![],
        side_effect_epoch: NativeValueId(0),
    }
}

fn set_value(state: &mut HostState, id: NativeValueId, value: Value) {
    let reference = state.arena.insert(value, Some(&state.runtime)).unwrap();
    state.values.insert(id, reference);
}

fn terminator_request(state: &mut HostState) -> NativeSiteRequest {
    let request = NativeSiteRequest {
        function: 0,
        block: 0,
        position: 0,
        phase: NativeSitePhase::TERMINATOR,
        ordinal: 0,
        reserved: 0,
    };
    state.current_request = Some(request);
    request
}

fn complete_pending(state: &mut HostState, request: NativeSiteRequest) {
    let pending = state
        .pending_call
        .as_ref()
        .expect("typed terminator pending");
    let identity = (pending.continuation, pending.generation);
    assert_eq!(pending.request, request);
    assert!(state.completed_call.is_none());
    assert_eq!(
        futures::executor::block_on(super::super::call_suspension::complete(
            state, identity.0, identity.1,
        ))
        .expect("complete typed terminator"),
        request
    );
}

#[test]
fn branch_ready_truth_suspends_without_polling_and_resumes_once() {
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
        let mut call = NativeCall::default();
        let mut exit = NativeExit::completed(0);
        assert!(matches!(
            execute_terminator(
                &mut state,
                &mut call,
                NativeBlockId(0),
                &terminator,
                &mut exit
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        assert!(state.pending_call.is_some());
        complete_pending(&mut state, request);
        let outcome = execute_terminator(
            &mut state,
            &mut call,
            NativeBlockId(0),
            &terminator,
            &mut exit,
        )
        .expect("resume branch once");
        assert_eq!(
            outcome.kind,
            runmat_runtime::native::NativeSiteOutcomeKind::EDGE
        );
        assert_eq!(outcome.edge, 0);
        assert!(state.pending_call.is_none());
        assert!(state.completed_call.is_none());
    });
}

#[test]
fn switch_reads_later_case_only_after_prior_equality_and_truth_finish() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = terminator_request(&mut state);
        set_value(&mut state, NativeValueId(1), Value::Num(9.0));
        set_value(&mut state, NativeValueId(2), Value::Num(1.0));
        set_value(&mut state, NativeValueId(3), Value::Num(2.0));
        let mut terminator = state.function.blocks[0].terminator.clone();
        terminator.kind = NativeTerminatorKind::Switch {
            discriminant: NativeValueId(1),
            cases: vec![(NativeValueId(2), edge()), (NativeValueId(3), edge())],
            otherwise: edge(),
        };
        let mut call = NativeCall::default();
        let mut exit = NativeExit::completed(0);
        assert!(matches!(
            execute_terminator(
                &mut state,
                &mut call,
                NativeBlockId(0),
                &terminator,
                &mut exit
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        complete_pending(&mut state, request);
        assert!(matches!(
            execute_terminator(
                &mut state,
                &mut call,
                NativeBlockId(0),
                &terminator,
                &mut exit
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        complete_pending(&mut state, request);
        // The second case is still unread. A callback from the first case can
        // update its SSA/workspace-visible reference before false truth resumes.
        set_value(&mut state, NativeValueId(3), Value::Num(9.0));
        assert!(matches!(
            execute_terminator(
                &mut state,
                &mut call,
                NativeBlockId(0),
                &terminator,
                &mut exit
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        complete_pending(&mut state, request);
        assert!(matches!(
            execute_terminator(
                &mut state,
                &mut call,
                NativeBlockId(0),
                &terminator,
                &mut exit
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        complete_pending(&mut state, request);
        let outcome = execute_terminator(
            &mut state,
            &mut call,
            NativeBlockId(0),
            &terminator,
            &mut exit,
        )
        .expect("second case takes its edge");
        assert_eq!(
            outcome.kind,
            runmat_runtime::native::NativeSiteOutcomeKind::EDGE
        );
        assert_eq!(outcome.edge, 1);
        assert!(state.pending_call.is_none());
        assert!(state.completed_call.is_none());
    });
}

#[test]
fn for_capture_and_next_restore_the_same_loop_before_column_publication() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = terminator_request(&mut state);
        assert!(matches!(
            super::super::call_suspension::begin_for_capture(&mut state, Value::Num(7.0)),
            Err(NativeExecutorError::CallSuspended)
        ));
        assert!(state.pending_call.is_some());
        complete_pending(&mut state, request);
        let Some((super::super::call_suspension::TerminatorReadyStage::ForCapture(iterator), None)) =
            super::super::call_suspension::take_completed_terminator(&mut state)
                .unwrap()
                .map(super::super::call_suspension::TerminatorReady::into_parts)
        else {
            panic!("captured iterable must have one typed completion");
        };
        state.start_for_loop(NativeBlockId(0), NativeBlockId(0), iterator);
        assert!(matches!(
            super::super::call_suspension::begin_for_next(&mut state, NativeBlockId(0)),
            Err(NativeExecutorError::CallSuspended)
        ));
        assert!(!state.has_for_loop(NativeBlockId(0)));
        complete_pending(&mut state, request);
        assert!(state.has_for_loop(NativeBlockId(0)));
        let Some((
            super::super::call_suspension::TerminatorReadyStage::ForNext(Some(reference)),
            None,
        )) = super::super::call_suspension::take_completed_terminator(&mut state)
            .unwrap()
            .map(super::super::call_suspension::TerminatorReady::into_parts)
        else {
            panic!("first loop column must be owned by the native arena");
        };
        assert_eq!(state.arena.get(reference).unwrap(), &Value::Num(7.0));
        assert!(matches!(
            super::super::call_suspension::begin_for_next(&mut state, NativeBlockId(0)),
            Err(NativeExecutorError::CallSuspended)
        ));
        complete_pending(&mut state, request);
        assert!(state.has_for_loop(NativeBlockId(0)));
        assert!(matches!(
            super::super::call_suspension::take_completed_terminator(&mut state)
                .unwrap()
                .map(super::super::call_suspension::TerminatorReady::into_parts),
            Some((
                super::super::call_suspension::TerminatorReadyStage::ForNext(None),
                None
            ))
        ));
    });
}

#[test]
fn typed_terminator_completion_rejects_other_site_without_consuming_it() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = terminator_request(&mut state);
        assert!(matches!(
            super::super::call_suspension::begin_branch_truth(&mut state, Value::Bool(false)),
            Err(NativeExecutorError::CallSuspended)
        ));
        complete_pending(&mut state, request);
        state.current_request = Some(NativeSiteRequest {
            ordinal: 1,
            ..request
        });
        assert!(matches!(
            super::super::call_suspension::take_completed_terminator(&mut state),
            Err(NativeExecutorError::Host(_))
        ));
        assert!(state.completed_call.is_some());
        state.current_request = Some(request);
        assert!(matches!(
            super::super::call_suspension::take_completed_terminator(&mut state)
                .unwrap()
                .map(super::super::call_suspension::TerminatorReady::into_parts),
            Some((
                super::super::call_suspension::TerminatorReadyStage::BranchTruth(false),
                None
            ))
        ));
    });
}
