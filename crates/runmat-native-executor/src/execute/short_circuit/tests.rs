use super::*;
use crate::execute::call_suspension;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_mir::{MirConstant, MirLocalId, MirRvalue};
use runmat_runtime::native::{NativeSitePhase, NativeSiteRequest};
use std::rc::Rc;

fn request(state: &mut HostState) -> NativeSiteRequest {
    let request = NativeSiteRequest {
        function: 0,
        block: 0,
        position: 0,
        phase: NativeSitePhase::RVALUE,
        ordinal: 0,
        reserved: 0,
    };
    state.current_request = Some(request);
    request
}

fn complete(state: &mut HostState, request: NativeSiteRequest) {
    let pending = state
        .pending_call
        .as_ref()
        .expect("short-circuit truth suspended");
    assert_eq!(pending.request, request);
    let (continuation, generation) = (pending.continuation, pending.generation);
    assert_eq!(
        futures::executor::block_on(call_suspension::complete(state, continuation, generation,))
            .unwrap(),
        request
    );
}

#[test]
fn false_left_skips_rhs_without_materializing_its_missing_local() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = request(&mut state);
        let left = MirOperand::Constant(MirConstant::Bool(false));
        let right = MirOperand::Local(MirLocalId(0));
        state.begin_contextual_site(request).unwrap();
        assert!(matches!(
            evaluate(&mut state, &left, MirShortCircuitOp::And, &[], &right),
            Err(NativeExecutorError::CallSuspended)
        ));
        complete(&mut state, request);
        state.begin_contextual_site(request).unwrap();
        let values = evaluate(&mut state, &left, MirShortCircuitOp::And, &[], &right).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(state.arena.get(values[0]).unwrap(), &Value::Bool(false));
        assert!(state.pending_call.is_none());
        state.finish_contextual_site().unwrap();
    });
}

#[test]
fn rhs_statement_is_not_replayed_when_final_truth_suspends() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = request(&mut state);
        let initial = state
            .arena
            .insert(Value::Num(0.0), Some(&state.runtime))
            .unwrap();
        state.locals.push(initial);
        let left = MirOperand::Constant(MirConstant::Bool(true));
        let right = MirOperand::Local(MirLocalId(0));
        let statements = [MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(0)),
                value: MirRvalue::Use(MirOperand::Constant(MirConstant::Number("7".into()))),
            },
            span: runmat_types::Span::default(),
        }];
        state.begin_contextual_site(request).unwrap();
        assert!(matches!(
            evaluate(
                &mut state,
                &left,
                MirShortCircuitOp::And,
                &statements,
                &right
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        complete(&mut state, request);
        state.begin_contextual_site(request).unwrap();
        assert!(matches!(
            evaluate(
                &mut state,
                &left,
                MirShortCircuitOp::And,
                &statements,
                &right
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        let assigned = state.locals[0];
        assert_ne!(assigned, initial);
        assert_eq!(state.arena.get(assigned).unwrap(), &Value::Num(7.0));
        complete(&mut state, request);
        state.begin_contextual_site(request).unwrap();
        let values = evaluate(
            &mut state,
            &left,
            MirShortCircuitOp::And,
            &statements,
            &right,
        )
        .expect("resume final truth without replaying assignment");
        assert_eq!(state.locals[0], assigned);
        assert_eq!(state.arena.get(values[0]).unwrap(), &Value::Bool(true));
        state.finish_contextual_site().unwrap();
    });
}
