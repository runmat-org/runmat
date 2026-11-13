use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use runmat_runtime::native::{NativeSitePhase, NativeSiteRequest};
use runmat_value::HandleRef;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn enter_left(state: &mut HostState, request: NativeSiteRequest) {
    state.current_request = Some(request);
    state.begin_contextual_site(request).unwrap();
    let (region, next, completed) = state.enter_contextual_region().unwrap();
    assert_eq!(next, 0);
    assert!(completed.is_none());
    state.enter_contextual_step(region, 0).unwrap();
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

#[test]
fn exact_region_step_rejects_other_site_without_consuming_truth() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = request();
        enter_left(&mut state, request);
        assert!(matches!(
            begin_short_circuit_truth(&mut state, ShortCircuitTruthStage::Left, Value::Bool(true)),
            Err(NativeExecutorError::CallSuspended)
        ));
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        assert_eq!(
            futures::executor::block_on(super::super::complete(
                &mut state,
                continuation,
                generation,
            ))
            .unwrap(),
            request
        );
        enter_left(&mut state, request);
        state.current_request = Some(NativeSiteRequest {
            ordinal: 1,
            ..request
        });
        assert!(matches!(
            take_completed_short_circuit_truth(&mut state, ShortCircuitTruthStage::Left),
            Err(NativeExecutorError::Host(_))
        ));
        assert!(state.completed_call.is_some());
        state.current_request = Some(request);
        assert!(matches!(
            take_completed_short_circuit_truth(
                &mut state,
                ShortCircuitTruthStage::Right { step: 1 },
            ),
            Err(NativeExecutorError::Host(_))
        ));
        assert!(state.completed_call.is_some());
        state.begin_contextual_site(request).unwrap();
        let (region, _, _) = state.enter_contextual_region().unwrap();
        state.enter_contextual_step(region, 1).unwrap();
        assert!(matches!(
            take_completed_short_circuit_truth(&mut state, ShortCircuitTruthStage::Left),
            Err(NativeExecutorError::Host(_))
        ));
        assert!(state.completed_call.is_some());
        enter_left(&mut state, request);
        assert_eq!(
            take_completed_short_circuit_truth(&mut state, ShortCircuitTruthStage::Left).unwrap(),
            Some(true)
        );
    });
}

#[test]
fn unpolled_truth_input_keeps_managed_target_until_abandonment() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::unpolled_truth_input_keeps_managed_target_until_abandonment"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        enter_left(&mut state, request());
        let allocation =
            runmat_gc::gc_allocate_rooted(Value::String("truth input".into())).unwrap();
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
                Value::String("truth input".into())
            );
        }))
        .unwrap();
        let input = Value::HandleObject(HandleRef {
            class_name: "TruthInput".into(),
            target: handle,
            valid: true,
        });
        assert!(matches!(
            begin_short_circuit_truth(&mut state, ShortCircuitTruthStage::Left, input),
            Err(NativeExecutorError::CallSuspended)
        ));
        allocation.unroot().unwrap();
        assert!(observed.load(Ordering::SeqCst));
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        drop(state);
        drop(observer);
        let before = runmat_gc::gc_collection_epoch();
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(runmat_gc::gc_collection_epoch(), before + 1);
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[test]
fn invalid_truth_preserves_runtime_identifier_without_advancing_checkpoint() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        enter_left(&mut state, request());
        assert!(matches!(
            begin_short_circuit_truth(
                &mut state,
                ShortCircuitTruthStage::Left,
                Value::String("invalid condition".into()),
            ),
            Err(NativeExecutorError::CallSuspended)
        ));
        let pending = state.pending_call.as_ref().unwrap();
        let (continuation, generation) = (pending.continuation, pending.generation);
        assert!(matches!(
            futures::executor::block_on(super::super::complete(
                &mut state,
                continuation,
                generation,
            )),
            Err(NativeExecutorError::Runtime(error))
                if error.identifier() == Some("RunMat:InvalidConditionType")
        ));
        assert!(state.pending_call.is_none());
        assert!(state.completed_call.is_none());
        state.begin_contextual_site(request()).unwrap();
        let (region, next, completed) = state.enter_contextual_region().unwrap();
        assert_eq!(region, 0);
        assert_eq!(next, 0);
        assert!(completed.is_none());
    });
}

#[test]
fn retained_expression_input_survives_source_replacement_and_abandonment_releases_it() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::retained_expression_input_survives_source_replacement_and_abandonment_releases_it"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let request = request();
        state.begin_contextual_site(request).unwrap();
        let allocation =
            runmat_gc::gc_allocate_rooted(Value::String("retained selector".into())).unwrap();
        let handle = allocation.handle();
        let original = state
            .arena
            .insert(
                Value::HandleObject(HandleRef {
                    class_name: "PreparedSelectorInput".into(),
                    target: handle,
                    valid: true,
                }),
                Some(&state.runtime),
            )
            .unwrap();
        state.retain_contextual_input(original).unwrap();
        state.arena.release(original).unwrap();
        let _replacement = state
            .arena
            .insert(Value::String("replacement".into()), Some(&state.runtime))
            .unwrap();
        allocation.unroot().unwrap();

        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        state.begin_contextual_site(request).unwrap();
        assert_eq!(state.take_contextual_input(), Some(original));

        state.abandon_site_continuations().unwrap();
        let before = runmat_gc::gc_collection_epoch();
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(runmat_gc::gc_collection_epoch(), before + 1);
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

#[test]
fn expression_site_retires_owned_region_results_but_preserves_borrowed_results() {
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        state.begin_contextual_site(request()).unwrap();

        let (owned_region, _, _) = state.enter_contextual_region().unwrap();
        let owned = state
            .arena
            .insert(Value::Num(11.0), Some(&state.runtime))
            .unwrap();
        state.finish_owned_contextual_reference(owned_region, owned);

        let (borrowed_region, _, _) = state.enter_contextual_region().unwrap();
        let borrowed = state
            .arena
            .insert(Value::Num(12.0), Some(&state.runtime))
            .unwrap();
        state.finish_contextual_region(borrowed_region, borrowed);

        state.finish_contextual_site().unwrap();
        assert!(matches!(
            state.arena.get(owned),
            Err(NativeExecutorError::StaleValue)
        ));
        assert_eq!(state.arena.get(borrowed).unwrap(), &Value::Num(12.0));
        state.arena.release(borrowed).unwrap();
    });
}
