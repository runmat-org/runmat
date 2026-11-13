use super::*;
use crate::execute::region::observation::test_support::{
    observed_state, observed_state_with_retirement,
};
use crate::region::placement_test_support::Probe;
use runmat_runtime::native::{NativeCall, NativeExit, NativeHostStatus};
use runmat_value::{HandleRef, Value};
use std::rc::Rc;

unsafe extern "C" fn failed_entry(_: *mut NativeCall, _: *mut NativeExit) -> NativeHostStatus {
    NativeHostStatus::HOST_FAILURE
}

#[test]
fn failed_entry_retires_observation_before_retained_invocation_is_dropped() {
    let probe = Rc::new(Probe::default());
    let retired = Rc::new(std::cell::Cell::new(0));
    let mut invocation = NativeInvocation::new(
        observed_state_with_retirement(&probe, retired.clone()),
        failed_entry,
        vec![],
        0,
        NativeResumeState::default(),
        None,
    );
    assert!(invocation.advance().is_err());
    assert!(invocation.active.is_none());
    assert_eq!(retired.get(), 1);
    assert!(probe.feedback.borrow().is_empty());
    drop(invocation);
    assert_eq!(retired.get(), 1);
    assert!(probe.feedback.borrow().is_empty());
}

unsafe extern "C" fn one_result_entry(
    call: *mut NativeCall,
    exit: *mut NativeExit,
) -> NativeHostStatus {
    // SAFETY: this fixture is entered only through the executor's checked ABI
    // record with one argument and one result slot.
    unsafe {
        *(*call).results = *(*call).arguments;
        *exit = NativeExit::completed(1);
    }
    NativeHostStatus::OK
}

#[test]
fn terminal_result_outlives_the_native_arena() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::terminal_result_outlives_the_native_arena"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let mut state = observed_state(&Rc::new(Probe::default()));
        let initial = runmat_gc::gc_allocate_rooted(Value::String("terminal result".into()))
            .expect("allocate terminal payload");
        let handle = initial.handle();
        let value = Value::HandleObject(HandleRef {
            class_name: "NativeTerminalHandle".into(),
            target: handle,
            valid: true,
        });
        let reference = state
            .arena
            .insert(value.clone(), Some(&state.runtime))
            .unwrap();
        initial.unroot().expect("transfer root to native arena");

        let mut invocation = NativeInvocation::new(
            state,
            one_result_entry,
            vec![reference],
            1,
            NativeResumeState::default(),
            None,
        );
        let NativeInvocationStep::Completed(execution) =
            invocation.advance().expect("complete native invocation")
        else {
            panic!("fixture must complete");
        };
        assert!(invocation.active.is_none());
        drop(invocation);
        runmat_gc::gc_collect_major().expect("collect after native arena teardown");
        assert_eq!(execution.outputs, vec![value]);
        assert_eq!(
            runmat_gc::gc_clone_value(&handle).expect("terminal carrier retains output"),
            Value::String("terminal result".into())
        );
        drop(execution);
        runmat_gc::gc_collect_major().expect("collect after terminal carrier drop");
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
