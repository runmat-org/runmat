use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use std::rc::Rc;

unsafe extern "C" fn failed_entry(_: *mut NativeCall, _: *mut NativeExit) -> NativeHostStatus {
    NativeHostStatus::HOST_FAILURE
}

#[test]
fn failed_entry_reports_before_retained_invocation_is_dropped() {
    let probe = Rc::new(Probe::default());
    let mut invocation = NativeInvocation::new(
        observed_state(&probe),
        failed_entry,
        vec![],
        0,
        NativeResumeState::default(),
        None,
    );
    assert!(invocation.advance().is_err());
    assert!(invocation.state.region_observation.is_none());
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(!probe.feedback.borrow()[0].succeeded);
    drop(invocation);
    assert_eq!(probe.feedback.borrow().len(), 1);
}
