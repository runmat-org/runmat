use super::*;
use crate::execute::region::observation::test_support::observed_state_with_retirement;
use crate::region::placement_test_support::Probe;
use std::rc::Rc;

#[test]
fn host_failures_close_observations_before_retained_state_is_dropped() {
    let probe = Rc::new(Probe::default());
    let retired = Rc::new(std::cell::Cell::new(0));
    for failure in 0..3 {
        let mut state = observed_state_with_retirement(&probe, retired.clone());
        let status = boundary((&mut state as *mut HostState).cast(), |_| match failure {
            0 => panic!("test host panic"),
            1 => Err(NativeExecutorError::Host("invalid request".into())),
            _ => Err(NativeExecutorError::StaleValue),
        });
        assert_ne!(status, NativeHostStatus::OK);
        assert!(state.region_observation.is_none());
        assert_eq!(retired.get(), failure + 1);
        assert!(probe.feedback.borrow().is_empty());
        drop(state);
        assert_eq!(retired.get(), failure + 1);
        assert!(probe.feedback.borrow().is_empty());
    }
}
