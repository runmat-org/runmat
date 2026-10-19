use super::*;
use crate::execute::region::observation::test_support::observed_state;
use crate::region::placement_test_support::Probe;
use std::rc::Rc;

#[test]
fn host_failures_close_observations_before_retained_state_is_dropped() {
    let probe = Rc::new(Probe::default());
    for failure in 0..3 {
        let mut state = observed_state(&probe);
        let status = boundary((&mut state as *mut HostState).cast(), |_| match failure {
            0 => panic!("test host panic"),
            1 => Err(NativeExecutorError::Host("invalid request".into())),
            _ => Err(NativeExecutorError::StaleValue),
        });
        assert_ne!(status, NativeHostStatus::OK);
        assert!(state.region_observation.is_none());
        assert_eq!(probe.feedback.borrow().len(), failure + 1);
        assert!(probe
            .feedback
            .borrow()
            .iter()
            .all(|sample| !sample.succeeded));
        drop(state);
        assert_eq!(probe.feedback.borrow().len(), failure + 1);
    }
}
