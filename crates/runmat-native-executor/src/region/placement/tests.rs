use super::test_support::*;
use super::*;
use runmat_runtime::execution::RuntimeExecutionService;
use std::rc::Rc;
#[test]
fn absent_authority_is_unavailable_without_disabling_future_choices() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    assert!(matches!(
        choose(&runtime, &plan(), &Value::Num(1.)),
        RegionPlacement::Unavailable
    ));
}
#[test]
fn selected_candidates_report_success_or_failure_exactly_once() {
    let probe = Rc::new(Probe::default());
    let runtime = runtime(&probe);
    let plan = plan();
    let RegionPlacement::Generic(observation) = choose(&runtime, &plan, &Value::Num(1.)) else {
        panic!("generic")
    };
    observation.complete();
    probe.vectorized.set(true);
    let RegionPlacement::Vectorized(observation) = choose(&runtime, &plan, &Value::Num(1.)) else {
        panic!("vectorized")
    };
    drop(observation);
    let feedback = probe.feedback.borrow();
    assert_eq!(feedback.len(), 2);
    assert!(feedback[0].succeeded);
    assert!(!feedback[1].succeeded);
    assert_ne!(feedback[0].candidate, feedback[1].candidate);
    assert!(feedback.iter().all(|sample| sample.total_elapsed_ns > 0));
}
#[test]
fn changed_shapes_and_resource_epochs_are_reconsidered_after_generic_choice() {
    let probe = Rc::new(Probe::default());
    probe.vectorized.set(true);
    probe.memory.set(Some(1));
    let runtime = runtime(&probe);
    let plan = plan();
    let first = Value::Tensor(runmat_value::Tensor::new(vec![1.; 2], vec![1, 2]).unwrap());
    let RegionPlacement::Generic(observation) = choose(&runtime, &plan, &first) else {
        panic!("memory pressure")
    };
    observation.complete();
    probe.memory.set(Some(10000));
    probe.epoch.set(1);
    let RegionPlacement::Vectorized(observation) = choose(&runtime, &plan, &first) else {
        panic!("capacity recovered")
    };
    observation.complete();
    let changed = Value::Tensor(runmat_value::Tensor::new(vec![1.; 7], vec![1, 7]).unwrap());
    let RegionPlacement::Vectorized(observation) = choose(&runtime, &plan, &changed) else {
        panic!("changed shape")
    };
    observation.complete();
    let requests = probe.requests.borrow();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].signature, requests[1].signature);
    assert_ne!(requests[0].resources, requests[1].resources);
    assert_ne!(
        requests[1].signature.runtime_facts,
        requests[2].signature.runtime_facts
    );
    assert_eq!(probe.feedback.borrow().len(), 3);
}
