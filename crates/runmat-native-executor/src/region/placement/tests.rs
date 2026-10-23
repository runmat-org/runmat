use super::test_support::*;
use super::*;
use runmat_runtime::execution::RuntimeExecutionService;
use std::rc::Rc;

#[test]
fn native_region_revision_uses_host_authority_not_provider_namespace() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let signature = signature(&runtime, &plan(), &[&Value::Num(1.)]);
    let digest = Digest::sha256(b"native-source-owner");
    assert_eq!(
        signature.revision.authority,
        PlacementExecutionAuthorityRevision::HostService(digest)
    );
    assert!(
        runmat_execution::PlacementInvalidation::HostService { digest }
            .matches_signature(&signature)
    );
    assert!(
        !runmat_execution::PlacementInvalidation::Provider { digest }.matches_signature(&signature)
    );
}

#[test]
fn absent_authority_is_unavailable_without_disabling_future_choices() {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    assert!(matches!(
        choose(&runtime, &plan(), &Value::Num(1.)),
        RegionPlacement::Unavailable
    ));
}
#[test]
fn unbound_source_and_abandoned_cpu_never_report_false_host_samples() {
    let probe = Rc::new(Probe::default());
    let runtime = runtime(&probe);
    let plan = plan();
    let RegionPlacement::Generic(observation) = choose(&runtime, &plan, &Value::Num(1.)) else {
        panic!("generic")
    };
    observation.complete();
    assert!(probe.feedback.borrow().is_empty());
    probe.vectorized.set(true);
    let RegionPlacement::Vectorized(observation) = choose(&runtime, &plan, &Value::Num(1.)) else {
        panic!("vectorized")
    };
    observation.complete();
    let RegionPlacement::Vectorized(observation) = choose(&runtime, &plan, &Value::Num(1.)) else {
        panic!("vectorized")
    };
    drop(observation);
    let feedback = probe.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert!(feedback[0].succeeded);
    assert_eq!(
        feedback[0].kind,
        ExecutionCandidateKind::VectorizedNativeCpu
    );
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
    assert!(matches!(
        choose(&runtime, &plan, &first),
        RegionPlacement::Unavailable
    ));
    // This graph-only fixture has no managed host ledger/selection owner.
    // It must not fabricate finite authority. Actual managed paths have their
    // existing separate Runtime/Accelerate ledger-backed oracles.
    probe.memory.set(None);
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
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].resources.epoch, 1);
    assert_ne!(
        requests[0].signature.runtime_facts,
        requests[1].signature.runtime_facts
    );
    assert_eq!(probe.feedback.borrow().len(), 2);
}
