use super::*;
use crate::region::placement_test_support::{choose, plan, runtime, Probe};
use crate::region::RegionPlacement;
use std::rc::Rc;

fn pending(probe: &Rc<Probe>) -> (GenericRegionObservation, OptimizedRegionPlan) {
    let plan = plan();
    let RegionPlacement::Generic(observation) =
        choose(&runtime(probe), &plan, &runmat_value::Value::Num(1.))
    else {
        panic!("generic")
    };
    (GenericRegionObservation::new(&plan, observation), plan)
}
fn exit_site(plan: &OptimizedRegionPlan) -> SiteIdentity {
    SiteIdentity {
        point: plan.exit,
        phase: runmat_native_codegen::NativeSitePhase::Terminator,
        ordinal: 0,
    }
}
#[test]
fn success_requires_all_sites_and_the_real_region_exit() {
    let probe = Rc::new(Probe::default());
    let (pending, plan) = pending(&probe);
    let pending = pending
        .after(plan.entry, &Ok(NativeSiteOutcome::continue_execution()))
        .unwrap();
    assert!(
        probe.feedback.borrow().is_empty(),
        "last instruction alone is not an observed region exit"
    );
    assert!(pending.before(exit_site(&plan), false, false).is_none());
    let feedback = probe.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert!(feedback[0].succeeded);
}
#[test]
fn premature_exit_cancellation_error_and_drop_never_report_success() {
    let probe = Rc::new(Probe::default());
    let (value, plan) = pending(&probe);
    assert!(value.before(exit_site(&plan), false, false).is_none());
    let (value, plan) = pending(&probe);
    assert!(value.before(plan.entry, false, true).is_none());
    let (value, plan) = pending(&probe);
    assert!(value
        .after(
            plan.entry,
            &Err(crate::NativeExecutorError::Host("failed".into()))
        )
        .is_none());
    let (value, plan) = pending(&probe);
    assert!(value
        .after(plan.entry, &Ok(NativeSiteOutcome::exit()))
        .is_none());
    drop(pending(&probe).0);
    assert_eq!(probe.feedback.borrow().len(), 5);
    assert!(probe
        .feedback
        .borrow()
        .iter()
        .all(|sample| !sample.succeeded));
}
#[test]
fn unexpected_jump_or_reentry_aborts_incomplete_measurement() {
    let probe = Rc::new(Probe::default());
    let (value, plan) = pending(&probe);
    let mut unexpected = plan.entry;
    unexpected.point.block = 99;
    assert!(value.before(unexpected, false, false).is_none());
    let (value, plan) = pending(&probe);
    let value = value
        .after(plan.entry, &Ok(NativeSiteOutcome::continue_execution()))
        .unwrap();
    assert!(value.before(plan.entry, false, false).is_none());
    assert_eq!(probe.feedback.borrow().len(), 2);
    assert!(probe
        .feedback
        .borrow()
        .iter()
        .all(|sample| !sample.succeeded));
}

#[test]
fn forward_linear_edge_keeps_one_observation_until_the_exit() {
    let probe = Rc::new(Probe::default());
    let (mut value, mut plan) = pending(&probe);
    let mut next = plan.entry;
    next.point.block = 1;
    value.remaining.insert(next);
    plan.exit.block = 1;
    value.exit = plan.exit;
    let value = value
        .after(plan.entry, &Ok(NativeSiteOutcome::continue_execution()))
        .unwrap();
    let mut edge = plan.entry;
    edge.phase = runmat_native_codegen::NativeSitePhase::Terminator;
    edge.point.position = 1;
    let value = value.before(edge, true, false).unwrap();
    let value = value.after(edge, &Ok(NativeSiteOutcome::edge(0))).unwrap();
    let value = value.before(next, false, false).unwrap();
    let value = value
        .after(next, &Ok(NativeSiteOutcome::continue_execution()))
        .unwrap();
    assert!(probe.feedback.borrow().is_empty());
    assert!(value.before(exit_site(&plan), false, false).is_none());
    assert_eq!(probe.feedback.borrow().len(), 1);
    assert!(probe.feedback.borrow()[0].succeeded);
}
