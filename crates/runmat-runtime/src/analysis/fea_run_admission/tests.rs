use super::*;
use crate::analysis::fea_run_admission::{FeaRunKind, FeaRunSite};
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
fn ops(value: u64) -> OperationCount {
    OperationCount::from_operations(value)
}
fn budget() -> FeaRunBudget {
    FeaRunBudget {
        budget_ops: ops(1_000_000_000),
        granted_ops: ops(1_000_000_000),
        expires_at_unix_millis: runmat_time::unix_timestamp_ms() as u64 + 60_000,
        heartbeat_millis: 30_000,
    }
}
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

#[derive(Default)]
struct Events {
    admitted: Vec<FeaRunRequest>,
    settled: Vec<(FeaRunOutcome, FeaRunUsage)>,
}
struct Host(Arc<Mutex<Events>>);
struct Permit(Arc<Mutex<Events>>);
impl FeaRunPermit for Permit {
    fn budget(&self) -> FeaRunBudget {
        budget()
    }
    fn heartbeat(&self, _: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        Ok(budget())
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        usage: &FeaRunUsage,
    ) -> Result<(), FeaRunDenied> {
        self.0
            .lock()
            .unwrap()
            .settled
            .push((outcome, usage.clone()));
        Ok(())
    }
}
impl FeaRunAdmission for Host {
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        self.0.lock().unwrap().admitted.push(request.clone());
        Ok(Box::new(Permit(Arc::clone(&self.0))))
    }
}
fn request() -> FeaRunRequest {
    let mut request = FeaRunRequest::local(
        FeaRunKind::MeshAndSolve,
        2000,
        runmat_execution::Digest::sha256(b"study"),
    )
    .unwrap();
    request.site = FeaRunSite::Dispatched;
    request
}

#[test]
fn retained_parent_control_and_supervised_child_conserve_one_grant() {
    let events = Arc::new(Mutex::new(Events::default()));
    let request = FeaRunRequest::local(
        FeaRunKind::Solve,
        1000,
        runmat_execution::Digest::sha256(b"retained-parent"),
    )
    .unwrap();
    let parent = FeaRunScope::admit_with(request.clone(), &Host(Arc::clone(&events))).unwrap();
    let control = parent.control();
    assert_eq!(control.exclusive_work_requirement(ops(7)).unwrap(), ops(7));
    assert_eq!(
        control.usage().unwrap().completed_ops.total().unwrap(),
        ops(0)
    );
    // The retained control can delegate outside an installed numerical scope.
    let mut child_owner = control.delegate(ops(0)).unwrap();
    assert_eq!(
        control.exclusive_work_requirement(ops(1)),
        Err(FeaRunDenied::Accounting(LedgerError::OutstandingWork))
    );
    assert!(matches!(
        parent.state.accounting.ledger().finish(),
        Err(LedgerError::OutstandingWork)
    ));
    let initial = child_owner.delegation.budget;
    let child =
        FeaRunScope::from_parent_supervised_delegation(request.clone(), initial, 0).unwrap();
    child_owner.reservation.increase_capacity(ops(7)).unwrap();
    child
        .renew_parent_supervised(FeaRunBudget {
            granted_ops: ops(7),
            ..initial
        })
        .unwrap();
    child.run_sync(|| {
        let context = UsageContext::capture().unwrap();
        let mut work = context
            .ledger()
            .reserve(UsageCategory::Solve, ops(5))
            .unwrap();
        work.complete(ops(5)).unwrap();
    });
    let child_control = child.control();
    child.settle(FeaRunOutcome::Completed).unwrap();
    child_owner
        .reservation
        .finish(child_control.settled_usage().unwrap().completed_ops)
        .unwrap();
    assert_eq!(control.exclusive_work_requirement(ops(3)).unwrap(), ops(8));
    assert!(matches!(
        control.exclusive_work_requirement(budget().budget_ops),
        Err(FeaRunDenied::Accounting(LedgerError::BudgetExceeded { .. }))
    ));
    parent.settle(FeaRunOutcome::Completed).unwrap();
    assert_eq!(control.settled_usage().unwrap().completed_ops.solve, ops(5));
    assert_eq!(events.lock().unwrap().settled.len(), 1);

    let wrong_role = FeaRunScope::from_parent_supervised_delegation(request, initial, 0).unwrap();
    assert_eq!(
        wrong_role.renew(FeaRunBudget {
            granted_ops: ops(1),
            ..initial
        }),
        Err(FeaRunDenied::LicenseInvalid)
    );
}

#[test]
fn nested_mesh_and_solve_share_one_permit_and_never_raise_its_allocation() {
    let events = Arc::new(Mutex::new(Events::default()));
    let scope = FeaRunScope::admit_with(request(), &Host(Arc::clone(&events))).unwrap();
    scope.run_sync(|| {
        let context = UsageContext::capture().expect("root installs its numerical ledger");
        let mut work = context
            .ledger()
            .reserve(UsageCategory::Solve, ops(5))
            .unwrap();
        work.complete(ops(5)).unwrap();
        drop(work);
        let nested = crate::analysis::fea_run_admission::entrypoints::run(
            FeaRunKind::Meshing,
            runmat_execution::Digest::sha256(b"nested-mesh"),
            "test.mesh",
            "1",
            &crate::operations::OperationContext::new(None, None),
            || Ok(7),
        )
        .unwrap();
        assert_eq!(
            nested, 7,
            "nested public root keeps the original allocation/site without another lease"
        );
        let mut child = request();
        child.kind = FeaRunKind::Meshing;
        FeaRunScope::admit(child.clone())
            .unwrap()
            .settle(FeaRunOutcome::Completed)
            .unwrap();
        child.allocated_cpu_millicores = 3000;
        assert!(matches!(
            FeaRunScope::admit(child),
            Err(FeaRunDenied::LicenseInvalid)
        ));
    });
    assert_eq!(FeaRunScope::current_allocation(), None);
    let control = scope.control();
    assert_eq!(
        control.settled_usage(),
        Err(FeaRunDenied::Accounting(LedgerError::OutstandingWork))
    );
    scope.settle(FeaRunOutcome::Cancelled).unwrap();
    assert_eq!(control.settled_usage().unwrap().completed_ops.solve, ops(5));
    let events = events.lock().unwrap();
    assert_eq!(events.admitted.len(), 1);
    assert_eq!(events.settled.len(), 1);
    assert_eq!(events.settled[0].0, FeaRunOutcome::Cancelled);
    assert_eq!(events.settled[0].1.completed_ops.solve, ops(5));
}

#[test]
fn pending_future_does_not_authorize_unrelated_tasks_and_drop_settles_failed_once() {
    let events = Arc::new(Mutex::new(Events::default()));
    let scope = FeaRunScope::admit_with(request(), &Host(Arc::clone(&events))).unwrap();
    let future = std::future::poll_fn(|_| {
        assert_eq!(FeaRunScope::current_allocation(), Some(2000));
        Poll::<()>::Pending
    });
    let mut future = scope.run_future(future);
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(Pin::new(&mut future).poll(&mut context).is_pending());
    assert_eq!(FeaRunScope::current_allocation(), None);
    drop(future);
    drop(scope);
    assert_eq!(events.lock().unwrap().settled[0].0, FeaRunOutcome::Failed);
    assert_eq!(events.lock().unwrap().settled.len(), 1);
}

#[test]
fn physical_resource_telemetry_preserves_allocated_time_without_pricing_it() {
    for (millicores, wall, expected) in [(2000, 123, 246), (1500, 3, 5), (1, 1, 1), (2000, 0, 0)] {
        let usage = allocated_usage(millicores, wall);
        assert_eq!(usage.wall_millis, wall);
        assert_eq!(usage.cpu_core_millis, expected);
        assert!(
            usage.accelerator_millis.is_empty(),
            "this CPU allocation observation does not invent GPU usage"
        );
    }
}

#[test]
fn invalid_transport_request_is_rejected_before_host_admission() {
    let events = Arc::new(Mutex::new(Events::default()));
    let host = Host(Arc::clone(&events));
    let mut invalid = request();
    invalid.run_id = "not an id".into();
    assert!(matches!(
        FeaRunScope::admit_with(invalid, &host),
        Err(FeaRunDenied::LicenseInvalid)
    ));
    let mut empty_identity = request();
    empty_identity.study_identity.clear();
    assert!(matches!(
        FeaRunScope::admit_with(empty_identity, &host),
        Err(FeaRunDenied::LicenseInvalid)
    ));
    assert!(events.lock().unwrap().admitted.is_empty());
    let mut wrong_site = request();
    wrong_site.site = FeaRunSite::Local;
    let scope = FeaRunScope::admit_with(request(), &host).unwrap();
    scope.run_sync(|| assert!(FeaRunScope::admit(wrong_site).is_err()));
    scope.settle(FeaRunOutcome::Completed).unwrap();
    assert_eq!(events.lock().unwrap().admitted.len(), 1);
}

#[test]
fn local_roots_need_an_installed_provider_and_plain_work_never_admits() {
    let _host_state = crate::analysis::tests::analysis_test_guard();
    let _none = super::super::replace_fea_run_admission(None).unwrap();
    let mut local = request();
    local.site = FeaRunSite::Local;
    assert!(matches!(
        FeaRunScope::admit(local.clone()),
        Err(FeaRunDenied::SignInRequired)
    ));
    let events = Arc::new(Mutex::new(Events::default()));
    let context = super::super::FeaRunAdmissionContext::new(Arc::new(Host(Arc::clone(&events))));
    assert_eq!(context.run_sync(|| 2 + 3), 5);
    assert!(events.lock().unwrap().admitted.is_empty());
    let scope = context.run_sync(|| FeaRunScope::admit(local)).unwrap();
    scope.settle(FeaRunOutcome::Completed).unwrap();
    let events = events.lock().unwrap();
    assert_eq!(events.admitted.len(), 1);
    assert_eq!(events.settled.len(), 1);
}

struct ControlledPermit {
    events: Arc<Mutex<Events>>,
    initial: FeaRunBudget,
    renewal: Result<FeaRunBudget, FeaRunDenied>,
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn private_worker_bounds_first_root_before_provider_and_keep_original_deadline() {
    use crate::analysis::fea_run_admission::FeaRunAdmissionContext;
    let events = Arc::new(Mutex::new(Events::default()));
    let provider = FeaRunAdmissionContext::new(Arc::new(Host(Arc::clone(&events))));
    let deadline = runmat_time::unix_timestamp_ms() as u64 + 5_000;
    let bounded = provider
        .run_sync(|| FeaRunAdmissionContext::with_current_resource_bounds(1000, deadline).unwrap());
    let mut root = request();
    root.site = FeaRunSite::Local;
    assert!(matches!(
        bounded.run_sync(|| FeaRunScope::admit(root.clone())),
        Err(FeaRunDenied::LicenseInvalid)
    ));
    assert!(events.lock().unwrap().admitted.is_empty());
    root.allocated_cpu_millicores = 1000;
    let scope = bounded.run_sync(|| {
        assert_eq!(
            super::super::current_fea_cpu_capacity_millicores(),
            Some(1000)
        );
        FeaRunScope::admit(root).unwrap()
    });
    let initial = scope.state.lease.lock().unwrap().authority;
    let LeaseAuthority::Metered(initial) = initial else {
        panic!("finite metered worker root")
    };
    assert_eq!(initial.expires_at_unix_millis, deadline);
    assert_eq!(initial.budget_ops, budget().budget_ops);
    assert_eq!(initial.granted_ops, budget().granted_ops);
    scope.state.report();
    let LeaseAuthority::Metered(renewed) = scope.state.lease.lock().unwrap().authority else {
        panic!("finite renewed worker root")
    };
    assert_eq!(renewed.expires_at_unix_millis, deadline);
    assert_eq!(renewed.budget_ops, initial.budget_ops);
    assert_eq!(renewed.granted_ops, initial.granted_ops);
    scope.settle(FeaRunOutcome::Completed).unwrap();
    assert_eq!(events.lock().unwrap().settled.len(), 1);
}
impl FeaRunPermit for ControlledPermit {
    fn budget(&self) -> FeaRunBudget {
        self.initial
    }
    fn heartbeat(&self, _: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        self.renewal.clone()
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        usage: &FeaRunUsage,
    ) -> Result<(), FeaRunDenied> {
        self.events
            .lock()
            .unwrap()
            .settled
            .push((outcome, usage.clone()));
        Ok(())
    }
}
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn numerical_batch_renews_the_original_root_and_preserves_a_host_denial() {
    for unavailable in [false, true] {
        let events = Arc::new(Mutex::new(Events::default()));
        let mut initial = budget();
        initial.granted_ops = ops(2);
        let mut renewed = initial;
        renewed.granted_ops = ops(8);
        let scope = FeaRunScope::from_admitted_permit(
            request(),
            Box::new(ControlledPermit {
                events: events.clone(),
                initial,
                renewal: if unavailable {
                    Err(FeaRunDenied::ServerUnreachable)
                } else {
                    Ok(renewed)
                },
            }),
        )
        .unwrap();
        scope.run_sync(|| {
            let context = UsageContext::capture().unwrap();
            let reserved = context.ledger().reserve(UsageCategory::Solve, ops(8));
            if unavailable {
                assert!(matches!(reserved, Err(LedgerError::GrantUnavailable)));
                assert_eq!(context.ledger().snapshot().unwrap().reserved, ops(0));
                assert_eq!(scope.check(), Err(FeaRunDenied::ServerUnreachable));
            } else {
                let mut reserved = reserved.unwrap();
                reserved.complete(ops(3)).unwrap();
                assert_eq!(context.ledger().snapshot().unwrap().granted, ops(8));
            }
        });
        let settled = scope.settle(FeaRunOutcome::Completed);
        if unavailable {
            assert_eq!(settled, Err(FeaRunDenied::ServerUnreachable));
        } else {
            settled.unwrap();
        }
        let events = events.lock().unwrap();
        assert_eq!(events.settled.len(), 1);
        assert_eq!(
            events.settled[0].1.completed_ops.solve,
            ops(if unavailable { 0 } else { 3 })
        );
    }
}
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn heartbeat_errors_latch_and_cannot_be_revived_or_settled_as_success() {
    for denial in [
        FeaRunDenied::ServerUnreachable,
        FeaRunDenied::LeaseExpired,
        FeaRunDenied::AllowanceUsed {
            refills_at_unix_seconds: Some(123),
        },
    ] {
        let events = Arc::new(Mutex::new(Events::default()));
        let scope = FeaRunScope::from_admitted_permit(
            request(),
            Box::new(ControlledPermit {
                events: events.clone(),
                initial: budget(),
                renewal: Err(denial.clone()),
            }),
        )
        .unwrap();
        scope.state.report();
        assert_eq!(scope.check(), Err(denial.clone()));
        assert!(scope.cancellation_token().is_cancelled());
        assert_eq!(scope.renew(budget()), Err(denial.clone()));
        assert_eq!(scope.settle(FeaRunOutcome::Completed), Err(denial));
        let events = events.lock().unwrap();
        assert_eq!(events.settled.len(), 1);
        assert_eq!(events.settled[0].0, FeaRunOutcome::Failed);
    }
}
#[test]
fn stale_and_reduced_grants_fail_closed_without_converting_elapsed_time_to_work() {
    let mut expired = budget();
    expired.expires_at_unix_millis = 1;
    assert!(matches!(
        FeaRunScope::from_delegated_budget(request(), expired, 0),
        Err(FeaRunDenied::LeaseExpired)
    ));
    let mut partial = budget();
    partial.granted_ops = ops(20);
    let delegated = FeaRunScope::from_delegated_budget(request(), partial, 10).unwrap();
    assert_eq!(
        delegated.usage().unwrap().completed_ops.total().unwrap(),
        ops(0)
    );
    delegated.run_sync(|| {
        let context = UsageContext::capture().unwrap();
        assert!(matches!(
            context.ledger().reserve(UsageCategory::Solve, ops(21)),
            Err(LedgerError::MoreGrantRequired { .. })
        ));
    });
    let scope = FeaRunScope::from_delegated_budget(request(), budget(), 123).unwrap();
    assert!(scope.usage().unwrap().resources.cpu_core_millis >= 246);
    let mut reduced = budget();
    reduced.granted_ops = ops(1);
    assert_eq!(scope.renew(reduced), Err(FeaRunDenied::LicenseInvalid));
}

#[test]
fn lease_loss_refuses_a_pending_future_and_never_polls_more_work() {
    let events = Arc::new(Mutex::new(Events::default()));
    let scope = FeaRunScope::admit_with(request(), &Host(events)).unwrap();
    let mut future = scope.run_checked_future(std::future::pending::<()>());
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    assert!(Pin::new(&mut future).poll(&mut cx).is_pending());
    scope.control().deny(FeaRunDenied::ServerUnreachable);
    assert!(matches!(
        Pin::new(&mut future).poll(&mut cx),
        Poll::Ready(Err(
            crate::analysis::fea_run_admission::FeaRunExecutionError::Denied(
                FeaRunDenied::ServerUnreachable
            )
        ))
    ));
    drop(future);
    assert_eq!(
        scope.settle(FeaRunOutcome::Completed),
        Err(FeaRunDenied::ServerUnreachable)
    );
}

#[test]
fn trusted_parent_supervised_role_computes_but_cannot_publish_or_renew() {
    let scope = FeaRunScope::from_parent_supervised_delegation(request(), budget(), 1000).unwrap();
    assert!(scope.usage().unwrap().resources.cpu_core_millis >= 2000);
    assert_eq!(scope.run_checked_sync(|| 7), Ok(7));
    assert_eq!(
        scope.run_sync(FeaRunScope::check_publication),
        Err(FeaRunDenied::LicenseInvalid)
    );
    assert!(scope.settle(FeaRunOutcome::Completed).is_err());
    let worker = FeaRunScope::from_parent_supervised_delegation(request(), budget(), 0).unwrap();
    assert_eq!(worker.renew(budget()), Err(FeaRunDenied::LicenseInvalid));
}

#[test]
fn inherited_generation_cancel_wakes_pending_work_and_settles_cancelled() {
    use crate::analysis::fea_run_admission::FeaRunExecutionError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Wake(AtomicUsize);
    impl futures::task::ArcWake for Wake {
        fn wake_by_ref(wake: &Arc<Self>) {
            wake.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let parent = CancellationSource::new();
    let _interrupt = crate::interrupt::replace_interrupt(Some(parent.token()));
    let events = Arc::new(Mutex::new(Events::default()));
    let scope = FeaRunScope::admit_with(request(), &Host(events.clone())).unwrap();
    let mut future = scope.run_checked_future(std::future::pending::<()>());
    let wake = Arc::new(Wake(AtomicUsize::new(0)));
    let waker = futures::task::waker(wake.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(Pin::new(&mut future).poll(&mut cx).is_pending());
    parent.cancel(runmat_execution::CancellationReason::User);
    assert!(wake.0.load(Ordering::SeqCst) > 0);
    assert!(matches!(
        Pin::new(&mut future).poll(&mut cx),
        Poll::Ready(Err(FeaRunExecutionError::Cancelled))
    ));
    assert!(
        scope.check().is_ok(),
        "user cancellation is not a lease denial"
    );
    let context = crate::operations::OperationContext::new(Some("cancelled-original".into()), None);
    let error = scope.run_sync(|| {
        FeaRunScope::check_publication_for_operation("publish", "1", &context).unwrap_err()
    });
    assert_eq!(
        error.error_type,
        crate::operations::OperationErrorType::Cancelled
    );
    assert_eq!(error.trace_id, context.trace_id);
    assert_eq!(error.operation, "publish");
    drop(future);
    scope.settle(FeaRunOutcome::Cancelled).unwrap();
    assert_eq!(
        events.lock().unwrap().settled[0].0,
        FeaRunOutcome::Cancelled
    );
}

#[test]
fn settlement_failure_preserves_original_typed_solver_error_but_refuses_success() {
    use crate::operations::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct FailingHost(Arc<AtomicUsize>);
    struct FailingPermit(Arc<AtomicUsize>);
    impl FeaRunAdmission for FailingHost {
        fn admit(&self, _: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
            Ok(Box::new(FailingPermit(self.0.clone())))
        }
    }
    impl FeaRunPermit for FailingPermit {
        fn budget(&self) -> FeaRunBudget {
            budget()
        }
        fn heartbeat(&self, _: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
            Ok(budget())
        }
        fn settle(self: Box<Self>, _: FeaRunOutcome, _: &FeaRunUsage) -> Result<(), FeaRunDenied> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(FeaRunDenied::ServerUnreachable)
        }
    }
    let count = Arc::new(AtomicUsize::new(0));
    let host = crate::analysis::fea_run_admission::FeaRunAdmissionContext::new(Arc::new(
        FailingHost(count.clone()),
    ));
    let context = OperationContext::new(Some("original-trace".into()), None);
    let primary = operation_error(
        "solver",
        "1",
        &context,
        OperationErrorSpec {
            error_code: "Test.OriginalSolver",
            error_type: OperationErrorType::Backend,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        "original solver failure",
        Default::default(),
    );
    let error = host
        .run_sync(|| {
            crate::analysis::fea_run_admission::entrypoints::run::<()>(
                FeaRunKind::Solve,
                runmat_execution::Digest::sha256(b"study"),
                "solver",
                "1",
                &context,
                || Err(primary.clone()),
            )
        })
        .unwrap_err();
    assert_eq!(error.error_code, primary.error_code);
    assert_eq!(error.error_type, primary.error_type);
    assert_eq!(error.message, primary.message);
    assert_eq!(error.trace_id, primary.trace_id);
    let secondary: OperationErrorEnvelope =
        serde_json::from_str(&error.context["fea_lease_error"]).unwrap();
    assert_eq!(secondary.error_code, "RM.FEA.ServerUnreachable");
    let refused = host
        .run_sync(|| {
            crate::analysis::fea_run_admission::entrypoints::run(
                FeaRunKind::Solve,
                runmat_execution::Digest::sha256(b"study"),
                "solver",
                "1",
                &context,
                || Ok(7),
            )
        })
        .unwrap_err();
    assert_eq!(refused.error_code, "RM.FEA.ServerUnreachable");
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn deadline_does_not_wait_for_a_blocked_transport_heartbeat() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Condvar,
    };
    struct BlockingPermit {
        initial: FeaRunBudget,
        entered: Arc<AtomicBool>,
        release: Arc<(Mutex<bool>, Condvar)>,
    }
    impl FeaRunPermit for BlockingPermit {
        fn budget(&self) -> FeaRunBudget {
            self.initial
        }
        fn heartbeat(&self, _: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
            self.entered.store(true, Ordering::Release);
            let (lock, wake) = &*self.release;
            let held = lock.lock().unwrap();
            let _released = wake
                .wait_timeout_while(held, std::time::Duration::from_secs(1), |released| {
                    !*released
                })
                .unwrap();
            Ok(self.initial)
        }
        fn settle(self: Box<Self>, _: FeaRunOutcome, _: &FeaRunUsage) -> Result<(), FeaRunDenied> {
            Ok(())
        }
    }
    let entered = Arc::new(AtomicBool::new(false));
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let mut initial = budget();
    initial.expires_at_unix_millis = runmat_time::unix_timestamp_ms() as u64 + 100;
    initial.heartbeat_millis = 5;
    let scope = FeaRunScope::from_admitted_permit(
        request(),
        Box::new(BlockingPermit {
            initial,
            entered: entered.clone(),
            release: release.clone(),
        }),
    )
    .unwrap();
    let token = scope.cancellation_token();
    let until = std::time::Instant::now() + std::time::Duration::from_millis(800);
    while !token.is_cancelled() && std::time::Instant::now() < until {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let cancelled_while_blocked = token.is_cancelled();
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    assert!(entered.load(Ordering::Acquire));
    assert!(
        cancelled_while_blocked,
        "deadline must not wait behind the permit transport mutex"
    );
    assert!(scope.settle(FeaRunOutcome::Completed).is_err());
}
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn lease_expiry_cancels_only_its_run_before_first_heartbeat() {
    let parent = CancellationSource::new();
    let _interrupt = crate::interrupt::replace_interrupt(Some(parent.token()));
    let events = Arc::new(Mutex::new(Events::default()));
    let mut initial = budget();
    initial.expires_at_unix_millis = runmat_time::unix_timestamp_ms() as u64 + 40;
    let scope = FeaRunScope::from_admitted_permit(
        request(),
        Box::new(ControlledPermit {
            events: events.clone(),
            initial,
            renewal: Ok(initial),
        }),
    )
    .unwrap();
    let token = scope.cancellation_token();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while !token.is_cancelled() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(
        token.is_cancelled(),
        "40ms lease expiry must not wait for the 30s heartbeat"
    );
    assert!(
        !parent.token().is_cancelled(),
        "lease loss must not cancel the caller or another execution"
    );
    assert!(scope.run_sync(FeaRunScope::check_current).is_err());
    assert!(scope
        .run_sync(|| crate::meshing::write_canonical_artifacts(&[], false))
        .is_err());
    assert!(scope.settle(FeaRunOutcome::Completed).is_err());
    assert_eq!(events.lock().unwrap().settled.len(), 1);
}
