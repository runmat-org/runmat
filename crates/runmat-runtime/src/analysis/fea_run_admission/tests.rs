use super::*;
use crate::analysis::fea_run_admission::{FeaRunKind, FeaRunSite};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

#[derive(Default)]
struct Events {
    admitted: Vec<FeaRunRequest>,
    settled: Vec<(FeaRunOutcome, ResourceUsage)>,
}
struct Host(Arc<Mutex<Events>>);
struct Permit(Arc<Mutex<Events>>);
impl FeaRunPermit for Permit {
    fn settle(self: Box<Self>, outcome: FeaRunOutcome, usage: &ResourceUsage) {
        self.0
            .lock()
            .unwrap()
            .settled
            .push((outcome, usage.clone()));
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
fn nested_mesh_and_solve_share_one_permit_and_never_raise_its_allocation() {
    let events = Arc::new(Mutex::new(Events::default()));
    let scope = FeaRunScope::admit_with(request(), &Host(Arc::clone(&events))).unwrap();
    scope.run_sync(|| {
        let mut child = request();
        child.kind = FeaRunKind::Meshing;
        FeaRunScope::admit(child.clone())
            .unwrap()
            .settle(FeaRunOutcome::Completed);
        child.allocated_cpu_millicores = 3000;
        assert!(matches!(
            FeaRunScope::admit(child),
            Err(FeaRunDenied::LicenseInvalid)
        ));
    });
    assert_eq!(FeaRunScope::current_allocation(), None);
    scope.settle(FeaRunOutcome::Cancelled);
    let events = events.lock().unwrap();
    assert_eq!(events.admitted.len(), 1);
    assert_eq!(events.settled.len(), 1);
    assert_eq!(events.settled[0].0, FeaRunOutcome::Cancelled);
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
fn allocated_time_not_utilization_drives_cumulative_usage_and_fractional_rounding() {
    for (millicores, wall, expected) in [(2000, 123, 246), (1500, 3, 5), (1, 1, 1), (2000, 0, 0)] {
        let usage = allocated_usage(millicores, wall);
        assert_eq!(usage.wall_millis, wall);
        assert_eq!(usage.cpu_core_millis, expected);
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
    let mut wrong_site = request();
    wrong_site.site = FeaRunSite::Local;
    let scope = FeaRunScope::admit_with(request(), &host).unwrap();
    scope.run_sync(|| assert!(FeaRunScope::admit(wrong_site).is_err()));
    scope.settle(FeaRunOutcome::Completed);
    assert_eq!(events.lock().unwrap().admitted.len(), 1);
}

#[test]
fn local_runs_never_consult_default_explicit_or_poll_local_billing_hosts() {
    struct ForbiddenHost;
    impl FeaRunAdmission for ForbiddenHost {
        fn admit(&self, _: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
            panic!("local execution must not call a billing host")
        }
    }
    for kind in [FeaRunKind::Meshing, FeaRunKind::Solve, FeaRunKind::MeshAndSolve] {
        let mut local = request();
        local.site = FeaRunSite::Local;
        local.kind = kind;
        let context = crate::analysis::fea_run_admission::FeaRunAdmissionContext::new(
            Arc::new(ForbiddenHost),
        );
        let scopes = [
            FeaRunScope::admit(local.clone()).expect("no host or sign-in needed"),
            FeaRunScope::admit_with(local.clone(), &ForbiddenHost).expect("explicit host bypassed"),
            context.run_sync(|| FeaRunScope::admit(local)).expect("poll-local host bypassed"),
        ];
        for scope in scopes {
            scope.run_sync(|| assert_eq!(FeaRunScope::current_allocation(), Some(2000)));
            scope.settle(FeaRunOutcome::Completed);
        }
    }
}
