//! Server-owned cluster allocation admission, installed only after bootstrap or
//! scheduler authority verification. It never starts a second local FEA lease.
use runmat_execution::cancellation::CancellationToken;
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunAdmission, FeaRunAdmissionContext, FeaRunBudget, FeaRunDenied, FeaRunOrigin,
    FeaRunOutcome, FeaRunPermit, FeaRunRequest, FeaRunSite, FeaRunUsage,
};
use std::sync::Arc;

#[derive(Clone)]
struct ClusterAdmission {
    cpu_millicores: u64,
    clock: AllocationClock,
    maximum_wall_millis: u64,
    origin: FeaRunOrigin,
    deadline_unix_millis: u64,
    cancellation: CancellationToken,
    lease: Option<Arc<runmat_fea_admission_native::NativeFeaAdmission>>,
    worker_control: Option<Arc<super::fea_control::WorkerControl>>,
}

#[derive(Clone, Copy)]
pub(super) struct AllocationClock {
    unix_millis: u64,
    monotonic: std::time::Instant,
}
impl AllocationClock {
    pub(super) fn begin() -> Self {
        Self {
            unix_millis: now(),
            monotonic: std::time::Instant::now(),
        }
    }
    fn elapsed_millis(&self) -> u64 {
        self.monotonic
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
    }
}

struct AllocationWindow {
    cpu_millicores: u64,
    maximum_wall_millis: u64,
    clock: AllocationClock,
    original_deadline: Option<u64>,
}

pub(super) fn for_verified_driver(
    authority: &runmat_execution_transport_native::control::DriverAuthority,
    allocation: &runmat_execution_transport_native::control::ResourceRequest,
    clock: AllocationClock,
    cancellation: CancellationToken,
) -> Result<FeaRunAdmissionContext, FeaRunDenied> {
    authority
        .validate()
        .map_err(|_| FeaRunDenied::LicenseInvalid)?;
    let org = runmat_server_client::ids::OrgId::parse(&authority.org_id)
        .map_err(|_| FeaRunDenied::LicenseInvalid)?;
    let driver =
        runmat_server_client::ids::ExecutionDriverLeaseId::parse(&authority.driver_lease_id)
            .map_err(|_| FeaRunDenied::LicenseInvalid)?;
    let lease = runmat_fea_admission_native::NativeFeaAdmission::for_driver(
        &authority.server_url,
        org,
        driver,
        authority.fencing_token,
        &authority.credential,
    )?;
    bounded(
        AllocationWindow {
            cpu_millicores: allocation.cpu_millicores,
            maximum_wall_millis: allocation.maximum_wall_millis,
            clock,
            original_deadline: None,
        },
        cancellation,
        Some(Arc::new(lease)),
        None,
        FeaRunOrigin::ClusterDriver {
            run_id: authority.run_id.clone(),
            allocation_lease_id: authority.allocation_lease_id.clone(),
            driver_lease_id: authority.driver_lease_id.clone(),
            fencing_token: authority.fencing_token,
        },
    )
}

pub(super) fn for_verified_attempt(
    run_id: &str,
    attempt: &runmat_execution_runner::AttemptRequest,
    clock: AllocationClock,
    cancellation: CancellationToken,
    worker_control: Arc<super::fea_control::WorkerControl>,
) -> Result<FeaRunAdmissionContext, FeaRunDenied> {
    attempt
        .validate()
        .map_err(|_| FeaRunDenied::LicenseInvalid)?;
    bounded(
        AllocationWindow {
            cpu_millicores: u64::from(attempt.task.resources.cpu_millicores),
            maximum_wall_millis: attempt.task.resources.max_wall_millis,
            clock,
            original_deadline: attempt.task.deadline_unix_millis,
        },
        cancellation,
        None,
        Some(worker_control),
        FeaRunOrigin::ClusterAttempt {
            run_id: run_id.to_owned(),
            scope_id: attempt.scope_id,
            pool_id: attempt.task.pool_id,
            task_id: attempt.task_id,
            attempt_id: attempt.id,
            driver_fence: attempt.driver_fence,
        },
    )
}

fn bounded(
    window: AllocationWindow,
    cancellation: CancellationToken,
    lease: Option<Arc<runmat_fea_admission_native::NativeFeaAdmission>>,
    worker_control: Option<Arc<super::fea_control::WorkerControl>>,
    origin: FeaRunOrigin,
) -> Result<FeaRunAdmissionContext, FeaRunDenied> {
    let AllocationWindow {
        cpu_millicores,
        maximum_wall_millis,
        clock,
        original_deadline,
    } = window;
    origin.validate(FeaRunSite::Dispatched)?;
    if !(1..=4_096_000).contains(&cpu_millicores)
        || maximum_wall_millis == 0
        || cancellation.is_cancelled()
    {
        return Err(FeaRunDenied::LicenseInvalid);
    }
    let deadline = clock
        .unix_millis
        .checked_add(maximum_wall_millis)
        .ok_or(FeaRunDenied::LicenseInvalid)?;
    let deadline = original_deadline.map_or(deadline, |original| original.min(deadline));
    let deadline = cancellation
        .deadline_unix_ms()
        .map_or(deadline, |original| original.min(deadline));
    if deadline <= now() {
        return Err(FeaRunDenied::LeaseExpired);
    }
    Ok(FeaRunAdmissionContext::new(Arc::new(ClusterAdmission {
        cpu_millicores,
        clock,
        maximum_wall_millis: deadline.saturating_sub(clock.unix_millis),
        origin,
        deadline_unix_millis: deadline,
        cancellation,
        lease,
        worker_control,
    })))
}

pub(super) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(u64::MAX, |duration| {
            duration.as_millis().min(u128::from(u64::MAX)) as u64
        })
}
impl FeaRunAdmission for ClusterAdmission {
    fn allocated_cpu_capacity_millicores(&self) -> Option<u64> {
        Some(self.cpu_millicores)
    }
    fn origin(&self) -> Option<FeaRunOrigin> {
        Some(self.origin.clone())
    }
    fn site(&self) -> FeaRunSite {
        FeaRunSite::Dispatched
    }
    fn admit(&self, request: &FeaRunRequest) -> Result<Box<dyn FeaRunPermit>, FeaRunDenied> {
        if request.site != FeaRunSite::Dispatched
            || request.allocated_cpu_millicores > self.cpu_millicores
            || request.allocated_cpu_millicores == 0
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        if self.cancellation.is_cancelled()
            || now() >= self.deadline_unix_millis
            || self.clock.elapsed_millis() >= self.maximum_wall_millis
        {
            return Err(FeaRunDenied::LeaseExpired);
        }
        // The verified driver holds the only Server credential. Worker attempts
        // must obtain disjoint capacity from the driver's authenticated private
        // parent control, never a second lease or a physical-resource conversion.
        let permit = match (&self.lease, &self.worker_control) {
            (Some(lease), None) => lease.admit(request)?,
            (None, Some(control)) => control.admit(request)?,
            _ => return Err(FeaRunDenied::LicenseInvalid),
        };
        Ok(Box::new(ClusterPermit {
            inner: permit,
            deadline_unix_millis: self.deadline_unix_millis,
            cancellation: self.cancellation.clone(),
        }))
    }
}
struct ClusterPermit {
    inner: Box<dyn FeaRunPermit>,
    deadline_unix_millis: u64,
    cancellation: CancellationToken,
}
impl ClusterPermit {
    fn physical_check(&self) -> Result<(), FeaRunDenied> {
        if self.cancellation.is_cancelled() || now() >= self.deadline_unix_millis {
            Err(FeaRunDenied::LeaseExpired)
        } else {
            Ok(())
        }
    }
    fn bounded(&self, mut budget: FeaRunBudget) -> Result<FeaRunBudget, FeaRunDenied> {
        budget.expires_at_unix_millis =
            budget.expires_at_unix_millis.min(self.deadline_unix_millis);
        budget.validate()?;
        Ok(budget)
    }
}
impl FeaRunPermit for ClusterPermit {
    fn budget(&self) -> FeaRunBudget {
        let mut budget = self.inner.budget();
        budget.expires_at_unix_millis =
            budget.expires_at_unix_millis.min(self.deadline_unix_millis);
        budget
    }
    fn heartbeat(&self, usage: &FeaRunUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        self.physical_check()?;
        self.bounded(self.inner.heartbeat(usage)?)
    }
    fn request_grant(
        &self,
        usage: &FeaRunUsage,
        required: runmat_simulation_usage::OperationCount,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        self.physical_check()?;
        self.bounded(self.inner.request_grant(usage, required)?)
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
        usage: &FeaRunUsage,
    ) -> Result<(), FeaRunDenied> {
        self.inner.settle(outcome, usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_allocation_cannot_mint_operation_capacity_from_cpu_or_wall_time() {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let deadline = now() + 60_000;
        let admission = ClusterAdmission {
            cpu_millicores: 2000,
            clock: AllocationClock::begin(),
            maximum_wall_millis: 60_000,
            origin: FeaRunOrigin::ClusterDriver {
                run_id: "run-cluster".into(),
                allocation_lease_id: "allocation".into(),
                driver_lease_id: "driver".into(),
                fencing_token: 1,
            },
            deadline_unix_millis: deadline,
            cancellation: cancellation.token(),
            lease: None,
            worker_control: None,
        };
        let mut request = FeaRunRequest::local(
            runmat_runtime::analysis::fea_run_admission::FeaRunKind::Solve,
            2000,
            runmat_execution::Digest::sha256(b"cluster-study"),
        )
        .unwrap();
        assert!(matches!(
            admission.admit(&request),
            Err(FeaRunDenied::LicenseInvalid)
        ));
        request.site = FeaRunSite::Dispatched;
        assert!(matches!(
            admission.admit(&request),
            Err(FeaRunDenied::LicenseInvalid)
        ));
        request.allocated_cpu_millicores = 3000;
        assert!(matches!(
            admission.admit(&request),
            Err(FeaRunDenied::LicenseInvalid)
        ));
        request.allocated_cpu_millicores = 2000;
        cancellation.cancel(runmat_execution::CancellationReason::Shutdown);
        assert!(matches!(
            admission.admit(&request),
            Err(FeaRunDenied::LeaseExpired)
        ));
    }
}
