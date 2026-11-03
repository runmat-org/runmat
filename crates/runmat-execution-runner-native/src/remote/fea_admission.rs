//! Server-owned cluster allocation admission, installed only after bootstrap or
//! scheduler authority verification. It never starts a second local FEA lease.
use runmat_execution::cancellation::CancellationToken;
use runmat_execution::resource::ResourceUsage;
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunAdmission, FeaRunAdmissionContext, FeaRunBudget, FeaRunDelegation,
    FeaRunDelegationSource, FeaRunDenied, FeaRunOrigin, FeaRunOutcome, FeaRunPermit, FeaRunRequest,
    FeaRunSite,
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

pub(super) fn for_verified_driver(
    authority: &runmat_execution_transport_native::control::DriverAuthority,
    allocation: &runmat_execution_transport_native::control::ResourceRequest,
    clock: AllocationClock,
    cancellation: CancellationToken,
) -> Result<FeaRunAdmissionContext, FeaRunDenied> {
    authority
        .validate()
        .map_err(|_| FeaRunDenied::LicenseInvalid)?;
    bounded(
        allocation.cpu_millicores,
        allocation.maximum_wall_millis,
        clock,
        None,
        cancellation,
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
) -> Result<FeaRunAdmissionContext, FeaRunDenied> {
    attempt
        .validate()
        .map_err(|_| FeaRunDenied::LicenseInvalid)?;
    bounded(
        u64::from(attempt.task.resources.cpu_millicores),
        attempt.task.resources.max_wall_millis,
        clock,
        attempt.task.deadline_unix_millis,
        cancellation,
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
    cpu_millicores: u64,
    maximum_wall_millis: u64,
    clock: AllocationClock,
    original_deadline: Option<u64>,
    cancellation: CancellationToken,
    origin: FeaRunOrigin,
) -> Result<FeaRunAdmissionContext, FeaRunDenied> {
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
    fn delegation_source(&self) -> Option<Arc<dyn FeaRunDelegationSource>> {
        Some(Arc::new(self.clone()))
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
        // Every root consumes only what remains of the SAME original wall grant.
        let remaining = self
            .maximum_wall_millis
            .saturating_sub(self.clock.elapsed_millis());
        let held =
            (u128::from(request.allocated_cpu_millicores) * u128::from(remaining)).div_ceil(1000);
        let budget = FeaRunBudget {
            held_core_millis: u64::try_from(held).map_err(|_| FeaRunDenied::LicenseInvalid)?,
            expires_at_unix_millis: self.deadline_unix_millis,
            heartbeat_millis: 30_000,
        };
        budget.validate()?;
        Ok(Box::new(ClusterPermit {
            budget,
            cancellation: self.cancellation.clone(),
        }))
    }
}
impl FeaRunDelegationSource for ClusterAdmission {
    fn snapshot(&self) -> Result<FeaRunDelegation, FeaRunDenied> {
        if self.cancellation.is_cancelled()
            || now() >= self.deadline_unix_millis
            || self.clock.elapsed_millis() >= self.maximum_wall_millis
        {
            return Err(FeaRunDenied::LeaseExpired);
        }
        let wall = self.maximum_wall_millis;
        let held = (u128::from(self.cpu_millicores) * u128::from(wall)).div_ceil(1000);
        let run_id = match &self.origin {
            FeaRunOrigin::ClusterDriver { run_id, .. }
            | FeaRunOrigin::ClusterAttempt { run_id, .. } => run_id.clone(),
            _ => return Err(FeaRunDenied::LicenseInvalid),
        };
        let grant = FeaRunDelegation {
            origin: self.origin.clone(),
            request: FeaRunRequest {
                run_id,
                kind: runmat_runtime::analysis::fea_run_admission::FeaRunKind::MeshAndSolve,
                site: FeaRunSite::Dispatched,
                allocated_cpu_millicores: self.cpu_millicores,
                estimated_wall_seconds: None,
                study_identity: "server-owned allocation".into(),
            },
            budget: FeaRunBudget {
                held_core_millis: u64::try_from(held).map_err(|_| FeaRunDenied::LicenseInvalid)?,
                expires_at_unix_millis: self.deadline_unix_millis,
                heartbeat_millis: 30_000,
            },
            elapsed_wall_millis: self.clock.elapsed_millis(),
            observed_at_unix_millis: now(),
            allocation_ceiling_millicores: self.cpu_millicores,
        };
        grant.validate()?;
        Ok(grant)
    }
    fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }
}
struct ClusterPermit {
    budget: FeaRunBudget,
    cancellation: CancellationToken,
}
impl FeaRunPermit for ClusterPermit {
    fn budget(&self) -> FeaRunBudget {
        self.budget
    }
    fn heartbeat(&self, _: &ResourceUsage) -> Result<FeaRunBudget, FeaRunDenied> {
        if self.cancellation.is_cancelled() {
            return Err(FeaRunDenied::LeaseExpired);
        }
        self.budget.validate()?;
        Ok(self.budget)
    }
    fn settle(self: Box<Self>, _: FeaRunOutcome, _: &ResourceUsage) -> Result<(), FeaRunDenied> {
        // The authenticated allocation owner already meters and settles cluster
        // CPU wall time. This permit owns no local HTTP lease to settle again.
        if self.cancellation.is_cancelled() {
            Err(FeaRunDenied::LeaseExpired)
        } else {
            self.budget.validate()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_cluster_grant_never_expands_and_rejects_wrong_site_cpu_or_expiry() {
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
        let permit = admission.admit(&request).unwrap();
        assert_eq!(permit.budget().expires_at_unix_millis, deadline);
        assert!(permit.budget().held_core_millis <= 120_000);
        assert_eq!(
            permit.heartbeat(&ResourceUsage::default()).unwrap(),
            permit.budget()
        );
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
        assert!(permit.heartbeat(&ResourceUsage::default()).is_err());
    }
}
