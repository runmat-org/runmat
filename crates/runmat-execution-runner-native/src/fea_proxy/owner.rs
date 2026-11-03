use super::*;
use runmat_execution::resource::ResourceRequest;
use std::sync::Mutex;
use std::time::Instant;

struct Active {
    root: u64,
    cpu: u64,
    started: Instant,
    last_wall: u64,
    permit: Box<dyn FeaRunPermit>,
    held_ceiling: u64,
}
impl Active {
    fn observed(&self) -> ResourceUsage {
        usage(
            self.cpu,
            self.started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        )
    }
}
struct State {
    sequence: u64,
    active: Option<Active>,
}
#[derive(Clone, Copy)]
struct Evidence {
    budget: FeaRunBudget,
    started: Instant,
    cpu: u64,
}
/// The parent retains one real permit. Blocking host calls never hold the
/// evidence lock checked by the async owner/cancellation loop.
pub struct Parent {
    provider: Provider,
    assignment: ProgramExecutionAssignment,
    resources: ResourceRequest,
    deadline: u64,
    started: Instant,
    root_minimum_bytes: u64,
    state: Mutex<State>,
    evidence: Mutex<Option<Evidence>>,
    denied: Mutex<Option<FeaRunDenied>>,
    stopped: Mutex<Option<Instant>>,
}
impl Parent {
    pub fn new(
        provider: Provider,
        assignment: ProgramExecutionAssignment,
        resources: ResourceRequest,
        original_deadline: Option<u64>,
        invocation_bytes: u64,
    ) -> Result<Arc<Self>, String> {
        assignment.validate().map_err(|e| e.to_string())?;
        assignment
            .resources
            .validate_for_request(&resources)
            .map_err(|e| e.to_string())?;
        resources.validate().map_err(|e| e.to_string())?;
        if resources.memory_bytes < storage_bytes(0, StorageRole::Proxy)? {
            return Err("FEA proxy metadata exceeds worker reservation".into());
        }
        let deadline = now()
            .checked_add(resources.max_wall_millis)
            .ok_or("FEA proxy deadline overflow")?;
        let deadline = original_deadline.map_or(deadline, |original| original.min(deadline));
        let root_minimum_bytes = storage_bytes(invocation_bytes, StorageRole::FirstRoot)?;
        Ok(Arc::new(Self {
            provider,
            assignment,
            resources,
            deadline,
            started: Instant::now(),
            root_minimum_bytes,
            state: Mutex::new(State {
                sequence: 0,
                active: None,
            }),
            evidence: Mutex::new(None),
            denied: Mutex::new(None),
            stopped: Mutex::new(None),
        }))
    }
    pub fn check(&self) -> Result<(), FeaRunDenied> {
        if self
            .stopped
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
        {
            return Err(FeaRunDenied::LeaseExpired);
        }
        if let Some(denied) = self
            .denied
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            return Err(denied);
        }
        if now() >= self.deadline
            || self.started.elapsed().as_millis() >= u128::from(self.resources.max_wall_millis)
        {
            return Err(FeaRunDenied::LeaseExpired);
        }
        if let Some(evidence) = *self.evidence.lock().unwrap_or_else(|e| e.into_inner()) {
            evidence.budget.validate()?;
            let wall = evidence
                .started
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64;
            if usage(evidence.cpu, wall).cpu_core_millis >= evidence.budget.held_core_millis {
                return Err(FeaRunDenied::AllowanceUsed {
                    refills_at_unix_seconds: None,
                });
            }
        }
        Ok(())
    }
    pub fn terminal_check(&self) -> Result<(), FeaRunDenied> {
        self.check()?;
        if self
            .evidence
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        Ok(())
    }
    pub fn close(&self) {
        self.stopped
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(Instant::now);
    }
    fn clamp(
        &self,
        mut budget: FeaRunBudget,
        held_ceiling: u64,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        budget.expires_at_unix_millis = budget.expires_at_unix_millis.min(self.deadline);
        budget.held_core_millis = budget.held_core_millis.min(held_ceiling);
        budget.validate()?;
        Ok(budget)
    }
    fn cumulative(
        &self,
        active: &mut Active,
        reported: &ResourceUsage,
    ) -> Result<ResourceUsage, FeaRunDenied> {
        let observed = active.observed();
        let expected = usage(active.cpu, reported.wall_millis);
        if reported.wall_millis < active.last_wall
            || reported.wall_millis > self.resources.max_wall_millis
            || reported.wall_millis > observed.wall_millis.saturating_add(1)
            || reported != &expected
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        active.last_wall = reported.wall_millis;
        Ok(usage(
            active.cpu,
            observed.wall_millis.max(reported.wall_millis),
        ))
    }
    /// Run on the blocking adapter lane; all parent I/O/physical-stop ticks stay runnable.
    pub fn call(&self, call: Call) -> Reply {
        let result = self.call_inner(call);
        match result {
            Ok(reply) => reply,
            Err(denied) => {
                if self
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .active
                    .is_some()
                {
                    *self.denied.lock().unwrap_or_else(|e| e.into_inner()) = Some(denied.clone());
                }
                Reply::Denied { denied }
            }
        }
    }
    fn call_inner(&self, call: Call) -> Result<Reply, FeaRunDenied> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if call.assignment != self.assignment
            || call.sequence == 0
            || state.sequence.checked_add(1) != Some(call.sequence)
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        state.sequence = call.sequence;
        // Settlement remains possible after a hold is lost so the original
        // owner can conserve final usage; loss cannot authorize new work.
        if !matches!(&call.operation, Operation::Settle { .. }) {
            self.check()?;
        }
        match call.operation {
            Operation::Start { request } => {
                request.validate()?;
                if self.resources.memory_bytes < self.root_minimum_bytes {
                    return Err(FeaRunDenied::LicenseInvalid);
                }
                if state.active.is_some()
                    || request.site != FeaRunSite::Local
                    || request.allocated_cpu_millicores > u64::from(self.resources.cpu_millicores)
                {
                    return Err(FeaRunDenied::LicenseInvalid);
                }
                let provider = self.provider.as_ref().map_err(Clone::clone)?;
                if provider.site() != FeaRunSite::Local {
                    return Err(FeaRunDenied::LicenseInvalid);
                }
                let remaining = self.deadline.saturating_sub(now()).min(
                    self.resources.max_wall_millis.saturating_sub(
                        self.started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
                    ),
                );
                if remaining == 0 {
                    return Err(FeaRunDenied::LeaseExpired);
                }
                let held_ceiling =
                    usage(request.allocated_cpu_millicores, remaining).cpu_core_millis;
                let permit = provider.admit(&request)?;
                let started = Instant::now();
                state.active = Some(Active {
                    root: call.sequence,
                    cpu: request.allocated_cpu_millicores,
                    started,
                    last_wall: 0,
                    permit,
                    held_ceiling,
                });
                let initial = state
                    .active
                    .as_ref()
                    .expect("retained original permit")
                    .permit
                    .budget();
                let budget = self.clamp(initial, held_ceiling)?;
                *self.evidence.lock().unwrap_or_else(|e| e.into_inner()) = Some(Evidence {
                    budget,
                    started,
                    cpu: request.allocated_cpu_millicores,
                });
                self.check()?;
                Ok(Reply::Budget { budget })
            }
            Operation::Heartbeat { root, usage } => {
                let active = state.active.as_mut().ok_or(FeaRunDenied::LicenseInvalid)?;
                if active.root != root {
                    return Err(FeaRunDenied::LicenseInvalid);
                }
                let cumulative = self.cumulative(active, &usage)?;
                let renewed = active.permit.heartbeat(&cumulative)?;
                self.check()?; // a late transport reply cannot revive an expired original hold
                let budget = self.clamp(renewed, active.held_ceiling)?;
                *self.evidence.lock().unwrap_or_else(|e| e.into_inner()) = Some(Evidence {
                    budget,
                    started: active.started,
                    cpu: active.cpu,
                });
                self.check()?;
                Ok(Reply::Budget { budget })
            }
            Operation::Settle {
                root,
                outcome,
                usage,
            } => {
                let active = state.active.as_mut().ok_or(FeaRunDenied::LicenseInvalid)?;
                if active.root != root {
                    return Err(FeaRunDenied::LicenseInvalid);
                }
                let cumulative = self.cumulative(active, &usage)?;
                let checked = self.check();
                let outcome = if checked.is_err() && outcome == FeaRunOutcome::Completed {
                    FeaRunOutcome::Failed
                } else {
                    outcome
                };
                let active = state.active.take().expect("validated original permit");
                *self.evidence.lock().unwrap_or_else(|e| e.into_inner()) = None;
                let settled = active.permit.settle(outcome, &cumulative);
                checked.and(settled)?;
                Ok(Reply::Settled)
            }
        }
    }
}
/// Own the consuming finish even if a queued blocking closure is discarded
/// during executor shutdown. Taking the permit before calling the host keeps
/// the fallback exactly once, including a host error or unwinding call.
struct FailedSettlement {
    permit: Option<Box<dyn FeaRunPermit>>,
    observed: ResourceUsage,
}
impl FailedSettlement {
    fn settle(&mut self) {
        if let Some(permit) = self.permit.take() {
            let _ = permit.settle(FeaRunOutcome::Failed, &self.observed);
        }
    }
}
impl Drop for FailedSettlement {
    fn drop(&mut self) {
        self.settle();
    }
}

impl Drop for Parent {
    fn drop(&mut self) {
        if let Some(active) = self
            .state
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .active
            .take()
        {
            let stopped = self
                .stopped
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .unwrap_or_else(Instant::now);
            let wall = stopped
                .saturating_duration_since(active.started)
                .as_millis()
                .min(u128::from(u64::MAX)) as u64;
            let observed = usage(active.cpu, wall.max(active.last_wall));
            let mut settlement = FailedSettlement {
                permit: Some(active.permit),
                observed,
            };
            let mut settle = move || {
                settlement.settle();
            };
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                // Loss cleanup must not block another window's async reader.
                // The provider's existing bounded transport still owns finish.
                drop(runtime.spawn_blocking(settle));
            } else {
                settle();
            }
        }
    }
}

pub struct Close(pub Arc<Parent>);
impl Drop for Close {
    fn drop(&mut self) {
        self.0.close();
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
