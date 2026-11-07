//! Bounded lease evidence on the existing authenticated private worker channel.
use runmat_execution::ProgramExecutionAssignment;
use runmat_runtime::analysis::fea_run_admission::{
    FeaRunBudget, FeaRunControl, FeaRunDelegation, FeaRunDelegationSource, FeaRunDenied,
    FeaRunScope,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
mod owner;
pub(crate) use owner::ChildGrant;

pub(crate) type Source = Arc<dyn FeaRunDelegationSource>;
pub(crate) const MAX_GRANT_BYTES: u64 = 16 * 1024;
pub(crate) const STORAGE_BYTES: u64 = MAX_GRANT_BYTES * 4
    // Keep the existing bounded grant wire/decode allowance. This is only the
    // additional fixed backing of the one replacement request envelope.
    + crate::protocol::FEA_LAUNCH_LAYOUT_GROWTH as u64
    + runmat_runtime::analysis::fea_run_admission::FEA_DEADLINE_MONITOR_STACK_BYTES as u64;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerFeaGrant {
    pub(crate) assignment: ProgramExecutionAssignment,
    pub(crate) resources: runmat_execution::resource::ResourceRequest,
    pub(crate) parent: FeaRunDelegation,
}
impl WorkerFeaGrant {
    pub(crate) fn install_with_parent(
        &self,
        actual: &ProgramExecutionAssignment,
        resources: &runmat_execution::resource::ResourceRequest,
        client: &Arc<crate::fea_proxy::Client>,
    ) -> Result<FeaRunScope, String> {
        self.validate(actual, resources)?;
        let permit = client
            .delegated_permit(self.parent.budget)
            .map_err(|error| error.to_string())?;
        FeaRunScope::from_private_worker_permit(
            self.parent.clone(),
            u64::from(resources.cpu_millicores),
            permit,
        )
        .map_err(|error| error.to_string())
    }
    fn validate(
        &self,
        actual: &ProgramExecutionAssignment,
        resources: &runmat_execution::resource::ResourceRequest,
    ) -> Result<(), String> {
        self.parent.validate().map_err(|error| error.to_string())?;
        actual.validate().map_err(|error| error.to_string())?;
        actual
            .resources
            .validate_for_request(resources)
            .map_err(|error| error.to_string())?;
        let _encoded = runmat_canonical_codec::encode_json_bounded(self, MAX_GRANT_BYTES)
            .map_err(|error| error.to_string())?;
        let minimum = crate::fea_proxy::storage_bytes(0, crate::fea_proxy::StorageRole::Delegated)?;
        if resources.memory_bytes < minimum {
            return Err("FEA grant metadata and deadline monitor exceed child memory".into());
        }
        resources.validate().map_err(|error| error.to_string())?;
        if &self.assignment != actual
            || &self.resources != resources
            || u64::from(resources.cpu_millicores) > self.parent.allocation_ceiling_millicores
        {
            return Err("FEA parent grant differs from the exact child assignment".into());
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn install(
        &self,
        actual: &ProgramExecutionAssignment,
        resources: &runmat_execution::resource::ResourceRequest,
    ) -> Result<FeaRunScope, String> {
        self.validate(actual, resources)?;
        FeaRunScope::from_private_worker_delegation(
            self.parent.clone(),
            u64::from(resources.cpu_millicores),
        )
        .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "update", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeaLeaseUpdate {
    Renewed { budget: FeaRunBudget },
    Denied { denied: FeaRunDenied },
}

struct Binding {
    run_id: String,
    attempt_id: runmat_execution::identity::AttemptId,
    control: FeaRunControl,
}
#[derive(Default)]
struct ObserverState {
    binding: Option<Binding>,
    pending: Option<(
        String,
        runmat_execution::identity::AttemptId,
        FeaLeaseUpdate,
    )>,
    invalid: bool,
}
#[derive(Default)]
pub(crate) struct LeaseObserver(Mutex<ObserverState>);
impl LeaseObserver {
    pub(crate) fn install(
        &self,
        grant: &WorkerFeaGrant,
        scope: &FeaRunScope,
    ) -> Result<(), String> {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state.invalid {
            return Err("invalid FEA parent control before worker admission".into());
        }
        if state.binding.is_some() {
            return Err("worker FEA parent was installed twice".into());
        }
        let binding = Binding {
            run_id: grant.parent.request.run_id.clone(),
            attempt_id: grant.assignment.attempt_id,
            control: scope.control(),
        };
        if let Some((run_id, attempt_id, update)) = state.pending.take() {
            apply(&binding, &run_id, attempt_id, update);
        }
        binding.control.check().map_err(|error| error.to_string())?;
        state.binding = Some(binding);
        Ok(())
    }
    /// Called by the existing parent reader thread, including while a kernel
    /// occupies the async executor. Other private controls retain their order.
    pub(crate) fn observe(&self, bytes: &[u8]) -> bool {
        #[derive(Deserialize)]
        struct Header<'a> {
            #[serde(borrow)]
            message: &'a str,
        }
        if !serde_json::from_slice::<Header<'_>>(bytes)
            .is_ok_and(|header| header.message == "fea_lease_update")
        {
            return false;
        }
        if bytes.len() as u64 > MAX_GRANT_BYTES {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            state.invalid = true;
            if let Some(binding) = &state.binding {
                binding.control.deny(FeaRunDenied::LicenseInvalid);
            }
            return true;
        }
        let parsed = serde_json::from_slice::<crate::protocol::WorkerDriverMessage>(bytes);
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match parsed {
            Ok(crate::protocol::WorkerDriverMessage::FeaLeaseUpdate {
                run_id,
                attempt_id,
                update,
            }) => {
                if let Some(binding) = &state.binding {
                    apply(binding, &run_id, attempt_id, update);
                } else {
                    if state
                        .pending
                        .as_ref()
                        .is_some_and(|(saved_run, saved_attempt, _)| {
                            saved_run != &run_id || saved_attempt != &attempt_id
                        })
                    {
                        state.invalid = true;
                    }
                    state.pending = Some((run_id, attempt_id, update));
                }
            }
            _ => {
                state.invalid = true;
                if let Some(binding) = &state.binding {
                    binding.control.deny(FeaRunDenied::LicenseInvalid);
                }
            }
        }
        true
    }
}
fn apply(
    binding: &Binding,
    run_id: &str,
    attempt_id: runmat_execution::identity::AttemptId,
    update: FeaLeaseUpdate,
) {
    if run_id != binding.run_id || attempt_id != binding.attempt_id {
        binding.control.deny(FeaRunDenied::LicenseInvalid);
        return;
    }
    match update {
        FeaLeaseUpdate::Renewed { budget } => {
            let _ = binding.control.renew(budget);
        }
        FeaLeaseUpdate::Denied { denied } => binding.control.deny(denied),
    }
}

pub(crate) fn capture() -> Result<Option<Source>, String> {
    FeaRunScope::current_delegation_source().map_err(|error| error.to_string())
}

#[cfg(test)]
pub(crate) fn test_grant() -> WorkerFeaGrant {
    tests::fixture()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_simulation_usage::{OperationCount, UsageContext};
    pub(super) fn fixture() -> WorkerFeaGrant {
        let request = runmat_runtime::analysis::fea_run_admission::FeaRunRequest::local(
            runmat_runtime::analysis::fea_run_admission::FeaRunKind::MeshAndSolve,
            2000,
            runmat_execution::Digest::sha256(b"paid-root"),
        )
        .unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        WorkerFeaGrant {
            assignment: ProgramExecutionAssignment {
                scope_id: runmat_execution::ExecutionScopeId::derive(&[b"child"]),
                pool_id: runmat_execution::PoolId::derive(&[b"child"]),
                task_id: runmat_execution::TaskId::derive(&[b"child"]),
                attempt_id: runmat_execution::identity::AttemptId::derive(&[b"child"]),
                worker_id: runmat_execution::identity::WorkerId::derive(&[b"child"]),
                backend: runmat_execution::PoolBackend::LocalProcesses,
                resources: Default::default(),
            },
            resources: runmat_execution::resource::ResourceRequest {
                cpu_millicores: 1000,
                memory_bytes: 64 * 1024 * 1024,
                scratch_bytes: 0,
                max_wall_millis: 60_000,
                max_artifact_bytes: 64 * 1024 * 1024,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: vec![],
                required_capabilities: Default::default(),
            },
            parent: FeaRunDelegation {
                origin: runmat_runtime::analysis::fea_run_admission::FeaRunOrigin::LocalLease {
                    run_id: request.run_id.clone(),
                },
                request,
                budget: FeaRunBudget {
                    budget_ops: OperationCount::from_operations(1_000_000),
                    granted_ops: OperationCount::from_operations(100_000),
                    expires_at_unix_millis: now + 60_000,
                    heartbeat_millis: 30_000,
                },
                elapsed_wall_millis: 500,
                observed_at_unix_millis: now,
                allocation_ceiling_millicores: 2000,
            },
        }
    }
    #[test]
    fn delegated_quote_covers_one_envelope_growth_at_original_memory_boundary() {
        let old_control = MAX_GRANT_BYTES * 4
            + runmat_runtime::analysis::fea_run_admission::FEA_DEADLINE_MONITOR_STACK_BYTES as u64;
        assert_eq!(
            STORAGE_BYTES - old_control,
            crate::protocol::FEA_LAUNCH_LAYOUT_GROWTH as u64
        );
        let minimum =
            crate::fea_proxy::storage_bytes(0, crate::fea_proxy::StorageRole::Delegated).unwrap();
        let mut grant = fixture();
        grant.resources.memory_bytes = minimum;
        grant.validate(&grant.assignment, &grant.resources).unwrap();
        grant.resources.memory_bytes -= 1;
        assert!(grant.validate(&grant.assignment, &grant.resources).is_err());
    }
    #[test]
    fn private_grant_binds_assignment_and_cpu_with_child_physical_usage() {
        let grant = fixture();
        let scope = grant.install(&grant.assignment, &grant.resources).unwrap();
        assert_eq!(scope.run_sync(FeaRunScope::current_allocation), Some(1000));
        let physical = scope.usage().unwrap().resources;
        assert_eq!(physical.cpu_core_millis, physical.wall_millis);
        assert_eq!(
            scope.usage().unwrap().completed_ops.total().unwrap(),
            OperationCount::ZERO
        );
        let mut other = grant.assignment.clone();
        other.attempt_id = runmat_execution::identity::AttemptId::derive(&[b"different"]);
        assert!(grant.install(&other, &grant.resources).is_err());
        let mut altered = grant.clone();
        altered.parent.allocation_ceiling_millicores = 500;
        assert!(altered
            .install(&grant.assignment, &grant.resources)
            .is_err());
        let mut expired = grant.clone();
        expired.parent.budget.expires_at_unix_millis = 1;
        assert!(expired
            .install(&grant.assignment, &grant.resources)
            .is_err());
    }
    #[test]
    fn reader_thread_renewal_preserves_identity_and_wrong_attempt_fails_closed() {
        let grant = fixture();
        let scope = grant.install(&grant.assignment, &grant.resources).unwrap();
        let observer = LeaseObserver::default();
        observer.install(&grant, &scope).unwrap();
        let mut budget = grant.parent.budget;
        budget.granted_ops = budget
            .granted_ops
            .checked_add(OperationCount::from_operations(1000))
            .unwrap();
        budget.expires_at_unix_millis += 1000;
        let bytes = serde_json::to_vec(&crate::protocol::WorkerDriverMessage::FeaLeaseUpdate {
            run_id: grant.parent.request.run_id.clone(),
            attempt_id: grant.assignment.attempt_id,
            update: FeaLeaseUpdate::Renewed { budget },
        })
        .unwrap();
        assert!(observer.observe(&bytes));
        let retained = scope.run_sync(|| {
            UsageContext::capture()
                .unwrap()
                .ledger()
                .snapshot()
                .unwrap()
        });
        assert_eq!(retained.budget, budget.budget_ops);
        assert_eq!(retained.granted, budget.granted_ops);
        let wrong = serde_json::to_vec(&crate::protocol::WorkerDriverMessage::FeaLeaseUpdate {
            run_id: grant.parent.request.run_id.clone(),
            attempt_id: runmat_execution::identity::AttemptId::derive(&[b"wrong"]),
            update: FeaLeaseUpdate::Renewed { budget },
        })
        .unwrap();
        assert!(observer.observe(&wrong));
        assert_eq!(scope.check(), Err(FeaRunDenied::LicenseInvalid));
    }
}
