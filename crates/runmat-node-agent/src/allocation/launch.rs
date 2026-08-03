use std::collections::BTreeMap;
use std::path::Path;

use runmat_execution_transport_native::control::NodeAllocation;
use runmat_process_host::{
    ChildLifetime, ChildProcess, HiddenMode, HostCommand, ResourceLimits, StdioPolicy,
};

use super::Sandbox;
use crate::{AgentError, AgentResult};

#[derive(Default)]
pub struct AllocationProcesses {
    children: BTreeMap<String, ManagedProcess>,
}

struct ManagedProcess {
    child: ChildProcess,
    process_id: u32,
    fencing_token: u64,
    expires_at_millis: i64,
    maximum_memory_bytes: u64,
    maximum_wall_millis: u64,
    started_at_millis: i64,
}

impl AllocationProcesses {
    pub async fn launch_driver(
        &mut self,
        runmat_executable: &Path,
        allocation: &NodeAllocation,
        sandbox: &Sandbox,
    ) -> AgentResult<u32> {
        if self.children.contains_key(&allocation.id) {
            return Err(AgentError::AllocationRejected(
                "allocation already has a driver".to_string(),
            ));
        }
        let mut command = HostCommand::new(runmat_executable);
        // This exact private mode remains the sole process argument. Lease
        // authority crosses in a sanitized environment; the Server cannot
        // choose an executable or arbitrary argument vector.
        command.arguments = vec![HiddenMode::ExecutionDriver.marker().to_string()];
        command.environment.insert(
            "RUNMAT_EXECUTION_ALLOCATION_ID".to_string(),
            allocation.id.clone(),
        );
        command.environment.insert(
            "RUNMAT_EXECUTION_FENCING_TOKEN".to_string(),
            allocation.fencing_token.to_string(),
        );
        command.working_directory = Some(sandbox.root.clone());
        command.lifetime = ChildLifetime::Owned;
        command.stdio = StdioPolicy::Files {
            stdout: sandbox.stdout.clone(),
            stderr: sandbox.stderr.clone(),
        };
        command.resource_limits = ResourceLimits {
            memory_bytes: Some(allocation.resources.memory_bytes),
            cpu_seconds: Some(
                allocation
                    .resources
                    .maximum_wall_millis
                    .div_ceil(1_000)
                    .max(1),
            ),
            process_count: Some(64),
        };
        let child = command.spawn().await?;
        let process_id = child.id().ok_or_else(|| {
            AgentError::AllocationRejected("driver process has no id".to_string())
        })?;
        self.children.insert(
            allocation.id.clone(),
            ManagedProcess {
                child,
                process_id,
                fencing_token: allocation.fencing_token,
                expires_at_millis: allocation.expires_at_millis,
                maximum_memory_bytes: allocation.resources.memory_bytes,
                maximum_wall_millis: allocation.resources.maximum_wall_millis,
                started_at_millis: chrono::Utc::now().timestamp_millis(),
            },
        );
        Ok(process_id)
    }

    pub async fn terminate(&mut self, allocation_id: &str) -> AgentResult<()> {
        if let Some(mut process) = self.children.remove(allocation_id) {
            process.child.terminate_tree().await?;
        }
        Ok(())
    }

    pub async fn terminate_all(&mut self) -> AgentResult<()> {
        let ids = self.children.keys().cloned().collect::<Vec<_>>();
        for id in ids {
            self.terminate(&id).await?;
        }
        Ok(())
    }

    pub fn reap_finished(&mut self) -> AgentResult<Vec<String>> {
        let mut completed = Vec::new();
        for (allocation_id, process) in &mut self.children {
            if process.child.try_wait()?.is_some() {
                completed.push(allocation_id.clone());
            }
        }
        for allocation_id in &completed {
            self.children.remove(allocation_id);
        }
        Ok(completed)
    }

    pub async fn fence_stale(
        &mut self,
        leases: &[NodeAllocation],
        now_millis: i64,
    ) -> AgentResult<Vec<String>> {
        let stale = self
            .children
            .iter()
            .filter_map(|(allocation_id, process)| {
                let authorized = leases.iter().any(|lease| {
                    lease.id == *allocation_id
                        && lease.fencing_token == process.fencing_token
                        && lease.expires_at_millis == process.expires_at_millis
                        && lease.expires_at_millis > now_millis
                        && matches!(lease.state.as_str(), "offered" | "active")
                });
                (!authorized).then(|| allocation_id.clone())
            })
            .collect::<Vec<_>>();
        for allocation_id in &stale {
            self.terminate(allocation_id).await?;
        }
        Ok(stale)
    }

    pub async fn enforce_local_limits(&mut self, now_millis: i64) -> AgentResult<Vec<String>> {
        let mut system = sysinfo::System::new();
        system.refresh_processes();
        let exceeded = self
            .children
            .iter()
            .filter_map(|(allocation_id, process)| {
                let wall_deadline = process
                    .started_at_millis
                    .saturating_add(i64::try_from(process.maximum_wall_millis).unwrap_or(i64::MAX));
                let memory_exceeded = system
                    .process(sysinfo::Pid::from_u32(process.process_id))
                    .is_some_and(|value| value.memory() > process.maximum_memory_bytes);
                (now_millis >= wall_deadline || memory_exceeded).then(|| allocation_id.clone())
            })
            .collect::<Vec<_>>();
        for allocation_id in &exceeded {
            self.terminate(allocation_id).await?;
        }
        Ok(exceeded)
    }

    pub fn active_count(&self) -> usize {
        self.children.len()
    }
}
