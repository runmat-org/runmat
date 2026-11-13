use std::collections::BTreeMap;
use std::path::PathBuf;

use runmat_execution_transport_native::control::DriverBootstrapCredential;
use runmat_execution_transport_native::control::NodeAllocation;
use runmat_execution_transport_native::control::WorkerBootstrapCredential;
use runmat_process_host::{
    clock::observe_monotonic,
    environment::{EnvironmentAllowlist, EnvironmentPolicy},
    ChildLifetime, ChildProcess, HiddenMode, HostCommand, ProcessSpawnEvidence, ResourceLimits,
    StdioPolicy,
};

use crate::{AgentError, AgentResult};

#[path = "launch/authority.rs"]
mod authority;
use authority::ProcessLaunchAuthority;
pub use authority::{AllocationLaunchContext, DriverClockScope, WorkerRetirementScope};

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
    worker_retirement: Option<WorkerRetirementScope>,
    driver_clock: Option<DriverClockScope>,
    /// The owned tree has retired, but its durable retirement record has not
    /// yet been committed by the service.
    retirement_prepared: bool,
}

struct WorkerJournalLaunch {
    process_launch_nonce: [u8; 16],
    clock_domain_nonce: [u8; 16],
    directory: PathBuf,
}

impl AllocationProcesses {
    pub async fn launch_worker(
        &mut self,
        context: AllocationLaunchContext<'_>,
        bootstrap: &WorkerBootstrapCredential,
        host: &runmat_execution::host::ExecutionHostInventory,
        spawn_evidence: &ProcessSpawnEvidence,
    ) -> AgentResult<u32> {
        use base64::Engine as _;
        let AllocationLaunchContext {
            runmat_executable,
            allocation,
            sandbox,
            server_url,
            clock_domain_nonce,
        } = context;

        if self.children.contains_key(&allocation.id) {
            return Err(AgentError::AllocationRejected(
                "allocation already has a worker".to_string(),
            ));
        }
        let mut command = HostCommand::new(runmat_executable);
        let process_launch_nonce = loop {
            let nonce: [u8; 16] = rand::random();
            if nonce.iter().any(|value| *value != 0) {
                break nonce;
            }
        };
        let journal_directory = sandbox.root.join("attempt-journal");
        std::fs::create_dir(&journal_directory)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&journal_directory, std::fs::Permissions::from_mode(0o700))?;
        }
        command.environment_policy =
            EnvironmentPolicy::Allow(EnvironmentAllowlist::platform_runtime());
        command.arguments = vec![HiddenMode::ExecutionWorker.marker().to_string()];
        command.environment.insert(
            "RUNMAT_EXECUTION_WORKER_REMOTE".to_string(),
            "1".to_string(),
        );
        if !allocation.resources.accelerators.is_empty() {
            command.environment.insert(
                "RUNMAT_EXECUTION_ACCELERATOR_REQUIRED".to_string(),
                "1".to_string(),
            );
        }
        if bootstrap.ready_v1 {
            command.environment.insert(
                "RUNMAT_EXECUTION_WORKER_READY_V1".to_string(),
                "1".to_string(),
            );
        }
        for (name, value) in [
            ("RUNMAT_EXECUTION_SERVER_URL", server_url.to_string()),
            ("RUNMAT_EXECUTION_RUN_ID", bootstrap.run_id.clone()),
            ("RUNMAT_EXECUTION_ORG_ID", bootstrap.org_id.clone()),
            ("RUNMAT_EXECUTION_PROJECT_ID", bootstrap.project_id.clone()),
            (
                "RUNMAT_EXECUTION_ALLOCATION_ID",
                bootstrap.allocation_lease_id.clone(),
            ),
            (
                "RUNMAT_EXECUTION_ALLOCATION_FENCING_TOKEN",
                bootstrap.allocation_fencing_token.to_string(),
            ),
            (
                "RUNMAT_EXECUTION_DRIVER_FENCING_TOKEN",
                bootstrap.driver_fencing_token.to_string(),
            ),
            (
                "RUNMAT_EXECUTION_ENDPOINT_FINGERPRINT",
                bootstrap.endpoint_fingerprint.clone(),
            ),
            (
                "RUNMAT_EXECUTION_RUN_KEY_ENVELOPE",
                base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .encode(&bootstrap.run_key_envelope),
            ),
            (
                "RUNMAT_EXECUTION_WORKER_RELAY_PATH",
                bootstrap.relay_path.clone(),
            ),
            (
                "RUNMAT_EXECUTION_WORKER_RELAY_PROTOCOL",
                bootstrap.relay_protocol.clone(),
            ),
            (
                "RUNMAT_EXECUTION_WORKER_RELAY_TICKET",
                bootstrap.relay_ticket.clone(),
            ),
            (
                "RUNMAT_EXECUTION_WORKER_RESOURCES",
                serde_json::to_string(
                    &runmat_execution_transport_native::control::AllocatedResources {
                        request: allocation.resources.clone(),
                        accelerator_devices: allocation.accelerator_devices.clone(),
                        host: host.clone(),
                    },
                )?,
            ),
            (
                "RUNMAT_EXECUTION_ENDPOINT_IDENTITY_FILE",
                sandbox
                    .root
                    .join("endpoint-identity.json")
                    .to_string_lossy()
                    .into_owned(),
            ),
            (
                "RUNMAT_EXECUTION_NODE_BUNDLE_CACHE",
                sandbox
                    .root
                    .parent()
                    .ok_or_else(|| {
                        AgentError::AllocationRejected(
                            "allocation sandbox has no node cache parent".into(),
                        )
                    })?
                    .join("bundle-cache")
                    .to_string_lossy()
                    .into_owned(),
            ),
            (
                "RUNMAT_EXECUTION_ATTEMPT_JOURNAL_DIRECTORY",
                journal_directory.to_string_lossy().into_owned(),
            ),
            (
                "RUNMAT_EXECUTION_PROCESS_LAUNCH_NONCE",
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(process_launch_nonce),
            ),
            (
                "RUNMAT_EXECUTION_CLOCK_DOMAIN_NONCE",
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(clock_domain_nonce),
            ),
        ] {
            command.environment.insert(name.to_string(), value);
        }
        command.working_directory = Some(sandbox.root.clone());
        command.lifetime = ChildLifetime::Owned;
        command.stdio = StdioPolicy::Files {
            stdout: sandbox.stdout.clone(),
            stderr: sandbox.stderr.clone(),
        };
        command.resource_limits = allocation_limits(allocation);
        let result = self
            .spawn(
                allocation,
                command,
                ProcessLaunchAuthority::Worker {
                    bootstrap,
                    spawn_evidence,
                    journal: WorkerJournalLaunch {
                        process_launch_nonce,
                        clock_domain_nonce,
                        directory: journal_directory.clone(),
                    },
                },
            )
            .await;
        if result.is_err() && !spawn_evidence.is_unverified() {
            // Only synchronous proof that no child was created permits
            // discarding its empty pre-dispatch journal directory.
            let _ = std::fs::remove_dir(&journal_directory);
        }
        result
    }

    pub async fn launch_driver(
        &mut self,
        context: AllocationLaunchContext<'_>,
        bootstrap: &DriverBootstrapCredential,
        node_identity_fingerprint: &str,
    ) -> AgentResult<u32> {
        use base64::Engine as _;
        let AllocationLaunchContext {
            runmat_executable,
            allocation,
            sandbox,
            server_url,
            clock_domain_nonce,
        } = context;
        if self.children.contains_key(&allocation.id) {
            return Err(AgentError::AllocationRejected(
                "allocation already has a driver".to_string(),
            ));
        }
        let mut command = HostCommand::new(runmat_executable);
        let process_launch_nonce: [u8; 16] = loop {
            let nonce: [u8; 16] = rand::random();
            if nonce.iter().any(|value| *value != 0) {
                break nonce;
            }
        };
        let request_directory = sandbox.root.join("clock-requests");
        let receipt_directory = sandbox.root.join("clock-receipts");
        std::fs::create_dir(&request_directory)?;
        std::fs::create_dir(&receipt_directory)?;
        command.environment_policy =
            EnvironmentPolicy::Allow(EnvironmentAllowlist::platform_runtime());
        // This exact private mode remains the sole process argument. Lease
        // authority crosses in a sanitized environment; the Server cannot
        // choose an executable or arbitrary argument vector.
        command.arguments = vec![HiddenMode::ExecutionDriver.marker().to_string()];
        command.environment.insert(
            "RUNMAT_EXECUTION_ALLOCATION_ID".to_string(),
            allocation.id.clone(),
        );
        command.environment.insert(
            "RUNMAT_EXECUTION_SERVER_URL".to_string(),
            server_url.to_string(),
        );
        command.environment.insert(
            "RUNMAT_EXECUTION_RUN_ID".to_string(),
            bootstrap.run_id.clone(),
        );
        command.environment.insert(
            "RUNMAT_EXECUTION_ORG_ID".to_string(),
            bootstrap.org_id.clone(),
        );
        command.environment.insert(
            "RUNMAT_EXECUTION_PROJECT_ID".to_string(),
            bootstrap.project_id.clone(),
        );
        command.environment.insert(
            "RUNMAT_EXECUTION_DRIVER_LEASE_ID".to_string(),
            bootstrap.driver_lease_id.clone(),
        );
        command.environment.insert(
            "RUNMAT_EXECUTION_DRIVER_FENCING_TOKEN".to_string(),
            bootstrap.fencing_token.to_string(),
        );
        command.environment.insert(
            "RUNMAT_EXECUTION_DRIVER_CREDENTIAL".to_string(),
            bootstrap.credential.clone(),
        );
        for (name, value) in [
            (
                "RUNMAT_EXECUTION_DRIVER_ALLOCATION_FENCE",
                allocation.fencing_token.to_string(),
            ),
            (
                "RUNMAT_EXECUTION_DRIVER_PROCESS_NONCE",
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(process_launch_nonce),
            ),
            (
                "RUNMAT_EXECUTION_CLOCK_DOMAIN_NONCE",
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(clock_domain_nonce),
            ),
            (
                "RUNMAT_EXECUTION_PARENT_NODE_FINGERPRINT",
                node_identity_fingerprint.to_string(),
            ),
            (
                "RUNMAT_EXECUTION_CLOCK_REQUEST_DIRECTORY",
                request_directory.to_string_lossy().into_owned(),
            ),
            (
                "RUNMAT_EXECUTION_CLOCK_RECEIPT_DIRECTORY",
                receipt_directory.to_string_lossy().into_owned(),
            ),
        ] {
            command.environment.insert(name.to_string(), value);
        }
        if bootstrap.ready_v1 {
            command.environment.insert(
                "RUNMAT_EXECUTION_SERVER_READY_V1".to_string(),
                "1".to_string(),
            );
        }
        command.environment.insert(
            "RUNMAT_EXECUTION_ENDPOINT_IDENTITY_FILE".to_string(),
            sandbox
                .root
                .join("endpoint-identity.json")
                .to_string_lossy()
                .into_owned(),
        );
        command.working_directory = Some(sandbox.root.clone());
        command.lifetime = ChildLifetime::Owned;
        command.stdio = StdioPolicy::Files {
            stdout: sandbox.stdout.clone(),
            stderr: sandbox.stderr.clone(),
        };
        command.resource_limits = allocation_limits(allocation);
        let clock = DriverClockScope {
            run_id: bootstrap.run_id.clone(),
            allocation_lease_id: allocation.id.clone(),
            allocation_fence: allocation.fencing_token,
            process_launch_nonce,
            clock_domain_nonce,
            request_directory,
            receipt_directory,
        };
        self.spawn(
            allocation,
            command,
            ProcessLaunchAuthority::Driver { clock },
        )
        .await
    }

    async fn spawn(
        &mut self,
        allocation: &NodeAllocation,
        command: HostCommand,
        authority: ProcessLaunchAuthority<'_>,
    ) -> AgentResult<u32> {
        let child = match &authority {
            ProcessLaunchAuthority::Worker { spawn_evidence, .. } => {
                command.spawn_observed(spawn_evidence).await?
            }
            ProcessLaunchAuthority::Driver { .. } => command.spawn().await?,
        };
        let process_id = child.id().ok_or_else(|| {
            let process_kind = match &authority {
                ProcessLaunchAuthority::Worker { .. } => "worker",
                ProcessLaunchAuthority::Driver { .. } => "driver",
            };
            AgentError::AllocationRejected(format!("{process_kind} process has no id"))
        })?;
        let (worker_retirement, driver_clock) = match authority {
            ProcessLaunchAuthority::Worker {
                bootstrap, journal, ..
            } => (
                Some(WorkerRetirementScope {
                    run_id: bootstrap.run_id.clone(),
                    allocation_lease_id: bootstrap.allocation_lease_id.clone(),
                    allocation_fencing_token: bootstrap.allocation_fencing_token,
                    driver_fencing_token: bootstrap.driver_fencing_token,
                    process_id,
                    process_launch_nonce: journal.process_launch_nonce,
                    clock_domain_nonce: journal.clock_domain_nonce,
                    journal_directory: journal.directory,
                    retired_monotonic: None,
                    retired_at_unix_millis: 0,
                }),
                None,
            ),
            ProcessLaunchAuthority::Driver { clock } => (None, Some(clock)),
        };
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
                worker_retirement,
                driver_clock,
                retirement_prepared: false,
            },
        );
        Ok(process_id)
    }

    pub async fn terminate(
        &mut self,
        allocation_id: &str,
        prepare: &mut impl FnMut(&str, Option<&WorkerRetirementScope>) -> AgentResult<()>,
    ) -> AgentResult<Option<WorkerRetirementScope>> {
        let Some(process) = self.children.get_mut(allocation_id) else {
            return Ok(None);
        };
        if process.retirement_prepared {
            return Ok(process.worker_retirement.clone());
        }
        // Publish the original process/clock ownership before the first
        // destructive process-tree action. A restart can then retain the
        // exact domain fail-closed without inventing a new clock reading.
        prepare(allocation_id, process.worker_retirement.as_ref())?;
        // Keep the owner and its charge on every failure. A release is not a
        // retirement certificate until the owned process tree has retired.
        process.child.terminate_tree().await?;
        if let Some(scope) = &mut process.worker_retirement {
            scope.retired_monotonic = Some(observe_monotonic()?);
            scope.retired_at_unix_millis = chrono::Utc::now()
                .timestamp_millis()
                .try_into()
                .unwrap_or(1);
        }
        process.retirement_prepared = true;
        Ok(process.worker_retirement.clone())
    }

    pub async fn terminate_all(
        &mut self,
        prepare: &mut impl FnMut(&str, Option<&WorkerRetirementScope>) -> AgentResult<()>,
    ) -> AgentResult<Vec<(String, Option<WorkerRetirementScope>)>> {
        let ids = self.children.keys().cloned().collect::<Vec<_>>();
        let mut retired = Vec::new();
        for id in ids {
            let scope = self.terminate(&id, prepare).await?;
            retired.push((id, scope));
        }
        Ok(retired)
    }

    pub async fn reap_finished(
        &mut self,
        prepare: &mut impl FnMut(&str, Option<&WorkerRetirementScope>) -> AgentResult<()>,
    ) -> AgentResult<Vec<(String, Option<WorkerRetirementScope>)>> {
        let mut completed = Vec::new();
        for (allocation_id, process) in &mut self.children {
            if process.retirement_prepared {
                completed.push(allocation_id.clone());
                continue;
            }
            // Observe without reaping the leader: wait_and_retire_tree still
            // owns the process group and certifies descendants.
            if process.child.try_wait_unreaped()?.is_some() {
                completed.push(allocation_id.clone());
            }
        }
        let mut retired = Vec::new();
        for allocation_id in completed {
            let process = self
                .children
                .get_mut(&allocation_id)
                .expect("observed owner");
            if process.retirement_prepared {
                retired.push((allocation_id, process.worker_retirement.clone()));
                continue;
            }
            prepare(&allocation_id, process.worker_retirement.as_ref())?;
            if process.child.wait_and_retire_tree().await.is_err() {
                // Keep the owner and its allocation for a later retirement
                // attempt; a leader exit alone cannot certify descendants.
                continue;
            }
            if let Some(value) = &mut process.worker_retirement {
                value.retired_monotonic = Some(observe_monotonic()?);
                value.retired_at_unix_millis = chrono::Utc::now()
                    .timestamp_millis()
                    .try_into()
                    .unwrap_or(1);
            }
            process.retirement_prepared = true;
            retired.push((allocation_id, process.worker_retirement.clone()));
        }
        Ok(retired)
    }

    pub async fn fence_stale(
        &mut self,
        leases: &[NodeAllocation],
        now_millis: i64,
        prepare: &mut impl FnMut(&str, Option<&WorkerRetirementScope>) -> AgentResult<()>,
    ) -> AgentResult<Vec<(String, Option<WorkerRetirementScope>)>> {
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
        let mut retired = Vec::new();
        for allocation_id in stale {
            let scope = self.terminate(&allocation_id, prepare).await?;
            retired.push((allocation_id, scope));
        }
        Ok(retired)
    }

    pub async fn enforce_local_limits(
        &mut self,
        now_millis: i64,
        prepare: &mut impl FnMut(&str, Option<&WorkerRetirementScope>) -> AgentResult<()>,
    ) -> AgentResult<Vec<(String, Option<WorkerRetirementScope>)>> {
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
        let mut retired = Vec::new();
        for allocation_id in exceeded {
            let scope = self.terminate(&allocation_id, prepare).await?;
            retired.push((allocation_id, scope));
        }
        Ok(retired)
    }

    pub fn commit_retired(&mut self, allocation_ids: &[String]) -> AgentResult<()> {
        for allocation_id in allocation_ids {
            let process = self.children.get(allocation_id).ok_or_else(|| {
                AgentError::AllocationRejected(
                    "durable retirement lost its live process owner".into(),
                )
            })?;
            if !process.retirement_prepared {
                return Err(AgentError::AllocationRejected(
                    "process owner was removed before tree retirement".into(),
                ));
            }
        }
        for allocation_id in allocation_ids {
            self.children.remove(allocation_id);
        }
        Ok(())
    }

    pub fn active_count(&self) -> usize {
        self.children.len()
    }

    pub fn driver_clock_scopes(&self) -> Vec<DriverClockScope> {
        self.children
            .values()
            .filter_map(|process| process.driver_clock.clone())
            .collect()
    }

    pub fn contains(&self, allocation_id: &str) -> bool {
        self.children.contains_key(allocation_id)
    }
}

fn allocation_limits(allocation: &NodeAllocation) -> ResourceLimits {
    ResourceLimits {
        memory_bytes: Some(allocation.resources.memory_bytes),
        cpu_seconds: Some(
            allocation
                .resources
                .maximum_wall_millis
                .div_ceil(1_000)
                .max(1),
        ),
        process_count: Some(64),
    }
}
