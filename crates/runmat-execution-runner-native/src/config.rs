use std::collections::BTreeSet;
use std::io;
use std::path::PathBuf;

use runmat_execution::{resource::Capability, ExecutionScopeId};

use crate::{NativeExecutionError, NativeExecutionResult};

/// Preserve a captured invalid selector as a denial at actual FEA admission.
/// Plain child programs do not consult the lazy account provider.
pub(crate) fn local_organization_environment(
    selector: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
) -> Option<String> {
    use runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization;
    match selector {
        FeaLocalOrganization::Default => None,
        FeaLocalOrganization::Selected(org) => Some(org.to_string()),
        FeaLocalOrganization::Invalid => Some("invalid-local-fea-organization".into()),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeExecutionConfig {
    /// Captured local lease account selector, never program environment or authority.
    #[serde(skip)]
    pub local_fea_organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    pub executable: PathBuf,
    pub worker_arguments: Vec<String>,
    pub max_workers: u32,
    /// Aggregate local capacity, shared by all registered worker slots.
    /// None is reserved for externally supplied/simulated inventories.
    #[serde(default)]
    pub local_cpu_capacity_millicores: Option<u32>,
    #[serde(default)]
    pub local_memory_capacity_bytes: Option<u64>,
    /// The authored per-worker resident ceiling (`execution { resources { memory_per_worker } }`).
    /// Analysis placement plans each worker within min(this, the step's authored ceiling, the
    /// host's admitted memory). Execution policy only: it never enters a study identity.
    #[serde(default)]
    pub authored_worker_memory_bytes: Option<u64>,
    pub max_message_bytes: u32,
    pub max_object_bytes: u64,
    pub max_stderr_bytes: usize,
    pub worker_memory_bytes: u64,
    pub worker_scratch_bytes: u64,
    pub worker_cpu_millicores: u32,
    pub store_root: PathBuf,
    /// Host-granted, bounded numerical reuse/diagnostics, outside transient
    /// session storage. Session constructors resolve None before adding a scope.
    #[serde(default)]
    pub meshing_subtask_cache_root: Option<PathBuf>,
    pub worker_capabilities: BTreeSet<Capability>,
    pub host_inventory: runmat_execution::host::ExecutionHostInventory,
    pub accelerator_devices: Vec<runmat_execution::resource::AcceleratorDevice>,
}

impl NativeExecutionConfig {
    pub(crate) fn scope_store(&mut self, scope: ExecutionScopeId) {
        self.meshing_subtask_cache_root
            .get_or_insert_with(|| self.store_root.join("meshing-subtask-cache/v1"));
        self.store_root.push(scope.to_string());
    }

    pub fn for_current_executable() -> std::io::Result<Self> {
        let store_root = session_store_root(
            std::env::var_os("RUNMAT_EXECUTION_STATE_DIR").map(PathBuf::from),
            dirs::cache_dir(),
        )?;
        let host_inventory = runmat_core::RunMatSession::with_options(false, false)
            .map_err(io::Error::other)?
            .execution_host_inventory(
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
            )
            .map_err(io::Error::other)?;
        let epoch = local_inventory_epoch();
        let mut accelerator_devices = runmat_accelerate_api::registered_providers()
            .into_iter()
            .map(|provider| provider.execution_accelerator_device(epoch))
            .filter_map(|result| result.transpose())
            .collect::<Result<Vec<_>, _>>()
            .map_err(io::Error::other)?;
        accelerator_devices.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(Self {
            local_fea_organization:
                runmat_runtime::analysis::fea_run_admission::current_local_fea_organization_hint(),
            executable: std::env::current_exe()?,
            worker_arguments: vec![runmat_process_host::HiddenMode::ExecutionWorker
                .marker()
                .to_string()],
            max_workers: std::thread::available_parallelism()
                .map_or(1, |count| count.get().min(32) as u32),
            local_cpu_capacity_millicores: Some(super::machine_capacity::cpu_millicores()),
            local_memory_capacity_bytes: Some(super::machine_capacity::memory_bytes()?),
            authored_worker_memory_bytes: None,
            max_message_bytes: 64 * 1024 * 1024,
            max_object_bytes: 512 * 1024 * 1024,
            max_stderr_bytes: 1024 * 1024,
            worker_memory_bytes: 1024 * 1024 * 1024,
            worker_scratch_bytes: 1024 * 1024 * 1024,
            worker_cpu_millicores: super::machine_capacity::cpu_millicores(),
            store_root,
            meshing_subtask_cache_root: None,
            worker_capabilities: BTreeSet::from([
                Capability::ProcessIsolation,
                Capability::GraphSelectedValuesV11,
                Capability::GraphCallableArgumentsV12,
                Capability::GraphProjectRuntimePolicyV13,
                Capability::GraphInvocationControlsV14,
            ]),
            host_inventory,
            accelerator_devices,
        })
    }

    pub fn enable_exact_meshing(
        &mut self,
        document: &runmat_geometry_core::GeometryDocument,
        request: &runmat_meshing_core::MeshingRequest,
        capability_cohort: Option<&str>,
    ) -> NativeExecutionResult<()> {
        self.worker_arguments = vec![runmat_process_host::HiddenMode::MeshingWorker
            .marker()
            .to_string()];
        self.worker_capabilities.extend(
            runmat_meshing_execution::exact_meshing_worker_capabilities(
                document,
                request,
                capability_cohort,
            )
            .map_err(|error| NativeExecutionError::Configuration(error.to_string()))?,
        );
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.executable.as_os_str().is_empty()
            || self.worker_arguments.is_empty()
            || self.max_workers == 0
            || self.local_cpu_capacity_millicores == Some(0)
            || self.local_memory_capacity_bytes == Some(0)
            || self.authored_worker_memory_bytes == Some(0)
            || self.max_message_bytes == 0
            || self.max_object_bytes == 0
            || self.max_stderr_bytes == 0
            || self.worker_memory_bytes == 0
            || self.worker_scratch_bytes == 0
            || self.worker_cpu_millicores == 0
            || self.store_root.as_os_str().is_empty()
            || self
                .meshing_subtask_cache_root
                .as_ref()
                .is_some_and(|root| !root.is_absolute())
            || !self
                .worker_capabilities
                .contains(&Capability::ProcessIsolation)
            || self.host_inventory.validate().is_err()
            || self
                .accelerator_devices
                .windows(2)
                .any(|pair| pair[0].id >= pair[1].id)
            || self
                .accelerator_devices
                .iter()
                .any(|device| device.validate().is_err())
        {
            return Err("native execution configuration contains an empty bound".into());
        }
        Ok(())
    }
}

fn local_inventory_epoch() -> u64 {
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    (u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX) ^ u64::from(std::process::id())).max(1)
}

fn session_store_root(
    configured_state_root: Option<PathBuf>,
    cache_root: Option<PathBuf>,
) -> io::Result<PathBuf> {
    let state_root = if let Some(root) = configured_state_root {
        if !root.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "RUNMAT_EXECUTION_STATE_DIR must be absolute",
            ));
        }
        root
    } else {
        cache_root
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "the per-user cache directory is unavailable",
                )
            })?
            .join("runmat")
            .join("execution")
    };
    Ok(state_root.join("sessions"))
}

pub(crate) fn fresh_scope_id(domain: &[u8], nonce: u64) -> ExecutionScopeId {
    let entropy = uuid::Uuid::new_v4();
    ExecutionScopeId::derive(&[
        domain,
        &std::process::id().to_be_bytes(),
        &nonce.to_be_bytes(),
        entropy.as_bytes(),
    ])
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{fresh_scope_id, session_store_root, NativeExecutionConfig};

    // witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Default numerical-cache placement is resolved before adding a transient scope,
    //   explicit host roots are preserved, and relative roots reject. Both native session constructors
    //   use this shared configuration operation. It does not independently establish payload
    //   admissibility, cross-process throughput or diagnostic replay success.
    #[test]
    fn meshing_cache_grant_is_resolved_before_transient_scope_and_preserves_explicit_root() {
        let directory = tempfile::tempdir().unwrap();
        for explicit in [false, true] {
            let mut config = NativeExecutionConfig::for_current_executable().unwrap();
            config.store_root = directory.path().join("sessions");
            let expected = if explicit {
                directory.path().join("explicit-numerical-cache")
            } else {
                config.store_root.join("meshing-subtask-cache/v1")
            };
            if explicit {
                config.meshing_subtask_cache_root = Some(expected.clone());
            }
            let scope = fresh_scope_id(b"retention-test", 1);
            config.scope_store(scope);
            assert_eq!(config.meshing_subtask_cache_root, Some(expected.clone()));
            assert!(!expected.starts_with(&config.store_root));
            assert!(config.validate().is_ok());
            config.meshing_subtask_cache_root = Some(PathBuf::from("relative-cache"));
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn default_session_store_is_scoped_to_the_current_user() {
        assert_eq!(
            session_store_root(None, Some(PathBuf::from("/user/cache"))).unwrap(),
            PathBuf::from("/user/cache")
                .join("runmat")
                .join("execution")
                .join("sessions")
        );
    }

    #[test]
    fn explicit_execution_state_root_is_shared_without_using_temp_state() {
        let temporary = tempfile::tempdir().unwrap();
        let state_root = temporary.path().join("state");
        assert_eq!(
            session_store_root(Some(state_root.clone()), None).unwrap(),
            state_root.join("sessions")
        );
        assert!(session_store_root(Some(PathBuf::from("relative")), None).is_err());
    }

    #[test]
    fn native_scope_ids_are_unique_across_process_lifetimes() {
        let first = fresh_scope_id(b"test-native-session", 1);
        let second = fresh_scope_id(b"test-native-session", 1);
        assert_ne!(first, second);
    }

    #[test]
    fn native_configuration_requires_explicit_worker_capacity() {
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.worker_memory_bytes = 0;
        assert!(config.validate().is_err());

        config.worker_memory_bytes = 1024;
        config.worker_scratch_bytes = 0;
        assert!(config.validate().is_err());
    }
}
