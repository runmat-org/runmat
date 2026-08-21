use std::path::{Path, PathBuf};

use runmat_execution_transport_native::control::{NodeAllocation, NodeInventory};

use crate::{AgentError, AgentResult};

#[derive(Debug, Clone)]
pub struct Sandbox {
    pub root: PathBuf,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
}

pub fn prepare(
    state_directory: &Path,
    allocation: &NodeAllocation,
    inventory: &NodeInventory,
) -> AgentResult<Sandbox> {
    let expected_runtime = inventory
        .capabilities
        .get("runmat.version")
        .ok_or_else(|| AgentError::AllocationRejected("runtime version is unknown".to_string()))?;
    if expected_runtime != env!("CARGO_PKG_VERSION") {
        return Err(AgentError::AllocationRejected(
            "runtime version does not match this agent".to_string(),
        ));
    }
    if !allocation
        .id
        .bytes()
        .all(|value| value.is_ascii_alphanumeric() || matches!(value, b'_' | b'-'))
    {
        return Err(AgentError::AllocationRejected(
            "lease id is unsafe for a sandbox path".to_string(),
        ));
    }
    let root = state_directory.join("allocations").join(&allocation.id);
    std::fs::create_dir_all(&root)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(Sandbox {
        stdout: root.join("stdout.log"),
        stderr: root.join("stderr.log"),
        root,
    })
}
