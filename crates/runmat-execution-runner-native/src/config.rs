use std::collections::BTreeSet;
use std::io;
use std::path::PathBuf;

use runmat_execution::{resource::Capability, ExecutionScopeId};

#[derive(Clone, Debug)]
pub struct NativeExecutionConfig {
    pub executable: PathBuf,
    pub worker_arguments: Vec<String>,
    pub max_workers: u32,
    pub max_message_bytes: u32,
    pub max_object_bytes: u64,
    pub max_stderr_bytes: usize,
    pub store_root: PathBuf,
    pub worker_capabilities: BTreeSet<Capability>,
}

impl NativeExecutionConfig {
    pub fn for_current_executable() -> std::io::Result<Self> {
        let store_root = session_store_root(
            std::env::var_os("RUNMAT_EXECUTION_STATE_DIR").map(PathBuf::from),
            dirs::cache_dir(),
        )?;
        Ok(Self {
            executable: std::env::current_exe()?,
            worker_arguments: vec![runmat_process_host::HiddenMode::ExecutionWorker
                .marker()
                .to_string()],
            max_workers: std::thread::available_parallelism()
                .map_or(1, |count| count.get().min(32) as u32),
            max_message_bytes: 64 * 1024 * 1024,
            max_object_bytes: 512 * 1024 * 1024,
            max_stderr_bytes: 1024 * 1024,
            store_root,
            worker_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
        })
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.executable.as_os_str().is_empty()
            || self.worker_arguments.is_empty()
            || self.max_workers == 0
            || self.max_message_bytes == 0
            || self.max_object_bytes == 0
            || self.max_stderr_bytes == 0
            || self.store_root.as_os_str().is_empty()
            || !self
                .worker_capabilities
                .contains(&Capability::ProcessIsolation)
        {
            return Err("native execution configuration contains an empty bound".into());
        }
        Ok(())
    }
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

    use super::{fresh_scope_id, session_store_root};

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
}
