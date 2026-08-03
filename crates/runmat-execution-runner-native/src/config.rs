use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct NativeExecutionConfig {
    pub executable: PathBuf,
    pub worker_arguments: Vec<String>,
    pub max_workers: u32,
    pub max_message_bytes: u32,
    pub max_stderr_bytes: usize,
    pub store_root: PathBuf,
}

impl NativeExecutionConfig {
    pub fn for_current_executable() -> std::io::Result<Self> {
        let mut store_root = std::env::temp_dir();
        store_root.push("runmat-execution");
        store_root.push(format!("session-{}", std::process::id()));
        Ok(Self {
            executable: std::env::current_exe()?,
            worker_arguments: vec![runmat_process_host::HiddenMode::ExecutionWorker
                .marker()
                .to_string()],
            max_workers: std::thread::available_parallelism()
                .map_or(1, |count| count.get().min(32) as u32),
            max_message_bytes: 64 * 1024 * 1024,
            max_stderr_bytes: 1024 * 1024,
            store_root,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.executable.as_os_str().is_empty()
            || self.worker_arguments.is_empty()
            || self.max_workers == 0
            || self.max_message_bytes == 0
            || self.max_stderr_bytes == 0
            || self.store_root.as_os_str().is_empty()
        {
            return Err("native execution configuration contains an empty bound".into());
        }
        Ok(())
    }
}
