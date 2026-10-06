//! Session environment state. A frozen snapshot never falls back to the host process.
use std::{collections::BTreeMap, io};

pub(crate) struct EnvironmentState(BTreeMap<String, String>);

impl std::fmt::Debug for EnvironmentState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EnvironmentState")
            .field("entry_count", &self.0.len())
            .finish_non_exhaustive()
    }
}

impl EnvironmentState {
    pub(crate) fn from_host() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let entries = std::env::vars_os()
            .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let entries = [("HOME", "/"), ("PATH", "/"), ("USER", "user")]
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        Self(entries)
    }

    pub(super) fn from_snapshot(entries: Vec<(String, String)>) -> io::Result<Self> {
        if entries.windows(2).any(|pair| pair[0].0 >= pair[1].0)
            || entries.iter().any(|(key, value)| {
                key.is_empty() || key.contains(['=', '\0']) || value.contains('\0')
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "environment snapshot names must be valid, sorted and unique",
            ));
        }
        Ok(Self(entries.into_iter().collect()))
    }

    pub(crate) fn var(&self, key: &str) -> io::Result<String> {
        self.0.get(key).cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "environment variable is not present in this session",
            )
        })
    }
    pub(crate) fn vars(&self) -> Vec<(String, String)> {
        self.0
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }
    pub(crate) fn set_var(&mut self, key: &str, value: &str) {
        self.0.insert(key.into(), value.into());
    }
    pub(crate) fn remove_var(&mut self, key: &str) {
        self.0.remove(key);
    }
}

impl super::RuntimeContext {
    /// Construct an isolated context without first collecting ambient host variables.
    pub fn with_frozen_environment(
        execution: std::rc::Rc<dyn crate::execution::RuntimeExecutionServices>,
        cancellation: std::sync::Arc<std::sync::atomic::AtomicBool>,
        entries: Vec<(String, String)>,
    ) -> io::Result<Self> {
        let environment = EnvironmentState::from_snapshot(entries)?;
        Ok(Self::with_environment_state(
            execution,
            cancellation,
            environment,
        ))
    }

    /// Replace all visible environment values, without reading or changing the process.
    /// Validation completes before replacement, so invalid imports leave state intact.
    pub fn install_environment_snapshot(&self, entries: Vec<(String, String)>) -> io::Result<()> {
        let snapshot = EnvironmentState::from_snapshot(entries)?;
        self.state().environment.replace(snapshot);
        Ok(())
    }

    pub fn environment_snapshot(&self) -> Vec<(String, String)> {
        self.state().environment.borrow().vars()
    }
}

#[cfg(test)]
mod tests;
