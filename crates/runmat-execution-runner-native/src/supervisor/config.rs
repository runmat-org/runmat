use super::SupervisorPaths;
use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Clone, Debug)]
pub struct LocalSupervisorConfig {
    pub execution: crate::NativeExecutionConfig,
    pub paths: SupervisorPaths,
}

impl LocalSupervisorConfig {
    pub fn new(
        mut execution: crate::NativeExecutionConfig,
        paths: SupervisorPaths,
    ) -> NativeExecutionResult<Self> {
        execution.store_root = paths.root.clone();
        execution
            .validate()
            .map_err(NativeExecutionError::Configuration)?;
        Ok(Self { execution, paths })
    }

    pub fn for_current_executable() -> NativeExecutionResult<Self> {
        if let Some(path) = std::env::var_os(super::bootstrap::PROFILE_ENV) {
            return super::bootstrap::read_profile(std::path::Path::new(&path));
        }
        super::bootstrap::resolve_default_profile(
            SupervisorPaths::platform_default()?,
            &std::env::current_exe().map_err(filesystem_error)?,
            || crate::NativeExecutionConfig::for_current_executable().map_err(filesystem_error),
        )
    }

    pub(super) fn identity(&self) -> NativeExecutionResult<runmat_execution::Digest> {
        self.validate()?;
        let bytes = runmat_canonical_codec::encode_json_bounded(
            &self.execution,
            super::bootstrap::PROFILE_MAX_BYTES,
        )?;
        Ok(runmat_execution::Digest::sha256(bytes))
    }

    pub(super) fn validate(&self) -> NativeExecutionResult<()> {
        self.execution
            .validate()
            .map_err(NativeExecutionError::Configuration)?;
        if self.execution.store_root != self.paths.root
            || self.paths != SupervisorPaths::new(self.paths.root.clone())?
        {
            return Err(NativeExecutionError::Configuration(
                "supervisor execution store differs from its authority root".into(),
            ));
        }
        Ok(())
    }
}

fn filesystem_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Configuration(error.to_string())
}
