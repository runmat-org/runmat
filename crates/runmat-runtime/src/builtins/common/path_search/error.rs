//! Lossless failures shared by source discovery and read-only path observation.

use std::{fmt, io, path::PathBuf};

#[derive(Debug)]
pub enum PathSearchError {
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    HomeUnavailable,
    TooManyRoots,
    Control(Box<crate::RuntimeError>),
}

impl PathSearchError {
    pub(super) fn io(operation: &'static str, path: &std::path::Path, source: io::Error) -> Self {
        Self::Io {
            operation,
            path: path.to_path_buf(),
            source,
        }
    }

    /// Operational discovery failures are terminal, not permission to try a
    /// different executable or replay the invocation through a gather fallback.
    pub fn into_runtime_error(self, builtin: &str) -> crate::RuntimeError {
        self.project(builtin, None)
    }

    /// Keep an existing caller diagnostic without replacing a selected
    /// control failure's identifier or source ownership.
    pub fn into_identified_runtime_error(
        self,
        builtin: &str,
        identifier: &str,
    ) -> crate::RuntimeError {
        let mut error = self.into_runtime_error(builtin);
        if error.identifier.is_none() {
            error.identifier = Some(identifier.to_owned());
        }
        error
    }

    pub fn into_builtin_error(
        self,
        builtin: &str,
        diagnostic: &runmat_builtins::BuiltinErrorDescriptor,
    ) -> crate::RuntimeError {
        self.project(builtin, Some(diagnostic))
    }

    fn project(
        self,
        builtin: &str,
        diagnostic: Option<&runmat_builtins::BuiltinErrorDescriptor>,
    ) -> crate::RuntimeError {
        if let Self::Control(mut error) = self {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            return *error;
        }
        let message = diagnostic.map_or_else(
            || format!("{builtin}: {self}"),
            |diagnostic| format!("{}: {self}", diagnostic.message),
        );
        let mut builder = crate::build_runtime_error(message)
            .with_builtin(builtin)
            .with_source(self)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
        if let Some(identifier) = diagnostic.and_then(|diagnostic| diagnostic.identifier) {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

impl fmt::Display for PathSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(f, "{operation} '{}': {source}", path.display()),
            Self::HomeUnavailable => f.write_str("unable to resolve home directory"),
            Self::TooManyRoots => f.write_str("search path has too many roots"),
            Self::Control(error) => f.write_str(error.message()),
        }
    }
}

impl std::error::Error for PathSearchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Control(error) => Some(error.as_ref()),
            Self::HomeUnavailable | Self::TooManyRoots => None,
        }
    }
}
