use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MexBuildError {
    #[error("C MEX compilation is unavailable for this target")]
    UnsupportedTarget,
    #[error("a MEX build requires at least one source file")]
    MissingSources,
    #[error("MEX source does not exist: {0}")]
    MissingSource(PathBuf),
    #[error("MEX output name must be a non-empty file stem")]
    InvalidOutputName,
    #[error("MEX preprocessor definition must be non-empty and contain no NUL bytes")]
    InvalidDefinition,
    #[error("failed to prepare the embedded MEX SDK at {path}: {source}")]
    PrepareSdk {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create MEX output directory {directory}: {source}")]
    CreateOutputDirectory {
        directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to invoke C compiler {compiler}: {source}")]
    CompilerLaunch {
        compiler: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("C MEX compilation failed\ncommand: {command}\n{diagnostics}")]
    CompilerFailure {
        command: String,
        diagnostics: String,
    },
}
