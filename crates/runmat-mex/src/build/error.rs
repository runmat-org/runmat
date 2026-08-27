use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MexBuildError {
    #[error("native MEX compilation is unavailable for this target")]
    UnsupportedTarget,
    #[error("MEX target `{triple}` is not a supported native RunMat target")]
    UnsupportedTargetIdentity { triple: String },
    #[error("C compiler family `{compiler_family}` is unsupported for MEX target `{triple}`")]
    UnsupportedCompilerForTarget {
        compiler_family: &'static str,
        triple: String,
    },
    #[error("Fortran MEX compiler `{compiler}` does not use a supported GNU-compatible ABI")]
    UnsupportedFortranCompiler { compiler: PathBuf },
    #[error("MEX cross-compilation for target `{triple}` is not available; run this build on the target host")]
    CrossCompilationUnavailable { triple: String },
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
    #[error("failed to invoke MEX compiler {compiler}: {source}")]
    CompilerLaunch {
        compiler: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("MEX compilation failed\ncommand: {command}\n{diagnostics}")]
    CompilerFailure {
        command: String,
        diagnostics: String,
    },
    #[error("failed to read compiled MEX module {module}: {source}")]
    ReadCompiledModule {
        module: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to encode the deterministic MEX artifact manifest: {source}")]
    ArtifactEncoding {
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to publish MEX artifact manifest {path}: {source}")]
    WriteArtifactManifest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("MEX artifact manifest does not match its module or supported host contract")]
    InvalidArtifactManifest,
}
