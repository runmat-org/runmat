pub type AnalysisBridgeResult<T> = Result<T, AnalysisBridgeError>;

/// Components of a rejected coordinator reservation, in bytes. The original
/// resource cause remains available without interpreting the diagnostic text.
#[derive(Debug, thiserror::Error)]
#[error("{cause}; decoded={decoded}, encoded={encoded}, coordinator={coordinator}, transport={transport}, caller={caller}, import_workspace={import_workspace}")]
pub struct CoordinatorRetentionLimit {
    #[source]
    pub cause: runmat_analysis_fea::planning::resource::ResourcePlanningError,
    pub decoded: u64,
    pub encoded: u64,
    pub coordinator: u64,
    pub transport: u64,
    pub caller: u64,
    pub import_workspace: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum AnalysisBridgeError {
    #[error(transparent)]
    RetentionCapacity(#[from] CoordinatorRetentionLimit),
    #[error("invalid analysis execution bridge contract: {0}")]
    Invalid(String),
    #[error("analysis execution bridge identity mismatch: {0}")]
    Identity(&'static str),
    #[error("analysis execution bridge artifact failure: {0}")]
    Artifact(#[from] runmat_execution_artifact::ArtifactError),
    #[error("analysis execution contract failure: {0}")]
    Analysis(#[from] runmat_analysis_fea::execution::AnalysisExecutionError),
    #[error(transparent)]
    FeaRun(#[from] runmat_analysis_fea::contracts::FeaRunError),
    #[error(transparent)]
    Resource(#[from] runmat_analysis_fea::planning::resource::ResourcePlanningError),
    #[error("analysis execution runner failure: {0}")]
    Runner(#[from] runmat_execution_runner::RunnerError),
    #[error("analysis execution platform contract failure: {0}")]
    Platform(#[from] runmat_execution::ContractError),
    #[error("analysis execution codec failure: {0}")]
    Codec(#[from] runmat_canonical_codec::CanonicalCodecError),
}
