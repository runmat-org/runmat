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
    #[error("analysis host execution failed: {0}")]
    HostExecution(String),
    #[error(transparent)]
    RetentionCapacity(#[from] CoordinatorRetentionLimit),
    #[error("invalid analysis execution bridge contract: {0}")]
    Invalid(String),
    #[error(transparent)]
    Observation(#[from] runmat_analysis_fea::results::observation::ObservationError),
    #[error("analysis execution bridge identity mismatch: {0}")]
    Identity(&'static str),
    #[error("analysis execution bridge artifact failure: {0}")]
    Artifact(#[from] runmat_execution_artifact::ArtifactError),
    #[error("analysis execution contract failure: {0}")]
    Analysis(#[from] runmat_analysis_fea::execution::AnalysisExecutionError),
    #[error(transparent)]
    FeaRun(#[from] runmat_analysis_fea::contracts::FeaRunError),
    #[error(transparent)]
    LinearAssembly(#[from] runmat_analysis_fea::assembly::LinearAssemblyError),
    #[error(transparent)]
    SolidAssembly(#[from] runmat_analysis_fea::assembly::solver_solid::SolverSolidAssemblyError),
    #[error(transparent)]
    Resource(#[from] runmat_analysis_fea::planning::resource::ResourcePlanningError),
    #[error("analysis execution runner failure: {0}")]
    Runner(#[from] runmat_execution_runner::RunnerError),
    #[error("analysis execution platform contract failure: {0}")]
    Platform(#[from] runmat_execution::ContractError),
    #[error("analysis execution codec failure: {0}")]
    Codec(#[from] runmat_canonical_codec::CanonicalCodecError),
}

impl AnalysisBridgeError {
    pub fn accounting_cause(&self) -> Option<&runmat_simulation_usage::LedgerError> {
        use runmat_analysis_fea::{
            assembly::{
                solver_solid::SolverSolidAssemblyError, LinearAssemblyError,
                StructuralAssemblyError,
            },
            constraints::ConstraintPlanError,
            FeaRunError,
        };
        match self {
            Self::Observation(
                runmat_analysis_fea::results::observation::ObservationError::Accounting(cause),
            ) => Some(cause),
            Self::FeaRun(FeaRunError::Usage(cause))
            | Self::Analysis(runmat_analysis_fea::execution::AnalysisExecutionError::Usage(
                cause,
            ))
            | Self::SolidAssembly(SolverSolidAssemblyError::Accounting(cause))
            | Self::LinearAssembly(LinearAssemblyError::SolidOperator(
                SolverSolidAssemblyError::Accounting(cause),
            ))
            | Self::LinearAssembly(LinearAssemblyError::Structural(
                StructuralAssemblyError::Accounting(cause),
            ))
            | Self::LinearAssembly(LinearAssemblyError::ConstraintPlan(
                ConstraintPlanError::Accounting(cause),
            )) => Some(cause),
            _ => None,
        }
    }
}

impl From<runmat_analysis_fea::assembly::StructuralAssemblyError> for AnalysisBridgeError {
    fn from(error: runmat_analysis_fea::assembly::StructuralAssemblyError) -> Self {
        Self::LinearAssembly(runmat_analysis_fea::assembly::LinearAssemblyError::Structural(error))
    }
}
