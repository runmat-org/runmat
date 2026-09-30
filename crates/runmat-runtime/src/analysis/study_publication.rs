//! Typed publication facade; each physics owner authenticates its own retained completion.
use super::{
    complete_study_run, execution_operation, meshing, structural_study, study_result,
    study_result_manifest, AcousticStudyCompletion, AcousticStudyObjectSource,
    AnalysisAcousticRunOptions, AnalysisBucklingRunOptions, AnalysisExecutionStep,
    AnalysisModalRunOptions, AnalysisRandomResponseRunOptions, AnalysisResponseSpectrumRunOptions,
    AnalysisRunOptions, AnalysisSolverOptions, AnalysisStructuralFrequencyRunOptions,
    AnalysisStructuralFrequencySolver, AnalysisStructuralTimeStepPolicy,
    AnalysisStudyCompletionIdentity, AnalysisStudyRunData, AnalysisThermalRunOptions,
    AnalysisTransientRunOptions, PrecisionMode, PreconditionerMode, QualityPolicy,
    StructuralStudyNumericalResult, StructuralStudyPrestressedFrequencyReductionResult,
    StructuralStudyResult, ThermalStudyCompletion, ThermalStudyObjectSource,
    ANALYSIS_RUN_STUDY_OPERATION, ANALYSIS_RUN_STUDY_OP_VERSION,
};

#[path = "acoustic_study_publication.rs"]
mod acoustic;
#[path = "typed_study_publication.rs"]
mod structural;
#[path = "thermal_study_publication.rs"]
pub(super) mod thermal;

pub use acoustic::publish_acoustic_study_result;
pub use structural::publish_structural_study_result;
pub use thermal::publish_thermal_study_result;
