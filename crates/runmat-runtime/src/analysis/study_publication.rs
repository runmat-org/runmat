//! Typed publication facade; each physics owner authenticates its own retained completion.
pub use super::contracts::AnalysisElectricalRunOptions;
#[path = "magnetic_run_options.rs"]
mod magnetic_run_options;
#[path = "maxwell_run_options.rs"]
mod maxwell_run_options;
use super::{
    complete_study_run, execution_operation, meshing, structural_study, study_result,
    study_result_manifest, AcousticStudyCompletion, AcousticStudyObjectSource,
    AnalysisAcousticRunOptions, AnalysisBucklingRunOptions, AnalysisExecutionStep,
    AnalysisModalRunOptions, AnalysisRandomResponseRunOptions, AnalysisResponseSpectrumRunOptions,
    AnalysisRunOptions, AnalysisSolverOptions, AnalysisStructuralFrequencyRunOptions,
    AnalysisStructuralFrequencySolver, AnalysisStructuralTimeStepPolicy,
    AnalysisStudyCompletionIdentity, AnalysisStudyRunData, AnalysisThermalRunOptions,
    AnalysisTransientRunOptions, ElectricalStudyCompletion, ElectricalStudyObjectSource,
    PrecisionMode, PreconditionerMode, QualityPolicy, StructuralStudyNumericalResult,
    StructuralStudyPrestressedFrequencyReductionResult, StructuralStudyResult,
    ThermalStudyCompletion, ThermalStudyObjectSource, ANALYSIS_RUN_STUDY_OPERATION,
    ANALYSIS_RUN_STUDY_OP_VERSION,
};
use super::{AnalysisCfdRunOptions, FluidStudyCompletion, FluidStudyObjectSource};
use super::{MagneticStudyCompletion, MagneticStudyObjectSource};
use super::{MaxwellStudyCompletion, MaxwellStudyObjectSource};
pub use magnetic_run_options::AnalysisMagneticRunOptions;
pub use maxwell_run_options::AnalysisMaxwellRunOptions;

#[path = "acoustic_study_publication.rs"]
mod acoustic;
#[path = "electrical_study_publication.rs"]
mod electrical;
pub use electrical::publish_electrical_study_result;
#[path = "magnetic_study_publication.rs"]
mod magnetic;
pub use magnetic::publish_magnetic_study_result;
#[path = "maxwell_study_publication.rs"]
mod maxwell;
pub use maxwell::publish_maxwell_study_result;
#[path = "fluid_study_publication.rs"]
mod fluid;
pub use fluid::publish_fluid_study_result;
#[path = "typed_study_publication.rs"]
mod structural;
#[path = "thermal_study_publication.rs"]
pub(super) mod thermal;

pub use acoustic::publish_acoustic_study_result;
pub(in crate::analysis) use acoustic::{
    AcousticFrequencyPersistence, CompletedAcousticFrequencyStep,
};
pub use structural::publish_structural_study_result;
pub use thermal::publish_thermal_study_result;
