//! Typed publication facade; each physics owner authenticates its own retained completion.
pub(in crate::analysis) mod catalog;
pub(in crate::analysis) mod preparation;
pub use super::contracts::AnalysisAcousticResponseRunOptions;
pub use super::contracts::AnalysisElectricalRunOptions;
use super::{AcousticResponseStudyCompletion, AcousticResponseStudyObjectSource};
#[path = "magnetic_run_options.rs"]
mod magnetic_run_options;
#[path = "maxwell_run_options.rs"]
mod maxwell_run_options;
use super::{
    execution_operation, meshing, structural_study, study_result, study_result_manifest,
    AcousticStudyCompletion, AcousticStudyObjectSource, AnalysisAcousticRunOptions,
    AnalysisBucklingRunOptions, AnalysisExecutionStep, AnalysisModalRunOptions,
    AnalysisRandomResponseRunOptions, AnalysisResponseSpectrumRunOptions, AnalysisRunOptions,
    AnalysisSolverOptions, AnalysisStructuralFrequencyRunOptions,
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
#[path = "acoustic_response_study_publication.rs"]
mod acoustic_response;
pub use acoustic_response::{
    prepare_acoustic_response_study_publication, publish_acoustic_response_study_result,
};
#[path = "electrical_study_publication.rs"]
mod electrical;
pub use electrical::{prepare_electrical_study_publication, publish_electrical_study_result};
#[path = "line_parameters_study_publication.rs"]
mod line_parameters;
pub use line_parameters::{
    prepare_line_parameters_study_publication, LineParametersStudyCompletion,
};
#[path = "magnetic_study_publication.rs"]
mod magnetic;
pub use magnetic::{prepare_magnetic_study_publication, publish_magnetic_study_result};
#[path = "maxwell_study_publication.rs"]
mod maxwell;
pub use maxwell::{prepare_maxwell_study_publication, publish_maxwell_study_result};
#[path = "fluid_study_publication.rs"]
mod fluid;
pub use fluid::{prepare_fluid_study_publication, publish_fluid_study_result};
#[path = "nonlinear_study_publication.rs"]
mod nonlinear;
pub use nonlinear::{
    prepare_nonlinear_solid_study_publication, publish_nonlinear_solid_study_result,
};
#[path = "typed_study_publication.rs"]
mod structural;
#[path = "thermal_study_publication.rs"]
pub(super) mod thermal;

pub use acoustic::{prepare_acoustic_study_publication, publish_acoustic_study_result};
pub(in crate::analysis) use acoustic::{
    AcousticFrequencyCompletionResults, AcousticFrequencyPersistence,
    CompletedAcousticFrequencyStep,
};
pub use structural::{prepare_structural_study_publication, publish_structural_study_result};
pub use thermal::{prepare_thermal_study_publication, publish_thermal_study_result};
