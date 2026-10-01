//! Placement-neutral compilation of realized studies onto exact solver topology.

mod acoustic_planning;
mod acoustic_product;
mod compilation;
mod electrical_planning;
mod electrical_product;
mod identity;
mod magnetic_planning;
mod magnetic_product;
mod maxwell_planning;
mod maxwell_product;
mod planning;
mod product;
mod thermal_planning;
mod thermal_product;

pub use acoustic_planning::{
    plan_acoustic_study, AcousticStudyPlan, AcousticStudyPlanningError,
    AcousticStudyPlanningPolicy, PlannedAcousticFrequency,
};
pub use acoustic_product::{
    build_acoustic_study_product, plan_acoustic_study_product, AcousticStudyProduct,
    AcousticStudyProductError, AcousticStudyProductInput,
};
pub use compilation::{
    compile_analysis_study, AnalysisStudyCompilationError, AnalysisStudyCompilationInput,
    CompiledAnalysisStudy,
};
pub use electrical_planning::{
    plan_electrical_study, ElectricalStudyPlanningError, ElectricalStudyPlanningPolicy,
};
pub use electrical_product::{
    build_electrical_study_product, ElectricalStudyProduct, ElectricalStudyProductError,
    ElectricalStudyProductInput,
};
pub use identity::{
    structural_study_source_digest, StructuralStudyMaterialBinding,
    StructuralStudySourceIdentityError,
};
pub use magnetic_planning::{
    plan_magnetic_study, MagneticStudyPlanningError, MagneticStudyPlanningPolicy,
};
pub use magnetic_product::{
    build_magnetic_study_product, plan_magnetic_study_product, MagneticStudyProduct,
    MagneticStudyProductError, MagneticStudyProductInput,
};
pub use maxwell_planning::{
    plan_maxwell_study, MaxwellStudyPlanningError, MaxwellStudyPlanningPolicy,
};
pub use maxwell_product::{
    build_maxwell_study_product, plan_maxwell_study_product, MaxwellStudyProduct,
    MaxwellStudyProductError, MaxwellStudyProductInput,
};
pub use planning::{
    plan_structural_study, StructuralNumericalPlanningPolicy, StructuralStudyPlanningError,
    StructuralStudyPlanningPolicy,
};
pub use product::{
    build_structural_study_product, StructuralStudyProduct, StructuralStudyProductError,
    StructuralStudyProductInput,
};
pub use thermal_planning::{
    plan_thermal_step, plan_thermal_study, thermal_step_initial_selection_supported,
    ThermalStudyPlanningError, ThermalStudyPlanningPolicy,
};
pub use thermal_product::{
    build_thermal_study_product, plan_thermal_study_product, ThermalStudyProduct,
    ThermalStudyProductError, ThermalStudyProductInput,
};
