//! Placement-neutral compilation of realized studies onto exact solver topology.

mod compilation;
mod identity;
mod planning;
mod product;
mod thermal_planning;
mod thermal_product;

pub use compilation::{
    compile_analysis_study, AnalysisStudyCompilationError, AnalysisStudyCompilationInput,
    CompiledAnalysisStudy,
};
pub use identity::{
    structural_study_source_digest, StructuralStudyMaterialBinding,
    StructuralStudySourceIdentityError,
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
    plan_thermal_step, plan_thermal_study, ThermalStudyPlanningError, ThermalStudyPlanningPolicy,
};
pub use thermal_product::{
    build_thermal_study_product, ThermalStudyProduct, ThermalStudyProductError,
    ThermalStudyProductInput,
};
