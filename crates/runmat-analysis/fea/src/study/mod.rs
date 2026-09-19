//! Placement-neutral compilation of realized studies onto exact solver topology.

mod compilation;
mod planning;
mod product;

pub use compilation::{
    compile_analysis_study, AnalysisStudyCompilationError, AnalysisStudyCompilationInput,
    CompiledAnalysisStudy,
};
pub use planning::{
    plan_structural_study, StructuralNumericalPlanningPolicy, StructuralStudyPlanningError,
    StructuralStudyPlanningPolicy,
};
pub use product::{
    build_structural_study_product, StructuralStudyProduct, StructuralStudyProductError,
    StructuralStudyProductInput,
};
