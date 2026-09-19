//! Placement-neutral compilation of realized studies onto exact solver topology.

mod compilation;
mod planning;

pub use compilation::{
    compile_analysis_study, AnalysisStudyCompilationError, AnalysisStudyCompilationInput,
    CompiledAnalysisStudy,
};
pub use planning::{
    plan_structural_study, StructuralStudyPlanningError, StructuralStudyPlanningPolicy,
};
