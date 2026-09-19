//! Placement-neutral compilation of realized studies onto exact solver topology.

mod compilation;

pub use compilation::{
    compile_analysis_study, AnalysisStudyCompilationError, AnalysisStudyCompilationInput,
    CompiledAnalysisStudy,
};
