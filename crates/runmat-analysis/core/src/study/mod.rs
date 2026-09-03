//! Solver-independent identity for an authored analysis and its realized formulation.

pub mod accuracy;
mod identity;
pub mod output;
pub mod sequence;

pub use identity::{
    AnalysisProcedure, AnalysisRunIdentity, CoupledPhysicsFamily, ExecutionFallbackPolicy,
    FormulationImplementation, FormulationMaturity, PhysicsFamily, SolutionSpace,
};
