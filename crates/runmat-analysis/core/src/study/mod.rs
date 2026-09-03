//! Solver-independent identity for an authored analysis and its realized formulation.

mod identity;

pub use identity::{
    AnalysisProcedure, AnalysisRunIdentity, CoupledPhysicsFamily, ExecutionFallbackPolicy,
    FormulationImplementation, FormulationMaturity, PhysicsFamily, SolutionSpace,
};
