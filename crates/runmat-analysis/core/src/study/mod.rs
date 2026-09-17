//! Solver-independent identity for an authored analysis and its realized formulation.

pub mod accuracy;
pub mod authoring;
mod compatibility;
mod formulation;
mod identity;
pub mod output;
pub mod sequence;

pub use authoring::{
    AuthoredMeshRef, AuthoredModelArgument, AuthoredModelArgumentValue, AuthoredModelInstance,
    AuthoredStudy, AuthoredStudyError, AuthoredStudyInputRef,
};
pub use compatibility::is_exposed_analysis_identity;
pub use formulation::{
    BoundaryTransferImplementation, ConstitutiveImplementation, CouplingImplementation,
    ExactFormulationComponent, ExactFormulationRealization, FormulationAdmissionRejectionReason,
    FormulationAxis, FormulationRequirement, RecoveryImplementation, SourceImplementation,
    SpatialOperatorImplementation, StructuralFormulationRealization,
    StructuralOperatorImplementation, StructuralRecoveryImplementation, TemporalImplementation,
    TopologyImplementation,
};
pub use identity::{
    AnalysisProcedure, CoupledPhysicsFamily, ExecutionFallbackPolicy, FormulationMaturity,
    PhysicsFamily, SolutionSpace,
};
