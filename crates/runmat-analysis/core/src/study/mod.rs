//! Solver-independent identity for an authored analysis and its realized formulation.

pub mod accuracy;
mod authored_observation;
mod authored_output;
pub mod authoring;
mod compatibility;
mod formulation;
mod identity;
pub mod output;
mod realization;
pub mod sequence;
mod sweep;

pub use authored_observation::{
    lower_authored_observations, AuthoredObservationError, LoweredAuthoredObservation,
};
pub use authored_output::{
    authored_output_types, validate_authored_outputs, AuthoredExtremaTiePolicy,
    AuthoredFieldDomain, AuthoredFieldProjection, AuthoredOutput, AuthoredOutputError,
    AuthoredOutputExpression, AuthoredOutputType, AuthoredOutputValue, AuthoredSpatialReduction,
    AuthoredVectorComponent,
};
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
pub use realization::{AuthoredStudyInputBinding, AuthoredStudyRealizationError, RealizedStudy};
pub use sweep::{
    AuthoredCaseOutputRef, AuthoredStudyRef, AuthoredSweep, AuthoredSweepAxis,
    AuthoredSweepAxisValue, AuthoredSweepBinding, AuthoredSweepCase, AuthoredSweepDefinition,
    AuthoredSweepError, AuthoredSweepOutput, AuthoredSweepOutputExpression, CaseOutputProjection,
    SweepAxisCombination, SweepExtremum,
};
