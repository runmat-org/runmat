//! Solver-independent identity for an authored analysis and its realized formulation.

pub mod accuracy;
pub mod assessment;
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
    attach_authored_outputs, lower_authored_observations, AuthoredObservationError,
    LoweredAuthoredObservation,
};
pub use authored_output::{
    authored_output_root, authored_output_types, retain_authored_outputs,
    validate_authored_outputs, AuthoredExtremaTiePolicy, AuthoredFieldDomain,
    AuthoredFieldProjection, AuthoredOutput, AuthoredOutputError, AuthoredOutputExpression,
    AuthoredOutputPolicy, AuthoredOutputType, AuthoredOutputValue, AuthoredPortReduction,
    AuthoredPrincipalSelection, AuthoredSeriesDomain, AuthoredSeriesReduction,
    AuthoredSpatialReduction, AuthoredVectorComponent,
};
pub use authoring::{
    geometry_parameter_input_type, AuthoredGeometryArgument, AuthoredGeometryArgumentValue,
    AuthoredGeometryInstance, AuthoredMeshRef, AuthoredModelArgument, AuthoredModelArgumentValue,
    AuthoredModelInstance, AuthoredStudy, AuthoredStudyError, AuthoredStudyInputRef,
};
pub use compatibility::{
    analysis_identity_status, family_label, is_exposed_analysis_identity, procedure_label,
    render_analysis_identity_matrix, AnalysisIdentity, AnalysisIdentityStatus, ANALYSIS_IDENTITIES,
};
pub use formulation::{
    BoundaryTransferImplementation, ConstitutiveImplementation, CouplingImplementation,
    ExactFormulationComponent, ExactFormulationRealization, FormulationAdmissionRejectionReason,
    FormulationAxis, RecoveryImplementation, SourceImplementation, SpatialOperatorImplementation,
    StructuralFormulationRealization, StructuralOperatorImplementation,
    StructuralRecoveryImplementation, TemporalImplementation, TopologyImplementation,
};
pub use identity::{AnalysisProcedure, CoupledPhysicsFamily, PhysicsFamily, SolutionSpace};
pub use realization::{
    AuthoredStudyInputBinding, AuthoredStudyRealizationError, RealizedGeometryInstance,
    RealizedStudy,
};
pub use sweep::{
    AuthoredCaseOutputRef, AuthoredStudyRef, AuthoredSweep, AuthoredSweepAxis,
    AuthoredSweepAxisValue, AuthoredSweepBinding, AuthoredSweepCase, AuthoredSweepDefinition,
    AuthoredSweepError, AuthoredSweepOutput, AuthoredSweepOutputExpression, CaseOutputProjection,
    SweepAxisCombination, SweepExtremum,
};
