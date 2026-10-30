//! Solver-agnostic analysis problem model and validation contracts.

pub mod problem {
    pub mod authored_constraints;
    pub mod authored_excitations;
    pub mod authored_expressions;
    pub mod authored_initial_conditions;
    pub mod authored_inputs;
    pub mod authored_model_physics;
    pub mod authored_parameterized_signals;
    pub mod authored_properties;
    pub mod acoustic_impedance;
    pub mod rheology;
    pub mod boundary;
    pub mod connector;
    pub mod constraint;
    pub mod coordinate_frame;
    pub mod domains;
    pub mod excitation;
    pub mod initial_condition;
    pub mod interfaces;
    pub mod material_assignment;
    pub mod materials;
    pub mod model;
    pub mod model_authoring;
    pub mod model_realization;
    pub mod prescribed_value;
    pub mod quantity;
    pub mod random_vibration;
    pub mod reference_point;
    pub mod response_spectrum;
    pub mod signal;
    pub mod structure;
    pub mod target;
}
mod execution_policy;
pub mod field;
pub mod study;
pub mod validate;
mod validate_accuracy;
mod validate_authored;
mod validate_domains;
mod validate_excitations;
mod validate_materials;
mod validate_outputs;
mod validate_sequence;
mod validate_signals;

pub use execution_policy::{
    AcceleratorAdmissionPolicy, AcceleratorPreference, AuthoredExecutionPolicy,
    AuthoredExecutionPolicyError, AuthoredExecutionPolicyRef, BackendPreference,
    CheckpointPreference, CooperativePolicy, ExecutionResourceLimits, FanoutPolicy,
    InclusiveCountRange, OperatorPartitioner, ParallelismLimit, PartitionBalance,
    PlacementPreference, ResiliencePolicy, MAX_CPU_THREADS_PER_WORKER,
};
pub use study::{
    attach_authored_outputs, authored_output_root, authored_output_types,
    geometry_parameter_input_type, lower_authored_observations, retain_authored_outputs,
    validate_authored_outputs, AuthoredCaseOutputRef, AuthoredExtremaTiePolicy,
    AuthoredFieldDomain, AuthoredFieldProjection, AuthoredGeometryArgument,
    AuthoredGeometryArgumentValue, AuthoredGeometryInstance, AuthoredMeshRef,
    AuthoredModelArgument, AuthoredModelArgumentValue, AuthoredModelInstance,
    AuthoredObservationError, AuthoredOutput, AuthoredOutputError, AuthoredOutputExpression,
    AuthoredOutputPolicy, AuthoredOutputType, AuthoredOutputValue, AuthoredPrincipalSelection,
    AuthoredPortReduction, AuthoredSeriesDomain, AuthoredSeriesReduction, AuthoredSpatialReduction, AuthoredStudy,
    AuthoredStudyError, AuthoredStudyInputBinding, AuthoredStudyInputRef,
    AuthoredStudyRealizationError, AuthoredStudyRef, AuthoredSweep, AuthoredSweepAxis,
    AuthoredSweepAxisValue, AuthoredSweepBinding, AuthoredSweepCase, AuthoredSweepDefinition,
    AuthoredSweepError, AuthoredSweepOutput, AuthoredSweepOutputExpression,
    AuthoredVectorComponent, CaseOutputProjection, LoweredAuthoredObservation,
    RealizedGeometryInstance, RealizedStudy, SweepAxisCombination, SweepExtremum,
};
pub use validate::physics_family_can_consume_state;
pub use validate_authored::validate_coordinate_frame;
pub use validate_outputs::spec::quantity_supported as procedure_produces_observation_quantity;
pub use validate_outputs::{
    observation_sampling_matches_domain, validate_authored_observation_request,
    validate_observation_sampling,
};
pub use validate_signals::validate_physical_signals;

pub use field::{AnalysisField, AnalysisFieldValues, DeviceFieldRef};
pub use problem::acoustic_impedance::AcousticImpedanceLaw;
pub use problem::authored_constraints::{
    authored_motion_acceleration_is_compatible, materialized_motion_acceleration_is_compatible,
    AuthoredConstraint, AuthoredConstraintError, AuthoredConstraintKind, AuthoredConstraintTarget,
    AuthoredPairedConstraint, AuthoredPairingPolicy, AuthoredPeriodicConstraint,
    AuthoredPeriodicTransform, AuthoredPhaseShiftedPeriodicConstraint,
    AuthoredPrescribedStructuralMotion, AuthoredReferencePointCoupling, AuthoredRotationDofValue,
    AuthoredStructuralConstraintEquation, AuthoredStructuralConstraintEquationTerm,
    AuthoredStructuralDofs, AuthoredTargetCardinality, AuthoredThermalJump,
    AuthoredTranslationDofValue,
};
pub use problem::authored_excitations::{
    validate_authored_excitations, AuthoredExcitation, AuthoredExcitationCase,
    AuthoredExcitationKind, AuthoredExcitationTarget,
};
pub use problem::authored_expressions::{
    AuthoredExpressionError, AuthoredNumericExpression, AuthoredNumericExpressionKind,
    AuthoredNumericOperator,
};
pub use problem::authored_initial_conditions::{
    AuthoredInitialCondition, AuthoredInitialConditionError, AuthoredInitialConditionKind,
    AuthoredInitialConditionTarget,
};
pub use problem::authored_inputs::{
    AuthoredInputDefinition, AuthoredInputError, AuthoredInputType, AuthoredInputValue,
};
pub use problem::authored_model_physics::{
    AuthoredBoundary, AuthoredBoundaryCoefficient, AuthoredBoundaryKind, AuthoredConnector,
    AuthoredConnectorKind, AuthoredInterface, AuthoredInterfaceKind, AuthoredModelPhysicsError,
};
pub use problem::authored_parameterized_signals::{
    AuthoredParameterizedSignal, AuthoredParameterizedSignalError, AuthoredSignalExpression,
};
pub use problem::authored_properties::{
    AuthoredLayupDefinition, AuthoredLayupSource, AuthoredOrientationDefinition,
    AuthoredOrientationSource, AuthoredPly, AuthoredPropertyAssignment,
    AuthoredPropertyDefinitionRef, AuthoredPropertyError, AuthoredSectionDefinition,
    AuthoredSectionSource, AxisOrientation, CartesianOrientation, CircularBeamSection,
    GeneralBeamSection, HomogeneousShellSection, RectangularBeamSection, TubularBeamSection,
};
pub use problem::boundary::{AnalysisBoundary, BoundaryOperator};
pub use problem::connector::{AnalysisConnector, ConnectorKind};
pub use problem::constraint::{
    AccelerationSignalId, AcousticConstraint, AcousticConstraintKind, AnalysisConstraint,
    ElectricalConstraint, ElectricalConstraintKind, FluidConstraint, FluidConstraintKind,
    MagneticConstraint, MagneticConstraintKind, PairedConstraint, PairingPolicy,
    PeriodicConstraint, PeriodicTransform, PhaseShiftedPeriodicConstraint,
    PrescribedStructuralDofs, PrescribedStructuralMotion, ReferencePointCoupling, RotationDofValue,
    StructuralConstraint, StructuralConstraintEquation, StructuralConstraintEquationTerm,
    StructuralConstraintKind, StructuralKinematicHistory, StructuralMotionReconstruction,
    StructuralRotationDof, StructuralTranslationDof, ThermalConstraint, ThermalConstraintKind,
    TranslationDofMotion, TranslationDofValue,
};
pub use problem::coordinate_frame::{AxialFrame, CartesianFrame, CoordinateFrame, SphericalFrame};
pub use problem::domains::{
    CfdDomain, ElectroThermalDomain, ElectromagneticDomain, MagnetostaticFormulation, PorousZone,
    StaticElectricalEquation, ThermalDomain, ThermoMechanicalDomain,
};
pub use problem::excitation::{
    AnalysisExcitation, Direction, ExcitationAction, ExcitationCase, ExcitationCaseActivation,
    ExcitationCaseId, ExcitationDistribution, ExcitationId, ExcitationKinematics, ExcitationTarget,
    ProjectionRequirement,
};
pub use problem::initial_condition::{
    InitialCondition, InitialConditionTarget, InitialConditionValue, InitialFieldArtifactRef,
    InitialValue,
};
pub use problem::interfaces::{
    AnalysisInterface, AnalysisInterfaceKind, ConjugateHeatTransferInterfaceModel, ContactFriction,
    ContactInterfaceModel, FluidStructureInterfaceModel, FluidStructureKinematics,
    ThermalInterfaceTransfer,
};
pub use problem::material_assignment::{
    EvidenceConfidence, MaterialAssignment, PropertyRegionScope, PropertyRegionSelection,
    PropertyRegionTarget,
};
pub use problem::materials::{
    ConductivityFrequencyPoint, FluidMaterialError, ImportedCadEnvironmentCondition,
    ImportedCadMaterialAssociation, ImportedCadMaterialAssociationTarget,
    ImportedCadMaterialProperty, ImportedCadMaterialPropertyKind, ImportedCadMaterialProvenance,
    IncompressibleNewtonianMaterial, MaterialAcousticModel, MaterialCapability,
    MaterialElasticityModel, MaterialElectricalModel, MaterialFluidModel, MaterialMechanicalModel,
    MaterialModel, MaterialOrthotropicLaminaModel, MaterialPlasticModel, MaterialProvenance,
    MaterialThermalModel, MechanicalDampingModel, MissingMaterialCapability, PorousMediumLaw,
    SolverMaterialModel,
};
pub use problem::model::{
    AnalysisModel, AnalysisModelId, SolverAnalysisModel, SolverAnalysisSequence,
    SolverMaterialAssignments, SolverMaterials, SolverProcedureStep, SolverSteps,
};
pub use problem::model_authoring::{
    AuthoredMaterialAssignment, AuthoredMaterialRef, AuthoredMaterialSource, AuthoredModel,
    AuthoredModelError, AuthoredModelInclude, AuthoredModelInputRef, AuthoredModelRef,
    AuthoredPropertyTarget, AuthoredRegionSource,
};
pub use problem::model_realization::{
    realize_analysis_model, AnalysisModelRealizationError, AnalysisModelRealizationInput,
    RealizedAnalysisModel, RealizedLayupDefinition, RealizedModelProperties,
    RealizedOrientationDefinition, RealizedPropertyAssignment, RealizedSectionDefinition,
};
pub use problem::prescribed_value::{PrescribedQuantity, PrescribedValue};
pub use problem::quantity::{
    Acceleration, AngularAcceleration, AngularVelocity, BodyForceDensity, CanonicalQuantity,
    ComplexPhase, Damping, Displacement, DynamicViscosity, ElectricCharge, ElectricChargeDensity,
    ElectricCurrent, ElectricCurrentDensity, ElectricPotential, Force, FoundationModulus,
    Frequency, FrictionCoefficient, HeatFlux, HeatGenerationDensity, HeatTransferCoefficient,
    Length, MagneticVectorPotential, MassDensity, MassFlowRate, Moment, Position, Power,
    Pressure, RayleighMassCoefficient, RayleighStiffnessCoefficient, Rotation, SpecificAcousticImpedance,
    Stiffness, SurfaceEmissivity, Temperature, TemperatureDifference, ThermalContactResistance,
    Traction, Translation, TurbulenceIntensity, Velocity, VolumeFlowRate,
};
pub use problem::random_vibration::{
    ComplexSpectralDensity, CrossPowerSpectralDensity, RandomExcitationChannel,
    RandomExcitationChannelId, RandomExcitationId, RandomExcitationQuantity,
    RandomExcitationSpectrum, RandomSpectrumEvidence, RandomSpectrumExtrapolation,
    RandomSpectrumInterpolation, SpectralDensityConvention,
};
pub use problem::reference_point::ReferencePoint;
pub use problem::response_spectrum::{
    DirectionalCombination, ModalCombination, ModalTruncationCorrection, ModalTruncationPolicy,
    ResponseSpectrum, ResponseSpectrumId, RigidResponsePolicy, ShockResponseSpectrumMethod,
    SpectrumAbscissa, SpectrumConstruction, SpectrumDamping, SpectrumDirection, SpectrumEnvelope,
    SpectrumEvidence, SpectrumExtrapolation, SpectrumInterpolation, SpectrumResponse,
};
pub use problem::rheology::GeneralizedNewtonianLaw;
pub use problem::signal::{
    AuthoredSignalFieldLocation, AuthoredSignalTopologySelector, GeneratorArgument,
    GeneratorCallable, GeneratorDomain, GeneratorEffectPolicy, GeneratorInvocation,
    GeneratorProvenance, ImportedFunctionIdentity, NumericalDataSource, PhysicalQuantityKind,
    PhysicalSignal, RandomDistribution, Signal, SignalApproximationEvidence, SignalArtifactDomain,
    SignalArtifactRef, SignalDomainKind, SignalExtrapolation, SignalGenerator, SignalId,
    SignalInterpolation, SignalKnot, SignalKnotValue, SignalOrdinate, SignalPointContinuity,
    SignalPrecision, SignalProvenance, SignalRepresentation, SignalSampleApproach,
    SignalSamplingPolicy, SignalShape, SignalValue, StandardWaveform, StandardWaveformKind,
    TabulatedSignal, TabulatedSignalDomain,
};
pub use problem::structure::{
    BeamElementModel, BeamSectionModel, ConformalSolidAttachment, HomogeneousShellReferenceSurface,
    HomogeneousShellSectionModel, LaminatePlyModel, LaminateShellSectionModel, PlyOrderDirection,
    ShellElementModel, ShellReferenceSurface, ShellSectionModel, StructuralElement,
    StructuralElementKind, StructuralModel, StructuralNode,
};
pub use problem::target::{
    EntityDimension, ExpectedTopology, RegionTarget, ResolvedTopologyTarget, TargetCardinality,
};
pub use study::accuracy::{
    AccuracyCriterion, AccuracyLimitAction, AccuracyPlan, AccuracyPlanId, AccuracyResourceCeiling,
    AccuracyTarget, AccuracyTolerance, ErrorContributionKind, NumericalEquivalencePolicy,
    RequiredAccuracyEvidence, UnavailableEvidencePolicy,
};
pub use study::{
    analysis_identity_status, family_label, is_exposed_analysis_identity, procedure_label,
    render_analysis_identity_matrix, AnalysisIdentity, AnalysisIdentityStatus,
    ANALYSIS_IDENTITIES,
};
pub use study::output::{
    AccuracyObservationEvent, ComplexRepresentation, CompressionPolicy, DiagnosticKind,
    DiagnosticObservation, ExtremaTiePolicy, FieldLocation, MonitorCadence, MonitorObservation,
    MonitorStopCondition, ObservationComponent, ObservationCostEstimate, ObservationCostPolicy,
    ObservationEvent, ObservationId, ObservationPurpose, ObservationQuantity, ObservationRequest,
    ObservationResultQuantity, ObservationSpec, ObservationStoragePolicy, ObservationTarget,
    ObservationThreshold, ObservationValueShape, PartialAccessPolicy, RecoveryPolicy,
    ReductionPolicy, RetentionPolicy, SamplingPolicy, SeriesObservationEvent, SeriesReduction,
    SeriesWeighting, SignConvention, SolverObservationEvent, SpatialReduction,
    sound_pressure_level_db, SOUND_PRESSURE_LEVEL_REFERENCE_PA,
};
pub use study::sequence::{
    ActivationLifecycle, AnalysisSequence, ExcitationCaseSelection, IndependentDomain,
    InitialStateConflictPolicy, ProcedureSettings, ProcedureStep, ProcedureStepId,
    ReferenceStatePolicy, RestartPolicy, SequenceBranch, SequenceBranchId, SequenceDependency,
    SequenceDependencyReason, StateCompleteness, StateId, StateInput, StateKind, StateOutput,
    StatePersistence, StateSource, StateTransition, StateTransitionMethod, StateUse,
    StepObjectActivation, StepObjectRef, StructuralKinematics, StructuralNonlinearSettings,
    StructuralRandomResponseSettings, TopologyTransitionPolicy, TransitionId,
    ValidatedAnalysisSequence,
};
pub use study::{
    AnalysisProcedure, BoundaryTransferImplementation, ConstitutiveImplementation,
    CoupledPhysicsFamily, CouplingImplementation, ExactFormulationComponent,
    ExactFormulationRealization, FormulationAdmissionRejectionReason, FormulationAxis,
    PhysicsFamily, RecoveryImplementation, SolutionSpace, SourceImplementation,
    SpatialOperatorImplementation, StructuralFormulationRealization,
    StructuralOperatorImplementation, StructuralRecoveryImplementation, TemporalImplementation,
    TopologyImplementation,
};
pub use validate::{
    dependency_graph_resident_bytes_upper_bound, validate_analysis_sequence,
    validate_material_definitions, validate_model, validate_model_against_geometry,
    validate_model_sequence, validate_model_with_dependency_graph_budget, AnalysisValidationError,
};

#[cfg(test)]
mod tests;

pub use study::assessment::*;
