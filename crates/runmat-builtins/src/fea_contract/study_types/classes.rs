//! Class identities produced or consumed by the typed study-contract
//! constructors. Spellings are declared once here; runtime bindings use
//! `display_name()` and catalog inference compares typed identities.

use runmat_types::StaticClassIdentity;

// Immutable values constructed by this family.
pub const FEA_OBSERVATION_TARGET_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationTarget");
pub const FEA_SAMPLING_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SamplingPolicy");
pub const FEA_OBSERVATION_EVENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationEvent");
pub const FEA_REDUCTION_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ReductionPolicy");
pub const FEA_SIGNAL_PRECISION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SignalPrecision");
pub const FEA_COMPRESSION_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.CompressionPolicy");
pub const FEA_RETENTION_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.RetentionPolicy");
pub const FEA_OBSERVATION_STORAGE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationStorage");
pub const FEA_OBSERVATION_COST_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationCost");
pub const FEA_MONITOR_CADENCE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MonitorCadence");
pub const FEA_MONITOR_STOP_CONDITION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.MonitorStopCondition");
pub const FEA_OBSERVATION_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Observation");
pub const FEA_INDEPENDENT_DOMAIN_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.IndependentDomain");
pub const FEA_RESTART_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.RestartPolicy");
pub const FEA_STATE_INPUT_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.StateInput");
pub const FEA_STATE_OUTPUT_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.StateOutput");
pub const FEA_SEQUENCE_DEPENDENCY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SequenceDependency");
pub const FEA_STATE_TRANSITION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StateTransition");
pub const FEA_SEQUENCE_BRANCH_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SequenceBranch");
pub const FEA_ANALYSIS_SEQUENCE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.AnalysisSequence");
pub const FEA_PROCEDURE_STEP_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ProcedureStep");

// Typed inputs consumed from sibling families.
pub const GEOMETRY_REGION_CLASS: StaticClassIdentity = StaticClassIdentity::new("geometry.Region");
pub const FEA_POSITION_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.Position");
pub const FEA_OBSERVATION_QUANTITY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationQuantity");
pub const FEA_FIELD_LOCATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.FieldLocation");
pub const FEA_RECOVERY_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.RecoveryPolicy");
pub const FEA_OBSERVATION_COMPONENT_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationComponent");
pub const FEA_SIGN_CONVENTION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SignConvention");
pub const FEA_COMPLEX_REPRESENTATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ComplexRepresentation");
pub const FEA_SPATIAL_REDUCTION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SpatialReduction");
pub const FEA_SERIES_REDUCTION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SeriesReduction");
pub const FEA_EXTREMA_TIE_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ExtremaTiePolicy");
pub const FEA_PARTIAL_ACCESS_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.PartialAccessPolicy");
pub const FEA_OBSERVATION_PURPOSE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationPurpose");
pub const FEA_DIAGNOSTIC_KIND_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.DiagnosticKind");
pub const FEA_STATE_KIND_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.StateKind");
pub const FEA_STATE_COMPLETENESS_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StateCompleteness");
pub const FEA_STATE_PERSISTENCE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StatePersistence");
pub const FEA_STATE_USE_CLASS: StaticClassIdentity = StaticClassIdentity::new("fea.StateUse");
pub const FEA_INITIAL_STATE_CONFLICT_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.InitialStateConflictPolicy");
pub const FEA_SEQUENCE_DEPENDENCY_REASON_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.SequenceDependencyReason");
pub const FEA_STATE_TRANSITION_METHOD_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StateTransitionMethod");
pub const FEA_TOPOLOGY_TRANSITION_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.TopologyTransitionPolicy");
pub const FEA_REFERENCE_STATE_POLICY_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ReferenceStatePolicy");
pub const FEA_RESPONSE_SPECTRUM_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.ResponseSpectrum");
pub const FEA_STEP_OBJECT_ACTIVATION_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.StepObjectActivation");
pub const FEA_ACCURACY_PLAN_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("fea.AccuracyPlan");

/// Family-specific procedure-step classes accepted by `fea.ProcedureStep.configure`
/// and `fea.AnalysisSequence.new`; the configured step keeps its family class.
pub const FEA_PROCEDURE_STEP_CLASSES: [StaticClassIdentity; 9] = [
    StaticClassIdentity::new("fea.StructuralStep"),
    StaticClassIdentity::new("fea.ThermalStep"),
    StaticClassIdentity::new("fea.ElectromagneticStep"),
    StaticClassIdentity::new("fea.AcousticStep"),
    StaticClassIdentity::new("fea.FluidStep"),
    StaticClassIdentity::new("fea.ThermoMechanicalStep"),
    StaticClassIdentity::new("fea.ElectroThermalStep"),
    StaticClassIdentity::new("fea.ConjugateHeatTransferStep"),
    StaticClassIdentity::new("fea.FluidStructureStep"),
];
