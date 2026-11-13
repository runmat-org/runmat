//! Class identities of typed FEA policy values and the typed inputs they consume.
//! Spellings match the runtime class registry; each is declared here once.

use runmat_types::StaticClassIdentity;

pub const ACCURACY_TARGET: StaticClassIdentity = StaticClassIdentity::new("fea.AccuracyTarget");
pub const ACCURACY_TOLERANCE: StaticClassIdentity =
    StaticClassIdentity::new("fea.AccuracyTolerance");
pub const NUMERICAL_EQUIVALENCE_POLICY: StaticClassIdentity =
    StaticClassIdentity::new("fea.NumericalEquivalencePolicy");
pub const ACCURACY_RESOURCE_CEILING: StaticClassIdentity =
    StaticClassIdentity::new("fea.AccuracyResourceCeiling");
pub const ACCURACY_CRITERION: StaticClassIdentity =
    StaticClassIdentity::new("fea.AccuracyCriterion");
pub const ACCURACY_PLAN: StaticClassIdentity = StaticClassIdentity::new("fea.AccuracyPlan");
pub const ERROR_CONTRIBUTION: StaticClassIdentity =
    StaticClassIdentity::new("fea.ErrorContribution");
pub const REQUIRED_ACCURACY_EVIDENCE: StaticClassIdentity =
    StaticClassIdentity::new("fea.RequiredAccuracyEvidence");
pub const UNAVAILABLE_EVIDENCE_POLICY: StaticClassIdentity =
    StaticClassIdentity::new("fea.UnavailableEvidencePolicy");
pub const ACCURACY_LIMIT_ACTION: StaticClassIdentity =
    StaticClassIdentity::new("fea.AccuracyLimitAction");
pub const COMPUTE_BACKEND: StaticClassIdentity = StaticClassIdentity::new("fea.ComputeBackend");
pub const SOLVER_OPTIONS: StaticClassIdentity = StaticClassIdentity::new("fea.SolverOptions");
pub const EXECUTION_STEP: StaticClassIdentity = StaticClassIdentity::new("fea.ExecutionStep");
pub const EXECUTION_PLAN: StaticClassIdentity = StaticClassIdentity::new("fea.ExecutionPlan");
pub const ANALYSIS_SEQUENCE: StaticClassIdentity = StaticClassIdentity::new("fea.AnalysisSequence");
pub const STRUCTURAL_STEP: StaticClassIdentity = StaticClassIdentity::new("fea.StructuralStep");
pub const THERMAL_STEP: StaticClassIdentity = StaticClassIdentity::new("fea.ThermalStep");
pub const ELECTROMAGNETIC_STEP: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectromagneticStep");
pub const ACOUSTIC_STEP: StaticClassIdentity = StaticClassIdentity::new("fea.AcousticStep");
pub const FLUID_STEP: StaticClassIdentity = StaticClassIdentity::new("fea.FluidStep");
pub const THERMO_MECHANICAL_STEP: StaticClassIdentity =
    StaticClassIdentity::new("fea.ThermoMechanicalStep");
pub const ELECTRO_THERMAL_STEP: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectroThermalStep");
pub const CONJUGATE_HEAT_TRANSFER_STEP: StaticClassIdentity =
    StaticClassIdentity::new("fea.ConjugateHeatTransferStep");
pub const FLUID_STRUCTURE_STEP: StaticClassIdentity =
    StaticClassIdentity::new("fea.FluidStructureStep");
pub const SIGNAL_DOMAIN: StaticClassIdentity = StaticClassIdentity::new("fea.SignalDomain");
pub const SIGNAL_SAMPLING: StaticClassIdentity = StaticClassIdentity::new("fea.SignalSampling");
pub const SIGNAL_KNOT: StaticClassIdentity = StaticClassIdentity::new("fea.SignalKnot");
pub const SIGNAL_POINT_CONTINUITY: StaticClassIdentity =
    StaticClassIdentity::new("fea.SignalPointContinuity");
pub const SIGNAL_INTERPOLATION: StaticClassIdentity =
    StaticClassIdentity::new("fea.SignalInterpolation");
pub const SIGNAL_EXTRAPOLATION: StaticClassIdentity =
    StaticClassIdentity::new("fea.SignalExtrapolation");
pub const DISPLACEMENT_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.DisplacementSignal");
pub const ROTATION_SIGNAL: StaticClassIdentity = StaticClassIdentity::new("fea.RotationSignal");
pub const VELOCITY_SIGNAL: StaticClassIdentity = StaticClassIdentity::new("fea.VelocitySignal");
pub const ACCELERATION_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.AccelerationSignal");
pub const ANGULAR_VELOCITY_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.AngularVelocitySignal");
pub const ANGULAR_ACCELERATION_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.AngularAccelerationSignal");
pub const FORCE_SIGNAL: StaticClassIdentity = StaticClassIdentity::new("fea.ForceSignal");
pub const MOMENT_SIGNAL: StaticClassIdentity = StaticClassIdentity::new("fea.MomentSignal");
pub const TRACTION_SIGNAL: StaticClassIdentity = StaticClassIdentity::new("fea.TractionSignal");
pub const PRESSURE_SIGNAL: StaticClassIdentity = StaticClassIdentity::new("fea.PressureSignal");
pub const BODY_FORCE_SIGNAL: StaticClassIdentity = StaticClassIdentity::new("fea.BodyForceSignal");
pub const TEMPERATURE_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.TemperatureSignal");
pub const TEMPERATURE_DIFFERENCE_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.TemperatureDifferenceSignal");
pub const HEAT_FLUX_SIGNAL: StaticClassIdentity = StaticClassIdentity::new("fea.HeatFluxSignal");
pub const HEAT_GENERATION_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.HeatGenerationSignal");
pub const ELECTRIC_POTENTIAL_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectricPotentialSignal");
pub const ELECTRIC_CURRENT_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectricCurrentSignal");
pub const ELECTRIC_CHARGE_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.ElectricChargeSignal");
pub const CURRENT_DENSITY_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.CurrentDensitySignal");
pub const CHARGE_DENSITY_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.ChargeDensitySignal");
pub const MAGNETIC_VECTOR_POTENTIAL_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.MagneticVectorPotentialSignal");
pub const MASS_FLOW_SIGNAL: StaticClassIdentity = StaticClassIdentity::new("fea.MassFlowSignal");
pub const VOLUME_FLOW_SIGNAL: StaticClassIdentity =
    StaticClassIdentity::new("fea.VolumeFlowSignal");
pub const RESPONSE_SPECTRUM: StaticClassIdentity = StaticClassIdentity::new("fea.ResponseSpectrum");
pub const SPECTRUM_AXIS: StaticClassIdentity = StaticClassIdentity::new("fea.SpectrumAxis");
pub const SPECTRUM_DAMPING: StaticClassIdentity = StaticClassIdentity::new("fea.SpectrumDamping");
pub const SPECTRUM_DIRECTION: StaticClassIdentity =
    StaticClassIdentity::new("fea.SpectrumDirection");
pub const SPECTRUM_RESPONSE: StaticClassIdentity = StaticClassIdentity::new("fea.SpectrumResponse");
pub const SPECTRUM_ENVELOPE: StaticClassIdentity = StaticClassIdentity::new("fea.SpectrumEnvelope");
pub const SPECTRUM_INTERPOLATION: StaticClassIdentity =
    StaticClassIdentity::new("fea.SpectrumInterpolation");
pub const SPECTRUM_EXTRAPOLATION: StaticClassIdentity =
    StaticClassIdentity::new("fea.SpectrumExtrapolation");
pub const MODAL_COMBINATION: StaticClassIdentity = StaticClassIdentity::new("fea.ModalCombination");
pub const DIRECTIONAL_COMBINATION: StaticClassIdentity =
    StaticClassIdentity::new("fea.DirectionalCombination");
pub const MODAL_TRUNCATION: StaticClassIdentity = StaticClassIdentity::new("fea.ModalTruncation");
pub const MODAL_TRUNCATION_CORRECTION: StaticClassIdentity =
    StaticClassIdentity::new("fea.ModalTruncationCorrection");
pub const RIGID_RESPONSE: StaticClassIdentity = StaticClassIdentity::new("fea.RigidResponse");
pub const SPECTRUM_CONSTRUCTION: StaticClassIdentity =
    StaticClassIdentity::new("fea.SpectrumConstruction");
pub const SPECTRUM_EVIDENCE: StaticClassIdentity = StaticClassIdentity::new("fea.SpectrumEvidence");
pub const COORDINATE_FRAME: StaticClassIdentity = StaticClassIdentity::new("fea.CoordinateFrame");
pub const OBSERVATION_QUANTITY: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationQuantity");
pub const FIELD_LOCATION: StaticClassIdentity = StaticClassIdentity::new("fea.FieldLocation");
pub const RECOVERY_POLICY: StaticClassIdentity = StaticClassIdentity::new("fea.RecoveryPolicy");
pub const OBSERVATION_COMPONENT: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationComponent");
pub const SIGN_CONVENTION: StaticClassIdentity = StaticClassIdentity::new("fea.SignConvention");
pub const COMPLEX_REPRESENTATION: StaticClassIdentity =
    StaticClassIdentity::new("fea.ComplexRepresentation");
pub const SPATIAL_REDUCTION: StaticClassIdentity = StaticClassIdentity::new("fea.SpatialReduction");
pub const SERIES_REDUCTION: StaticClassIdentity = StaticClassIdentity::new("fea.SeriesReduction");
pub const SERIES_WEIGHTING: StaticClassIdentity = StaticClassIdentity::new("fea.SeriesWeighting");
pub const EXTREMA_TIE_POLICY: StaticClassIdentity =
    StaticClassIdentity::new("fea.ExtremaTiePolicy");
pub const PARTIAL_ACCESS_POLICY: StaticClassIdentity =
    StaticClassIdentity::new("fea.PartialAccessPolicy");
pub const OBSERVATION_PURPOSE: StaticClassIdentity =
    StaticClassIdentity::new("fea.ObservationPurpose");
pub const DIAGNOSTIC_KIND: StaticClassIdentity = StaticClassIdentity::new("fea.DiagnosticKind");
pub const POSITION: StaticClassIdentity = StaticClassIdentity::new("fea.Position");

/// Every typed procedure-step class accepted where a single procedure step is required.
pub const PROCEDURE_STEP_CLASSES: [StaticClassIdentity; 9] = [
    STRUCTURAL_STEP,
    THERMAL_STEP,
    ELECTROMAGNETIC_STEP,
    ACOUSTIC_STEP,
    FLUID_STEP,
    THERMO_MECHANICAL_STEP,
    ELECTRO_THERMAL_STEP,
    CONJUGATE_HEAT_TRANSFER_STEP,
    FLUID_STRUCTURE_STEP,
];
