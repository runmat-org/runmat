use serde::{Deserialize, Serialize};

use crate::{CoordinateFrame, ExcitationTarget, Position, SignalPrecision};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ObservationId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationRequest {
    Field(ObservationSpec),
    History(ObservationSpec),
    Probe(ObservationSpec),
    Reduction(ObservationSpec),
    Diagnostic(DiagnosticObservation),
    Monitor(MonitorObservation),
}

impl ObservationRequest {
    pub fn spec(&self) -> Option<&ObservationSpec> {
        match self {
            Self::Field(value)
            | Self::History(value)
            | Self::Probe(value)
            | Self::Reduction(value) => Some(value),
            Self::Diagnostic(_) | Self::Monitor(_) => None,
        }
    }

    pub fn observation_id(&self) -> &ObservationId {
        match self {
            Self::Field(value)
            | Self::History(value)
            | Self::Probe(value)
            | Self::Reduction(value) => &value.observation_id,
            Self::Diagnostic(value) => &value.observation_id,
            Self::Monitor(value) => &value.observation_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationSpec {
    pub observation_id: ObservationId,
    pub quantity: ObservationQuantity,
    pub value_shape: ObservationValueShape,
    pub target: ObservationTarget,
    pub source_location: FieldLocation,
    pub output_location: FieldLocation,
    pub recovery: RecoveryPolicy,
    pub component: ObservationComponent,
    pub frame: CoordinateFrame,
    pub sign_convention: SignConvention,
    pub complex_representation: ComplexRepresentation,
    pub sampling: SamplingPolicy,
    pub reduction: Option<ReductionPolicy>,
    pub storage: ObservationStoragePolicy,
    pub purpose: ObservationPurpose,
    pub cost: ObservationCostEstimate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationQuantity {
    Displacement,
    Rotation,
    Velocity,
    Acceleration,
    Strain,
    Stress,
    ReactionForce,
    ReactionMoment,
    StrainEnergy,
    Temperature,
    TemperatureGradient,
    HeatFlux,
    HeatRate,
    ElectricPotential,
    ElectricField,
    ElectricCurrent,
    ElectricCurrentDensity,
    ElectricCharge,
    MagneticVectorPotential,
    MagneticFluxDensity,
    MagneticFieldIntensity,
    AcousticPressure,
    AcousticParticleVelocity,
    SoundPressureLevel,
    FluidPressure,
    FluidVelocity,
    Vorticity,
    MassFlowRate,
    WallShearStress,
    ModalFrequency,
    ModalParticipation,
    Eigenvalue,
    Residual,
    Balance,
    ConstraintViolation,
    ContactGap,
    ContactPressure,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationValueShape {
    Scalar,
    Vector3,
    SymmetricTensor3,
    Tensor { extents: Vec<usize> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationTarget {
    Spatial(ExcitationTarget),
    Point { position: Position },
    Path { positions: Vec<Position> },
    Connector { connector_id: String },
    Constraint { constraint_id: String },
    WholeModel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldLocation {
    Node,
    Cell,
    Element,
    IntegrationPoint,
    Face,
    Edge,
    Interface,
    Connector,
    ReferencePoint,
    Mode,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryPolicy {
    Native,
    Extrapolate,
    VolumeWeightedAverage,
    AreaWeightedAverage,
    ConservativeProjection,
    ShapeFunctionInterpolation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationComponent {
    All,
    X,
    Y,
    Z,
    Normal,
    Tangent1,
    Tangent2,
    Trace,
    Hydrostatic,
    Deviatoric,
    VonMises,
    MaximumPrincipal,
    MinimumPrincipal,
    Magnitude,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignConvention {
    Native,
    TensionPositive,
    CompressionPositive,
    OutwardPositive,
    InwardPositive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplexRepresentation {
    Real,
    Imaginary,
    Magnitude,
    PhaseRadians,
    Complex,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingPolicy {
    Final,
    EveryAcceptedIncrement,
    TimeSeconds(Vec<f64>),
    TimeIntervalSeconds {
        start: f64,
        end: f64,
        interval: f64,
    },
    FrequencyHertz(Vec<f64>),
    FrequencyIntervalHertz {
        start: f64,
        end: f64,
        interval: f64,
    },
    LoadFactors(Vec<f64>),
    LoadFactorInterval {
        start: f64,
        end: f64,
        interval: f64,
    },
    ModeIndices(Vec<usize>),
    Event(ObservationEvent),
    Adaptive {
        minimum_interval: f64,
        maximum_samples: usize,
        relative_change: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationEvent {
    FirstYield,
    ContactStatusChange,
    PeakResponse,
    ConvergenceFailure,
    AccuracyCriterionCrossing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReductionPolicy {
    pub spatial: SpatialReduction,
    pub series: SeriesReduction,
    pub extrema_ties: ExtremaTiePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialReduction {
    None,
    Integral,
    Average,
    RootMeanSquare,
    EuclideanNorm,
    Minimum,
    Maximum,
    Resultant,
    Moment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesReduction {
    None,
    Minimum,
    Maximum,
    AbsoluteMaximum,
    RootMeanSquare,
    Mean,
    Variance,
    Envelope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtremaTiePolicy {
    FirstCanonical,
    AllLocations,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationStoragePolicy {
    pub precision: SignalPrecision,
    pub compression: CompressionPolicy,
    pub retention: RetentionPolicy,
    pub partial_access: PartialAccessPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionPolicy {
    Lossless,
    ErrorBounded { maximum_relative_error: f64 },
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionPolicy {
    Canonical,
    UntilDependentsComplete,
    LatestSamples { count: usize },
    ExplicitSamples { maximum_count: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialAccessPolicy {
    ManifestAndChunks,
    WholeArtifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationPurpose {
    Publication,
    SolverControl,
    Visualization,
    DownstreamState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationCostEstimate {
    pub operation_count: u64,
    pub peak_resident_bytes: u64,
    pub stored_bytes: u64,
    pub transfer_bytes: u64,
    pub collective_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticObservation {
    pub observation_id: ObservationId,
    pub diagnostic: DiagnosticKind,
    pub sampling: SamplingPolicy,
    pub retention: RetentionPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticKind {
    Convergence,
    Balance,
    Constraint,
    Contact,
    Adaptation,
    Resource,
    ExecutionEquivalence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonitorObservation {
    pub observation_id: ObservationId,
    pub source_observation_id: ObservationId,
    pub cadence: MonitorCadence,
    pub stop: Option<MonitorStopCondition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MonitorCadence {
    AcceptedIncrements { interval: usize },
    WallSeconds { interval: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MonitorStopCondition {
    LessThan {
        threshold: ObservationThreshold,
    },
    GreaterThan {
        threshold: ObservationThreshold,
    },
    AbsoluteChangeBelow {
        threshold: ObservationThreshold,
        consecutive: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationThreshold {
    pub quantity: ObservationQuantity,
    pub canonical_value: f64,
}
