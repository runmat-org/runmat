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

    pub fn spec_mut(&mut self) -> Option<&mut ObservationSpec> {
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
    pub cost: ObservationCostPolicy,
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
    VonMisesStress,
    ReactionForce,
    ReactionMoment,
    StrainEnergy,
    BeamAxialForce,
    BeamShearForce,
    BeamTorsionMoment,
    BeamBendingMoment,
    BeamBendingStress,
    BeamTorsionStress,
    ShellMembraneForce,
    ShellBendingMoment,
    ShellTransverseShear,
    ShellVonMises,
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

impl ObservationQuantity {
    /// Resolve the stable source-language spelling of an observation quantity.
    ///
    /// Keeping this closed mapping beside the canonical enum prevents frontends from growing
    /// independent string-dispatch tables for the same scientific quantities.
    pub fn from_source_name(name: &str) -> Option<Self> {
        Some(match name {
            "displacement" => Self::Displacement,
            "rotation" => Self::Rotation,
            "velocity" => Self::Velocity,
            "acceleration" => Self::Acceleration,
            "strain" => Self::Strain,
            "stress" => Self::Stress,
            "von_mises_stress" => Self::VonMisesStress,
            "reaction_force" => Self::ReactionForce,
            "reaction_moment" => Self::ReactionMoment,
            "strain_energy" => Self::StrainEnergy,
            "beam_axial_force" => Self::BeamAxialForce,
            "beam_shear_force" => Self::BeamShearForce,
            "beam_torsion_moment" => Self::BeamTorsionMoment,
            "beam_bending_moment" => Self::BeamBendingMoment,
            "beam_bending_stress" => Self::BeamBendingStress,
            "beam_torsion_stress" => Self::BeamTorsionStress,
            "shell_membrane_force" => Self::ShellMembraneForce,
            "shell_bending_moment" => Self::ShellBendingMoment,
            "shell_transverse_shear" => Self::ShellTransverseShear,
            "shell_von_mises" => Self::ShellVonMises,
            "temperature" => Self::Temperature,
            "temperature_gradient" => Self::TemperatureGradient,
            "heat_flux" => Self::HeatFlux,
            "heat_rate" => Self::HeatRate,
            "electric_potential" => Self::ElectricPotential,
            "electric_field" => Self::ElectricField,
            "electric_current" => Self::ElectricCurrent,
            "electric_current_density" => Self::ElectricCurrentDensity,
            "electric_charge" => Self::ElectricCharge,
            "magnetic_vector_potential" => Self::MagneticVectorPotential,
            "magnetic_flux_density" => Self::MagneticFluxDensity,
            "magnetic_field_intensity" => Self::MagneticFieldIntensity,
            "acoustic_pressure" => Self::AcousticPressure,
            "acoustic_particle_velocity" => Self::AcousticParticleVelocity,
            "sound_pressure_level" => Self::SoundPressureLevel,
            "fluid_pressure" => Self::FluidPressure,
            "fluid_velocity" => Self::FluidVelocity,
            "vorticity" => Self::Vorticity,
            "mass_flow_rate" => Self::MassFlowRate,
            "wall_shear_stress" => Self::WallShearStress,
            "modal_frequency" => Self::ModalFrequency,
            "modal_participation" => Self::ModalParticipation,
            "eigenvalue" => Self::Eigenvalue,
            "residual" => Self::Residual,
            "balance" => Self::Balance,
            "constraint_violation" => Self::ConstraintViolation,
            "contact_gap" => Self::ContactGap,
            "contact_pressure" => Self::ContactPressure,
            _ => return None,
        })
    }

    /// Canonical component shape of this physical quantity.
    ///
    /// This is the shared authority used by authored-output validation, physics recovery, and
    /// observation evaluation. Storage may use an equivalent explicit tensor extent, but it must
    /// not redefine the quantity's semantics.
    pub fn canonical_value_shape(self) -> ObservationValueShape {
        match self {
            Self::Displacement
            | Self::Rotation
            | Self::Velocity
            | Self::Acceleration
            | Self::ReactionForce
            | Self::ReactionMoment
            | Self::TemperatureGradient
            | Self::HeatFlux
            | Self::ElectricField
            | Self::ElectricCurrentDensity
            | Self::MagneticVectorPotential
            | Self::MagneticFluxDensity
            | Self::MagneticFieldIntensity
            | Self::AcousticParticleVelocity
            | Self::FluidVelocity
            | Self::Vorticity
            | Self::WallShearStress => ObservationValueShape::Vector3,
            Self::Strain | Self::Stress => ObservationValueShape::SymmetricTensor3,
            Self::BeamShearForce
            | Self::BeamBendingMoment
            | Self::BeamBendingStress
            | Self::ShellTransverseShear => ObservationValueShape::Tensor { extents: vec![2] },
            Self::ShellMembraneForce | Self::ShellBendingMoment => {
                ObservationValueShape::Tensor { extents: vec![3] }
            }
            _ => ObservationValueShape::Scalar,
        }
    }
}

impl ObservationSpec {
    /// Physical meaning of the projected result, distinct from its recovered source quantity.
    pub fn result_quantity(&self) -> ObservationResultQuantity {
        if self.component == ObservationComponent::VonMises {
            ObservationResultQuantity::Physical(ObservationQuantity::VonMisesStress)
        } else if self.complex_representation == ComplexRepresentation::PhaseRadians {
            ObservationResultQuantity::Dimensionless
        } else {
            ObservationResultQuantity::Physical(self.quantity)
        }
    }

    /// Canonical value shape after component projection and spatial reduction.
    ///
    /// This is distinct from `value_shape`, which describes the recovered source quantity.
    pub fn result_value_shape(&self) -> ObservationValueShape {
        let projected = match self.component {
            ObservationComponent::All | ObservationComponent::Deviatoric => {
                self.value_shape.clone()
            }
            _ => ObservationValueShape::Scalar,
        };
        match self.reduction.as_ref().map(|reduction| reduction.spatial) {
            Some(
                SpatialReduction::EuclideanNorm
                | SpatialReduction::Minimum
                | SpatialReduction::Maximum,
            ) => ObservationValueShape::Scalar,
            Some(SpatialReduction::Resultant | SpatialReduction::Moment { .. }) => {
                ObservationValueShape::Vector3
            }
            _ => projected,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "quantity")]
pub enum ObservationResultQuantity {
    Physical(ObservationQuantity),
    Dimensionless,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationValueShape {
    Scalar,
    Vector3,
    SymmetricTensor3,
    Tensor { extents: Vec<usize> },
}

impl ObservationValueShape {
    pub fn component_count(&self) -> Option<usize> {
        match self {
            Self::Scalar => Some(1),
            Self::Vector3 => Some(3),
            Self::SymmetricTensor3 => Some(6),
            Self::Tensor { extents } if !extents.is_empty() && !extents.contains(&0) => extents
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent)),
            Self::Tensor { .. } => None,
        }
    }

    pub fn is_storage_equivalent_to(&self, canonical: &Self) -> bool {
        match (self, canonical) {
            (Self::Scalar, Self::Scalar)
            | (Self::Vector3, Self::Vector3)
            | (Self::SymmetricTensor3, Self::SymmetricTensor3) => true,
            (Self::Tensor { extents }, Self::Vector3) => extents == &[3],
            (Self::Tensor { extents }, Self::SymmetricTensor3) => extents == &[3, 3],
            (Self::Tensor { extents: left }, Self::Tensor { extents: right }) => left == right,
            _ => false,
        }
    }
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
    /// A geometric sample position evaluated from an owning finite element.
    Point,
    Cell,
    Element,
    IntegrationPoint,
    Face,
    Edge,
    Interface,
    Connector,
    /// One authored constraint group in the canonical executed constraint plan.
    Constraint,
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
    StudyParameterValues(Vec<f64>),
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
    Solver(SolverObservationEvent),
    Series(SeriesObservationEvent),
    Accuracy(AccuracyObservationEvent),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverObservationEvent {
    FirstYield,
    ContactStatusChange,
    ConvergenceFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesObservationEvent {
    PeakResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyObservationEvent {
    CriterionCrossing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReductionPolicy {
    pub spatial: SpatialReduction,
    pub series: SeriesReduction,
    pub extrema_ties: ExtremaTiePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
    Moment { about: Position },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesReduction {
    None,
    Minimum,
    Maximum,
    AbsoluteMaximum,
    RootMeanSquare { weighting: SeriesWeighting },
    Mean { weighting: SeriesWeighting },
    Variance { weighting: SeriesWeighting },
    Envelope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeriesWeighting {
    UniformSamples,
    TrapezoidalIndependentDomain,
}

impl SeriesReduction {
    pub const fn weighting(self) -> Option<SeriesWeighting> {
        match self {
            Self::RootMeanSquare { weighting }
            | Self::Mean { weighting }
            | Self::Variance { weighting } => Some(weighting),
            Self::None | Self::Minimum | Self::Maximum | Self::AbsoluteMaximum | Self::Envelope => {
                None
            }
        }
    }
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

/// Governs when an observation's resource cost becomes authoritative.
///
/// Source-authored requests normally use `Automatic`: topology-aware planning computes and admits
/// the exact cost before solve work starts. Integrations with a stricter external budget may
/// provide an explicit upper bound, which the resolved plan must not exceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode", content = "limit")]
pub enum ObservationCostPolicy {
    Automatic,
    UpperBound(ObservationCostEstimate),
}

impl ObservationCostPolicy {
    pub const fn upper_bound(self) -> Option<ObservationCostEstimate> {
        match self {
            Self::Automatic => None,
            Self::UpperBound(limit) => Some(limit),
        }
    }
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
