use runmat_types::PhysicalDimension;
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

mod quantity;
pub use quantity::ObservationQuantity;

impl ObservationSpec {
    /// Physical meaning of the projected result, distinct from its recovered source quantity.
    pub fn result_quantity(&self) -> ObservationResultQuantity {
        if self.complex_representation == ComplexRepresentation::PhaseRadians {
            return ObservationResultQuantity::Derived(
                runmat_types::physical::DynamicDimension::quantity_type(
                    runmat_types::physical::QuantityKind::ComplexPhase,
                ),
            );
        }
        if self.complex_representation == ComplexRepresentation::SoundPressureLevel {
            return ObservationResultQuantity::Physical(ObservationQuantity::SoundPressureLevel);
        }
        if self.spatial_reduction() == SpatialReduction::TransmissionLoss {
            return ObservationResultQuantity::Physical(ObservationQuantity::TransmissionLoss);
        }
        let quantity = if self.component == ObservationComponent::VonMises {
            ObservationQuantity::VonMisesStress
        } else {
            self.quantity
        };
        match self.result_descriptor() {
            Some(dimension) if Some(dimension) != quantity.quantity_descriptor() => {
                ObservationResultQuantity::Derived(dimension)
            }
            _ => ObservationResultQuantity::Physical(quantity),
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
                | SpatialReduction::Maximum
                | SpatialReduction::NormalFlux
                | SpatialReduction::MeanDifference
                | SpatialReduction::TransmissionLoss,
            ) => ObservationValueShape::Scalar,
            Some(SpatialReduction::Resultant | SpatialReduction::Moment { .. }) => {
                ObservationValueShape::Vector3
            }
            _ => projected,
        }
    }

    pub fn spatial_reduction(&self) -> SpatialReduction {
        self.reduction
            .as_ref()
            .map_or(SpatialReduction::None, |reduction| reduction.spatial)
    }

    /// SI base dimension after representation and spatial reduction.
    pub fn result_dimension(&self) -> Option<PhysicalDimension> {
        Some(self.result_descriptor()?.dimension())
    }

    pub fn result_descriptor(&self) -> Option<runmat_types::physical::DynamicDimension> {
        if matches!(
            self.complex_representation,
            ComplexRepresentation::PhaseRadians | ComplexRepresentation::SoundPressureLevel
        ) {
            return Some(runmat_types::physical::DynamicDimension::quantity_type(
                runmat_types::physical::QuantityKind::ComplexPhase,
            ));
        }
        let source = self.quantity.quantity_descriptor()?;
        let spatial = self
            .reduction
            .as_ref()
            .map_or(SpatialReduction::None, |reduction| reduction.spatial);
        let measure = self.target.measure_dimension();
        let spatial_dimension = match spatial {
            SpatialReduction::Integral | SpatialReduction::NormalFlux => {
                source.multiply(measure?.into())
            }
            SpatialReduction::Resultant => {
                if matches!(
                    self.quantity,
                    ObservationQuantity::ReactionForce | ObservationQuantity::ReactionMoment
                ) {
                    Some(source)
                } else {
                    source.multiply(measure?.into())
                }
            }
            SpatialReduction::Moment { .. } => {
                let resultant = if self.quantity == ObservationQuantity::ReactionForce {
                    source
                } else {
                    source.multiply(measure?.into())?
                };
                resultant.multiply(PhysicalDimension::LENGTH.into())
            }
            SpatialReduction::None
            | SpatialReduction::Average
            | SpatialReduction::RootMeanSquare
            | SpatialReduction::EuclideanNorm
            | SpatialReduction::Minimum
            | SpatialReduction::Maximum
            | SpatialReduction::MeanDifference => Some(source),
            // A level in dB of a power ratio.
            SpatialReduction::TransmissionLoss => Some(PhysicalDimension::DIMENSIONLESS.into()),
        }?;
        match self.reduction.as_ref().map(|reduction| reduction.series) {
            Some(SeriesReduction::Variance { .. }) => spatial_dimension.power(2),
            _ => Some(spatial_dimension),
        }
    }
}

impl ObservationTarget {
    fn measure_dimension(&self) -> Option<PhysicalDimension> {
        let Self::Spatial(ExcitationTarget::Region(region)) = self else {
            return None;
        };
        Some(match region.entity_dimension {
            crate::EntityDimension::Point => return None,
            crate::EntityDimension::Edge => PhysicalDimension::LENGTH,
            crate::EntityDimension::Face => PhysicalDimension::AREA,
            crate::EntityDimension::Volume => PhysicalDimension::VOLUME,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "quantity")]
pub enum ObservationResultQuantity {
    Physical(ObservationQuantity),
    Derived(runmat_types::physical::DynamicDimension),
    Dimensionless,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationValueShape {
    Scalar,
    Vector3,
    SymmetricTensor3,
    Tensor {
        #[serde(with = "runmat_types::wire_integer::vec")]
        extents: Vec<usize>,
    },
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
    /// Two face regions read by one paired reduction, in order (from/to, incident/transmitted).
    RegionPair {
        first: crate::RegionTarget,
        second: crate::RegionTarget,
    },
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
    /// Sound pressure level in dB re 20 uPa of a peak pressure phasor:
    /// 20 log10(|p| / (sqrt(2) * 20e-6 Pa)), i.e. referred to the RMS pressure.
    SoundPressureLevel,
}

/// Reference pressure of `ComplexRepresentation::SoundPressureLevel` (air, 20 uPa).
pub const SOUND_PRESSURE_LEVEL_REFERENCE_PA: f64 = 20.0e-6;

/// Sound pressure level in dB re 20 uPa of a peak pressure phasor magnitude `|p|`, referred to
/// the RMS pressure `|p| / sqrt(2)`: `20 log10(|p| / (sqrt(2) * 20 uPa))`. This is the one
/// definition every acoustic result path uses. It is not clamped: levels below the reference
/// are negative, and zero pressure has no finite level (`-inf`).
pub fn sound_pressure_level_db(peak_pressure_magnitude_pa: f64) -> f64 {
    20.0 * (peak_pressure_magnitude_pa.abs()
        / (std::f64::consts::SQRT_2 * SOUND_PRESSURE_LEVEL_REFERENCE_PA))
        .log10()
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
    ModeIndices(#[serde(with = "runmat_types::wire_integer::vec")] Vec<usize>),
    Event(ObservationEvent),
    Adaptive {
        minimum_interval: f64,
        #[serde(with = "runmat_types::wire_integer")]
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
    Moment {
        about: Position,
    },
    /// Flux of a vector field through a face region along its outward normal, ∫ v·n dA.
    NormalFlux,
    /// Area-weighted mean over the first face region of a `RegionPair` target minus the
    /// area-weighted mean over the second: the static pressure drop between two ports.
    MeanDifference,
    /// 10·log10(∫ first / ∫ second) of a power-density field over the two face regions of a
    /// `RegionPair` target, in dB: transmission loss from the incident to the transmitted port.
    TransmissionLoss,
}

impl SpatialReduction {
    /// Reductions that read two face regions (a `RegionPair` target) into one scalar.
    pub const fn is_paired(self) -> bool {
        matches!(self, Self::MeanDifference | Self::TransmissionLoss)
    }
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
    LatestSamples {
        #[serde(with = "runmat_types::wire_integer")]
        count: usize,
    },
    ExplicitSamples {
        #[serde(with = "runmat_types::wire_integer")]
        maximum_count: usize,
    },
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
    #[serde(with = "runmat_types::wire_integer")]
    pub operation_count: u64,
    #[serde(with = "runmat_types::wire_integer")]
    pub peak_resident_bytes: u64,
    #[serde(with = "runmat_types::wire_integer")]
    pub stored_bytes: u64,
    #[serde(with = "runmat_types::wire_integer")]
    pub transfer_bytes: u64,
    #[serde(with = "runmat_types::wire_integer")]
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
    AcceptedIncrements {
        #[serde(with = "runmat_types::wire_integer")]
        interval: usize,
    },
    WallSeconds {
        interval: f64,
    },
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
        #[serde(with = "runmat_types::wire_integer")]
        consecutive: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationThreshold {
    pub quantity: ObservationQuantity,
    pub canonical_value: f64,
}
