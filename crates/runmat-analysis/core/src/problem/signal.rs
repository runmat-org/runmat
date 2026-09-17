use serde::{Deserialize, Serialize};

use super::{
    coordinate_frame::CoordinateFrame,
    quantity::{
        Acceleration, AngularAcceleration, AngularVelocity, BodyForceDensity, Displacement,
        ElectricCharge, ElectricChargeDensity, ElectricCurrent, ElectricCurrentDensity,
        ElectricPotential, Force, HeatFlux, HeatGenerationDensity, MagneticVectorPotential,
        MassFlowRate, Moment, Pressure, Rotation, Temperature, TemperatureDifference, Traction,
        Velocity,
    },
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SignalId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalQuantityKind {
    Displacement,
    Rotation,
    Velocity,
    Acceleration,
    AngularVelocity,
    AngularAcceleration,
    Force,
    Moment,
    Traction,
    Pressure,
    BodyForceDensity,
    Temperature,
    TemperatureDifference,
    HeatFlux,
    HeatGenerationDensity,
    ElectricPotential,
    ElectricCurrent,
    ElectricCharge,
    ElectricCurrentDensity,
    ElectricChargeDensity,
    MagneticVectorPotential,
    MassFlowRate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalDomainKind {
    Constant,
    Time,
    Frequency,
    StudyParameter,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalSignal {
    Displacement(Signal<Displacement>),
    Rotation(Signal<Rotation>),
    Velocity(Signal<Velocity>),
    Acceleration(Signal<Acceleration>),
    AngularVelocity(Signal<AngularVelocity>),
    AngularAcceleration(Signal<AngularAcceleration>),
    Force(Signal<Force>),
    Moment(Signal<Moment>),
    Traction(Signal<Traction>),
    Pressure(Signal<Pressure>),
    BodyForceDensity(Signal<BodyForceDensity>),
    Temperature(Signal<Temperature>),
    TemperatureDifference(Signal<TemperatureDifference>),
    HeatFlux(Signal<HeatFlux>),
    HeatGenerationDensity(Signal<HeatGenerationDensity>),
    ElectricPotential(Signal<ElectricPotential>),
    ElectricCurrent(Signal<ElectricCurrent>),
    ElectricCharge(Signal<ElectricCharge>),
    ElectricCurrentDensity(Signal<ElectricCurrentDensity>),
    ElectricChargeDensity(Signal<ElectricChargeDensity>),
    MagneticVectorPotential(Signal<MagneticVectorPotential>),
    MassFlowRate(Signal<MassFlowRate>),
}

impl PhysicalSignal {
    pub fn signal_id(&self) -> &SignalId {
        match self {
            Self::Displacement(value) => &value.signal_id,
            Self::Rotation(value) => &value.signal_id,
            Self::Velocity(value) => &value.signal_id,
            Self::Acceleration(value) => &value.signal_id,
            Self::AngularVelocity(value) => &value.signal_id,
            Self::AngularAcceleration(value) => &value.signal_id,
            Self::Force(value) => &value.signal_id,
            Self::Moment(value) => &value.signal_id,
            Self::Traction(value) => &value.signal_id,
            Self::Pressure(value) => &value.signal_id,
            Self::BodyForceDensity(value) => &value.signal_id,
            Self::Temperature(value) => &value.signal_id,
            Self::TemperatureDifference(value) => &value.signal_id,
            Self::HeatFlux(value) => &value.signal_id,
            Self::HeatGenerationDensity(value) => &value.signal_id,
            Self::ElectricPotential(value) => &value.signal_id,
            Self::ElectricCurrent(value) => &value.signal_id,
            Self::ElectricCharge(value) => &value.signal_id,
            Self::ElectricCurrentDensity(value) => &value.signal_id,
            Self::ElectricChargeDensity(value) => &value.signal_id,
            Self::MagneticVectorPotential(value) => &value.signal_id,
            Self::MassFlowRate(value) => &value.signal_id,
        }
    }

    pub fn quantity(&self) -> PhysicalQuantityKind {
        match self {
            Self::Displacement(_) => PhysicalQuantityKind::Displacement,
            Self::Rotation(_) => PhysicalQuantityKind::Rotation,
            Self::Velocity(_) => PhysicalQuantityKind::Velocity,
            Self::Acceleration(_) => PhysicalQuantityKind::Acceleration,
            Self::AngularVelocity(_) => PhysicalQuantityKind::AngularVelocity,
            Self::AngularAcceleration(_) => PhysicalQuantityKind::AngularAcceleration,
            Self::Force(_) => PhysicalQuantityKind::Force,
            Self::Moment(_) => PhysicalQuantityKind::Moment,
            Self::Traction(_) => PhysicalQuantityKind::Traction,
            Self::Pressure(_) => PhysicalQuantityKind::Pressure,
            Self::BodyForceDensity(_) => PhysicalQuantityKind::BodyForceDensity,
            Self::Temperature(_) => PhysicalQuantityKind::Temperature,
            Self::TemperatureDifference(_) => PhysicalQuantityKind::TemperatureDifference,
            Self::HeatFlux(_) => PhysicalQuantityKind::HeatFlux,
            Self::HeatGenerationDensity(_) => PhysicalQuantityKind::HeatGenerationDensity,
            Self::ElectricPotential(_) => PhysicalQuantityKind::ElectricPotential,
            Self::ElectricCurrent(_) => PhysicalQuantityKind::ElectricCurrent,
            Self::ElectricCharge(_) => PhysicalQuantityKind::ElectricCharge,
            Self::ElectricCurrentDensity(_) => PhysicalQuantityKind::ElectricCurrentDensity,
            Self::ElectricChargeDensity(_) => PhysicalQuantityKind::ElectricChargeDensity,
            Self::MagneticVectorPotential(_) => PhysicalQuantityKind::MagneticVectorPotential,
            Self::MassFlowRate(_) => PhysicalQuantityKind::MassFlowRate,
        }
    }

    pub fn domain(&self) -> SignalDomainKind {
        match self {
            Self::Displacement(value) => signal_domain(value),
            Self::Rotation(value) => signal_domain(value),
            Self::Velocity(value) => signal_domain(value),
            Self::Acceleration(value) => signal_domain(value),
            Self::AngularVelocity(value) => signal_domain(value),
            Self::AngularAcceleration(value) => signal_domain(value),
            Self::Force(value) => signal_domain(value),
            Self::Moment(value) => signal_domain(value),
            Self::Traction(value) => signal_domain(value),
            Self::Pressure(value) => signal_domain(value),
            Self::BodyForceDensity(value) => signal_domain(value),
            Self::Temperature(value) => signal_domain(value),
            Self::TemperatureDifference(value) => signal_domain(value),
            Self::HeatFlux(value) => signal_domain(value),
            Self::HeatGenerationDensity(value) => signal_domain(value),
            Self::ElectricPotential(value) => signal_domain(value),
            Self::ElectricCurrent(value) => signal_domain(value),
            Self::ElectricCharge(value) => signal_domain(value),
            Self::ElectricCurrentDensity(value) => signal_domain(value),
            Self::ElectricChargeDensity(value) => signal_domain(value),
            Self::MagneticVectorPotential(value) => signal_domain(value),
            Self::MassFlowRate(value) => signal_domain(value),
        }
    }

    pub fn shape(&self) -> &SignalShape {
        match self {
            Self::Displacement(value) => &value.shape,
            Self::Rotation(value) => &value.shape,
            Self::Velocity(value) => &value.shape,
            Self::Acceleration(value) => &value.shape,
            Self::AngularVelocity(value) => &value.shape,
            Self::AngularAcceleration(value) => &value.shape,
            Self::Force(value) => &value.shape,
            Self::Moment(value) => &value.shape,
            Self::Traction(value) => &value.shape,
            Self::Pressure(value) => &value.shape,
            Self::BodyForceDensity(value) => &value.shape,
            Self::Temperature(value) => &value.shape,
            Self::TemperatureDifference(value) => &value.shape,
            Self::HeatFlux(value) => &value.shape,
            Self::HeatGenerationDensity(value) => &value.shape,
            Self::ElectricPotential(value) => &value.shape,
            Self::ElectricCurrent(value) => &value.shape,
            Self::ElectricCharge(value) => &value.shape,
            Self::ElectricCurrentDensity(value) => &value.shape,
            Self::ElectricChargeDensity(value) => &value.shape,
            Self::MagneticVectorPotential(value) => &value.shape,
            Self::MassFlowRate(value) => &value.shape,
        }
    }

    pub fn frame(&self) -> Option<&CoordinateFrame> {
        match self {
            Self::Displacement(value) => value.frame.as_ref(),
            Self::Rotation(value) => value.frame.as_ref(),
            Self::Velocity(value) => value.frame.as_ref(),
            Self::Acceleration(value) => value.frame.as_ref(),
            Self::AngularVelocity(value) => value.frame.as_ref(),
            Self::AngularAcceleration(value) => value.frame.as_ref(),
            Self::Force(value) => value.frame.as_ref(),
            Self::Moment(value) => value.frame.as_ref(),
            Self::Traction(value) => value.frame.as_ref(),
            Self::Pressure(value) => value.frame.as_ref(),
            Self::BodyForceDensity(value) => value.frame.as_ref(),
            Self::Temperature(value) => value.frame.as_ref(),
            Self::TemperatureDifference(value) => value.frame.as_ref(),
            Self::HeatFlux(value) => value.frame.as_ref(),
            Self::HeatGenerationDensity(value) => value.frame.as_ref(),
            Self::ElectricPotential(value) => value.frame.as_ref(),
            Self::ElectricCurrent(value) => value.frame.as_ref(),
            Self::ElectricCharge(value) => value.frame.as_ref(),
            Self::ElectricCurrentDensity(value) => value.frame.as_ref(),
            Self::ElectricChargeDensity(value) => value.frame.as_ref(),
            Self::MagneticVectorPotential(value) => value.frame.as_ref(),
            Self::MassFlowRate(value) => value.frame.as_ref(),
        }
    }

    pub fn generator(&self) -> Option<&SignalGenerator> {
        macro_rules! generator {
            ($signal:expr) => {
                match &$signal.representation {
                    SignalRepresentation::Generator(generator) => Some(generator),
                    _ => None,
                }
            };
        }
        match self {
            Self::Displacement(signal) => generator!(signal),
            Self::Rotation(signal) => generator!(signal),
            Self::Velocity(signal) => generator!(signal),
            Self::Acceleration(signal) => generator!(signal),
            Self::AngularVelocity(signal) => generator!(signal),
            Self::AngularAcceleration(signal) => generator!(signal),
            Self::Force(signal) => generator!(signal),
            Self::Moment(signal) => generator!(signal),
            Self::Traction(signal) => generator!(signal),
            Self::Pressure(signal) => generator!(signal),
            Self::BodyForceDensity(signal) => generator!(signal),
            Self::Temperature(signal) => generator!(signal),
            Self::TemperatureDifference(signal) => generator!(signal),
            Self::HeatFlux(signal) => generator!(signal),
            Self::HeatGenerationDensity(signal) => generator!(signal),
            Self::ElectricPotential(signal) => generator!(signal),
            Self::ElectricCurrent(signal) => generator!(signal),
            Self::ElectricCharge(signal) => generator!(signal),
            Self::ElectricCurrentDensity(signal) => generator!(signal),
            Self::ElectricChargeDensity(signal) => generator!(signal),
            Self::MagneticVectorPotential(signal) => generator!(signal),
            Self::MassFlowRate(signal) => generator!(signal),
        }
    }

    pub fn precision(&self) -> SignalPrecision {
        match self {
            Self::Displacement(signal) => signal.precision,
            Self::Rotation(signal) => signal.precision,
            Self::Velocity(signal) => signal.precision,
            Self::Acceleration(signal) => signal.precision,
            Self::AngularVelocity(signal) => signal.precision,
            Self::AngularAcceleration(signal) => signal.precision,
            Self::Force(signal) => signal.precision,
            Self::Moment(signal) => signal.precision,
            Self::Traction(signal) => signal.precision,
            Self::Pressure(signal) => signal.precision,
            Self::BodyForceDensity(signal) => signal.precision,
            Self::Temperature(signal) => signal.precision,
            Self::TemperatureDifference(signal) => signal.precision,
            Self::HeatFlux(signal) => signal.precision,
            Self::HeatGenerationDensity(signal) => signal.precision,
            Self::ElectricPotential(signal) => signal.precision,
            Self::ElectricCurrent(signal) => signal.precision,
            Self::ElectricCharge(signal) => signal.precision,
            Self::ElectricCurrentDensity(signal) => signal.precision,
            Self::ElectricChargeDensity(signal) => signal.precision,
            Self::MagneticVectorPotential(signal) => signal.precision,
            Self::MassFlowRate(signal) => signal.precision,
        }
    }
}

fn signal_domain<Q>(signal: &Signal<Q>) -> SignalDomainKind {
    match &signal.representation {
        SignalRepresentation::Constant(_) => SignalDomainKind::Constant,
        SignalRepresentation::Tabulated(table) => match &table.domain {
            TabulatedSignalDomain::TimeSeconds => SignalDomainKind::Time,
            TabulatedSignalDomain::FrequencyHertz => SignalDomainKind::Frequency,
            TabulatedSignalDomain::StudyParameter { .. } => SignalDomainKind::StudyParameter,
        },
        SignalRepresentation::Waveform(_) => SignalDomainKind::Time,
        SignalRepresentation::Artifact(artifact) => artifact.domain.kind(),
        SignalRepresentation::Generator(generator) => generator.domain.kind(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Signal<Q> {
    pub signal_id: SignalId,
    pub shape: SignalShape,
    pub frame: Option<CoordinateFrame>,
    pub representation: SignalRepresentation<Q>,
    pub precision: SignalPrecision,
    pub evidence: SignalApproximationEvidence,
    pub provenance: SignalProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalShape {
    Scalar,
    Vector3,
    Tensor(Vec<usize>),
    TopologyField {
        topology_id: String,
        component_count: usize,
    },
}

impl SignalShape {
    pub fn component_extents(&self) -> Option<Vec<usize>> {
        match self {
            Self::Scalar => Some(Vec::new()),
            Self::Vector3 => Some(vec![3]),
            Self::Tensor(shape) if !shape.is_empty() => Some(shape.clone()),
            Self::TopologyField {
                topology_id,
                component_count,
            } if !topology_id.trim().is_empty() && *component_count > 0 => {
                Some(vec![*component_count])
            }
            _ => None,
        }
    }

    /// Number of canonical values carried by one sample after topology resolution.
    pub fn realized_component_count(&self, topology_entity_count: Option<usize>) -> Option<usize> {
        match self {
            Self::Scalar if topology_entity_count.is_none() => Some(1),
            Self::Vector3 if topology_entity_count.is_none() => Some(3),
            Self::Tensor(extents)
                if topology_entity_count.is_none()
                    && !extents.is_empty()
                    && extents.iter().all(|extent| *extent > 0) =>
            {
                extents
                    .iter()
                    .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
            }
            Self::TopologyField {
                topology_id,
                component_count,
            } if !topology_id.trim().is_empty() && *component_count > 0 => {
                topology_entity_count.and_then(|entities| entities.checked_mul(*component_count))
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalRepresentation<Q> {
    Constant(SignalValue<Q>),
    Tabulated(TabulatedSignal<Q>),
    Waveform(StandardWaveform<Q>),
    Artifact(SignalArtifactRef),
    Generator(SignalGenerator),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalValue<Q> {
    Scalar(Q),
    Vector3([Q; 3]),
    Tensor { shape: Vec<usize>, values: Vec<Q> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalOrdinate<Q> {
    Real(SignalValue<Q>),
    Complex {
        real: SignalValue<Q>,
        imaginary: SignalValue<Q>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TabulatedSignal<Q> {
    pub domain: TabulatedSignalDomain,
    pub knots: Vec<SignalKnot<Q>>,
    pub interpolation: SignalInterpolation,
    pub extrapolation: SignalExtrapolation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TabulatedSignalDomain {
    TimeSeconds,
    FrequencyHertz,
    StudyParameter { parameter_id: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalKnot<Q> {
    pub abscissa: f64,
    pub value: SignalKnotValue<Q>,
}

impl<Q> SignalKnot<Q> {
    pub fn continuous(abscissa: f64, value: SignalOrdinate<Q>) -> Self {
        Self {
            abscissa,
            value: SignalKnotValue::Continuous(value),
        }
    }

    pub fn jump(
        abscissa: f64,
        left_limit: SignalOrdinate<Q>,
        right_limit: SignalOrdinate<Q>,
        continuity: SignalPointContinuity,
    ) -> Self {
        Self {
            abscissa,
            value: SignalKnotValue::Jump {
                left_limit,
                right_limit,
                continuity,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalKnotValue<Q> {
    Continuous(SignalOrdinate<Q>),
    Jump {
        left_limit: SignalOrdinate<Q>,
        right_limit: SignalOrdinate<Q>,
        continuity: SignalPointContinuity,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalPointContinuity {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalSampleApproach {
    At,
    LeftLimit,
    RightLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalInterpolation {
    LeftEndpoint,
    RightEndpoint,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalExtrapolation {
    Reject,
    HoldEndpoint,
    Zero,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StandardWaveform<Q> {
    pub kind: StandardWaveformKind,
    pub amplitude: SignalValue<Q>,
    pub offset: Option<SignalValue<Q>>,
    pub start_seconds: f64,
    pub duration_seconds: Option<f64>,
    pub frequency_hz: Option<f64>,
    pub phase_radians: f64,
    pub duty_cycle: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StandardWaveformKind {
    Step,
    Ramp,
    Sine,
    HalfSinePulse,
    TriangularPulse,
    RectangularPulse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalPrecision {
    Real32,
    Real64,
    Complex64,
    Complex128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalArtifactRef {
    pub artifact_id: String,
    pub content_digest: String,
    pub schema: String,
    pub domain: SignalArtifactDomain,
    pub topology_digest: Option<String>,
    pub sample_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalArtifactDomain {
    Constant,
    TimeSeconds {
        start: f64,
        end: f64,
    },
    FrequencyHertz {
        start: f64,
        end: f64,
    },
    StudyParameter {
        parameter_id: String,
        start: f64,
        end: f64,
    },
}

impl SignalArtifactDomain {
    pub fn kind(&self) -> SignalDomainKind {
        match self {
            Self::Constant => SignalDomainKind::Constant,
            Self::TimeSeconds { .. } => SignalDomainKind::Time,
            Self::FrequencyHertz { .. } => SignalDomainKind::Frequency,
            Self::StudyParameter { .. } => SignalDomainKind::StudyParameter,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalApproximationEvidence {
    pub max_absolute_error: Option<f64>,
    pub max_relative_error: Option<f64>,
    pub derivative_max_absolute_error: Option<f64>,
    pub sample_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalProvenance {
    Authored,
    Imported { source_digest: String },
    Generated(GeneratorProvenance),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorProvenance {
    pub project_revision: String,
    pub callable_id: String,
    pub frozen_input_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalGenerator {
    pub invocation: GeneratorInvocation,
    pub domain: GeneratorDomain,
    pub sampling: SignalSamplingPolicy,
    pub interpolation: SignalInterpolation,
    pub extrapolation: SignalExtrapolation,
}

/// A portable invocation of one frozen RunMat function.
///
/// The invocation is independent of the numerical object that consumes its output. Signal,
/// response-spectrum, and random-spectrum authoring add their own domain, shape, interpolation,
/// and evidence contracts around this shared callable boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorInvocation {
    pub callable: GeneratorCallable,
    pub arguments: Vec<GeneratorArgument>,
    pub effects: GeneratorEffectPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalDataSource<T> {
    Authored {
        values: T,
        provenance_digest: String,
    },
    Imported {
        values: T,
        source_digest: String,
    },
    Generator(GeneratorInvocation),
    Materialized {
        values: T,
        provenance: GeneratorProvenance,
        content_digest: String,
    },
}

impl<T> NumericalDataSource<T> {
    pub fn authored(values: T, provenance_digest: impl Into<String>) -> Self {
        Self::Authored {
            values,
            provenance_digest: provenance_digest.into(),
        }
    }

    pub fn imported(values: T, source_digest: impl Into<String>) -> Self {
        Self::Imported {
            values,
            source_digest: source_digest.into(),
        }
    }

    pub fn values(&self) -> Option<&T> {
        match self {
            Self::Authored { values, .. }
            | Self::Imported { values, .. }
            | Self::Materialized { values, .. } => Some(values),
            Self::Generator(_) => None,
        }
    }

    pub fn values_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Authored { values, .. }
            | Self::Imported { values, .. }
            | Self::Materialized { values, .. } => Some(values),
            Self::Generator(_) => None,
        }
    }

    pub fn generator(&self) -> Option<&GeneratorInvocation> {
        match self {
            Self::Generator(generator) => Some(generator),
            Self::Authored { .. } | Self::Imported { .. } | Self::Materialized { .. } => None,
        }
    }

    /// Stable identity admitted for this numerical realization.
    ///
    /// Authored and imported sources retain their admitted provenance/source digest. Generator
    /// declarations have no realization identity until execution has produced and validated their
    /// values; materialized generators use the digest of their canonical numerical content rather
    /// than the identity of the callable that happened to produce it.
    pub fn numerical_digest(&self) -> Result<&str, String> {
        match self {
            Self::Authored {
                provenance_digest, ..
            } => Ok(provenance_digest),
            Self::Imported { source_digest, .. } => Ok(source_digest),
            Self::Materialized { content_digest, .. } => Ok(content_digest),
            Self::Generator(_) => Err(
                "generated numerical data must be materialized before its values have an identity"
                    .into(),
            ),
        }
    }

    pub fn validate_provenance(&self) -> Result<(), String> {
        if let Self::Generator(generator) = self {
            return crate::validate_signals::validate_generator_invocation(generator);
        }
        let valid = match self {
            Self::Authored {
                provenance_digest, ..
            } => !provenance_digest.trim().is_empty(),
            Self::Imported { source_digest, .. } => !source_digest.trim().is_empty(),
            Self::Generator(_) => unreachable!("generator returned before provenance validation"),
            Self::Materialized {
                provenance,
                content_digest,
                ..
            } => {
                !provenance.project_revision.trim().is_empty()
                    && !provenance.callable_id.trim().is_empty()
                    && !provenance.frozen_input_digest.trim().is_empty()
                    && !content_digest.trim().is_empty()
            }
        };
        valid
            .then_some(())
            .ok_or_else(|| "numerical data-source provenance must be non-empty".into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratorCallable {
    ImportedFunction(Box<ImportedFunctionIdentity>),
    ProjectFunction {
        qualified_name: String,
    },
    BoundFunction {
        function: runmat_types::ProgramFunctionId,
        display_name: String,
    },
}

impl GeneratorCallable {
    pub fn display_name(&self) -> &str {
        match self {
            Self::ImportedFunction(identity) => &identity.symbol,
            Self::ProjectFunction { qualified_name } => qualified_name,
            Self::BoundFunction { display_name, .. } => display_name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedFunctionIdentity {
    pub module_path: String,
    pub symbol: String,
    pub source_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratorDomain {
    TimeSeconds,
    FrequencyHertz,
    StudyParameter { parameter_id: String },
}

impl GeneratorDomain {
    pub fn kind(&self) -> SignalDomainKind {
        match self {
            Self::TimeSeconds => SignalDomainKind::Time,
            Self::FrequencyHertz => SignalDomainKind::Frequency,
            Self::StudyParameter { .. } => SignalDomainKind::StudyParameter,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratorArgument {
    Boolean(bool),
    SignedInteger(i64),
    UnsignedInteger(u64),
    Real(f64),
    Text(String),
    RealVector(Vec<f64>),
    Artifact(SignalArtifactRef),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalSamplingPolicy {
    Uniform {
        start: f64,
        end: f64,
        sample_count: usize,
    },
    Adaptive {
        start: f64,
        end: f64,
        max_absolute_error: f64,
        max_samples: usize,
    },
    CanonicalPoints(Vec<f64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratorEffectPolicy {
    Pure,
    DeclaredImmutableArtifacts,
    SeededRandom {
        seed: u64,
        distribution: RandomDistribution,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomDistribution {
    Uniform,
    Normal,
    LogNormal,
}
