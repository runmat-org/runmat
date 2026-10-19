use serde::{Deserialize, Serialize};

mod shape;
pub use shape::{AuthoredSignalFieldLocation, AuthoredSignalTopologySelector, SignalShape};
mod physical;
mod value;
pub use value::SignalValue;

use super::{
    coordinate_frame::CoordinateFrame,
    quantity::{
        Acceleration, AngularAcceleration, AngularVelocity, BodyForceDensity, Displacement,
        ElectricCharge, ElectricChargeDensity, ElectricCurrent, ElectricCurrentDensity,
        ElectricPotential, Force, HeatFlux, HeatGenerationDensity, MagneticVectorPotential,
        MassFlowRate, Moment, Pressure, Rotation, Temperature, TemperatureDifference, Traction,
        Velocity, VolumeFlowRate,
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
    VolumeFlowRate,
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
    VolumeFlowRate(Signal<VolumeFlowRate>),
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

mod provenance;
pub use provenance::{GeneratorProvenance, SignalProvenance};

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
