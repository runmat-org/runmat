//! Physical signal introspection over the shared typed signal contract.
use super::*;

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
            Self::HeatRate(value) => &value.signal_id,
            Self::ElectricPotential(value) => &value.signal_id,
            Self::ElectricCurrent(value) => &value.signal_id,
            Self::ElectricCharge(value) => &value.signal_id,
            Self::ElectricCurrentDensity(value) => &value.signal_id,
            Self::ElectricChargeDensity(value) => &value.signal_id,
            Self::MagneticVectorPotential(value) => &value.signal_id,
            Self::MassFlowRate(value) => &value.signal_id,
            Self::VolumeFlowRate(value) => &value.signal_id,
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
            Self::HeatRate(_) => PhysicalQuantityKind::Power,
            Self::ElectricPotential(_) => PhysicalQuantityKind::ElectricPotential,
            Self::ElectricCurrent(_) => PhysicalQuantityKind::ElectricCurrent,
            Self::ElectricCharge(_) => PhysicalQuantityKind::ElectricCharge,
            Self::ElectricCurrentDensity(_) => PhysicalQuantityKind::ElectricCurrentDensity,
            Self::ElectricChargeDensity(_) => PhysicalQuantityKind::ElectricChargeDensity,
            Self::MagneticVectorPotential(_) => PhysicalQuantityKind::MagneticVectorPotential,
            Self::MassFlowRate(_) => PhysicalQuantityKind::MassFlowRate,
            Self::VolumeFlowRate(_) => PhysicalQuantityKind::VolumeFlowRate,
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
            Self::HeatRate(value) => signal_domain(value),
            Self::ElectricPotential(value) => signal_domain(value),
            Self::ElectricCurrent(value) => signal_domain(value),
            Self::ElectricCharge(value) => signal_domain(value),
            Self::ElectricCurrentDensity(value) => signal_domain(value),
            Self::ElectricChargeDensity(value) => signal_domain(value),
            Self::MagneticVectorPotential(value) => signal_domain(value),
            Self::MassFlowRate(value) => signal_domain(value),
            Self::VolumeFlowRate(value) => signal_domain(value),
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
            Self::HeatRate(value) => &value.shape,
            Self::ElectricPotential(value) => &value.shape,
            Self::ElectricCurrent(value) => &value.shape,
            Self::ElectricCharge(value) => &value.shape,
            Self::ElectricCurrentDensity(value) => &value.shape,
            Self::ElectricChargeDensity(value) => &value.shape,
            Self::MagneticVectorPotential(value) => &value.shape,
            Self::MassFlowRate(value) => &value.shape,
            Self::VolumeFlowRate(value) => &value.shape,
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
            Self::HeatRate(value) => value.frame.as_ref(),
            Self::ElectricPotential(value) => value.frame.as_ref(),
            Self::ElectricCurrent(value) => value.frame.as_ref(),
            Self::ElectricCharge(value) => value.frame.as_ref(),
            Self::ElectricCurrentDensity(value) => value.frame.as_ref(),
            Self::ElectricChargeDensity(value) => value.frame.as_ref(),
            Self::MagneticVectorPotential(value) => value.frame.as_ref(),
            Self::MassFlowRate(value) => value.frame.as_ref(),
            Self::VolumeFlowRate(value) => value.frame.as_ref(),
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
            Self::HeatRate(signal) => generator!(signal),
            Self::ElectricPotential(signal) => generator!(signal),
            Self::ElectricCurrent(signal) => generator!(signal),
            Self::ElectricCharge(signal) => generator!(signal),
            Self::ElectricCurrentDensity(signal) => generator!(signal),
            Self::ElectricChargeDensity(signal) => generator!(signal),
            Self::MagneticVectorPotential(signal) => generator!(signal),
            Self::MassFlowRate(signal) => generator!(signal),
            Self::VolumeFlowRate(signal) => generator!(signal),
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
            Self::HeatRate(signal) => signal.precision,
            Self::ElectricPotential(signal) => signal.precision,
            Self::ElectricCurrent(signal) => signal.precision,
            Self::ElectricCharge(signal) => signal.precision,
            Self::ElectricCurrentDensity(signal) => signal.precision,
            Self::ElectricChargeDensity(signal) => signal.precision,
            Self::MagneticVectorPotential(signal) => signal.precision,
            Self::MassFlowRate(signal) => signal.precision,
            Self::VolumeFlowRate(signal) => signal.precision,
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
