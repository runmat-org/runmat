use super::SignalShape;
use crate::PhysicalSignal;

impl PhysicalSignal {
    /// Clone a typed declaration with another declared shape, preserving every other
    /// contract field. This is not mesh resolution or validation; callers must authenticate
    /// and validate any resulting topology binding before numerical admission.
    pub fn with_shape(&self, shape: SignalShape) -> Self {
        macro_rules! replace {
            ($signal:expr, $variant:ident) => {{
                let mut signal = $signal.clone();
                signal.shape = shape;
                Self::$variant(signal)
            }};
        }
        match self {
            Self::Displacement(signal) => replace!(signal, Displacement),
            Self::Rotation(signal) => replace!(signal, Rotation),
            Self::Velocity(signal) => replace!(signal, Velocity),
            Self::Acceleration(signal) => replace!(signal, Acceleration),
            Self::AngularVelocity(signal) => replace!(signal, AngularVelocity),
            Self::AngularAcceleration(signal) => replace!(signal, AngularAcceleration),
            Self::Force(signal) => replace!(signal, Force),
            Self::Moment(signal) => replace!(signal, Moment),
            Self::Traction(signal) => replace!(signal, Traction),
            Self::Pressure(signal) => replace!(signal, Pressure),
            Self::BodyForceDensity(signal) => replace!(signal, BodyForceDensity),
            Self::Temperature(signal) => replace!(signal, Temperature),
            Self::TemperatureDifference(signal) => replace!(signal, TemperatureDifference),
            Self::HeatFlux(signal) => replace!(signal, HeatFlux),
            Self::HeatGenerationDensity(signal) => replace!(signal, HeatGenerationDensity),
            Self::HeatRate(signal) => replace!(signal, HeatRate),
            Self::ElectricPotential(signal) => replace!(signal, ElectricPotential),
            Self::ElectricCurrent(signal) => replace!(signal, ElectricCurrent),
            Self::ElectricCharge(signal) => replace!(signal, ElectricCharge),
            Self::ElectricCurrentDensity(signal) => replace!(signal, ElectricCurrentDensity),
            Self::ElectricChargeDensity(signal) => replace!(signal, ElectricChargeDensity),
            Self::MagneticVectorPotential(signal) => replace!(signal, MagneticVectorPotential),
            Self::MassFlowRate(signal) => replace!(signal, MassFlowRate),
            Self::VolumeFlowRate(signal) => replace!(signal, VolumeFlowRate),
        }
    }
}
