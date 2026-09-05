use serde::{Deserialize, Serialize};

pub trait FiniteQuantity {
    fn is_finite(&self) -> bool;
}

macro_rules! scalar_quantity {
    ($name:ident, $field:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name {
            pub $field: f64,
        }

        impl $name {
            pub const fn new($field: f64) -> Self {
                Self { $field }
            }

            pub fn is_finite(self) -> bool {
                self.$field.is_finite()
            }
        }

        impl FiniteQuantity for $name {
            fn is_finite(&self) -> bool {
                self.$field.is_finite()
            }
        }
    };
}

scalar_quantity!(Length, meters);
scalar_quantity!(Displacement, meters);
scalar_quantity!(Rotation, radians);
scalar_quantity!(Temperature, kelvin);
scalar_quantity!(TemperatureDifference, kelvin);
scalar_quantity!(Pressure, pascals);
scalar_quantity!(ElectricPotential, volts);
scalar_quantity!(ElectricCurrent, amperes);
scalar_quantity!(ElectricCharge, coulombs);
scalar_quantity!(MagneticVectorPotential, webers_per_meter);
scalar_quantity!(Velocity, meters_per_second);
scalar_quantity!(Acceleration, meters_per_second_squared);
scalar_quantity!(AngularVelocity, radians_per_second);
scalar_quantity!(AngularAcceleration, radians_per_second_squared);
scalar_quantity!(Force, newtons);
scalar_quantity!(Moment, newton_meters);
scalar_quantity!(Traction, pascals);
scalar_quantity!(BodyForceDensity, newtons_per_cubic_meter);
scalar_quantity!(HeatFlux, watts_per_square_meter);
scalar_quantity!(HeatGenerationDensity, watts_per_cubic_meter);
scalar_quantity!(HeatTransferCoefficient, watts_per_square_meter_kelvin);
scalar_quantity!(SpecificAcousticImpedance, pascal_seconds_per_meter);
scalar_quantity!(MassFlowRate, kilograms_per_second);
scalar_quantity!(ElectricCurrentDensity, amperes_per_square_meter);
scalar_quantity!(ElectricChargeDensity, coulombs_per_cubic_meter);
scalar_quantity!(Stiffness, newtons_per_meter);
scalar_quantity!(Damping, newton_seconds_per_meter);
scalar_quantity!(ThermalContactResistance, square_meter_kelvin_per_watt);
scalar_quantity!(FrictionCoefficient, ratio);
scalar_quantity!(Frequency, hertz);
scalar_quantity!(MassDensity, kilograms_per_cubic_meter);
scalar_quantity!(DynamicViscosity, pascal_seconds);
scalar_quantity!(TurbulenceIntensity, ratio);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Translation {
    pub meters: [f64; 3],
}

impl Translation {
    pub const fn new(meters: [f64; 3]) -> Self {
        Self { meters }
    }

    pub fn is_finite(self) -> bool {
        self.meters.into_iter().all(f64::is_finite)
    }
}

impl FiniteQuantity for Translation {
    fn is_finite(&self) -> bool {
        self.meters.into_iter().all(f64::is_finite)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComplexPhase {
    pub radians: f64,
}

impl ComplexPhase {
    pub const fn new(radians: f64) -> Self {
        Self { radians }
    }

    pub fn is_finite(self) -> bool {
        self.radians.is_finite()
    }
}

impl FiniteQuantity for ComplexPhase {
    fn is_finite(&self) -> bool {
        self.radians.is_finite()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Position {
    pub meters: [f64; 3],
}

impl Position {
    pub const fn new(meters: [f64; 3]) -> Self {
        Self { meters }
    }

    pub fn is_finite(self) -> bool {
        self.meters.into_iter().all(f64::is_finite)
    }
}

impl FiniteQuantity for Position {
    fn is_finite(&self) -> bool {
        self.meters.into_iter().all(f64::is_finite)
    }
}
