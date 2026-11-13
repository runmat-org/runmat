//! Deliberately admitted semantic operations. Dimensions are necessary but do
//! not choose between torque and work, or absolute and differential temperature.
use super::*;
use std::ops::{Add, Div, Mul, Sub};
impl Add<TemperatureDifference> for Temperature {
    type Output = Temperature;
    fn add(self, v: TemperatureDifference) -> Self::Output {
        Temperature::new(self.canonical_value() + v.canonical_value())
    }
}
impl Add<Temperature> for TemperatureDifference {
    type Output = Temperature;
    fn add(self, v: Temperature) -> Self::Output {
        v + self
    }
}
impl Sub<Temperature> for Temperature {
    type Output = TemperatureDifference;
    fn sub(self, v: Temperature) -> Self::Output {
        TemperatureDifference::new(self.canonical_value() - v.canonical_value())
    }
}
impl Sub<TemperatureDifference> for Temperature {
    type Output = Temperature;
    fn sub(self, v: TemperatureDifference) -> Self::Output {
        Temperature::new(self.canonical_value() - v.canonical_value())
    }
}
macro_rules! product {
    ($left:ty,$right:ty=>$out:ty) => {
        impl Mul<$right> for $left {
            type Output = $out;
            fn mul(self, rhs: $right) -> Self::Output {
                <$out>::new(self.canonical_value() * rhs.canonical_value())
            }
        }
        impl Mul<$left> for $right {
            type Output = $out;
            fn mul(self, rhs: $left) -> Self::Output {
                rhs * self
            }
        }
        impl Div<$left> for $out {
            type Output = $right;
            fn div(self, rhs: $left) -> Self::Output {
                <$right>::new(self.canonical_value() / rhs.canonical_value())
            }
        }
        impl Div<$right> for $out {
            type Output = $left;
            fn div(self, rhs: $right) -> Self::Output {
                <$left>::new(self.canonical_value() / rhs.canonical_value())
            }
        }
    };
}
product!(HeatFlux,Area=>Power);
product!(HeatGenerationDensity,Volume=>Power);
product!(HeatTransferCoefficient,TemperatureDifference=>HeatFlux);
product!(ElectricCurrent,ElectricPotential=>Power);
product!(ElectricCurrentDensity,Area=>ElectricCurrent);
product!(ElectricalConductivity,ElectricField=>ElectricCurrentDensity);
product!(MassDensity,Volume=>Mass);
product!(Power,Time=>Energy);
impl Force {
    pub fn work_over(self, displacement: Displacement) -> Energy {
        Energy::new(self.canonical_value() * displacement.canonical_value())
    }
    pub fn moment_with_arm(self, arm: Length) -> Moment {
        Moment::new(self.canonical_value() * arm.canonical_value())
    }
}
impl Moment {
    /// Rotational work is a semantic contraction, not generic dimension guessing.
    pub fn work_over(self, rotation: Rotation) -> Energy {
        Energy::new(self.canonical_value() * rotation.canonical_value())
    }
}
impl<K: QuantityMarker> Quantity<K> {
    /// A ratio uses the same semantic quantity kind on both sides. In particular
    /// an energy cannot be used as a torque normalization reference.
    pub fn ratio_to(self, reference: Self) -> Dimensionless {
        Dimensionless::new(self.canonical_value() / reference.canonical_value())
    }
}

/// Hertz counts full cycles; angular frequency retains radians explicitly.
impl Frequency {
    pub fn angular_frequency(self) -> AngularFrequency {
        AngularFrequency::new(self.canonical_value() * std::f64::consts::TAU)
    }
}
impl AngularFrequency {
    pub fn frequency(self) -> Frequency {
        Frequency::new(self.canonical_value() / std::f64::consts::TAU)
    }
}
product!(AngularFrequency,Time=>Angle);

impl AngularFrequency {
    /// The exp(i*phase) generator contributes inverse radians. Its action
    /// cancels omega's nominal angle and yields the real magnitude of a time
    /// derivative coefficient. This is not Hertz and must not be relabelled as
    /// cycles/second: Frequency -> AngularFrequency multiplies 2π once before
    /// this conversion. The complex kernel separately supplies the factor i.
    pub fn phasor_derivative_rate(self) -> super::TimeDerivativeRate {
        super::TimeDerivativeRate::new(self.canonical_value())
    }
}
product!(TimeDerivativeRate,Time=>Dimensionless);
