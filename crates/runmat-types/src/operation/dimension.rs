use serde::{Deserialize, Serialize};

/// SI base-dimension exponents carried by statically checked operations.
///
/// Shared quantity descriptors carry nominal and semantic metadata separately. This composable form is
/// the authority for derived results such as a pressure integrated over an area or a resultant
/// moment about a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PhysicalDimension {
    pub length: i8,
    pub mass: i8,
    pub time: i8,
    pub current: i8,
    pub temperature: i8,
    pub amount: i8,
    pub luminous_intensity: i8,
    pub information: i8,
}

impl PhysicalDimension {
    pub const DIMENSIONLESS: Self = Self::new([0, 0, 0, 0, 0, 0, 0, 0]);
    pub const LENGTH: Self = Self::new([1, 0, 0, 0, 0, 0, 0, 0]);
    pub const AREA: Self = Self::new([2, 0, 0, 0, 0, 0, 0, 0]);
    pub const VOLUME: Self = Self::new([3, 0, 0, 0, 0, 0, 0, 0]);
    pub const TIME: Self = Self::new([0, 0, 1, 0, 0, 0, 0, 0]);
    pub const FREQUENCY: Self = Self::new([0, 0, -1, 0, 0, 0, 0, 0]);
    pub const MASS: Self = Self::new([0, 1, 0, 0, 0, 0, 0, 0]);
    pub const DENSITY: Self = Self::new([-3, 1, 0, 0, 0, 0, 0, 0]);
    pub const TEMPERATURE: Self = Self::new([0, 0, 0, 0, 1, 0, 0, 0]);
    pub const VELOCITY: Self = Self::new([1, 0, -1, 0, 0, 0, 0, 0]);
    pub const ACCELERATION: Self = Self::new([1, 0, -2, 0, 0, 0, 0, 0]);
    pub const FORCE: Self = Self::new([1, 1, -2, 0, 0, 0, 0, 0]);
    pub const MOMENT: Self = Self::new([2, 1, -2, 0, 0, 0, 0, 0]);
    pub const PRESSURE: Self = Self::new([-1, 1, -2, 0, 0, 0, 0, 0]);
    pub const ENERGY: Self = Self::new([2, 1, -2, 0, 0, 0, 0, 0]);
    pub const POWER: Self = Self::new([2, 1, -3, 0, 0, 0, 0, 0]);
    pub const HEAT_FLUX: Self = Self::new([0, 1, -3, 0, 0, 0, 0, 0]);
    pub const THERMAL_CONDUCTIVITY: Self = Self::new([1, 1, -3, 0, -1, 0, 0, 0]);
    pub const HEAT_TRANSFER_COEFFICIENT: Self = Self::new([0, 1, -3, 0, -1, 0, 0, 0]);
    pub const SPECIFIC_HEAT: Self = Self::new([2, 0, -2, 0, -1, 0, 0, 0]);
    pub const ELECTRICAL_CONDUCTIVITY: Self = Self::new([-3, -1, 3, 2, 0, 0, 0, 0]);
    pub const ELECTRIC_CURRENT: Self = Self::new([0, 0, 0, 1, 0, 0, 0, 0]);
    pub const ELECTRIC_POTENTIAL: Self = Self::new([2, 1, -3, -1, 0, 0, 0, 0]);
    pub const MAGNETIC_FLUX_DENSITY: Self = Self::new([0, 1, -2, -1, 0, 0, 0, 0]);
    pub const INFORMATION: Self = Self::new([0, 0, 0, 0, 0, 0, 0, 1]);

    pub const fn new(exponents: [i8; 8]) -> Self {
        Self {
            length: exponents[0],
            mass: exponents[1],
            time: exponents[2],
            current: exponents[3],
            temperature: exponents[4],
            amount: exponents[5],
            luminous_intensity: exponents[6],
            information: exponents[7],
        }
    }

    pub fn multiply(self, other: Self) -> Option<Self> {
        Some(Self {
            length: self.length.checked_add(other.length)?,
            mass: self.mass.checked_add(other.mass)?,
            time: self.time.checked_add(other.time)?,
            current: self.current.checked_add(other.current)?,
            temperature: self.temperature.checked_add(other.temperature)?,
            amount: self.amount.checked_add(other.amount)?,
            luminous_intensity: self
                .luminous_intensity
                .checked_add(other.luminous_intensity)?,
            information: self.information.checked_add(other.information)?,
        })
    }

    pub fn divide(self, other: Self) -> Option<Self> {
        self.multiply(other.power(-1)?)
    }

    pub fn power(self, exponent: i8) -> Option<Self> {
        Some(Self {
            length: self.length.checked_mul(exponent)?,
            mass: self.mass.checked_mul(exponent)?,
            time: self.time.checked_mul(exponent)?,
            current: self.current.checked_mul(exponent)?,
            temperature: self.temperature.checked_mul(exponent)?,
            amount: self.amount.checked_mul(exponent)?,
            luminous_intensity: self.luminous_intensity.checked_mul(exponent)?,
            information: self.information.checked_mul(exponent)?,
        })
    }
}
