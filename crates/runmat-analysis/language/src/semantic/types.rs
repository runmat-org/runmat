use crate::FeaSymbolKind;
use runmat_types::QuantityDimension;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    pub const SPECIFIC_HEAT: Self = Self::new([2, 0, -2, 0, -1, 0, 0, 0]);
    pub const ELECTRICAL_CONDUCTIVITY: Self = Self::new([-3, -1, 3, 2, 0, 0, 0, 0]);
    pub const ELECTRIC_CURRENT: Self = Self::new([0, 0, 0, 1, 0, 0, 0, 0]);
    pub const ELECTRIC_POTENTIAL: Self = Self::new([2, 1, -3, -1, 0, 0, 0, 0]);
    pub const MAGNETIC_FLUX_DENSITY: Self = Self::new([0, 1, -2, -1, 0, 0, 0, 0]);
    pub const INFORMATION: Self = Self::new([0, 0, 0, 0, 0, 0, 0, 1]);

    const fn new(exponents: [i8; 8]) -> Self {
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

impl From<QuantityDimension> for PhysicalDimension {
    fn from(value: QuantityDimension) -> Self {
        match value {
            QuantityDimension::Dimensionless => Self::DIMENSIONLESS,
            QuantityDimension::Length => Self::LENGTH,
            QuantityDimension::Area => Self::AREA,
            QuantityDimension::Volume => Self::VOLUME,
            QuantityDimension::Time => Self::TIME,
            QuantityDimension::Frequency => Self::FREQUENCY,
            QuantityDimension::Mass => Self::MASS,
            QuantityDimension::Density => Self::DENSITY,
            QuantityDimension::Temperature => Self::TEMPERATURE,
            QuantityDimension::Velocity => Self::VELOCITY,
            QuantityDimension::Acceleration => Self::ACCELERATION,
            QuantityDimension::Force => Self::FORCE,
            QuantityDimension::Moment => Self::MOMENT,
            QuantityDimension::Pressure => Self::PRESSURE,
            QuantityDimension::Energy => Self::ENERGY,
            QuantityDimension::Power => Self::POWER,
            QuantityDimension::HeatFlux => Self::HEAT_FLUX,
            QuantityDimension::ThermalConductivity => Self::THERMAL_CONDUCTIVITY,
            QuantityDimension::ElectricPotential => Self::ELECTRIC_POTENTIAL,
            QuantityDimension::ElectricCurrent => Self::ELECTRIC_CURRENT,
            QuantityDimension::MagneticFluxDensity => Self::MAGNETIC_FLUX_DENSITY,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FeaTopologyRole {
    Vertex,
    Edge,
    Face,
    Solid,
    Cell,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeaType {
    Unknown,
    Error,
    Boolean,
    Integer,
    Real,
    Complex,
    String,
    Enum,
    Quantity(PhysicalDimension),
    Vector {
        element: Box<FeaType>,
        length: Option<usize>,
    },
    List(Box<FeaType>),
    Record(BTreeMap<String, FeaType>),
    GeometryFeature,
    Region(FeaTopologyRole),
    Declaration(FeaSymbolKind),
    Field(Box<FeaType>),
    History(Box<FeaType>),
    AttributedExtremum(Box<FeaType>),
    CaseTable(Box<FeaType>),
}

impl FeaType {
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::Integer | Self::Real | Self::Complex | Self::Quantity(_)
        )
    }

    pub fn accepts(&self, value: &Self) -> bool {
        self == value
            || matches!(
                (self, value),
                (Self::Real, Self::Integer) | (Self::Complex, Self::Integer | Self::Real)
            )
            || matches!(self, Self::Unknown | Self::Error)
            || matches!(value, Self::Unknown | Self::Error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedUnit {
    pub dimension: PhysicalDimension,
    pub scale_bits: u64,
    pub offset_bits: u64,
}

impl ParsedUnit {
    pub fn scale(self) -> f64 {
        f64::from_bits(self.scale_bits)
    }

    pub fn offset(self) -> f64 {
        f64::from_bits(self.offset_bits)
    }
}
