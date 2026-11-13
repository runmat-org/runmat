//! Single semantic catalog for physical quantities. Equal dimensions do not imply equal kinds.
use super::{Quantity, QuantityStorage, QuantityValueError};
use crate::PhysicalDimension;
use serde::{Deserialize, Serialize};

pub(super) mod sealed {
    pub trait Sealed {}
    pub trait Spec {}
}
/// Sealed descriptor carried by the single quantity storage representation.
/// Static markers know a semantic kind; dynamic descriptors retain only facts actually known.
pub trait QuantitySpec: sealed::Spec + Copy + std::fmt::Debug + PartialEq {
    fn physical_dimension(self) -> PhysicalDimension;
    fn semantic_kind(self) -> Option<QuantityKind>;
    fn nominal_angle_power(self) -> i8;
}
pub trait QuantityMarker: sealed::Sealed + QuantitySpec + Copy + Send + Sync + 'static {
    const MARKER: Self;
    const KIND: QuantityKind;
    const CANONICAL_FIELD: &'static str;
    const AFFINE: bool = false;
}
macro_rules! catalog {
    ($( $name:ident, $field:literal, $dimension:expr, $affine:expr; )*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[cfg_attr(feature="tool-schema", derive(schemars::JsonSchema))]
        #[serde(rename_all="snake_case")]
        pub enum QuantityKind { $( $name, )* }
        impl QuantityKind {
            pub const fn dimension(self) -> PhysicalDimension { match self { $( Self::$name => PhysicalDimension::new($dimension), )* } }
            pub const fn angle_power(self) -> i8 { match self {
                Self::Angle | Self::Rotation | Self::ComplexPhase | Self::AngularVelocity | Self::AngularAcceleration | Self::AngularFrequency => 1,
                _ => 0,
            } }
            pub const fn canonical_field(self) -> &'static str { match self { $( Self::$name => $field, )* } }
            pub const fn permits_affine_unit(self) -> bool { match self { $( Self::$name => $affine, )* } }
        }
        #[derive(Debug,Clone,Copy,PartialEq,Serialize,Deserialize)]
        #[cfg_attr(feature="tool-schema",derive(schemars::JsonSchema))]
        #[serde(tag="kind",content="value",rename_all="snake_case",bound(serialize="S: Serialize + QuantityStorage",deserialize="S: Deserialize<'de> + QuantityStorage"))]
        pub enum AnyQuantity<S=f64> { $( $name(Quantity<kind::$name,S>), )* }
        impl<S:QuantityStorage> AnyQuantity<S> {
            pub fn try_new(kind:QuantityKind,value:S)->Result<Self,QuantityValueError>{Ok(match kind { $( QuantityKind::$name=>Self::$name(Quantity::try_new(value)?), )* })}
        }
        impl<S> AnyQuantity<S> {
            pub const fn kind(&self)->QuantityKind {match self { $( Self::$name(_)=>QuantityKind::$name, )* }}
            pub const fn dimension(&self)->PhysicalDimension {self.kind().dimension()}
            pub fn canonical_values(&self)->&S {match self { $( Self::$name(value)=>value.canonical_values(), )* }}
        }
        impl AnyQuantity {
            pub fn canonical_value(&self)->f64 {*self.canonical_values()}
            pub fn typed<K:QuantityMarker>(&self)->Option<Quantity<K>> { (self.kind()==K::KIND).then(||Quantity::new(self.canonical_value())) }
        }
        pub mod kind { $(
            #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            pub struct $name;
            impl super::sealed::Sealed for $name {}
            impl super::sealed::Spec for $name {}
            impl super::QuantitySpec for $name {
                fn physical_dimension(self) -> crate::PhysicalDimension { super::QuantityKind::$name.dimension() }
                fn semantic_kind(self) -> Option<super::QuantityKind> { Some(super::QuantityKind::$name) }
                fn nominal_angle_power(self) -> i8 { super::QuantityKind::$name.angle_power() }
            }
            impl super::QuantityMarker for $name {
                const MARKER: Self = Self;
                const KIND: super::QuantityKind=super::QuantityKind::$name;
                const CANONICAL_FIELD: &'static str=$field;
                const AFFINE: bool=$affine;
            }
        )* }
    }
}
catalog! {
    Length, "meters", [1, 0, 0, 0, 0, 0, 0, 0], false;
    Displacement, "meters", [1, 0, 0, 0, 0, 0, 0, 0], false;
    Position, "meters", [1, 0, 0, 0, 0, 0, 0, 0], false;
    Translation, "meters", [1, 0, 0, 0, 0, 0, 0, 0], false;
    Rotation, "radians", [0, 0, 0, 0, 0, 0, 0, 0], false;
    ComplexPhase, "radians", [0, 0, 0, 0, 0, 0, 0, 0], false;
    Temperature, "kelvin", [0, 0, 0, 0, 1, 0, 0, 0], true;
    TemperatureDifference, "kelvin", [0, 0, 0, 0, 1, 0, 0, 0], false;
    Pressure, "pascals", [-1, 1, -2, 0, 0, 0, 0, 0], false;
    ElectricPotential, "volts", [2, 1, -3, -1, 0, 0, 0, 0], false;
    ElectricCurrent, "amperes", [0, 0, 0, 1, 0, 0, 0, 0], false;
    ElectricCharge, "coulombs", [0, 0, 1, 1, 0, 0, 0, 0], false;
    MagneticVectorPotential, "webers_per_meter", [1, 1, -2, -1, 0, 0, 0, 0], false;
    Velocity, "meters_per_second", [1, 0, -1, 0, 0, 0, 0, 0], false;
    AccelerationPowerSpectralDensity, "square_meters_per_cubic_second", [2, 0, -3, 0, 0, 0, 0, 0], false;
    Acceleration, "meters_per_second_squared", [1, 0, -2, 0, 0, 0, 0, 0], false;
    AngularVelocity, "radians_per_second", [0, 0, -1, 0, 0, 0, 0, 0], false;
    AngularAcceleration, "radians_per_second_squared", [0, 0, -2, 0, 0, 0, 0, 0], false;
    Force, "newtons", [1, 1, -2, 0, 0, 0, 0, 0], false;
    Moment, "newton_meters", [2, 1, -2, 0, 0, 0, 0, 0], false;
    Traction, "pascals", [-1, 1, -2, 0, 0, 0, 0, 0], false;
    BodyForceDensity, "newtons_per_cubic_meter", [-2, 1, -2, 0, 0, 0, 0, 0], false;
    HeatFlux, "watts_per_square_meter", [0, 1, -3, 0, 0, 0, 0, 0], false;
    HeatGenerationDensity, "watts_per_cubic_meter", [-1, 1, -3, 0, 0, 0, 0, 0], false;
    HeatTransferCoefficient, "watts_per_square_meter_kelvin", [0, 1, -3, 0, -1, 0, 0, 0], false;
    SurfaceEmissivity, "fraction", [0, 0, 0, 0, 0, 0, 0, 0], false;
    SpecificAcousticImpedance, "pascal_seconds_per_meter", [-2, 1, -1, 0, 0, 0, 0, 0], false;
    MassFlowRate, "kilograms_per_second", [0, 1, -1, 0, 0, 0, 0, 0], false;
    VolumeFlowRate, "cubic_meters_per_second", [3, 0, -1, 0, 0, 0, 0, 0], false;
    ElectricCurrentDensity, "amperes_per_square_meter", [-2, 0, 0, 1, 0, 0, 0, 0], false;
    ElectricChargeDensity, "coulombs_per_cubic_meter", [-3, 0, 1, 1, 0, 0, 0, 0], false;
    Stiffness, "newtons_per_meter", [0, 1, -2, 0, 0, 0, 0, 0], false;
    Damping, "newton_seconds_per_meter", [0, 1, -1, 0, 0, 0, 0, 0], false;
    FoundationModulus, "newtons_per_cubic_meter", [-2, 1, -2, 0, 0, 0, 0, 0], false;
    RayleighMassCoefficient, "per_second", [0, 0, -1, 0, 0, 0, 0, 0], false;
    RayleighStiffnessCoefficient, "seconds", [0, 0, 1, 0, 0, 0, 0, 0], false;
    ThermalContactResistance, "square_meter_kelvin_per_watt", [0, -1, 3, 0, 1, 0, 0, 0], false;
    FrictionCoefficient, "ratio", [0, 0, 0, 0, 0, 0, 0, 0], false;
    Frequency, "hertz", [0, 0, -1, 0, 0, 0, 0, 0], false;
    AngularFrequency, "radians_per_second", [0, 0, -1, 0, 0, 0, 0, 0], false;
    MassDensity, "kilograms_per_cubic_meter", [-3, 1, 0, 0, 0, 0, 0, 0], false;
    DynamicViscosity, "pascal_seconds", [-1, 1, -1, 0, 0, 0, 0, 0], false;
    TurbulenceIntensity, "ratio", [0, 0, 0, 0, 0, 0, 0, 0], false;
    Area, "square_meters", [2, 0, 0, 0, 0, 0, 0, 0], false;
    Volume, "cubic_meters", [3, 0, 0, 0, 0, 0, 0, 0], false;
    SecondMomentOfArea, "meters_fourth", [4, 0, 0, 0, 0, 0, 0, 0], false;
    GeometricVolumeSecondMoment, "meters_fifth", [5, 0, 0, 0, 0, 0, 0, 0], false;
    SurfaceEnergyDensity, "joules_per_square_meter", [0, 1, -2, 0, 0, 0, 0, 0], false;
    TractionDisplacementStiffness, "pascals_per_meter", [-2, 1, -2, 0, 0, 0, 0, 0], false;
    Curvature, "per_meter", [-1, 0, 0, 0, 0, 0, 0, 0], false;
    TimeDerivativeRate, "per_second", [0, 0, -1, 0, 0, 0, 0, 0], false;
    Vorticity, "per_second", [0, 0, -1, 0, 0, 0, 0, 0], false;
    Time, "seconds", [0, 0, 1, 0, 0, 0, 0, 0], false;
    Mass, "kilograms", [0, 1, 0, 0, 0, 0, 0, 0], false;
    Energy, "joules", [2, 1, -2, 0, 0, 0, 0, 0], false;
    Power, "watts", [2, 1, -3, 0, 0, 0, 0, 0], false;
    ThermalConductivity, "watts_per_meter_kelvin", [1, 1, -3, 0, -1, 0, 0, 0], false;
    SpecificHeat, "joules_per_kilogram_kelvin", [2, 0, -2, 0, -1, 0, 0, 0], false;
    InverseTemperature, "per_kelvin", [0, 0, 0, 0, -1, 0, 0, 0], false;
    ElectricalConductivity, "siemens_per_meter", [-3, -1, 3, 2, 0, 0, 0, 0], false;
    ElectricPermittivity, "farads_per_meter", [-3, -1, 4, 2, 0, 0, 0, 0], false;
    MagneticPermeability, "henries_per_meter", [1, 1, -2, -2, 0, 0, 0, 0], false;
    EnergyDensity, "joules_per_cubic_meter", [-1, 1, -2, 0, 0, 0, 0, 0], false;
    TemperatureGradient, "kelvin_per_meter", [-1, 0, 0, 0, 1, 0, 0, 0], false;
    VolumetricHeatCapacity, "joules_per_cubic_meter_kelvin", [-1, 1, -2, 0, -1, 0, 0, 0], false;
    HeatCapacity, "joules_per_kelvin", [2, 1, -2, 0, -1, 0, 0, 0], false;
    ThermalConductance, "watts_per_kelvin", [2, 1, -3, 0, -1, 0, 0, 0], false;
    ElectricDisplacement, "coulombs_per_square_meter", [-2, 0, 1, 1, 0, 0, 0, 0], false;
    MagneticFieldIntensity, "amperes_per_meter", [-1, 0, 0, 1, 0, 0, 0, 0], false;
    MagneticFlux, "webers", [2, 1, -2, -1, 0, 0, 0, 0], false;
    AcousticIntensity, "watts_per_square_meter", [0, 1, -3, 0, 0, 0, 0, 0], false;
    Compressibility, "per_pascal", [1, -1, 2, 0, 0, 0, 0, 0], false;
    ReactivePower, "vars", [2, 1, -3, 0, 0, 0, 0, 0], false;
    ApparentPower, "volt_amperes", [2, 1, -3, 0, 0, 0, 0, 0], false;
    ElectricField, "volts_per_meter", [1, 1, -3, -1, 0, 0, 0, 0], false;
    MagneticFluxDensity, "teslas", [0, 1, -2, -1, 0, 0, 0, 0], false;
    Dimensionless, "ratio", [0, 0, 0, 0, 0, 0, 0, 0], false;
    Angle, "radians", [0, 0, 0, 0, 0, 0, 0, 0], false;
    YoungsModulus, "pascals", [-1, 1, -2, 0, 0, 0, 0, 0], false;
    PoissonRatio, "ratio", [0, 0, 0, 0, 0, 0, 0, 0], false;
    ThermalExpansion, "per_kelvin", [0, 0, 0, 0, -1, 0, 0, 0], false;
    SpeedOfSound, "meters_per_second", [1, 0, -1, 0, 0, 0, 0, 0], false;
    DampingRatio, "ratio", [0, 0, 0, 0, 0, 0, 0, 0], false;
    RelativePermittivity, "ratio", [0, 0, 0, 0, 0, 0, 0, 0], false;
    RelativePermeability, "ratio", [0, 0, 0, 0, 0, 0, 0, 0], false;
    YieldStress, "pascals", [-1, 1, -2, 0, 0, 0, 0, 0], false;
    HardeningModulus, "pascals", [-1, 1, -2, 0, 0, 0, 0, 0], false;
    ShearModulus, "pascals", [-1, 1, -2, 0, 0, 0, 0, 0], false;
    ShearCorrection, "ratio", [0, 0, 0, 0, 0, 0, 0, 0], false;
}

pub type Length = Quantity<kind::Length>;
pub type Displacement = Quantity<kind::Displacement>;
pub type Position = Quantity<kind::Position, [f64; 3]>;
pub type Translation = Quantity<kind::Translation, [f64; 3]>;
pub type Rotation = Quantity<kind::Rotation>;
pub type ComplexPhase = Quantity<kind::ComplexPhase>;
pub type Temperature = Quantity<kind::Temperature>;
pub type TemperatureDifference = Quantity<kind::TemperatureDifference>;
pub type Pressure = Quantity<kind::Pressure>;
pub type ElectricPotential = Quantity<kind::ElectricPotential>;
pub type ElectricCurrent = Quantity<kind::ElectricCurrent>;
pub type ElectricCharge = Quantity<kind::ElectricCharge>;
pub type MagneticVectorPotential = Quantity<kind::MagneticVectorPotential>;
pub type Velocity = Quantity<kind::Velocity>;
pub type Acceleration = Quantity<kind::Acceleration>;
pub type AngularVelocity = Quantity<kind::AngularVelocity>;
pub type AngularAcceleration = Quantity<kind::AngularAcceleration>;
pub type Force = Quantity<kind::Force>;
pub type Moment = Quantity<kind::Moment>;
pub type Traction = Quantity<kind::Traction>;
pub type BodyForceDensity = Quantity<kind::BodyForceDensity>;
pub type HeatFlux = Quantity<kind::HeatFlux>;
pub type HeatGenerationDensity = Quantity<kind::HeatGenerationDensity>;
pub type HeatTransferCoefficient = Quantity<kind::HeatTransferCoefficient>;
pub type SurfaceEmissivity = Quantity<kind::SurfaceEmissivity>;
pub type SpecificAcousticImpedance = Quantity<kind::SpecificAcousticImpedance>;
pub type MassFlowRate = Quantity<kind::MassFlowRate>;
pub type VolumeFlowRate = Quantity<kind::VolumeFlowRate>;
pub type ElectricCurrentDensity = Quantity<kind::ElectricCurrentDensity>;
pub type ElectricChargeDensity = Quantity<kind::ElectricChargeDensity>;
pub type Stiffness = Quantity<kind::Stiffness>;
pub type Damping = Quantity<kind::Damping>;
pub type FoundationModulus = Quantity<kind::FoundationModulus>;
pub type RayleighMassCoefficient = Quantity<kind::RayleighMassCoefficient>;
pub type RayleighStiffnessCoefficient = Quantity<kind::RayleighStiffnessCoefficient>;
pub type ThermalContactResistance = Quantity<kind::ThermalContactResistance>;
pub type FrictionCoefficient = Quantity<kind::FrictionCoefficient>;
pub type Frequency = Quantity<kind::Frequency>;
pub type MassDensity = Quantity<kind::MassDensity>;
pub type DynamicViscosity = Quantity<kind::DynamicViscosity>;
pub type TurbulenceIntensity = Quantity<kind::TurbulenceIntensity>;
pub type Area = Quantity<kind::Area>;
pub type Volume = Quantity<kind::Volume>;
pub type SecondMomentOfArea = Quantity<kind::SecondMomentOfArea>;
pub type Time = Quantity<kind::Time>;
pub type Mass = Quantity<kind::Mass>;
pub type Energy = Quantity<kind::Energy>;
pub type Power = Quantity<kind::Power>;
pub type ThermalConductivity = Quantity<kind::ThermalConductivity>;
pub type SpecificHeat = Quantity<kind::SpecificHeat>;
pub type InverseTemperature = Quantity<kind::InverseTemperature>;
pub type ElectricalConductivity = Quantity<kind::ElectricalConductivity>;
pub type ElectricField = Quantity<kind::ElectricField>;
pub type MagneticFluxDensity = Quantity<kind::MagneticFluxDensity>;
pub type Dimensionless = Quantity<kind::Dimensionless>;
pub type Angle = Quantity<kind::Angle>;
pub type YoungsModulus = Quantity<kind::YoungsModulus>;
pub type PoissonRatio = Quantity<kind::PoissonRatio>;
pub type ThermalExpansion = Quantity<kind::ThermalExpansion>;
pub type SpeedOfSound = Quantity<kind::SpeedOfSound>;
pub type DampingRatio = Quantity<kind::DampingRatio>;
pub type RelativePermittivity = Quantity<kind::RelativePermittivity>;
pub type RelativePermeability = Quantity<kind::RelativePermeability>;
pub type YieldStress = Quantity<kind::YieldStress>;
pub type HardeningModulus = Quantity<kind::HardeningModulus>;
pub type ShearModulus = Quantity<kind::ShearModulus>;
pub type ShearCorrection = Quantity<kind::ShearCorrection>;

/// Kinds admitting ordinary addition/subtraction; affine absolute values and
/// spatial positions require their explicitly named delta operations.
pub trait AdditiveQuantity: QuantityMarker {}
impl AdditiveQuantity for kind::Length {}
impl AdditiveQuantity for kind::Displacement {}
impl AdditiveQuantity for kind::Translation {}
impl AdditiveQuantity for kind::Rotation {}
impl AdditiveQuantity for kind::ComplexPhase {}
impl AdditiveQuantity for kind::TemperatureDifference {}
impl AdditiveQuantity for kind::Pressure {}
impl AdditiveQuantity for kind::ElectricPotential {}
impl AdditiveQuantity for kind::ElectricCurrent {}
impl AdditiveQuantity for kind::ElectricCharge {}
impl AdditiveQuantity for kind::MagneticVectorPotential {}
impl AdditiveQuantity for kind::Velocity {}
impl AdditiveQuantity for kind::Acceleration {}
impl AdditiveQuantity for kind::AngularVelocity {}
impl AdditiveQuantity for kind::AngularAcceleration {}
impl AdditiveQuantity for kind::Force {}
impl AdditiveQuantity for kind::Moment {}
impl AdditiveQuantity for kind::Traction {}
impl AdditiveQuantity for kind::BodyForceDensity {}
impl AdditiveQuantity for kind::HeatFlux {}
impl AdditiveQuantity for kind::HeatGenerationDensity {}
impl AdditiveQuantity for kind::HeatTransferCoefficient {}
impl AdditiveQuantity for kind::SurfaceEmissivity {}
impl AdditiveQuantity for kind::SpecificAcousticImpedance {}
impl AdditiveQuantity for kind::MassFlowRate {}
impl AdditiveQuantity for kind::VolumeFlowRate {}
impl AdditiveQuantity for kind::ElectricCurrentDensity {}
impl AdditiveQuantity for kind::ElectricChargeDensity {}
impl AdditiveQuantity for kind::Stiffness {}
impl AdditiveQuantity for kind::Damping {}
impl AdditiveQuantity for kind::FoundationModulus {}
impl AdditiveQuantity for kind::RayleighMassCoefficient {}
impl AdditiveQuantity for kind::RayleighStiffnessCoefficient {}
impl AdditiveQuantity for kind::ThermalContactResistance {}
impl AdditiveQuantity for kind::FrictionCoefficient {}
impl AdditiveQuantity for kind::Frequency {}
impl AdditiveQuantity for kind::MassDensity {}
impl AdditiveQuantity for kind::DynamicViscosity {}
impl AdditiveQuantity for kind::TurbulenceIntensity {}
impl AdditiveQuantity for kind::Area {}
impl AdditiveQuantity for kind::Volume {}
impl AdditiveQuantity for kind::SecondMomentOfArea {}
impl AdditiveQuantity for kind::Time {}
impl AdditiveQuantity for kind::Mass {}
impl AdditiveQuantity for kind::Energy {}
impl AdditiveQuantity for kind::Power {}
impl AdditiveQuantity for kind::ThermalConductivity {}
impl AdditiveQuantity for kind::SpecificHeat {}
impl AdditiveQuantity for kind::InverseTemperature {}
impl AdditiveQuantity for kind::ElectricalConductivity {}
impl AdditiveQuantity for kind::ElectricField {}
impl AdditiveQuantity for kind::MagneticFluxDensity {}
impl AdditiveQuantity for kind::Dimensionless {}
impl AdditiveQuantity for kind::Angle {}
impl AdditiveQuantity for kind::YoungsModulus {}
impl AdditiveQuantity for kind::PoissonRatio {}
impl AdditiveQuantity for kind::ThermalExpansion {}
impl AdditiveQuantity for kind::SpeedOfSound {}
impl AdditiveQuantity for kind::DampingRatio {}
impl AdditiveQuantity for kind::RelativePermittivity {}
impl AdditiveQuantity for kind::RelativePermeability {}
impl AdditiveQuantity for kind::YieldStress {}
impl AdditiveQuantity for kind::HardeningModulus {}
impl AdditiveQuantity for kind::ShearModulus {}
impl AdditiveQuantity for kind::ShearCorrection {}

pub type ElectricPermittivity = Quantity<kind::ElectricPermittivity>;
impl AdditiveQuantity for kind::ElectricPermittivity {}

pub type MagneticPermeability = Quantity<kind::MagneticPermeability>;
impl AdditiveQuantity for kind::MagneticPermeability {}

pub type EnergyDensity = Quantity<kind::EnergyDensity>;
impl AdditiveQuantity for kind::EnergyDensity {}

pub type TemperatureGradient = Quantity<kind::TemperatureGradient>;
impl AdditiveQuantity for kind::TemperatureGradient {}

pub type VolumetricHeatCapacity = Quantity<kind::VolumetricHeatCapacity>;
impl AdditiveQuantity for kind::VolumetricHeatCapacity {}

pub type HeatCapacity = Quantity<kind::HeatCapacity>;
impl AdditiveQuantity for kind::HeatCapacity {}

pub type ThermalConductance = Quantity<kind::ThermalConductance>;
impl AdditiveQuantity for kind::ThermalConductance {}

pub type AngularFrequency = Quantity<kind::AngularFrequency>;
impl AdditiveQuantity for kind::AngularFrequency {}

pub type GeometricVolumeSecondMoment = Quantity<kind::GeometricVolumeSecondMoment>;
impl AdditiveQuantity for kind::GeometricVolumeSecondMoment {}

pub type ElectricDisplacement = Quantity<kind::ElectricDisplacement>;
impl AdditiveQuantity for kind::ElectricDisplacement {}

pub type MagneticFieldIntensity = Quantity<kind::MagneticFieldIntensity>;
impl AdditiveQuantity for kind::MagneticFieldIntensity {}

pub type MagneticFlux = Quantity<kind::MagneticFlux>;
impl AdditiveQuantity for kind::MagneticFlux {}

pub type AcousticIntensity = Quantity<kind::AcousticIntensity>;
impl AdditiveQuantity for kind::AcousticIntensity {}

pub type Compressibility = Quantity<kind::Compressibility>;
impl AdditiveQuantity for kind::Compressibility {}

pub type ReactivePower = Quantity<kind::ReactivePower>;
impl AdditiveQuantity for kind::ReactivePower {}

pub type ApparentPower = Quantity<kind::ApparentPower>;
impl AdditiveQuantity for kind::ApparentPower {}

pub type TimeDerivativeRate = Quantity<kind::TimeDerivativeRate>;
impl AdditiveQuantity for kind::TimeDerivativeRate {}

/// Curl of a velocity field (twice the local rotation rate), distinct from rigid angular velocity.
pub type Vorticity = Quantity<kind::Vorticity>;
impl AdditiveQuantity for kind::Vorticity {}

pub type Curvature = Quantity<kind::Curvature>;
impl AdditiveQuantity for kind::Curvature {}

pub type SurfaceEnergyDensity = Quantity<kind::SurfaceEnergyDensity>;
impl AdditiveQuantity for kind::SurfaceEnergyDensity {}
pub type TractionDisplacementStiffness = Quantity<kind::TractionDisplacementStiffness>;
impl AdditiveQuantity for kind::TractionDisplacementStiffness {}

/// One-sided acceleration squared per ordinary hertz, not amplitude spectral density.
pub type AccelerationPowerSpectralDensity = Quantity<kind::AccelerationPowerSpectralDensity>;
