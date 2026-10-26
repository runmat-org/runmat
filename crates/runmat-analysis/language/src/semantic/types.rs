use crate::FeaSymbolKind;
use runmat_analysis_core::{FieldLocation, ObservationQuantity};
pub use runmat_types::PhysicalDimension;
use std::collections::BTreeMap;

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
    Quantity(runmat_types::physical::DynamicDimension),
    Vector {
        element: Box<FeaType>,
        length: Option<usize>,
    },
    List(Box<FeaType>),
    Record(BTreeMap<String, FeaType>),
    GeometryFeature,
    Region(FeaTopologyRole),
    Section,
    Layup,
    Orientation,
    Constraint,
    Load,
    LoadCase,
    Interface,
    Connector,
    Signal,
    Excitation,
    ResponseSpectrum,
    RandomExcitation,
    InitialCondition,
    Declaration(FeaSymbolKind),
    Field(FeaFieldType),
    History(FeaHistoryType),
    AttributedExtremum(Box<FeaType>),
    Envelope(Box<FeaType>),
    CaseTable(Box<FeaType>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaFieldType {
    pub value: Box<FeaType>,
    pub location: Option<FieldLocation>,
    pub quantity: Option<ObservationQuantity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaHistoryType {
    pub sample: Box<FeaType>,
    pub domain: Option<FeaSeriesDomain>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeaSeriesDomain {
    Time,
    Frequency,
    LoadFactor,
    StudyParameter(String),
    Mode,
}

impl FeaType {
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::Integer | Self::Real | Self::Complex | Self::Quantity(_)
        )
    }

    pub fn is_quantity(&self, expected: runmat_types::physical::DynamicDimension) -> bool {
        matches!(self, Self::Quantity(actual) if actual.is_compatible_with(expected))
    }
    pub fn quantity_descriptor(&self) -> Option<runmat_types::physical::DynamicDimension> {
        match self {
            Self::Quantity(spec) => Some(*spec),
            Self::Vector { element, .. } => element.quantity_descriptor(),
            _ => None,
        }
    }
    pub fn with_quantity_kind(&self, kind: runmat_types::physical::QuantityKind) -> Option<Self> {
        Some(match self {
            Self::Quantity(spec) => Self::Quantity(spec.with_semantic_kind(Some(kind)).ok()?),
            Self::Vector { element, length } => Self::Vector {
                element: Box::new(element.with_quantity_kind(kind)?),
                length: *length,
            },
            _ => return None,
        })
    }
    pub fn accepts(&self, value: &Self) -> bool {
        self == value
            || matches!((self, value), (Self::Quantity(a),Self::Quantity(b)) if a.is_compatible_with(*b))
            || matches!(
                (self, value),
                (Self::Real, Self::Integer) | (Self::Complex, Self::Integer | Self::Real)
            )
            // A bare number is a dimensionless quantity (emissivity, ratios, factors).
            || matches!((self, value), (Self::Quantity(dimension), Self::Integer | Self::Real)
                if dimension.dimension() == PhysicalDimension::DIMENSIONLESS)
            || matches!(self, Self::Unknown | Self::Error)
            || matches!(value, Self::Unknown | Self::Error)
    }
}

pub use runmat_types::physical::UnitDefinition as ParsedUnit;
