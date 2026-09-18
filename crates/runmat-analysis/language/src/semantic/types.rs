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
    Quantity(PhysicalDimension),
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
    InitialCondition,
    Declaration(FeaSymbolKind),
    Field(FeaFieldType),
    History(Box<FeaType>),
    AttributedExtremum(Box<FeaType>),
    CaseTable(Box<FeaType>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaFieldType {
    pub value: Box<FeaType>,
    pub location: Option<FieldLocation>,
    pub quantity: Option<ObservationQuantity>,
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
