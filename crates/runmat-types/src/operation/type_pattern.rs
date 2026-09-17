use super::QuantityDimension;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AnalysisValueType {
    Geometry,
    GeometryFeature,
    Region,
    Material,
    Model,
    Study,
    Sweep,
    Field,
    History,
    AttributedExtremum,
    CaseTable,
    ExecutionProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperationTypePattern {
    Any,
    NumericScalar,
    Quantity(QuantityDimension),
    VectorQuantity {
        dimension: QuantityDimension,
        length: Option<usize>,
    },
    IntegerScalar,
    StringScalar,
    NumericArray,
    LogicalArray,
    Size,
    Prototype,
    Axes,
    Style,
    PropertyName,
    PropertyValue,
    Analysis(AnalysisValueType),
    List(Box<OperationTypePattern>),
}
