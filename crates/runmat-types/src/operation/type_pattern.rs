use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AnalysisValueType {
    Geometry,
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
