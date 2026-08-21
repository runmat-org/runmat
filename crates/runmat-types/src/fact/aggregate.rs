use super::ValueFact;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellFact {
    /// Conservative common fact for arbitrary element access.
    pub element: Box<ValueFact>,
    /// Position-preserving facts when the cell contents are statically known.
    pub elements: Vec<ValueFact>,
    pub elements_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructFact {
    pub fields: BTreeMap<String, ValueFact>,
    pub fields_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputListFact {
    pub outputs: Vec<ValueFact>,
    pub variadic: bool,
}
