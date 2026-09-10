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
    /// Conservative, order-abstract field facts for arbitrary element access.
    pub fields: BTreeMap<String, ValueFact>,
    pub fields_complete: bool,
    /// Position-preserving field facts in column-major order when known.
    #[serde(skip)]
    pub elements: Vec<BTreeMap<String, ValueFact>>,
    #[serde(skip)]
    pub elements_complete: bool,
}

impl StructFact {
    pub fn scalar(fields: BTreeMap<String, ValueFact>, fields_complete: bool) -> Self {
        Self {
            elements: fields_complete
                .then(|| fields.clone())
                .into_iter()
                .collect(),
            elements_complete: fields_complete,
            fields,
            fields_complete,
        }
    }

    pub fn array(
        fields: BTreeMap<String, ValueFact>,
        fields_complete: bool,
        elements: Vec<BTreeMap<String, ValueFact>>,
        elements_complete: bool,
    ) -> Self {
        Self {
            fields,
            fields_complete,
            elements,
            elements_complete,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputListFact {
    pub outputs: Vec<ValueFact>,
    pub variadic: bool,
}
