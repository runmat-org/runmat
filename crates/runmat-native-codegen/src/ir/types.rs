use runmat_types::ValueFact;
use serde::{Deserialize, Serialize};

/// Native IR never reinfers types. A value is either deliberately generic or
/// carries the exact shared fact supplied by canonical MIR analysis.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "fact")]
pub enum NativeValueType {
    Generic,
    Analyzed(Box<ValueFact>),
}
