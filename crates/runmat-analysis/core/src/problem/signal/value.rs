//! Solver-facing canonical values, independent of runtime numeric-array storage.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalValue<Q> {
    Scalar(Q),
    Vector3([Q; 3]),
    /// Canonical row-major values: the last shape axis varies fastest.
    Tensor {
        #[serde(with = "runmat_types::wire_integer::vec")]
        shape: Vec<usize>,
        values: Vec<Q>,
    },
}
