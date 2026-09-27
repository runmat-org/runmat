//! Request-local matrix execution structure. Scalar controls use exact IEEE
//! bits so signed zeros participate in query identity without lossy JSON floats.
use crate::{MatmulAffine, ScaleOp};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatmulScaleWork {
    pub input: usize,
    pub operation: ScaleOp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatmulWorkload {
    pub left: usize,
    pub right: usize,
    pub rows: u64,
    pub inner: u64,
    pub columns: u64,
    pub affine: MatmulAffine<u64>,
    pub row: Option<MatmulScaleWork>,
    pub column: Option<MatmulScaleWork>,
    pub clamp_min: Option<u64>,
    pub clamp_max: Option<u64>,
    pub exponent: Option<u64>,
    pub diagonal: bool,
}
