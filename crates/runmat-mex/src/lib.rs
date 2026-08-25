//! MATLAB-compatible C Matrix and MEX boundary.
//!
//! `runmat-mex` owns compatibility objects and adapter lifecycle. It does not
//! define a second RunMat value model: every call copies between the canonical
//! [`runmat_value::Value`] representation and a call-owned [`MxArray`] arena.

pub mod arena;
pub mod conversion;
pub mod mxarray;

pub use arena::{MxArena, MxArenaError};
pub use conversion::{value_from_mx, value_to_mx, MxConversionError};
pub use mxarray::{
    MxApiMode, MxArray, MxClassId, MxComplexity, MxInterleaved, MxNumeric, MxSparse, MxSparseValues,
};
