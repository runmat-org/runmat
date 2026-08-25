//! MATLAB-compatible C Matrix and MEX boundary.
//!
//! `runmat-mex` owns compatibility objects and adapter lifecycle. It does not
//! define a second RunMat value model: every call copies between the canonical
//! [`runmat_value::Value`] representation and a call-owned [`MxArray`] arena.

pub mod arena;
pub mod build;
pub mod conversion;
pub mod host;
pub mod libmx;
#[cfg(not(target_family = "wasm"))]
pub mod loader;
pub mod mxarray;

pub use arena::{MxArena, MxArenaError};
pub use build::{
    mex_suffix, MexArgumentError, MexBuild, MexBuildError, MexBuildInvocation, MexBuildOutput,
    MexBuildPlan, MEX_EXTENSIONS,
};
pub use conversion::{value_from_mx, value_to_mx, MxConversionError};
pub use host::{
    MexCallState, MexDiagnostic, MexHostApiV1, MexHostServices, UnavailableMexHostServices,
    MEX_HOST_ABI_VERSION,
};
pub use libmx::MxApi;
#[cfg(not(target_family = "wasm"))]
pub use loader::{MexInvocation, MexLoadError, MexModule};
pub use mxarray::{
    MxApiMode, MxArray, MxClassId, MxComplexity, MxInterleaved, MxNumeric, MxSparse, MxSparseValues,
};
