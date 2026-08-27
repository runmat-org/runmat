//! MATLAB-compatible C Matrix and MEX boundary.
//!
//! `runmat-mex` owns compatibility objects and adapter lifecycle. It does not
//! define a second RunMat value model. Compatible in-process arrays retain the
//! canonical [`runmat_value::Value`] host allocation through invocation leases;
//! incompatible layouts cross through explicit, accounted conversions.

pub mod arena;
pub mod build;
pub mod compatibility;
pub mod conversion;
pub mod host;
pub mod libmx;
#[cfg(not(target_family = "wasm"))]
pub mod loader;
pub mod mxarray;

pub use arena::{MxArena, MxArenaError};
pub use build::{
    mex_suffix, MexApi, MexArgumentError, MexArtifactIdentity, MexArtifactManifest, MexBuild,
    MexBuildError, MexBuildInvocation, MexBuildOutput, MexBuildPlan, MexTarget, MEX_ADAPTER_ID,
    MEX_ADAPTER_VERSION, MEX_ARTIFACT_SCHEMA_VERSION, MEX_EXTENSIONS,
};
pub use compatibility::{MexApiAvailability, MexApiSymbol, C_MATRIX_API, C_MEX_API};
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
