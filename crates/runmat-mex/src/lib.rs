//! MATLAB-compatible C Matrix, C MEX, and modern C++ MEX/Data API boundary.
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
    MexBuildError, MexBuildInvocation, MexBuildOutput, MexBuildPlan, MexBuildStep,
    MexSourceLanguage, MexTarget, MEX_ADAPTER_ID, MEX_ADAPTER_VERSION, MEX_ARTIFACT_SCHEMA_VERSION,
    MEX_EXTENSIONS,
};
pub use compatibility::{
    MexApiAvailability, MexApiSymbol, C_MATRIX_API, C_MEX_API, FORTRAN_MATRIX_API, FORTRAN_MEX_API,
};
pub use conversion::{value_from_mx, value_to_mx, MxConversionError, MxValueContext};
pub(crate) use conversion::{value_from_mx_in_context, value_to_mx_for_interface_in_context};
pub use host::{
    ConcurrentMexBoundaryHostServices, DirectMexBoundaryHostServices, MexAsyncHostServices,
    MexAsyncOperation, MexAsyncResult, MexBoundaryHostServices, MexCallState, MexCancellationScope,
    MexDiagnostic, MexEngineCompletion, MexHostApiV1, MexHostServices, UnavailableMexHostServices,
    MEX_HOST_ABI_VERSION,
};
pub use libmx::MxApi;
#[cfg(not(target_family = "wasm"))]
pub use loader::{MexInvocation, MexLoadError, MexModule, MexNativeInvocation};
pub use mxarray::{
    MxApiMode, MxArray, MxBoundaryInterface, MxClassId, MxComplexity, MxHandleToken, MxInterleaved,
    MxNumeric, MxSparse, MxSparseValues,
};
