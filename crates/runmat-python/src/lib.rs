//! Native CPython interoperability for RunMat.
//!
//! This crate owns interpreter discovery, dynamic stable-ABI loading, the
//! interpreter execution lane, Python object resources, and Python-native
//! errors. Runtime policy and conversion to or from RunMat values remain in
//! `runmat-runtime`.

#![deny(unsafe_op_in_unsafe_fn)]

mod artifact;
#[cfg(not(target_arch = "wasm32"))]
mod cpython;
mod environment;
mod error;
mod object;
#[cfg(not(target_arch = "wasm32"))]
mod session;
mod value;

#[cfg(not(target_arch = "wasm32"))]
pub use artifact::InstalledPythonArtifacts;
pub use artifact::{
    PythonArtifactBundle, PythonArtifactBundleEntry, PythonArtifactError, PythonArtifactIdentity,
    PythonEnvironmentIdentity, PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE, PYTHON_WHEEL_MEDIA_TYPE,
};
#[cfg(not(target_arch = "wasm32"))]
pub use environment::discover_python;
pub use environment::{
    PythonDiscoveryRequest, PythonExecutionMode, PythonInstallation, PythonStatus, PythonVersion,
};
pub use error::{PythonError, PythonFrame};
pub use object::{PythonCallbackInvocation, PythonObjectHandle, PythonObjectMetadata};
#[cfg(not(target_arch = "wasm32"))]
pub use session::{
    PythonCall, PythonLaneObservation, PythonLaneObserver, PythonReentryAuthority, PythonSession,
    PythonSessionConfig,
};
pub use value::{
    PythonArray, PythonBufferOwner, PythonDType, PythonDateTime, PythonTimeDelta, PythonValue,
};

pub const PYTHON_ADAPTER_ID: &str = "python";
pub const PYTHON_ADAPTER_VERSION: u32 = 1;
