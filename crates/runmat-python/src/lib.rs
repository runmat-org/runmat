//! Native CPython interoperability for RunMat.
//!
//! This crate owns interpreter discovery, dynamic stable-ABI loading, the
//! interpreter execution lane, Python object resources, and Python-native
//! errors. Runtime policy and conversion to or from RunMat values remain in
//! `runmat-runtime`.

#![deny(unsafe_op_in_unsafe_fn)]

mod cpython;
mod environment;
mod error;
mod object;
mod session;
mod value;

pub use environment::{
    discover_python, PythonDiscoveryRequest, PythonExecutionMode, PythonInstallation, PythonStatus,
    PythonVersion,
};
pub use error::{PythonError, PythonFrame};
pub use object::{PythonCallbackInvocation, PythonObjectHandle, PythonObjectMetadata};
pub use session::{PythonCall, PythonSession, PythonSessionConfig};
pub use value::{PythonArray, PythonBufferOwner, PythonDType, PythonValue};

pub const PYTHON_ADAPTER_ID: &str = "python";
pub const PYTHON_ADAPTER_VERSION: u32 = 1;
