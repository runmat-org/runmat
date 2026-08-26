//! Native Java interoperability for RunMat.
//!
//! This crate owns JVM discovery, startup, classpath state, Java reflection,
//! and JNI object resources. Runtime session policy and RunMat values remain
//! owned by `runmat-runtime` and `runmat-value` respectively.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod classpath;
pub mod exception;
pub mod jvm;
pub mod object;
pub mod resolve;

pub const JAVA_ADAPTER_ID: &str = "java";
pub const JAVA_ADAPTER_VERSION: u32 = 1;

pub use classpath::{ClasspathError, ClasspathLayer, ClasspathSnapshot, SessionClasspath};
pub use exception::{JavaException, JavaStackFrame};
#[cfg(not(target_family = "wasm"))]
pub use jvm::{discover_jvm, JavaDiscoveryRequest, JvmInstallation, JvmProcess};
pub use jvm::{JvmConfig, JvmError, JvmVersion};
pub use object::{JavaObjectHandle, JavaObjectMetadata, JavaObjectRegistry, ObjectRegistryError};
pub use resolve::{
    select_overload, JavaArgumentType, JavaCallableCandidate, JavaParameterType, OverloadError,
    SelectedOverload,
};
