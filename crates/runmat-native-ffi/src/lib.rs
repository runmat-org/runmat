//! Native shared-library interoperability for RunMat.
//!
//! Prototype metadata is portable and safe to inspect on every target. Native
//! library loading, calling-convention work, and raw pointer access are kept in
//! target-gated modules with small audited `unsafe` boundaries.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod metadata;
pub mod model;

#[cfg(not(target_family = "wasm"))]
pub mod invoke;
#[cfg(not(target_family = "wasm"))]
pub mod loader;
#[cfg(not(target_family = "wasm"))]
pub mod value;

#[cfg(not(target_family = "wasm"))]
pub use loader::{LoadedLibrary, LoadedSymbol, LoaderError};

#[cfg(not(target_family = "wasm"))]
pub use invoke::{
    invoke_symbol, invoke_symbol_with_bindings, invoke_symbol_with_callbacks, CallbackBinding,
    CallbackDispatch, InvocationError, InvocationResult, InvocationValue, NativePointerResource,
    PointerBinding,
};
pub use metadata::{
    artifact_identity, normalize_metadata, validate_metadata, MetadataError,
    NativeInterfaceArtifactBundle, NativeInterfaceArtifactBundleEntry,
    NativeInterfaceArtifactError, NativeInterfaceArtifactIdentity, NativeInterfaceArtifactManifest,
    NativeLibraryMetadata, NATIVE_FFI_ADAPTER_ID, NATIVE_FFI_ADAPTER_VERSION,
    NATIVE_FFI_ARTIFACT_SCHEMA_VERSION, NATIVE_FFI_METADATA_SCHEMA_VERSION,
    NATIVE_INTERFACE_BUNDLE_SCHEMA_VERSION, NATIVE_INTERFACE_MANIFEST_MEDIA_TYPE,
    NATIVE_LIBRARY_MEDIA_TYPE,
};
#[cfg(not(target_family = "wasm"))]
pub use metadata::{prepare_header, HeaderPreparation, HeaderPreparationError};
pub use model::*;
#[cfg(not(target_family = "wasm"))]
pub use value::NativePointer;
