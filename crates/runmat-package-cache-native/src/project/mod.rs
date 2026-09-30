mod captured;
pub use captured::{resolve_native_captured_project, NativeCapturedProject};
mod service;

pub use service::{
    resolve_native_project, NativeProjectError, NativeProjectResolveRequest, NativeResolvedProject,
};
