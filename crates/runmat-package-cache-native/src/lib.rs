//! Native adapters for the portable RunMat package cache.

pub mod backend;
pub mod concurrency;
pub mod config;
pub mod error;
pub mod filesystem;
pub mod gc;
pub mod git;
pub mod materialize;
pub mod project;
mod provider;
pub mod registry;
pub mod server;

pub use backend::SqliteCacheBackend;
pub use concurrency::NativeCacheLease;
pub use config::NativeCacheConfig;
pub use error::NativeCacheError;
pub use project::{
    resolve_native_project, NativeProjectError, NativeProjectResolveRequest, NativeResolvedProject,
};
pub use provider::{NativeGitPackageProvider, NativePackageSourceProvider};
