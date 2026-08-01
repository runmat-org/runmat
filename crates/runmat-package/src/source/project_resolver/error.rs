use crate::{GitPolicyError, GraphError, LockError};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProjectResolveError {
    #[error("failed to load project manifest {path}: {reason}")]
    Manifest { path: PathBuf, reason: String },
    #[error("failed to build source inventory for package `{package}`: {reason}")]
    SourceInventory { package: String, reason: String },
    #[error("failed to read source {path}: {reason}")]
    SourceRead { path: PathBuf, reason: String },
    #[error("dependency `{dependency}` in package `{package}` points to missing manifest {path}")]
    MissingManifest {
        package: String,
        dependency: String,
        path: PathBuf,
    },
    #[error("dependency cycle detected: {cycle}")]
    Cycle { cycle: String },
    #[error("Git acquisition failed for dependency `{dependency}` of `{package}`: {reason}")]
    GitAcquire {
        package: String,
        dependency: String,
        reason: String,
    },
    #[error("dependency source `{kind}` is not implemented by this resolver")]
    UnsupportedSource { kind: &'static str },
    #[error("dependency `{dependency}` of `{package}` requires {requirement}, but package `{target}` {actual}")]
    Version {
        package: String,
        dependency: String,
        requirement: String,
        target: String,
        actual: String,
    },
    #[error("invalid package project: {0}")]
    Invalid(String),
    #[error(transparent)]
    GitPolicy(#[from] GitPolicyError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Lock(#[from] LockError),
}
