use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IdentityError {
    #[error("invalid {kind} `{value}`: {reason}")]
    InvalidName {
        kind: &'static str,
        value: String,
        reason: &'static str,
    },
    #[error("invalid digest `{value}`: {reason}")]
    InvalidDigest { value: String, reason: &'static str },
    #[error("invalid version `{value}`: {reason}")]
    InvalidVersion { value: String, reason: String },
    #[error("invalid relative package path `{value}`: {reason}")]
    InvalidRelativePath { value: String, reason: &'static str },
    #[error("invalid Git source `{value}`: {reason}")]
    InvalidGitSource { value: String, reason: &'static str },
    #[error("invalid Git object id `{value}`: {reason}")]
    InvalidGitObjectId { value: String, reason: &'static str },
    #[error("invalid Server project source `{value}`: {reason}")]
    InvalidServerProjectSource { value: String, reason: &'static str },
}

#[derive(Debug, Error)]
pub enum PackageError {
    #[error(transparent)]
    Identity(#[from] IdentityError),
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(transparent)]
    Lock(#[from] LockError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Resolve(#[from] ResolveError),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResolveError {
    #[error("{0}")]
    Conflict(String),
    #[error("candidate metadata provider failed: {0}")]
    Provider(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GraphError {
    #[error("invalid package graph: {0}")]
    Invalid(String),
    #[error(
        "package graph requires unavailable capabilities at {dependency_path}: {capabilities}"
    )]
    UnavailableCapabilities {
        dependency_path: String,
        capabilities: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ManifestError {
    #[error("invalid dependency `{alias}` in [{table}]: {reason}")]
    InvalidDependency {
        table: String,
        alias: String,
        reason: String,
    },
    #[error("invalid package metadata: {0}")]
    InvalidPackage(String),
    #[error("invalid feature `{feature}`: {reason}")]
    InvalidFeature { feature: String, reason: String },
    #[error("invalid capability `{0}`")]
    InvalidCapability(String),
    #[error("invalid target predicate `{value}`: {reason}")]
    InvalidTarget { value: String, reason: String },
    #[error("invalid registry `{registry}`: {reason}")]
    InvalidRegistry { registry: String, reason: String },
    #[error("invalid source replacement `{registry}`: {reason}")]
    InvalidSourceReplacement { registry: String, reason: String },
}

#[derive(Debug, Error)]
pub enum LockError {
    #[error("failed to decode runmat.lock: {0}")]
    Decode(#[from] toml::de::Error),
    #[error("failed to encode runmat.lock: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("invalid runmat.lock: {0}")]
    Invalid(String),
    #[error("incompatible runmat.lock: {0}")]
    Incompatible(String),
}
