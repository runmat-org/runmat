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
