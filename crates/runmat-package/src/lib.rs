//! Portable package-domain authority for RunMat.

pub mod error;
pub mod identity;
pub mod manifest;
pub mod policy;

pub use error::{IdentityError, ManifestError, PackageError};
pub use identity::{
    CanonicalPackageId, ContentDigest, DigestAlgorithm, GitCommitId, GitObjectAlgorithm,
    GitRepositoryUrl, GitSourceId, NormalizedRelativePath, PackageAlias, PackageInstanceId,
    PackageVersion, PathSourceId, RegistryId, RegistrySourceId, ServerProjectSourceId, SourceId,
};
pub use manifest::{
    DependencyGroup, DependencyLocator, DependencySpec, GitSelector, HostCapability,
    PackageManifest, PublicationDeclaration, RegistryDeclaration, SourceReplacement,
    TargetEnvironment, TargetPredicate,
};
