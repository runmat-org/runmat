mod artifacts;
mod client;
mod metadata;

pub use client::RegistryClient;
pub use metadata::{
    RegistryAdvisory, RegistryArtifact, RegistryCandidate, RegistryCandidateArtifact,
    RegistryCandidateList, RegistryCandidateOutcome, RegistryCandidateResponse,
    RegistryClientError, RegistryDependency, RegistryReleaseCore, RegistryReleaseMetadata,
    RegistryReleaseOutcome, RegistryReleaseResponse,
};
