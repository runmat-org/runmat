use std::str::FromStr;
use std::sync::Arc;

use runmat_package::{
    CanonicalPackageId, ContentDigest, DependencyGroup, PackageVersion, RegistryAcquisitionPlan,
    RegistryCandidatePlan, RegistryCandidateRecord, RegistryOrigin, RegistryPackageReference,
    RegistryReleaseDependency, RegistryReleaseId, RegistryReleaseMetadata, RegistrySourceId,
};
use runmat_server_client::packages::{
    RegistryCandidate, RegistryCandidateOutcome, RegistryClient, RegistryClientError,
    RegistryReleaseCore, RegistryReleaseMetadata as RegistryReleaseResponse,
    RegistryReleaseOutcome,
};

use crate::registry::{RegistryArtifactTransfer, RegistryTransport};

use super::AccessTokenProvider;

const MAX_REGISTRY_ARTIFACT_BYTES: u64 = 6 * 1024 * 1024 * 1024;

pub struct HttpRegistryTransport {
    credentials: Arc<dyn AccessTokenProvider>,
}

impl HttpRegistryTransport {
    pub fn new(credentials: Arc<dyn AccessTokenProvider>) -> Self {
        Self { credentials }
    }
}

impl RegistryTransport for HttpRegistryTransport {
    fn candidates<'a>(
        &'a self,
        plan: &'a RegistryCandidatePlan,
    ) -> futures::future::LocalBoxFuture<'a, Result<Vec<RegistryCandidateRecord>, String>> {
        Box::pin(async move {
            let mut credentials = self.credentials.snapshot(&plan.index).await?;
            let response = match candidates(plan, credentials.token()).await {
                Err(RegistryClientError::Unauthorized | RegistryClientError::Forbidden) => {
                    credentials = self
                        .credentials
                        .refresh_after_rejection(&plan.index, credentials.generation)
                        .await?;
                    candidates(plan, credentials.token()).await
                }
                response => response,
            }
            .map_err(|error| error.to_string())?;
            response
                .iter()
                .map(|candidate| candidate_record(&plan.package, candidate))
                .collect()
        })
    }

    fn fetch<'a>(
        &'a self,
        plan: &'a RegistryAcquisitionPlan,
    ) -> futures::future::LocalBoxFuture<'a, Result<RegistryArtifactTransfer, String>> {
        Box::pin(async move {
            let mut credentials = self.credentials.snapshot(&plan.index).await?;
            match fetch(plan, credentials.token()).await {
                Err(RegistryClientError::Unauthorized | RegistryClientError::Forbidden) => {
                    credentials = self
                        .credentials
                        .refresh_after_rejection(&plan.index, credentials.generation)
                        .await?;
                    fetch(plan, credentials.token()).await
                }
                response => response,
            }
            .map_err(|error| error.to_string())
        })
    }
}

async fn candidates(
    plan: &RegistryCandidatePlan,
    token: Option<&str>,
) -> Result<Vec<RegistryCandidate>, RegistryClientError> {
    let outcome = RegistryClient::new(&plan.index, token.map(str::to_owned))?
        .candidates(plan.package.organization(), plan.package.name(), None)
        .await?;
    match outcome {
        RegistryCandidateOutcome::Candidates(response) => Ok(response.candidates),
        RegistryCandidateOutcome::NotModified => Err(RegistryClientError::InvalidResponse(
            "unexpected not-modified candidate response without an ETag".into(),
        )),
    }
}

async fn fetch(
    plan: &RegistryAcquisitionPlan,
    token: Option<&str>,
) -> Result<RegistryArtifactTransfer, RegistryClientError> {
    let client = RegistryClient::new(&plan.index, token.map(str::to_owned))?;
    let outcome = if let Some(expected) = &plan.expected {
        client.resolve_exact(expected.release.as_str(), None).await
    } else {
        client
            .resolve(
                plan.package.organization(),
                plan.package.name(),
                &plan.requirement.to_string(),
                None,
            )
            .await
    }?;
    let response = match outcome {
        RegistryReleaseOutcome::Release(response) => response,
        RegistryReleaseOutcome::NotModified => {
            return Err(RegistryClientError::InvalidResponse(
                "unexpected not-modified registry response without an ETag".into(),
            ))
        }
    };
    let source = source_from_metadata(plan, &response.metadata)
        .map_err(RegistryClientError::InvalidResponse)?;
    let metadata = release_metadata(&response.metadata.release)
        .map_err(RegistryClientError::InvalidResponse)?;
    let artifact_bytes = client
        .download_artifact(&response.metadata.artifact, MAX_REGISTRY_ARTIFACT_BYTES)
        .await?;
    Ok(RegistryArtifactTransfer {
        package_id: response.metadata.release.package_id.clone(),
        source,
        metadata,
        artifact_bytes,
        key_envelopes: response.metadata.artifact.key_envelopes,
    })
}

fn candidate_record(
    package: &CanonicalPackageId,
    candidate: &RegistryCandidate,
) -> Result<RegistryCandidateRecord, String> {
    let source = source_from_parts(
        package,
        &candidate.release,
        &candidate.artifact.digest,
        &candidate.artifact.tree_digest,
    )?;
    let metadata = release_metadata(&candidate.release)?;
    metadata.verify_supply_chain(&candidate.release.package_id, &source)?;
    Ok(RegistryCandidateRecord {
        source,
        metadata,
        yanked: candidate.yanked,
    })
}

fn release_metadata(metadata: &RegistryReleaseCore) -> Result<RegistryReleaseMetadata, String> {
    Ok(RegistryReleaseMetadata {
        singleton: metadata.singleton,
        runmat_requirement: metadata.runmat_requirement.clone(),
        dependencies: metadata
            .dependencies
            .iter()
            .map(|dependency| {
                Ok(RegistryReleaseDependency {
                    alias: dependency.alias.clone(),
                    package: RegistryPackageReference {
                        registry: RegistryOrigin::from_str(&dependency.registry)
                            .map_err(|error| error.to_string())?,
                        namespace: dependency.namespace.clone(),
                        name: dependency.name.clone(),
                    },
                    requirement: dependency.requirement.clone(),
                    group: match dependency.group.as_str() {
                        "runtime" => DependencyGroup::Runtime,
                        "development" => DependencyGroup::Development,
                        "test" => DependencyGroup::Test,
                        _ => return Err("registry dependency group is invalid".to_string()),
                    },
                    target: dependency.target.clone(),
                    optional: dependency.optional,
                    default_features: dependency.default_features,
                    features: dependency.features.clone(),
                })
            })
            .collect::<Result<_, String>>()?,
        features: metadata.features.clone(),
        required_capabilities: metadata.required_capabilities.clone(),
        optional_capabilities: metadata.optional_capabilities.clone(),
        readme_digest: metadata.readme_digest.clone(),
        license: metadata.license.clone(),
        encryption: metadata.encryption.clone(),
        supply_chain: metadata.supply_chain.clone(),
    })
}

fn source_from_metadata(
    plan: &RegistryAcquisitionPlan,
    metadata: &RegistryReleaseResponse,
) -> Result<RegistrySourceId, String> {
    if metadata.release.namespace != plan.package.organization()
        || metadata.release.name != plan.package.name()
    {
        return Err("registry returned a different package identity".into());
    }
    source_from_parts(
        &plan.package,
        &metadata.release,
        &metadata.artifact.digest,
        &metadata.artifact.tree_digest,
    )
}

fn source_from_parts(
    package: &CanonicalPackageId,
    metadata: &RegistryReleaseCore,
    artifact_digest: &str,
    tree_digest: &str,
) -> Result<RegistrySourceId, String> {
    Ok(RegistrySourceId {
        registry_origin: RegistryOrigin::from_str(&metadata.registry)
            .map_err(|error| error.to_string())?,
        package: package.clone(),
        release: RegistryReleaseId::from_str(&metadata.release_id)
            .map_err(|error| error.to_string())?,
        version: PackageVersion::from_str(&metadata.version).map_err(|error| error.to_string())?,
        release_digest: ContentDigest::from_str(&metadata.release_digest)
            .map_err(|error| error.to_string())?,
        artifact_digest: ContentDigest::from_str(artifact_digest)
            .map_err(|error| error.to_string())?,
        tree_digest: ContentDigest::from_str(tree_digest).map_err(|error| error.to_string())?,
    })
}
